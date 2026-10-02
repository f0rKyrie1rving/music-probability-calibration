"""Regression reports preserve claims, row identities and discrete outcomes."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_reports', PROJECT / 'scripts/compare_reports.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ReportRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference, self.actual = self.root / 'reference', self.root / 'actual'
        self.reference.mkdir()
        baseline = PROJECT / 'reports/budget_v1'
        # Small real-schema fixtures keep these corruption checks fast. The CLI
        # is separately executed against all complete published reports.
        for name in comparison.FILES:
            if name.endswith('.csv'):
                with (baseline / name).open(newline='', encoding='utf-8') as stream:
                    values = list(csv.reader(stream))[:3]
                with (self.reference / name).open('w', newline='', encoding='utf-8') as stream:
                    csv.writer(stream).writerows(values)
            else:
                value = json.loads((baseline / name).read_text(encoding='utf-8'))
                if name == 'summary.json':
                    value['cells'] = value['cells'][:2]
                (self.reference / name).write_text(json.dumps(value), encoding='utf-8')
        shutil.copytree(self.reference, self.actual)
        self.receipt = self.root / 'run/regression.json'

    def read_csv(self, name):
        with (self.actual / name).open(newline='', encoding='utf-8') as stream:
            return list(csv.reader(stream))

    def write_csv(self, name, rows):
        with (self.actual / name).open('w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)

    def mutate_csv(self, name, column, value):
        rows = self.read_csv(name)
        rows[1][rows[0].index(column)] = value
        self.write_csv(name, rows)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex((ValueError, TypeError), pattern):
            comparison.compare_reports(self.reference, self.actual, self.receipt)
        self.assertEqual(json.loads(self.receipt.read_text())['status'], 'failed')

    def test_all_seven_reports_pass_and_receipt_binds_files(self):
        result = comparison.compare_reports(self.reference, self.actual, self.receipt)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(set(result['reference_sha256']), set(comparison.FILES))
        self.assertEqual(result['reference_sha256'], result['actual_sha256'])
        self.assertGreater(result['numeric_comparisons'], 0)
        self.assertEqual(result['maximum_absolute_error'], 0)

    def test_tiny_roundoff_and_negative_zero_pass(self):
        rows = self.read_csv('case_metrics.csv')
        brier = rows[0].index('macro_brier')
        rows[1][brier] = str(float(rows[1][brier]) + 2e-10)
        rows[1][rows[0].index('delta_raw')] = '-0.0'
        self.write_csv('case_metrics.csv', rows)
        result = comparison.compare_reports(self.reference, self.actual, self.receipt)
        self.assertGreater(result['maximum_absolute_error'], 0)
        self.assertLess(result['maximum_absolute_error'], 1e-8)

    def test_real_metric_drift_is_rejected(self):
        rows = self.read_csv('label_metrics.csv')
        value = rows[0].index('brier')
        rows[1][value] = str(float(rows[1][value]) + .001)
        self.write_csv('label_metrics.csv', rows)
        self.assert_rejected('numeric drift')

    def test_count_is_exact_even_under_large_tolerance(self):
        self.mutate_csv('summary_cells.csv', 'n', '21')
        with self.assertRaisesRegex(ValueError, 'exact value/type changed'):
            comparison.compare_reports(self.reference, self.actual, self.receipt, atol=100.)

    def test_count_cannot_change_to_float_string(self):
        self.mutate_csv('summary_cells.csv', 'n', '20.0')
        self.assert_rejected('nonnegative integer')

    def test_candidate_category_is_exact(self):
        self.mutate_csv('ridge_selections.csv', 'selected_penalty', '0.010000000001')
        self.assert_rejected('exact value/type changed')

    def test_missing_file_is_rejected(self):
        (self.actual / 'fallbacks.json').unlink()
        self.assert_rejected('Missing actual report')

    def test_missing_column_is_rejected(self):
        rows = self.read_csv('label_metrics.csv')
        self.write_csv('label_metrics.csv', [row[:-1] for row in rows])
        self.assert_rejected('columns')

    def test_dropped_duplicated_and_reordered_rows_are_rejected(self):
        original = self.read_csv('case_metrics.csv')
        variants = [original[:-1], original + [original[-1]], [original[0], original[2], original[1]]]
        for variant in variants:
            with self.subTest(variant=variant):
                self.write_csv('case_metrics.csv', variant)
                self.assert_rejected('row count|duplicated row|exact value/type')

    def test_nan_and_infinite_measurements_are_rejected(self):
        for value in ['nan', 'inf', '-inf', '1e999']:
            with self.subTest(value=value):
                self.mutate_csv('label_metrics.csv', 'brier', value)
                self.assert_rejected('nonfinite')

    def test_json_integer_type_and_counts_are_exact(self):
        path = self.actual / 'summary.json'
        original = json.loads(path.read_text())
        for value in [float(original['cases']), True, str(original['cases']), original['cases'] + 1]:
            altered = {**original, 'cases': value}
            path.write_text(json.dumps(altered))
            with self.subTest(value=value):
                self.assert_rejected('exact value/type')

    def test_json_duplicate_keys_and_nonfinite_are_rejected(self):
        path = self.actual / 'summary.json'
        original = path.read_text()
        for altered in ['{"cases": 1, "cases": 2}', '{"cases": NaN}']:
            path.write_text(altered)
            with self.subTest(altered=altered):
                self.assert_rejected('duplicate JSON key|nonfinite JSON')
        path.write_text(original)

    def test_json_cell_order_and_duplicates_are_rejected(self):
        path = self.actual / 'summary.json'
        original = json.loads(path.read_text())
        for cells in [original['cells'][::-1], [original['cells'][0], original['cells'][0]]]:
            path.write_text(json.dumps({**original, 'cells': cells}))
            with self.subTest(cells=cells):
                self.assert_rejected('exact value/type|duplicated row')

    def test_null_measurement_cannot_become_zero(self):
        filename = 'example_reliability.csv'
        for folder in [self.reference, self.actual]:
            with (folder / filename).open(newline='') as stream:
                rows = list(csv.reader(stream))
            rows[1][rows[0].index('mean_probability')] = ''
            rows[1][rows[0].index('count')] = '0'
            with (folder / filename).open('w', newline='') as stream:
                csv.writer(stream).writerows(rows)
        self.mutate_csv(filename, 'mean_probability', '0.0')
        self.assert_rejected('exact value/type')

    def test_receipt_cannot_overwrite_reference_or_actual_input(self):
        original = (self.reference / 'summary.json').read_bytes()
        for out in [self.reference / 'regression.json', self.actual / 'summary.json']:
            with self.subTest(out=out), self.assertRaisesRegex(ValueError, 'Receipt cannot'):
                comparison.compare_reports(self.reference, self.actual, out)
        self.assertEqual((self.reference / 'summary.json').read_bytes(), original)

    def test_file_override_rejects_unsafe_unknown_and_duplicate_names(self):
        for files in [['../summary.json'], ['/tmp/summary.json'], ['unknown.csv'],
                      ['summary.json', 'summary.json'], []]:
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, 'supported report basenames'):
                comparison.compare_reports(self.reference, self.actual, self.receipt, files=files)

    def test_summary_only_override_does_not_require_default_csv_files(self):
        (self.actual / 'case_metrics.csv').unlink()
        result = comparison.compare_reports(self.reference, self.actual, self.receipt, files=['summary.json'])
        self.assertEqual(result['compared_files'], ['summary.json'])

    def test_explicit_stability_schemas_and_exact_counts(self):
        for name, schema in comparison.STABILITY_SCHEMAS.items():
            columns, integers, measurements, _, _ = schema
            values = []
            for index in range(2):
                row = {}
                for column in columns:
                    if column in integers:
                        row[column] = str(index + 1)
                    elif column in measurements:
                        row[column] = '0.125'
                    elif column in comparison.BOOL_COLUMNS:
                        row[column] = 'False'
                    else:
                        row[column] = {'representation': ['maest', 'mert_v0'][index],
                            'weighting': 'unweighted', 'fraction': '0.25', 'method': 'ridge_platt',
                            'label': 'ambient', 'selected_penalty': 'identity',
                            'selected_strengths': '["identity"]'}.get(column, 'fixture')
                values.append([row[column] for column in columns])
            for folder in [self.reference, self.actual]:
                with (folder / name).open('w', newline='', encoding='utf-8') as stream:
                    csv.writer(stream).writerows([columns, *values])
        for folder in [self.reference, self.actual]:
            (folder / 'controls.json').write_text(json.dumps({'full_budget_source_cases': 120,
                'historical_full_max_absolute_error': 0.0,
                'all_children_independently_verified': True}), encoding='utf-8')
        files = ['summary.json', 'controls.json', *comparison.STABILITY_SCHEMAS]
        result = comparison.compare_reports(self.reference, self.actual, self.receipt, files=files)
        self.assertEqual(result['status'], 'passed')
        self.mutate_csv('summary_cells.csv', 'outer_splits', '99')
        with self.assertRaisesRegex(ValueError, 'exact value/type changed'):
            comparison.compare_reports(self.reference, self.actual, self.receipt, atol=100., files=files)

    def test_cli_failure_returns_nonzero_and_records_failure(self):
        self.mutate_csv('ridge_selections.csv', 'label', 'wrong_label')
        result = subprocess.run([sys.executable, str(PROJECT / 'scripts/compare_reports.py'),
                                 '--reference', str(self.reference), '--actual', str(self.actual),
                                 '--out', str(self.receipt)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Report regression failed', result.stderr)
        self.assertEqual(json.loads(self.receipt.read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
