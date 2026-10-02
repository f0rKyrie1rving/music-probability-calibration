"""Portability diagnostics enumerate mismatches without granting acceptance."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stability_portability', ROOT / 'scripts/audit_stability_portability.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PortabilityAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = ROOT / 'reports/stability_v1'
        self.actual = self.root / 'actual'
        self.actual.mkdir()
        for name in module.FILES + module.CONTEXT_FILES:
            shutil.copyfile(self.reference / name, self.actual / name)
        self.out = self.root / 'audit.json'

    def csv_rows(self, name):
        with (self.actual / name).open(newline='') as stream:
            return list(csv.reader(stream))

    def save_csv(self, name, rows):
        with (self.actual / name).open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)

    def mutate_csv(self, name, field, value):
        rows = self.csv_rows(name)
        rows[1][rows[0].index(field)] = value
        self.save_csv(name, rows)

    def run_audit(self):
        return module.audit(self.reference, self.actual, self.out)

    def test_identical_complete_reports_match_and_keep_all_ridge_directions(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['difference_count'], 0)
        self.assertEqual(len(result['ridge_vs_raw_and_platt']), 18)
        self.assertEqual(result['ridge_direction_changed_cells'], 0)
        self.assertEqual(result['reference_sha256'], result['actual_sha256'])

    def test_all_numeric_and_selected_differences_are_enumerated(self):
        path = self.actual / 'summary.json'
        summary = json.loads(path.read_text())
        summary['cells'][117]['mean_brier'] += 1.619e-6
        path.write_text(json.dumps(summary))
        self.mutate_csv('case_metrics.csv', 'macro_brier', '0.3')
        self.mutate_csv('ridge_selections.csv', 'selected_penalty', '0.1')
        self.mutate_csv('ridge_selections.csv', 'inner_fallback_count', '1')
        result = self.run_audit()
        self.assertEqual(result['status'], 'not_reproduced')
        self.assertEqual(result['difference_count'], 4)
        self.assertEqual(result['difference_kinds'], {'numeric': 2, 'discrete': 2})
        self.assertTrue(any(row['field'] == 'selected_penalty' for row in result['differences']))

    def test_roundoff_within_fixed_tolerance_is_explicitly_counted(self):
        rows = self.csv_rows('case_metrics.csv')
        index = rows[0].index('macro_brier')
        rows[1][index] = str(float(rows[1][index]) + 1e-10)
        self.save_csv('case_metrics.csv', rows)
        result = self.run_audit()
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['per_file']['case_metrics.csv']['numeric_roundoff_within_tolerance'], 1)
        self.assertGreater(result['maximum_absolute_error'], 0)

    def test_missing_duplicate_reordered_or_new_identity_is_an_error(self):
        original = self.csv_rows('case_metrics.csv')
        variants = [original[:-1], original + [original[-1]], [original[0], original[2], original[1], *original[3:]]]
        unknown = [row[:] for row in original]
        unknown[1][unknown[0].index('representation')] = 'unknown'
        variants.append(unknown)
        for variant in variants:
            with self.subTest(rows=len(variant)):
                self.save_csv('case_metrics.csv', variant)
                with self.assertRaises(ValueError):
                    self.run_audit()
                self.assertEqual(json.loads(self.out.read_text())['status'], 'error')

    def test_bad_schema_nonfinite_and_missing_file_are_errors(self):
        original = self.csv_rows('case_metrics.csv')
        self.save_csv('case_metrics.csv', [row[:-1] for row in original])
        with self.assertRaises(ValueError):
            self.run_audit()
        self.save_csv('case_metrics.csv', original)
        self.mutate_csv('case_metrics.csv', 'macro_brier', 'nan')
        with self.assertRaises(ValueError):
            self.run_audit()
        (self.actual / 'case_metrics.csv').unlink()
        with self.assertRaises(OSError):
            self.run_audit()

    def test_config_must_remain_bound_to_its_freeze(self):
        path = self.actual / 'config.json'
        config = json.loads(path.read_text())
        config['budget_seeds'][0] += 100
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_ridge_direction_changes_are_reported(self):
        rows = self.csv_rows('summary_cells.csv')
        method = rows[0].index('method')
        index = rows[0].index('mean_delta_raw')
        target = next(row for row in rows[1:] if row[method] == 'ridge_platt')
        target[index] = str(-float(target[index]))
        self.save_csv('summary_cells.csv', rows)
        result = self.run_audit()
        self.assertEqual(result['status'], 'not_reproduced')
        self.assertEqual(result['ridge_direction_changed_cells'], 1)

    def test_cli_mismatch_returns_two_and_does_not_claim_acceptance(self):
        self.mutate_csv('case_metrics.csv', 'macro_brier', '0.3')
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stability_portability.py'),
                                 '--reference', str(self.reference), '--actual', str(self.actual),
                                 '--out', str(self.out)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('not_reproduced', result.stdout)
        self.assertIn('does not override', result.stdout)
        self.assertEqual(json.loads(self.out.read_text())['status'], 'not_reproduced')

    def test_cannot_overwrite_any_report_input(self):
        for out in [self.reference / 'audit.json', self.actual / 'summary.json']:
            with self.subTest(out=out), self.assertRaises(ValueError):
                module.audit(self.reference, self.actual, out)


if __name__ == '__main__':
    unittest.main()
