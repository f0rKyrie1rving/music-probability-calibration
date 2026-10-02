"""Portable v2 exports bind all eval/OOF probabilities and exact fit policy."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('portable_runs', ROOT / 'scripts/compare_portable_runs.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class PortableRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference, self.actual = self.root / 'reference', self.root / 'actual'
        self.reference.mkdir()
        self.receipt = self.root / 'result.json'
        self.runs = {}
        for name in module.RUNS:
            run = self.root / 'runs' / name
            self.runs[name] = run
            case_path = 'cases/2026092800/maest/ambient_balanced/budget_0'
            case = run / case_path
            case.mkdir(parents=True)
            write(run / 'config.json', {'expected_source_cases': 1, 'budgets': [0.25],
                'methods': module.METHODS, 'ridge_penalties': [0.0, 0.001, 0.01, 0.1, 1.0], 'inner_folds': 3})
            write(run / 'plans.json', {'2026092800': [{'n_tracks': 3, 'track_ids': ['c1', 'c2', 'c3'], 'folds': [0, 1, 2]}]})
            write(case / 'case.json', {'id': '2026092800/maest/ambient_balanced', 'seed': 2026092800,
                'budget_index': 0, 'calibration_tracks': 3})
            fits = {method: [{'fallback': False, 'reason': None} for _ in module.LABELS] for method in module.METHODS}
            for fit in fits['ridge_platt']:
                fit.update(selected_penalty='identity', oof_predictions={c: [.2, .4, .7] for c in module.CANDIDATES},
                    oof_scores=[{'penalty': c if c == 'identity' else float(c), 'brier': .2,
                        'fold_fits': [] if c == 'identity' else [{'fold': k, 'fit': {'fallback': False, 'reason': None}} for k in range(3)]}
                        for c in module.CANDIDATES])
            write(case / 'calibrators.json', fits)
            probabilities = {m: np.array([[.2, .3, .4, .5], [.6, .7, .8, .9]]) for m in module.METHODS}
            np.savez_compressed(case / 'predictions.npz', ids=np.array(['e1', 'e2']), artists=np.array(['a1', 'a2']),
                                y=np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), **probabilities)
            write(case / 'metrics.json', {})
            write(case / 'status.json', {'state': 'complete', 'hashes': {f: module._reports._sha(case / f)
                for f in ['case.json', 'calibrators.json', 'predictions.npz', 'metrics.json']}})
            write(run / 'completed.json', {'state': 'complete', 'cases': [case_path],
                'status_hashes': {case_path: module._reports._sha(case / 'status.json')}})
            report = self.reference / name
            report.mkdir()
            for filename in module._reports.FILES:
                source = ROOT / 'reports/budget_v1' / filename
                if filename.endswith('.csv'):
                    with source.open(newline='') as stream:
                        rows = list(csv.reader(stream))
                    # The first seven case-metric rows are one case's seven methods.
                    rows = rows[:8] if filename == 'case_metrics.csv' else rows[:3]
                    with (report / filename).open('w', newline='') as stream:
                        csv.writer(stream).writerows(rows)
                else:
                    data = json.loads(source.read_text())
                    if filename == 'summary.json':
                        data['cells'] = data['cells'][:2]
                    write(report / filename, data)
        self.export = module.export_run_signatures(self.runs, self.reference)
        shutil.copytree(self.reference, self.actual)

    def index(self):
        return json.loads((self.actual / module.INDEX_FILE).read_text())

    def save_index(self, index):
        write(self.actual / module.INDEX_FILE, index)

    def mutate_array(self, key, index, value=None, delta=None, dtype=None):
        path = self.actual / module.ARRAY_FILE
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if dtype is not None:
            arrays[key] = arrays[key].astype(dtype)
        else:
            arrays[key][index] = value if value is not None else arrays[key][index] + delta
        np.savez_compressed(path, **arrays)

    def compare(self, **kwargs):
        return module.compare_portable_runs(self.reference, self.actual, self.receipt, **kwargs)

    def test_export_and_full_compare_cover_all_six_runs_and_candidates(self):
        result = self.compare()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['fit_records'], 6 * 7 * 4)
        self.assertEqual(result['probabilities']['eval_probabilities']['values_compared'], 6 * 7 * 4 * 2)
        self.assertEqual(result['probabilities']['oof_probabilities']['values_compared'], 6 * 4 * 6 * 3)
        self.assertEqual(set(result['reports']), set(module.RUNS))
        self.assertEqual(self.export['records'], result['fit_records'])

    def test_unselected_candidate_oof_drift_is_rejected(self):
        ridge = next(r for r in self.index()['records'] if '/ridge_platt/' in r['id'])
        candidate = next(c for c in ridge['oof'] if c['candidate'] == '0.1')
        self.mutate_array('oof_probabilities', candidate['start'], delta=2e-6)
        with self.assertRaisesRegex(ValueError, 'oof_probabilities: probability drift'):
            self.compare()
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['probabilities']['oof_probabilities']['worst_candidate'], '0.1')

    def test_eval_probability_drift_is_rejected(self):
        self.mutate_array('eval_probabilities', 3, delta=2e-6)
        with self.assertRaisesRegex(ValueError, 'eval_probabilities: probability drift'):
            self.compare()

    def test_tiny_probability_and_oof_score_roundoff_is_allowed(self):
        self.mutate_array('eval_probabilities', 3, delta=1e-10)
        self.mutate_array('oof_probabilities', 2, delta=1e-10)
        index = self.index()
        next(r for r in index['records'] if r['oof_scores'])['oof_scores'][0]['brier'] += 1e-10
        self.save_index(index)
        self.assertEqual(self.compare()['status'], 'passed')

    def test_selected_penalty_and_each_inner_fallback_are_exact(self):
        original = self.index()
        for field in ['selected_penalty', 'candidate_fold_fallbacks']:
            index = json.loads(json.dumps(original))
            record = next(r for r in index['records'] if r['oof_scores'])
            if field == 'selected_penalty':
                record[field] = '0.1'
            else:
                record[field][-1].update(fallback=True, reason='single_class_calibration')
            self.save_index(index)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exact value/type changed'):
                self.compare()

    def test_numerical_failure_fallback_is_rejected_even_if_identical(self):
        index = self.index()
        record = next(r for r in index['records'] if r['candidate_fold_fallbacks'])
        record['candidate_fold_fallbacks'][0].update(fallback=True, reason='optimizer_nonconvergence')
        self.save_index(index)
        write(self.reference / module.INDEX_FILE, index)
        with self.assertRaisesRegex(ValueError, 'numerical-failure fallback'):
            self.compare()

    def test_oof_score_drift_is_rejected(self):
        index = self.index()
        next(r for r in index['records'] if r['oof_scores'])['oof_scores'][3]['brier'] += 1e-5
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, 'numeric drift'):
            self.compare()

    def test_bad_dtype_nonfinite_and_probability_bounds_are_rejected(self):
        original = (self.actual / module.ARRAY_FILE).read_bytes()
        for kwargs in [{'dtype': np.float32}, {'value': np.nan}, {'value': 1.2}]:
            (self.actual / module.ARRAY_FILE).write_bytes(original)
            self.mutate_array('eval_probabilities', 0, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'float64|nonfinite|out-of-range'):
                self.compare()

    def test_duplicate_reordered_and_missing_ids_are_rejected(self):
        original = self.index()
        for kind in ['duplicate', 'reorder', 'missing']:
            index = json.loads(json.dumps(original))
            if kind == 'duplicate':
                index['records'][1]['id'] = index['records'][0]['id']
            elif kind == 'reorder':
                index['records'][0], index['records'][1] = index['records'][1], index['records'][0]
            else:
                index['records'].pop()
            self.save_index(index)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.compare()

    def test_slice_overlap_and_unindexed_tail_are_rejected(self):
        original = self.index()
        for change in [-1, 1]:
            index = json.loads(json.dumps(original))
            index['records'][1]['eval_start'] += change
            self.save_index(index)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'gap/overlap'):
                self.compare()

    def test_row_target_and_fold_alignment_hashes_are_exact(self):
        original = self.index()
        for key in ['eval_ids_sha256', 'cal_ids_sha256', 'eval_targets_sha256', 'cal_folds_sha256']:
            index = json.loads(json.dumps(original))
            for record in index['records']:
                record[key] = 'a' * 64
            self.save_index(index)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'exact value/type changed'):
                self.compare()

    def test_strict_reports_reject_even_unselected_diagnostic_drift(self):
        path = self.actual / module.RUNS[0] / 'ridge_selections.csv'
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        rows[1][rows[0].index('inner_fallback_count')] = '1'
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, 'exact value/type changed'):
            self.compare()

    def test_reports_only_explicitly_omits_signatures_and_claim(self):
        (self.actual / module.ARRAY_FILE).unlink()
        (self.actual / module.INDEX_FILE).unlink()
        result = self.compare(reports_only=True)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['reports_only'])
        self.assertNotIn('probabilities', result)
        self.assertIn('signatures were not compared', result['scope'])
        with self.assertRaises(OSError):
            self.compare()

    def test_export_refuses_changed_source_or_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'already exist'):
            module.export_run_signatures(self.runs, self.reference)
        path = self.runs[module.RUNS[0]] / 'cases/2026092800/maest/ambient_balanced/budget_0/calibrators.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed calibrators'):
            module.export_run_signatures(self.runs, self.root / 'new_export')

    def test_receipt_cannot_overwrite_inputs(self):
        for out in [self.reference / 'audit.json', self.actual / module.ARRAY_FILE]:
            with self.subTest(out=out), self.assertRaises(ValueError):
                module.compare_portable_runs(self.reference, self.actual, out)

    def test_cli_returns_failure_for_probability_drift(self):
        self.mutate_array('eval_probabilities', 0, delta=.001)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/compare_portable_runs.py'),
            '--reference', str(self.reference), '--actual', str(self.actual), '--out', str(self.receipt)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('probability drift', result.stdout)


if __name__ == '__main__':
    unittest.main()
