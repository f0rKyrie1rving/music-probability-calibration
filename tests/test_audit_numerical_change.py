"""Paired diagnostics preserve improvements, deterioration and lineage."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('numerical_change', ROOT / 'scripts/audit_numerical_change.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def seal(run):
    cases = sorted(p.parent.relative_to(run).as_posix() for p in (run / 'cases').rglob('case.json'))
    for case in cases:
        directory = run / case
        write(directory / 'status.json', {'state': 'complete', 'hashes': {name: module._io._sha(directory / name)
            for name in ['case.json', 'calibrators.json', 'predictions.npz', 'metrics.json']}})
    write(run / 'completed.json', {'state': 'complete', 'cases': cases,
                                  'status_hashes': {case: module._io._sha(run / case / 'status.json') for case in cases}})


class NumericalChangeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.budget, self.stability, self.portable = [self.root / name for name in ['budget', 'stability', 'portable']]
        self.output = self.root / 'audit.json'
        write(self.stability / 'config.json', {'budget_seeds': list(range(1001, 1006))})
        self.pairs = {}
        # Deliberately unrelated folder names prove that matching uses config seeds.
        for seed in range(1000, 1006):
            old = self.budget if seed == 1000 else self.stability / 'children' / ('old_' + str(2000 - seed))
            new = self.portable / 'children' / ('new_' + str(seed * 7))
            self.pairs[seed] = (old, new)
            for side, run in [('v1', old), ('v2', new)]:
                config = {'expected_source_cases': 1, 'budgets': [.25, .5, 1.0], 'budget_seed': seed,
                    'inner_seed': 99, 'inner_folds': 3, 'methods': ['raw', 'ridge_platt'],
                    'ridge_penalties': [1.0], 'identity_candidate': True}
                write(run / 'config.json', config)
                write(run / 'plans.json', {'7': [{'track_ids': ['c0', 'c1'], 'folds': [0, 1]} for _ in range(3)]})
                write(run / 'freeze.json', {'bundle_manifest_sha256': 'a' * 64, 'bundle_metadata_hashes': {'manifest.json': 'a' * 64}})
                for index, fraction in enumerate(config['budgets']):
                    directory = run / ('cases/7/example/unweighted/budget_' + str(index))
                    directory.mkdir(parents=True)
                    write(directory / 'case.json', {'id': '7/example/unweighted', 'seed': 7,
                        'representation': 'example', 'weighting': 'unweighted', 'path': 'cases/input.npz',
                        'sha256': 'b' * 64, 'budget_index': index, 'fraction': fraction,
                        'calibration_artists': 2, 'calibration_tracks': 2})
                    fits = {method: [{'fallback': False, 'reason': None} for _ in range(4)] for method in config['methods']}
                    for fit in fits['ridge_platt']:
                        fit.update(selected_penalty='identity', final_fit_fallback=False,
                            oof_scores=[{'penalty': 'identity', 'brier': .25, 'inner_fallback_count': 0, 'fold_fits': []},
                                {'penalty': 1.0, 'brier': .26, 'inner_fallback_count': 0,
                                 'fold_fits': [{'fold': fold, 'fit': {'fallback': False, 'reason': None}} for fold in range(3)]}])
                    probabilities = {method: np.full((2, 4), .5, dtype=np.float64) for method in config['methods']}
                    if side == 'v2' and seed == 1000 and index < 2:
                        probabilities['ridge_platt'][0, 0] = .4 if index == 0 else .6
                    if side == 'v2' and seed == 1000 and index == 0:
                        fits['ridge_platt'][2]['selected_penalty'] = 1.0
                    if side == 'v1' and seed == 1000 and index == 1:
                        candidate = fits['ridge_platt'][2]['oof_scores'][1]
                        candidate['inner_fallback_count'] = 1
                        candidate['fold_fits'][1]['fit'] = {'fallback': True, 'reason': 'optimizer_nonconvergence'}
                    write(directory / 'calibrators.json', fits)
                    y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
                    np.savez_compressed(directory / 'predictions.npz', ids=np.array(['e0', 'e1']),
                        artists=np.array(['a0', 'a1']), y=y, **probabilities)
                    write(directory / 'metrics.json', {method: {'macro_brier': float(np.mean((p - y) ** 2))}
                        for method, p in probabilities.items()})
                seal(run)

    def run_audit(self):
        return module.audit(self.budget, self.stability, self.portable, self.output)

    def test_all_six_plans_counted_and_full_endpoints_deduplicated(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'diagnosed')
        self.assertEqual(result['counts']['executed_cases'], 18)
        self.assertEqual(result['counts']['unique_source_subset_cases'], 13)
        self.assertEqual(result['counts']['changed_label_choices'], 1)
        self.assertEqual(result['counts']['changed_fallback_locations'], 1)
        self.assertEqual(result['counts']['evaluation_probability_values_compared'], 18 * 2 * 8)
        self.assertEqual([row['budget_seed'] for row in result['input_runs']], list(range(1000, 1006)))

    def test_improved_and_worse_brier_changes_both_retained_without_equality_gate(self):
        result = self.run_audit()
        rows = [row for case in result['changed_cases'] for row in case['methods'] if row['method'] == 'ridge_platt']
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]['delta_v2_minus_v1'], 0)
        self.assertGreater(rows[1]['delta_v2_minus_v1'], 0)
        summary = next(x for x in result['unique_summary_by_method'] if x['method'] == 'ridge_platt')
        self.assertEqual((summary['improved'], summary['deteriorated'], summary['ties']), (1, 1, 11))
        self.assertNotIn(str(self.root), self.output.read_text())
        self.assertNotIn(json.dumps(str(self.root))[1:-1], self.output.read_text())

    def test_target_or_fold_misalignment_refuses_diagnosis(self):
        run = self.pairs[1002][1]
        path = run / 'plans.json'
        plans = json.loads(path.read_text())
        plans['7'][0]['folds'] = [1, 0]
        write(path, plans)
        with self.assertRaisesRegex(ValueError, 'IDs/folds/plans changed'):
            self.run_audit()
        self.assertEqual(json.loads(self.output.read_text())['status'], 'error')

    def test_changed_source_case_hash_is_rejected(self):
        run = self.pairs[1002][1]
        path = run / 'cases/7/example/unweighted/budget_0/case.json'
        case = json.loads(path.read_text())
        case['sha256'] = 'c' * 64
        write(path, case)
        seal(run)
        with self.assertRaisesRegex(ValueError, 'source/case identity'):
            self.run_audit()

    def test_corruption_and_incomplete_execution_are_rejected(self):
        run = self.pairs[1002][1]
        path = run / 'cases/7/example/unweighted/budget_0/calibrators.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
            self.run_audit()

    def test_full_endpoint_output_difference_cannot_be_silently_deduplicated(self):
        run = self.pairs[1002][1]
        directory = run / 'cases/7/example/unweighted/budget_2'
        with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays['ridge_platt'][0, 0] = .6
        np.savez_compressed(directory / 'predictions.npz', **arrays)
        write(directory / 'metrics.json', {m: {'macro_brier': float(np.mean((arrays[m] - arrays['y']) ** 2))}
            for m in ['raw', 'ridge_platt']})
        seal(run)
        with self.assertRaisesRegex(ValueError, 'Repeated full endpoint changed'):
            self.run_audit()

    def test_saved_brier_must_match_actual_probabilities(self):
        run = self.pairs[1002][1]
        path = run / 'cases/7/example/unweighted/budget_0/metrics.json'
        metrics = json.loads(path.read_text())
        metrics['raw']['macro_brier'] += .01
        write(path, metrics)
        seal(run)
        with self.assertRaisesRegex(ValueError, 'Saved macro Brier inconsistent'):
            self.run_audit()

    def test_brier_change_below_probability_threshold_is_still_retained(self):
        run = self.pairs[1002][1]
        directory = run / 'cases/7/example/unweighted/budget_0'
        with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays['ridge_platt'][0, 0] += 5e-9
        np.savez_compressed(directory / 'predictions.npz', **arrays)
        write(directory / 'metrics.json', {m: {'macro_brier': float(np.mean((arrays[m] - arrays['y']) ** 2))}
            for m in ['raw', 'ridge_platt']})
        seal(run)
        result = self.run_audit()
        changed = next(case for case in result['changed_cases'] if case['case'].startswith('budget_seed_1002/'))
        ridge = next(row for row in changed['methods'] if row['method'] == 'ridge_platt')
        self.assertEqual(ridge['changed_probability_values'], 0)
        self.assertGreater(ridge['delta_v2_minus_v1'], module.BRIER_TOLERANCE)
        self.assertEqual(changed['changed_brier_methods'], ['ridge_platt'])


if __name__ == '__main__':
    unittest.main()
