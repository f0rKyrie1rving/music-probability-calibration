"""Data integrity, artist grouping and calibration/evaluation separation guards."""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from music_calibration.data import audit_bundle, plan_budgets
from music_calibration.experiment import fit_methods
from music_calibration.verify import Checks, _map, _metrics, _ridge


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class BundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arrays = {}
        for part, count in [('fit', 4), ('cal', 8), ('eval', 4)]:
            self.arrays[part + '_ids'] = np.array([f'{part}_track_{i}' for i in range(count)])
            self.arrays[part + '_artists'] = np.array([f'{part}_artist_{i//2}' for i in range(count)])
            self.arrays[part + '_y'] = np.tile([[0, 1, 0, 1], [1, 0, 1, 0]], (count // 2, 1))
            if part != 'fit':
                self.arrays[part + '_logits'] = np.zeros((count, 4))
        self.arrays['offsets'] = np.zeros(4)
        for method in ['raw', 'platt', 'temperature', 'prior_offset']:
            self.arrays['reference_' + method] = np.full((4, 4), .5)
        self.case = {'id': '7/maest/unweighted', 'seed': 7, 'representation': 'maest',
                     'weighting': 'unweighted', 'path': 'case.npz'}
        self.provenance = self.root / 'source.json'
        write_json(self.provenance, {'source': 'fixture'})
        self.manifest = {'schema_version': 1, 'labels': ['electronic', 'pop', 'ambient', 'rock'],
                         'cases': [self.case], 'provenance_hashes': {'source.json': digest(self.provenance)}}
        self.update_case()

    def update_case(self):
        np.savez_compressed(self.root / 'case.npz', **self.arrays)
        self.case['sha256'] = digest(self.root / 'case.npz')
        write_json(self.root / 'manifest.json', self.manifest)

    def test_valid_bundle_is_accepted(self):
        _, audit = audit_bundle(self.root)
        self.assertEqual(audit['unique_tracks'], 16)
        self.assertTrue(audit['all_hashes_and_artist_boundaries_valid'])

    def test_modified_score_file_is_rejected(self):
        with (self.root / 'case.npz').open('ab') as stream:
            stream.write(b'altered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            audit_bundle(self.root)

    def test_modified_provenance_file_is_rejected(self):
        write_json(self.provenance, {'source': 'altered'})
        with self.assertRaisesRegex(ValueError, 'provenance|Provenance|metadata|Metadata'):
            audit_bundle(self.root)

    def test_group_leakage_is_rejected_even_with_valid_checksum(self):
        self.arrays['eval_artists'][0] = self.arrays['fit_artists'][0]
        self.update_case()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_bundle(self.root)

    def test_paired_target_misalignment_is_rejected(self):
        other = {key: value.copy() for key, value in self.arrays.items()}
        other['eval_y'][0, 0] = 1 - other['eval_y'][0, 0]
        np.savez_compressed(self.root / 'other.npz', **other)
        self.manifest['cases'].append({**self.case, 'id': '7/mert_v0/unweighted',
            'representation': 'mert_v0', 'path': 'other.npz', 'sha256': digest(self.root / 'other.npz')})
        write_json(self.root / 'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'different split rows'):
            audit_bundle(self.root)


class BudgetAndFittingTests(unittest.TestCase):
    def test_budgets_are_nested_whole_artists_with_disjoint_inner_folds(self):
        artists = np.repeat([f'artist_{i:02d}' for i in range(20)], [1, 3] * 10)
        config = {'budget_seed': 19, 'inner_seed': 23, 'budgets': [.25, .5, 1.], 'inner_folds': 3}
        plans = plan_budgets(artists, 7, config)
        self.assertEqual(plans, plan_budgets(artists, 7, config))
        previous = set()
        for plan in plans:
            indices, folds = np.asarray(plan['indices']), np.asarray(plan['folds'])
            chosen = set(plan['artist_ids'])
            self.assertTrue(previous.issubset(chosen))
            np.testing.assert_array_equal(indices, np.flatnonzero(np.isin(artists, list(chosen))))
            for artist in chosen:
                self.assertEqual(len(set(folds[artists[indices] == artist])), 1)
            for fold in np.unique(folds):
                self.assertFalse(set(artists[indices[folds == fold]]) & set(artists[indices[folds != fold]]))
            previous = chosen
        self.assertEqual(previous, set(artists))

    def test_evaluation_labels_cannot_enter_fitting_or_selection(self):
        rng = np.random.default_rng(41)
        data = {'cal_logits': rng.normal(size=(30, 4)), 'cal_y': rng.integers(0, 2, size=(30, 4)),
                'eval_y': rng.integers(0, 2, size=(20, 4))}
        config = {'methods': ['raw', 'platt', 'ridge_platt'],
                  'ridge_penalties': [0., .1], 'selection_tie_tolerance': 1e-12}
        folds = np.repeat(np.arange(3), 10)
        before = fit_methods(data['cal_logits'], data['cal_y'], folds, np.zeros(4), config)
        data['eval_y'][:] = 1 - data['eval_y']
        after = fit_methods(data['cal_logits'], data['cal_y'], folds, np.zeros(4), config)
        self.assertEqual(before, after)
        self.assertFalse(any('eval' in name for name in inspect.signature(fit_methods).parameters))
        for label, record in enumerate(before['ridge_platt']):
            _ridge(Checks(), data['cal_logits'][:, label], data['cal_y'][:, label], folds, record, config)

    def test_independent_verifier_rejects_wrong_oof_selection(self):
        rng = np.random.default_rng(43)
        z, y = rng.normal(size=(24, 4)), rng.integers(0, 2, size=(24, 4))
        folds = np.repeat(np.arange(3), 8)
        config = {'methods': ['ridge_platt'], 'ridge_penalties': [.1], 'selection_tie_tolerance': 1e-12}
        record = fit_methods(z, y, folds, np.zeros(4), config)['ridge_platt'][0]
        record['selected_penalty'] = 999
        with self.assertRaisesRegex(ValueError, 'tie selection'):
            _ridge(Checks(), z[:, 0], y[:, 0], folds, record, config)

    def test_verifier_metrics_handle_empty_bins_and_zero_one_probabilities(self):
        y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        scores = _metrics(y, y.astype(float))
        self.assertEqual(scores['macro_brier'], 0.)
        self.assertTrue(np.isfinite(scores['macro_log_loss']))
        self.assertEqual(scores['reliability_5'][0][2]['count'], 0)
        self.assertIsNone(scores['reliability_5'][0][2]['mean_probability'])
        self.assertEqual(scores['reliability_5'][0][-1]['count'], 1)
        np.testing.assert_array_equal(_map(np.array([-10., 10.]), {'method': 'isotonic',
            'x_thresholds': [-1., 1.], 'y_thresholds': [.2, .8]}), [.2, .8])


if __name__ == '__main__':
    unittest.main()
