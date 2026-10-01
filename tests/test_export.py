"""Protect the migration boundaries: artist leakage, row order and fit-only priors."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('export_legacy',
    Path(__file__).resolve().parents[1] / 'scripts/export_legacy.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class ExportGuards(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([f'track_{i}' for i in range(12)])
        self.artists = np.repeat([f'artist_{i}' for i in range(6)], 2)
        self.y = np.tile([[0, 1, 0, 1], [1, 0, 1, 0]], (6, 1))
        self.split = {'indices': {'fit': [0, 1, 2, 3],
                                 'calibration': [4, 5, 6, 7],
                                 'evaluation': [8, 9, 10, 11]}}

    def test_disjoint_partition_is_accepted(self):
        result = legacy.check_split(self.split, self.ids, self.artists, self.y)
        self.assertEqual(len(result['fit']), 4)

    def test_cross_split_artist_leakage_is_rejected(self):
        self.artists[4:6] = self.artists[0]
        with self.assertRaisesRegex(ValueError, 'Artist leakage'):
            legacy.check_split(self.split, self.ids, self.artists, self.y)

    def test_repeated_and_omitted_track_is_rejected(self):
        self.split['indices']['evaluation'][-1] = 10
        with self.assertRaisesRegex(ValueError, 'Repeated split row'):
            legacy.check_split(self.split, self.ids, self.artists, self.y)

    def test_same_shape_reordered_rows_are_rejected(self):
        idx = np.array([0, 1, 2, 3])
        saved = {'ids': self.ids[idx[::-1]], 'artists': self.artists[idx],
                 'y': self.y[idx], 'logits': np.zeros((4, 4))}
        with self.assertRaisesRegex(ValueError, 'Misaligned saved ids'):
            legacy.check_rows(saved, idx, self.ids, self.artists, self.y)

    def test_offset_matches_classifier_fit_counts_only(self):
        fit_y = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                          [1, 0, 0, 0], [0, 1, 0, 1]])
        heads = {'n_fit': np.array(4), 'positive_fit': fit_y.sum(0)}
        offset = legacy.derived_offsets(heads, fit_y, 'ambient_balanced')
        np.testing.assert_allclose(offset, [0, 0, np.log(1 / 3), 0])
        heads['positive_fit'] = np.array([2, 2, 2, 2])
        with self.assertRaisesRegex(ValueError, 'positive counts'):
            legacy.derived_offsets(heads, fit_y, 'ambient_balanced')

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'marker'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                legacy.export('/missing/old/project', directory)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
