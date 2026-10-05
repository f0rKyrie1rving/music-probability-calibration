"""Integrity and leakage regression tests for the published confirmation bundle."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_public_verifier', ROOT / 'scripts/verify_fresh_confirmation.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class FreshPublicTests(unittest.TestCase):
    def test_published_hashes_and_original_gzip_bytes(self):
        result = v.verify_integrity(ROOT)
        self.assertEqual(result['calibration_tracks'], 143)
        self.assertEqual(result['evaluation_tracks'], 637)

    def test_modified_score_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data/fresh_confirmation_v1'
            data.mkdir(parents=True)
            scores = data / 'calibration_scores.npz'
            scores.write_bytes(b'original numeric bytes')
            name = 'data/fresh_confirmation_v1/calibration_scores.npz'
            old_hash = v.sha(scores)
            manifest = {'schema_version': 1, 'source_case_count': 440,
                        'exports': {name: {'operation': 'exact bytes', 'original_sha256': old_hash, 'export_sha256': old_hash}},
                        'numerical_source_hashes': {}}
            (data / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            scores.write_bytes(b'changed numeric bytes')
            with self.assertRaisesRegex(ValueError, 'Public artifact changed'):
                v.verify_integrity(root)

    def test_manifest_cannot_read_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                v.checked_path(Path(tmp), '../outside.json')
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                v.checked_path(Path(tmp), '/outside.json')

    def test_evaluation_labels_and_artist_leakage_rejected(self):
        data = ROOT / 'data/fresh_confirmation_v1'
        cal = v.load(data / 'calibration_scores.npz')
        ev = v.load(data / 'evaluation_inputs.npz')
        labels = v.load(data / 'evaluation_labels.npz')
        config = v.read(ROOT / 'research/fresh_confirmation_v1/config.json')
        with self.assertRaisesRegex(ValueError, 'contains labels'):
            v.validate_arrays(cal, {**ev, 'y': labels['y']}, labels, config)
        changed_artists = ev['artists'].copy()
        changed_artists[0] = cal['artists'][0]
        with self.assertRaisesRegex(ValueError, 'artist overlap'):
            v.validate_arrays(cal, {**ev, 'artists': changed_artists}, labels, config)
        swapped = {**labels, 'ids': np.roll(labels['ids'], 1)}
        with self.assertRaisesRegex(ValueError, 'row order'):
            v.validate_arrays(cal, ev, swapped, config)


if __name__ == '__main__':
    unittest.main()
