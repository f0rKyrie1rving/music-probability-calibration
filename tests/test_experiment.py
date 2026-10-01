import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from music_calibration import experiment
from music_calibration.data import audit_bundle
from music_calibration.io import digest, read, write


class ExperimentGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        (self.project / "music_calibration").mkdir(parents=True)
        (self.project / "protocols").mkdir()
        (self.project / "music_calibration" / "__init__.py").write_text("# frozen source\n")
        (self.project / "protocols" / "budget_v1.md").write_text("Frozen test protocol\n")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        cases = [{"id": f"1/{representation}/{weighting}", "seed": 1,
                  "representation": representation, "weighting": weighting}
                 for representation in ["maest", "mert_v0", "mert_v1"]
                 for weighting in ["ambient_balanced", "unweighted"]]
        self.case = cases[0]
        self.manifest = {"cases": cases}
        write(self.bundle / "manifest.json", self.manifest)
        self.config = {"expected_source_cases": 6, "budgets": [.5, 1.],
                       "budget_seed": 2, "inner_seed": 3, "inner_folds": 3}
        write(self.project / "protocols" / "budget_v1.json", self.config)
        self.arrays = {"cal_ids": np.array([f"track{i}" for i in range(6)]),
                       "cal_artists": np.array([f"artist{i}" for i in range(6)])}
        self.out = self.root / "run"
        self.root_patch = patch.object(experiment, "ROOT", self.project)
        self.root_patch.start()
        self.audit_patch = patch.object(experiment, "audit_bundle", return_value=(self.manifest, {}))
        self.audit_patch.start()
        self.load_patch = patch.object(experiment, "load_case", return_value=self.arrays)
        self.load_patch.start()

    def tearDown(self):
        self.load_patch.stop()
        self.audit_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def test_freeze_refuses_any_existing_output_directory(self):
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("original result")
        with self.assertRaises(FileExistsError):
            experiment.freeze(self.bundle, self.out)
        self.assertEqual(sentinel.read_text(), "original result")

    def test_freeze_records_and_verifies_source_and_plans(self):
        experiment.freeze(self.bundle, self.out)
        bundle, manifest, config, plans = experiment.verify_frozen(self.out)
        self.assertEqual(bundle, self.bundle.resolve())
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(config, self.config)
        self.assertEqual(len(plans["1"]), 2)
        receipt = read(self.out / "freeze.json")
        self.assertIn("music_calibration/__init__.py", receipt["source_hashes"])
        self.assertEqual(plans["1"][1]["track_ids"], self.arrays["cal_ids"].tolist())

    def test_freeze_rejects_missing_design_cell_before_creating_output(self):
        self.manifest["cases"].pop()
        self.config["expected_source_cases"] = 5
        write(self.project / "protocols" / "budget_v1.json", self.config)
        with self.assertRaisesRegex(ValueError, "Missing representation/weighting"):
            experiment.freeze(self.bundle, self.out)
        self.assertFalse(self.out.exists())

    def test_freeze_cannot_write_inside_input_bundle(self):
        with self.assertRaisesRegex(ValueError, "outside the input bundle"):
            experiment.freeze(self.bundle, self.bundle / "run")
        self.assertFalse((self.bundle / "run").exists())

    def test_case_identity_cannot_escape_output_root(self):
        for identifier in ["../../outside", "/tmp/absolute_case", "1/maest/../../../outside"]:
            with self.subTest(identifier=identifier):
                manifest = {"schema_version": 1,
                            "labels": ["electronic", "pop", "ambient", "rock"],
                            "provenance_hashes": {},
                            "cases": [{**self.case, "id": identifier}]}
                write(self.bundle / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "Invalid case identity"):
                    audit_bundle(self.bundle)
        self.assertFalse((self.root / "outside").exists())

    def test_frozen_plan_tampering_is_rejected(self):
        experiment.freeze(self.bundle, self.out)
        altered = read(self.out / "plans.json")
        altered["1"][0]["folds"][0] = 99
        write(self.out / "plans.json", altered)
        with self.assertRaisesRegex(ValueError, "Frozen plan changed"):
            experiment.verify_frozen(self.out)

    def test_snapshot_and_live_source_tampering_are_rejected(self):
        experiment.freeze(self.bundle, self.out)
        source = self.project / "music_calibration" / "__init__.py"
        original = source.read_bytes()
        for path in [source, self.out / "source" / "music_calibration" / "__init__.py"]:
            with self.subTest(path=path):
                path.write_bytes(b"# changed source\n")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    experiment.verify_frozen(self.out)
                path.write_bytes(original)

    def test_added_live_or_snapshot_modules_are_rejected(self):
        experiment.freeze(self.bundle, self.out)
        roots = [self.project, self.out / "source"]
        for root in roots:
            for relative in ["extra.py", "nested/extra.py"]:
                path = root / "music_calibration" / relative
                with self.subTest(path=path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# added after freeze\n")
                    with self.assertRaisesRegex(ValueError, "source inventory changed"):
                        experiment.verify_frozen(self.out)
                    path.unlink()

    def test_removed_live_or_snapshot_modules_are_rejected(self):
        experiment.freeze(self.bundle, self.out)
        for root in [self.project, self.out / "source"]:
            path = root / "music_calibration" / "__init__.py"
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, "source inventory changed"):
                    experiment.verify_frozen(self.out)
                path.write_bytes(original)

    def test_incomplete_case_is_preserved_and_cannot_resume(self):
        destination = self.out / "cases" / self.case["id"] / "budget_0"
        write(destination / "status.json", {"state": "failed", "error": "original error"})
        before = digest(destination / "status.json")
        prepared = (self.bundle, self.manifest, {"budgets": [1.]}, {"1": [{}]})
        with patch.object(experiment, "verify_frozen", return_value=prepared):
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                experiment.run(self.out)
        self.assertEqual(digest(destination / "status.json"), before)

    def test_fitting_error_is_recorded_and_raised(self):
        self.arrays.update({"cal_logits": np.zeros((6, 4)), "cal_y": np.zeros((6, 4)),
                            "offsets": np.zeros(4)})
        plan = {"indices": list(range(6)), "track_ids": self.arrays["cal_ids"].tolist(),
                "folds": [0, 1, 2, 0, 1, 2]}
        prepared = (self.bundle, self.manifest, {"budgets": [1.]}, {"1": [plan]})
        with patch.object(experiment, "verify_frozen", return_value=prepared), \
                patch.object(experiment, "fit_methods", side_effect=RuntimeError("implementation error")):
            with self.assertRaisesRegex(RuntimeError, "implementation error"):
                experiment.run(self.out)
        status = read(self.out / "cases" / self.case["id"] / "budget_0" / "status.json")
        self.assertEqual(status["state"], "failed")
        self.assertIn("implementation error", status["error"])


if __name__ == "__main__":
    unittest.main()
