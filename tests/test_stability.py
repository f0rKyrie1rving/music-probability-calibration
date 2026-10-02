import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from music_calibration.data import plan_budgets
from music_calibration.io import digest, read, write
from scripts import stability


BASE = {
    "expected_source_cases": 120, "budget_seed": 2026100101,
    "budgets": [.25, .5, 1.], "inner_seed": 2026100102, "inner_folds": 3,
}


def plans_for(budget_seed, outer_seeds=(1,)):
    artists = np.array([f"artist_{i:03d}" for i in range(94)])
    result = {}
    for seed in outer_seeds:
        plans = plan_budgets(artists, seed, {**BASE, "budget_seed": budget_seed})
        for plan in plans:
            plan["track_ids"] = [f"track_{i:03d}" for i in plan["indices"]]
        result[str(seed)] = plans
    return result


class StabilitySummaryTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for seed in [1, 2]:
            for fraction in [.25, .5, 1.]:
                for draw in range(5):
                    delta = (draw - 2) * .001 if fraction < 1 else -.01
                    if seed == 2:
                        delta = -.002 if fraction < 1 else -.01
                    rows.append({"draw": draw, "seed": seed, "representation": "maest",
                        "weighting": "unweighted", "fraction": fraction, "method": "ridge_platt",
                        "macro_brier": .1 + delta, "macro_log_loss": .2 + delta,
                        "delta_raw": delta, "delta_platt": delta - .004, "fallback_labels": 0})
        return rows

    def test_full_endpoint_counted_once_and_partial_draws_preserved(self):
        splits, cells = stability.aggregate_rows(self.make_rows())
        self.assertEqual(len(splits), 6)
        self.assertEqual(len(cells), 3)
        full = next(c for c in cells if c["fraction"] == 1)
        self.assertEqual(full["draws_per_split"], 1)
        self.assertEqual(full["evaluated_distinct_cases"], 2)
        self.assertEqual(full["mean_within_split_range_raw"], 0)
        partial = next(c for c in cells if c["fraction"] == .25)
        self.assertEqual(partial["draws_per_split"], 5)
        self.assertEqual(partial["evaluated_distinct_cases"], 10)
        self.assertEqual(partial["sign_flip_splits_raw"], 1)
        self.assertEqual(partial["all_draws_improve_splits_raw"], 1)
        self.assertAlmostEqual(partial["mean_delta_raw"], -.001)
        self.assertAlmostEqual(partial["mean_within_split_range_raw"], .002)

    def test_full_control_changes_are_rejected_not_averaged(self):
        rows = self.make_rows()
        next(r for r in rows if r["fraction"] == 1 and r["draw"] == 3)["macro_brier"] += 1e-12
        with self.assertRaisesRegex(ValueError, "full-budget values"):
            stability.aggregate_rows(rows)

    def test_missing_and_duplicate_draws_are_rejected(self):
        rows = self.make_rows()
        for altered in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaisesRegex(ValueError, "Missing or duplicate draw"):
                stability.aggregate_rows(altered)
        rows[1]["draw"] = rows[0]["draw"]
        with self.assertRaisesRegex(ValueError, "Missing or duplicate draw"):
            stability.aggregate_rows(rows)

    def test_tie_does_not_create_a_sign_flip(self):
        result = stability._contrast([-.1, 0, -1e-13, 1e-13], 1e-12)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 3, 0))
        self.assertFalse(result["sign_flip"])

    def test_five_nested_plans_preserve_common_full_endpoint(self):
        original = plans_for(BASE["budget_seed"])
        children = [plans_for(seed) for seed in range(2026100201, 2026100206)]
        result = stability.check_child_plans(children, original)
        self.assertEqual(result["distinct_split_budget_subsets"], 11)
        for plans in children:
            small, middle, full = [set(p["track_ids"]) for p in plans["1"]]
            self.assertTrue(small < middle < full)

    def test_duplicate_partial_draw_and_changed_full_fold_rejected(self):
        original = plans_for(BASE["budget_seed"])
        children = [plans_for(seed) for seed in range(2026100201, 2026100206)]
        duplicated = copy.deepcopy(children)
        duplicated[1] = duplicated[0]
        with self.assertRaisesRegex(ValueError, "repeated partial subset"):
            stability.check_child_plans(duplicated, original)
        children[0]["1"][2]["folds"][0] = 99
        with self.assertRaisesRegex(ValueError, "Full-budget plan"):
            stability.check_child_plans(children, original)

    def test_old_draw_and_invalid_seeds_are_not_accepted(self):
        base = BASE.copy()
        study = read(stability.ROOT / "protocols/stability_v1.json")
        stability._validate_config(study, base)
        for seeds in [[BASE["budget_seed"], 1, 2, 3, 4], [1, 1, 2, 3, 4],
                      [1, 2, 3], [True, 2, 3, 4, 5]]:
            with self.assertRaises(ValueError):
                stability._validate_config({**study, "budget_seeds": seeds}, base)

    def test_all_requires_report_before_any_execution(self):
        with patch("sys.argv", ["stability.py", "all", "--out", "unused"]), \
                patch.object(stability, "freeze_study") as freeze:
            with self.assertRaises(SystemExit):
                stability.main()
            freeze.assert_not_called()


class StabilityFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.project = self.root / "project"
        self.data = self.project / "data/legacy_scores"
        self.data.mkdir(parents=True)
        self.out = self.root / "study"
        self.study = read(stability.ROOT / "protocols/stability_v1.json")
        for name, text in {
            "scripts/stability.py": "# study execution source\n",
            "music_calibration/__init__.py": "# unchanged engine\n",
            "protocols/stability_v1.md": "Study protocol\n",
            "protocols/budget_v1.md": "Original protocol\n",
        }.items():
            p = self.project / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        write(self.project / "protocols/stability_v1.json", self.study)
        write(self.project / "protocols/budget_v1.json", BASE)
        self.manifest = {"cases": [{"id": i} for i in range(120)]}
        write(self.data / "manifest.json", self.manifest)
        write(self.project / "reports/budget_v1/freeze.json", {
            "source_hashes": {"music_calibration/__init__.py": digest(self.project / "music_calibration/__init__.py")},
            "local_hashes": {"config.json": digest(self.project / "protocols/budget_v1.json"),
                             "protocol.md": digest(self.project / "protocols/budget_v1.md")},
            "bundle_manifest_sha256": digest(self.data / "manifest.json")})
        write(self.project / "reports/budget_v1/plans.json", plans_for(BASE["budget_seed"], range(20)))
        self.root_patch = patch.object(stability, "ROOT", self.project)
        self.root_patch.start()
        self.audit_patch = patch.object(stability, "audit_bundle", return_value=(self.manifest, {"cases": 120, "outer_splits": 20}))
        self.audit_patch.start()
        self.freeze_patch = patch.object(stability, "freeze_child", side_effect=self.freeze_child)
        self.freeze_mock = self.freeze_patch.start()
        self.verify_patch = patch.object(stability, "verify_frozen", side_effect=self.verify_child)
        self.verify_patch.start()

    def tearDown(self):
        self.verify_patch.stop()
        self.freeze_patch.stop()
        self.audit_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def freeze_child(self, data, out, config_path):
        out = Path(out)
        config = read(config_path)
        write(out / "config.json", config)
        write(out / "plans.json", plans_for(config["budget_seed"], range(20)))
        write(out / "freeze.json", {"unit_test_child": True})

    def verify_child(self, out):
        return self.data, self.manifest, read(out / "config.json"), read(out / "plans.json")

    def test_all_children_frozen_before_any_fitting_and_verified(self):
        with patch.object(stability, "run_child") as fit:
            stability.freeze_study(self.data, self.out)
            fit.assert_not_called()
        self.assertEqual(self.freeze_mock.call_count, 5)
        record, _, _ = stability.verify_study_freeze(self.out)
        self.assertEqual(record["executed_cases"], 1800)
        self.assertEqual(record["distinct_subset_cases"], 1320)
        self.assertFalse(record["original_draw_included"])
        self.assertEqual(read(self.out / "plan_audit.json")["distinct_subset_cases"], 1320)

    def test_existing_study_and_bundle_containment_refused(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            stability.freeze_study(self.data, self.out)
        with self.assertRaisesRegex(ValueError, "outside the input bundle"):
            stability.freeze_study(self.data, self.data / "study")
        self.freeze_mock.assert_not_called()

    def test_original_engine_changed_refused_before_output(self):
        (self.project / "music_calibration/__init__.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Original budget_v1 engine changed"):
            stability.freeze_study(self.data, self.out)
        self.assertFalse(self.out.exists())

    def test_changed_ridge_grid_is_rejected_before_creating_output(self):
        original = read(self.project / "protocols/budget_v1.json")
        altered = {**original, "ridge_penalties": [0, 10, 100]}
        write(self.project / "protocols/budget_v1.json", altered)
        with self.assertRaisesRegex(ValueError, "Original budget_v1 config/protocol changed"):
            stability.freeze_study(self.data, self.out)
        self.assertFalse(self.out.exists())
        self.freeze_mock.assert_not_called()

    def test_base_config_source_formatting_matches_original_serialized_receipt(self):
        # Original freeze stored a normalized JSON serialization, not source bytes.
        config_path = self.project / "protocols/budget_v1.json"
        config_path.write_text(json.dumps(BASE, separators=(",", ":")), encoding="utf-8")
        stability.freeze_study(self.data, self.out)
        self.assertEqual(read(self.out / "base_config.json"), BASE)

    def test_changed_base_protocol_is_rejected_before_creating_output(self):
        (self.project / "protocols/budget_v1.md").write_text("Changed selection protocol\n")
        with self.assertRaisesRegex(ValueError, "Original budget_v1 config/protocol changed"):
            stability.freeze_study(self.data, self.out)
        self.assertFalse(self.out.exists())

    def test_new_script_tampering_and_child_config_change_rejected(self):
        stability.freeze_study(self.data, self.out)
        script = self.project / "scripts/stability.py"
        content = script.read_bytes()
        script.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen study source"):
            stability.verify_study_freeze(self.out)
        script.write_bytes(content)
        child = self.out / "children/draw_00"
        config = read(child / "config.json")
        config["inner_seed"] += 1
        write(child / "config.json", config)
        with self.assertRaisesRegex(ValueError, "changed more than budget_seed"):
            stability.verify_study_freeze(self.out)


if __name__ == "__main__":
    unittest.main()
