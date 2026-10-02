"""Failure-injection tests for the automation, using the unchanged experiment loop."""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from music_calibration import experiment
from music_calibration.io import write, read, digest
from scripts import reproduce as runner


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.project = self.root / "project"
        (self.project / "music_calibration").mkdir(parents=True)
        (self.project / "protocols").mkdir()
        (self.project / "music_calibration/__init__.py").write_text("# frozen fixture\n")
        (self.project / "protocols/budget_v1.md").write_text("test protocol\n")
        self.bundle = self.root / "data"
        self.bundle.mkdir()
        self.manifest = {"cases": [{"id": f"1/{rep}/{weight}", "seed": 1,
            "representation": rep, "weighting": weight} for rep in ["maest", "mert_v0", "mert_v1"]
            for weight in ["ambient_balanced", "unweighted"]]}
        write(self.bundle / "manifest.json", self.manifest)
        config = {"expected_source_cases": 6, "budgets": [1.], "budget_seed": 2,
                  "inner_seed": 3, "inner_folds": 3, "historical_prediction_tolerance": 1e-8,
                  "methods": ["raw", "prior_offset", "platt", "temperature"]}
        write(self.project / "protocols/budget_v1.json", config)
        self.arrays = {"cal_ids": np.array([f"cal{i}" for i in range(6)]),
            "cal_artists": np.array([f"ca{i}" for i in range(6)]),
            "cal_logits": np.zeros((6, 4)), "cal_y": np.tile([0, 1, 0, 1, 0, 1], (4, 1)).T,
            "eval_ids": np.array([f"eval{i}" for i in range(4)]),
            "eval_artists": np.array([f"ea{i}" for i in range(4)]),
            "eval_logits": np.zeros((4, 4)), "eval_y": np.tile([0, 1, 0, 1], (4, 1)).T,
            "offsets": np.zeros(4)}
        self.arrays.update({"reference_" + method: np.full((4, 4), .5) for method in config["methods"]})
        self.out, self.report = self.root / "run", self.root / "report"
        self.patches = [patch.object(experiment, "ROOT", self.project),
            patch.object(runner, "ROOT", self.project),
            patch.object(experiment, "audit_bundle", return_value=(self.manifest, {})),
            patch.object(experiment, "load_case", return_value=self.arrays),
            patch.object(runner, "summarize", side_effect=self.summarize),
            patch.object(runner, "verify", side_effect=self.verify)]
        for item in self.patches:
            item.start()
        self.addCleanup(self.temp.cleanup)
        for item in self.patches:
            self.addCleanup(item.stop)

    def summarize(self, out, report):
        summary = {"cases": len(read(out / "completed.json")["cases"])}
        write(out / "summary.json", summary)
        write(report / "summary.json", summary)

    def verify(self, out):
        self.assertEqual(len(read(out / "completed.json")["cases"]), 6)
        result = {"status": "passed", "cases": 6}
        write(out / "verification.json", result)
        return result

    def run_it(self, **kwargs):
        return runner.reproduce(self.bundle, self.out, self.report, **kwargs)

    def first_case(self):
        return self.out / "cases" / self.manifest["cases"][0]["id"] / "budget_0"

    def test_keyboard_interrupt_retains_and_archives_case_then_finishes_every_case(self):
        with patch.object(experiment, "fit_methods", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_it()
        status_bytes = (self.first_case() / "status.json").read_bytes()
        self.assertEqual(read(self.first_case() / "status.json")["state"], "running")
        self.assertFalse(self.report.exists())
        result = self.run_it()
        self.assertEqual(result["completed_cases"], 6)
        archived = self.out / result["recovered_cases"][0]
        self.assertEqual((archived / "original/status.json").read_bytes(), status_bytes)
        evidence = read(archived / "recovery.json")
        self.assertEqual(evidence["reason"], "interrupted_running_case")
        self.assertEqual(evidence["files"]["status.json"], digest(archived / "original/status.json"))
        attempts = [read(path)["state"] for path in (self.out / "automation/attempts").glob("*.json")]
        self.assertCountEqual(attempts, ["interrupted", "complete"])

    def test_failed_case_requires_explicit_retry_and_preserves_original_error(self):
        with patch.object(experiment, "fit_methods", side_effect=RuntimeError("programming bug")):
            with self.assertRaisesRegex(RuntimeError, "programming bug"):
                self.run_it()
        before = runner.inventory(self.first_case())
        with self.assertRaisesRegex(RuntimeError, "not retried automatically"):
            self.run_it()
        self.assertEqual(runner.inventory(self.first_case()), before)
        result = self.run_it(retry_failed=True)
        recovery = self.out / result["recovered_cases"][0]
        record = read(recovery / "recovery.json")
        self.assertIn("programming bug", record["original_status"]["error"])
        self.assertEqual(record["reason"], "explicit_failed_retry")
        self.assertEqual(runner.inventory(recovery / "original"), before)

    def test_valid_completed_cases_are_reused_without_refitting_and_report_is_archived(self):
        self.run_it()
        completed = read(self.out / "completed.json")
        old_statuses = completed["status_hashes"]
        old_report = runner.inventory(self.report)
        with patch.object(experiment, "fit_methods", side_effect=AssertionError("must not refit")):
            result = self.run_it()
        self.assertEqual(read(self.out / "completed.json")["status_hashes"], old_statuses)
        self.assertEqual(runner.inventory(self.report.parent / result["previous_report_archive"]), old_report)

    def test_changed_frozen_source_data_plan_or_version_blocks_recovery(self):
        experiment.freeze(self.bundle, self.out)
        write(self.first_case() / "status.json", {"state": "running"})
        source = self.project / "music_calibration/__init__.py"
        for path in [source, self.bundle / "manifest.json", self.out / "plans.json"]:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "changed"):
                self.run_it()
            path.write_bytes(original)
            self.assertTrue(self.first_case().exists())
        current = importlib.metadata.version
        with patch.object(runner.importlib.metadata, "version", side_effect=lambda name: "0.invalid" if name == "numpy" else current(name)):
            with self.assertRaisesRegex(ValueError, "Dependency version changed"):
                self.run_it()
        self.assertFalse((self.out / "automation").exists())
        self.assertEqual(read(self.first_case() / "status.json")["state"], "running")

    def test_tampered_completed_case_stops_before_retrying_running_case(self):
        self.run_it()
        (self.out / "completed.json").unlink()
        first = self.first_case()
        (first / "metrics.json").write_text("{}")
        second = self.out / "cases" / self.manifest["cases"][1]["id"] / "budget_0"
        write(second / "status.json", {"state": "running"})
        before = runner.inventory(second)
        with self.assertRaisesRegex(ValueError, "Completed case changed"):
            self.run_it()
        self.assertEqual(runner.inventory(second), before)

    def test_failed_independent_verification_never_publishes_over_a_valid_report(self):
        self.run_it()
        before = runner.inventory(self.report)
        with patch.object(runner, "verify", side_effect=ValueError("independent numeric mismatch")):
            with self.assertRaisesRegex(ValueError, "independent numeric mismatch"):
                self.run_it()
        self.assertEqual(runner.inventory(self.report), before)
        self.assertFalse(list(self.report.parent.glob(".report.archive-*")))
        self.assertTrue(list(self.report.parent.glob(".report.staging-*")))
        attempts = [read(path) for path in (self.out / "automation/attempts").glob("*.json")]
        failed = next(record for record in attempts if record["state"] == "failed")
        self.assertEqual(failed["failed_stage"], "verifying")

    def test_unrelated_or_modified_report_is_never_overwritten(self):
        experiment.freeze(self.bundle, self.out)
        self.report.mkdir()
        (self.report / "notes.txt").write_text("keep my notes")
        with self.assertRaisesRegex(FileExistsError, "not owned"):
            self.run_it()
        self.assertEqual((self.report / "notes.txt").read_text(), "keep my notes")
        (self.report / "notes.txt").unlink()
        self.report.rmdir()
        self.run_it()
        (self.report / "summary.json").write_text("changed by a person")
        with self.assertRaisesRegex(ValueError, "report was changed"):
            self.run_it()
        self.assertEqual((self.report / "summary.json").read_text(), "changed by a person")

    def test_overlapping_and_source_outputs_are_rejected_without_touching_files(self):
        for out, report in [(self.bundle / "run", self.report), (self.out, self.out / "report"),
                            (self.project, self.report), (self.out, self.project / "music_calibration/report")]:
            with self.subTest(out=out, report=report), self.assertRaises(ValueError):
                runner.reproduce(self.bundle, out, report)
        self.assertFalse(self.out.exists())

    def test_symlink_output_and_completed_artifact_are_rejected(self):
        link = self.root / "linked-output"
        try:
            link.symlink_to(self.bundle, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symlinks is not permitted on this host")
        with self.assertRaisesRegex(ValueError, "Symbolic links"):
            runner.reproduce(self.bundle, link, self.report)
        self.run_it()
        artifact = self.first_case() / "metrics.json"
        original = artifact.read_bytes()
        artifact.unlink()
        outside = self.root / "outside.json"
        outside.write_bytes(original)
        artifact.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.run_it()
        self.assertEqual(outside.read_bytes(), original)

    def test_corrupt_or_unknown_case_evidence_is_not_silently_dropped(self):
        experiment.freeze(self.bundle, self.out)
        self.first_case().mkdir(parents=True)
        (self.first_case() / "status.json").write_text('{"state":')
        with self.assertRaisesRegex(ValueError, "corrupt case status"):
            self.run_it()
        write(self.first_case() / "status.json", {"state": "running"})
        (self.first_case() / "unknown.txt").write_text("unrelated evidence")
        with self.assertRaisesRegex(ValueError, "Unexpected case artifact"):
            self.run_it()
        self.assertTrue((self.first_case() / "unknown.txt").exists())

    def test_unowned_running_case_is_not_taken_over_even_with_retry_flag(self):
        experiment.freeze(self.bundle, self.out)
        write(self.first_case() / "status.json", {"state": "running"})
        before = runner.inventory(self.first_case())
        for retry in [False, True]:
            with self.subTest(retry=retry), self.assertRaisesRegex(RuntimeError, "Unowned running cases"):
                self.run_it(retry_failed=retry)
        self.assertEqual(runner.inventory(self.first_case()), before)
        self.assertFalse((self.out / "automation/recoveries").exists())

    def test_live_writer_receipt_blocks_recovery_even_when_os_lock_is_free(self):
        with patch.object(experiment, "fit_methods", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_it()
        writer = read(self.out / "automation/writer.json")
        attempt_path = self.out / "automation/attempts" / (writer["attempt"] + ".json")
        attempt = read(attempt_path)
        attempt["state"] = "running"  # Same-host PID really is alive.
        write(attempt_path, attempt)
        before = runner.inventory(self.first_case())
        with self.assertRaisesRegex(RuntimeError, "Previous writer is alive"):
            self.run_it()
        self.assertEqual(runner.inventory(self.first_case()), before)

    def test_dead_writer_checkpoint_recovers_after_abrupt_process_exit(self):
        with patch.object(experiment, "fit_methods", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_it()
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=30)
        self.assertFalse(runner.process_alive(child.pid))
        writer = read(self.out / "automation/writer.json")
        writer["pid"] = child.pid
        write(self.out / "automation/writer.json", writer)
        attempt_path = self.out / "automation/attempts" / (writer["attempt"] + ".json")
        attempt = read(attempt_path)
        attempt.update(state="running", writer=writer)
        write(attempt_path, attempt)
        result = self.run_it()
        self.assertEqual(result["completed_cases"], 6)
        self.assertEqual(len(result["recovered_cases"]), 1)

    def test_process_lock_blocks_live_owner_and_recovers_actual_dead_owner(self):
        with runner.Lock(self.out):
            with self.assertRaisesRegex(RuntimeError, "Another reproduction process"):
                with runner.Lock(self.out):
                    self.fail("Concurrent lock was acquired")
        # The child exits without running __exit__, like sudden termination.
        code = "from pathlib import Path; from scripts.reproduce import Lock; import os; lock=Lock(Path(os.environ['TEST_LOCK_TARGET'])); lock.__enter__(); os._exit(0)"
        env = dict(os.environ, TEST_LOCK_TARGET=str(self.out), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        with runner.Lock(self.out) as acquired:
            self.assertEqual(acquired.recovered["state"], "held")
            self.assertNotEqual(acquired.recovered["pid"], os.getpid())
        self.assertEqual(read(acquired.path)["state"], "released")

    def test_foreign_host_and_unrecognized_locks_are_preserved(self):
        path = self.out.with_name("." + self.out.name + ".reproduce.lock")
        for record in [{"schema": runner.SCHEMA, "state": "held", "host": socket.gethostname() + "-other", "pid": 123},
                       {"unrelated": "keep"}]:
            write(path, record)
            before = path.read_bytes()
            with self.assertRaises((RuntimeError, ValueError)):
                with runner.Lock(self.out):
                    self.fail("Unsafe lock was accepted")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
