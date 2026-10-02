"""Resume a frozen benchmark safely, publish its report, and verify its numbers.

No fitting or statistical code lives here. Historical frozen research sources are
left unchanged. A dead same-host process can be recovered automatically; corrupt
or failed cases require investigation (and failed retries require an explicit flag).
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import stat
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music_calibration import experiment
from music_calibration.report import summarize
from music_calibration.verify import verify

SCHEMA = "music-calibration-reproduction-v1"
CASE_FILES = {"status.json", "calibrators.json", "predictions.npz", "metrics.json", "case.json"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_path(value):
    path = Path(os.path.abspath(value))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise ValueError("Symbolic links are not allowed in managed paths: " + str(component))
    return path


def no_links(root):
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a real directory: " + str(root))
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(base) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("Unsupported file or symbolic link: " + str(path))
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise ValueError("Hard-linked files are not supported: " + str(path))


def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_paths(data, out, report):
    data, out, report = [safe_path(path) for path in [data, out, report]]
    if overlaps(out, report) or overlaps(out, data) or overlaps(report, data):
        raise ValueError("Data, run, and report directories must not overlap")
    for path, container in [(out, ROOT / "runs"), (report, ROOT / "reports")]:
        if ROOT.is_relative_to(path) or (path.is_relative_to(ROOT) and not path.is_relative_to(container)):
            raise ValueError("Inside this project, outputs belong under runs/ or reports/")
        if path == container:
            raise ValueError("Use a named child directory, not the entire output container")
    no_links(data)
    no_links(out)
    no_links(report)
    return data, out, report


def process_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("Invalid process identifier in lock")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied is conservatively alive.
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Lock:
    """Stable-inode OS advisory lock; process exit releases it on all supported OSes.

    The file is deliberately retained after release: unlinking lock files allows
    two processes to lock different inodes under the same pathname.
    """
    def __init__(self, target):
        self.path = target.with_name("." + target.name + ".reproduce.lock")
        self.stream = None
        self.recovered = None

    def save(self, record):
        # Metadata may be interrupted; OS locking still protects the inode.
        self.stream.seek(0)
        self.stream.write((json.dumps(record) + "\n").encode("utf-8"))
        self.stream.truncate()
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __enter__(self):
        safe_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        self.stream = os.fdopen(descriptor, "r+b", buffering=0)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            self.stream.close()
            raise ValueError("Unsafe lock file: " + str(self.path))
        try:
            if os.name == "nt":
                import msvcrt
                if info.st_size == 0:
                    self.stream.write(b"\n")
                self.stream.seek(0)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("Another reproduction process holds " + str(self.path)) from error
        try:
            self.stream.seek(0)
            raw = self.stream.read().strip()
            if raw:
                previous = json.loads(raw)
                if previous.get("schema") != SCHEMA:
                    raise ValueError("Unrecognized lock file; refusing to overwrite it")
                if previous.get("state") == "held":
                    if previous.get("host") != socket.gethostname() or process_alive(previous.get("pid")):
                        raise RuntimeError("Lock owner is alive or on another host; cannot prove it is stale")
                    self.recovered = previous
            self.record = {"schema": SCHEMA, "state": "held", "host": socket.gethostname(),
                           "pid": os.getpid(), "token": uuid.uuid4().hex, "started_utc": stamp()}
            if self.recovered:
                self.record["recovered_dead_owner"] = self.recovered
            self.save(self.record)
            return self
        except BaseException:
            self.stream.close()
            raise

    def __exit__(self, kind, error, traceback):
        self.record.update(state="released", finished_utc=stamp())
        if error is not None:
            self.record["error"] = repr(error)
        try:
            self.save(self.record)
        finally:
            self.stream.close()


def inventory(root):
    no_links(root)
    return {path.relative_to(root).as_posix(): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def validate_report(report, frozen_hash):
    if not report.exists():
        return
    marker = report / ".reproduction.json"
    if not marker.is_file():
        raise FileExistsError("Existing report is not owned by this automation: " + str(report))
    record = read(marker)
    if record.get("schema") != SCHEMA or record.get("freeze_sha256") != frozen_hash:
        raise ValueError("Existing report belongs to a different frozen run")
    actual = inventory(report)
    actual.pop(".reproduction.json")
    if record.get("files") != actual:
        raise ValueError("Existing report was changed; preserving it instead of overwriting")


def preflight(out, data):
    no_links(out)
    bundle, manifest, config, plans = experiment.verify_frozen(out)
    if bundle != data:
        raise ValueError("Requested data differs from the frozen data bundle")
    for name, version in read(out / "freeze.json")["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Dependency version changed after freezing: " + name)
    expected = {"cases/" + case["id"] + "/budget_" + str(index)
                for case in manifest["cases"] for index in range(len(plans[str(case["seed"])]))}
    # Unknown case paths must never disappear into a retry or be omitted from the report.
    case_root = out / "cases"
    if case_root.exists():
        for path in case_root.rglob("*"):
            relative = path.relative_to(out).as_posix()
            if path.is_dir():
                if not any(key == relative or key.startswith(relative + "/") for key in expected):
                    raise ValueError("Unexpected case directory: " + relative)
            elif path.parent.relative_to(out).as_posix() not in expected or path.name not in CASE_FILES:
                raise ValueError("Unexpected case artifact: " + relative)
    pending = []
    complete = set()
    for relative in sorted(expected):
        destination = out / relative
        if not destination.exists():
            continue
        try:
            status = read(destination / "status.json")
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("Missing or corrupt case status; preserving all evidence: " + relative) from error
        if status.get("state") == "complete":
            if set(status.get("hashes", {})) != CASE_FILES - {"status.json"}:
                raise ValueError("Incomplete artifact inventory in completed case: " + relative)
            for name, expected_hash in status["hashes"].items():
                if sha(destination / name) != expected_hash:
                    raise ValueError("Completed case changed: " + relative)
            complete.add(relative)
        elif status.get("state") in {"running", "failed"}:
            pending.append((relative, status))
        else:
            raise ValueError("Unknown case state: " + relative)
    if (out / "completed.json").exists():
        receipt = read(out / "completed.json")
        if receipt.get("state") != "complete" or set(receipt.get("cases", [])) != expected or len(receipt["cases"]) != len(expected):
            raise ValueError("Completed run inventory changed")
        if complete != expected or set(receipt.get("status_hashes", {})) != expected:
            raise ValueError("Completed run now contains missing or incomplete cases")
        for relative, expected_hash in receipt["status_hashes"].items():
            if sha(out / relative / "status.json") != expected_hash:
                raise ValueError("Completed case status changed: " + relative)
    return expected, pending


def validate_recovery_owner(out, pending):
    """A legacy engine process does not hold our lock; never take over its work."""
    if not any(status["state"] == "running" for _, status in pending):
        return
    automation = out / "automation"
    writer_path = automation / "writer.json"
    if not writer_path.is_file():
        raise RuntimeError("Unowned running cases cannot be recovered automatically; a legacy engine process may still be writing")
    writer = read(writer_path)
    attempt = writer.get("attempt", "")
    if not isinstance(attempt, str) or len(attempt) != 32 or any(c not in "0123456789abcdef" for c in attempt):
        raise ValueError("Invalid writer provenance")
    previous = read(automation / "attempts" / (attempt + ".json"))
    if (writer.get("schema") != SCHEMA or writer.get("freeze_sha256") != sha(out / "freeze.json")
            or previous.get("attempt") != attempt or previous.get("freeze_sha256") != writer["freeze_sha256"]
            or previous.get("writer") != writer):
        raise ValueError("Running case ownership no longer matches its frozen attempt")
    terminal_interruption = (previous.get("state") == "interrupted"
        or (previous.get("state") == "failed" and previous.get("failed_stage") == "running"))
    if terminal_interruption:
        return  # Previous wrapper exited its engine and released the OS lock.
    if (previous.get("state") != "running" or writer.get("host") != socket.gethostname()
            or process_alive(writer.get("pid"))):
        raise RuntimeError("Previous writer is alive or cannot be proved dead; running cases are preserved")


def recover(out, pending, retry_failed, attempt):
    failed = [relative for relative, status in pending if status["state"] == "failed"]
    if failed and not retry_failed:
        raise RuntimeError("Failed cases are retained, not retried automatically: " + ", ".join(failed)
                           + ". Investigate the recorded error first; use --retry-failed only for an intentional retry.")
    archived = []
    for index, (relative, status) in enumerate(pending):
        directory = out / "automation" / "recoveries" / (attempt + "-" + str(index))
        directory.mkdir(parents=True, exist_ok=False)
        record = {"schema": SCHEMA, "state": "prepared", "case": relative, "original_status": status,
                  "reason": "interrupted_running_case" if status["state"] == "running" else "explicit_failed_retry",
                  "recorded_utc": stamp(), "freeze_sha256": sha(out / "freeze.json"),
                  "files": inventory(out / relative)}
        atomic_json(directory / "recovery.json", record)
        os.rename(out / relative, directory / "original")
        record["state"] = "archived_for_refit"
        atomic_json(directory / "recovery.json", record)
        archived.append(directory.relative_to(out).as_posix())
    return archived


def reproduce(data=None, out=None, report=None, retry_failed=False):
    data, out, report = validate_paths(data or ROOT / "data/legacy_scores",
                                     out or ROOT / "runs/reproduction", report or ROOT / "reports/reproduction")
    attempt = uuid.uuid4().hex
    checkpoint = None
    state = {"schema": SCHEMA, "attempt": attempt, "started_utc": stamp(), "state": "starting",
             "automation_sha256": sha(__file__), "retry_failed_explicitly_requested": retry_failed}
    with ExitStack() as stack:
        locks = [stack.enter_context(Lock(target)) for target in sorted([out, report])]
        state["recovered_dead_locks"] = [lock.recovered for lock in locks if lock.recovered]
        try:
            if not out.exists():
                if report.exists():
                    raise FileExistsError("New runs require a new report destination")
                temporary = out.with_name("." + out.name + ".freeze-" + attempt)
                state["freeze_staging_name"] = temporary.name
                experiment.freeze(data, temporary)
                # The sibling rename keeps all relative data references valid.
                preflight(temporary, data)
                os.rename(temporary, out)
            expected, pending = preflight(out, data)
            frozen_hash = sha(out / "freeze.json")
            validate_report(report, frozen_hash)
            automation = out / "automation"
            if automation.exists():
                if not (automation / "owner.json").exists() or read(automation / "owner.json") != {"schema": SCHEMA}:
                    raise ValueError("Unrecognized automation directory; refusing to overwrite it")
            validate_recovery_owner(out, pending)
            if not automation.exists():
                atomic_json(automation / "owner.json", {"schema": SCHEMA})
            checkpoint = automation / "attempts" / (attempt + ".json")
            state.update(state="recovering", freeze_sha256=frozen_hash, expected_cases=len(expected))
            atomic_json(checkpoint, state)
            state["recovered_cases"] = recover(out, pending, retry_failed, attempt)
            run_lock = next(lock for lock in locks if lock.path == out.with_name("." + out.name + ".reproduce.lock"))
            writer = {"schema": SCHEMA, "attempt": attempt, "freeze_sha256": frozen_hash,
                      "host": socket.gethostname(), "pid": os.getpid(), "lock_token": run_lock.record["token"]}
            state.update(state="running", writer=writer)
            atomic_json(checkpoint, state)
            atomic_json(automation / "writer.json", writer)
            experiment.run(out)
            remaining_expected, remaining_pending = preflight(out, data)
            if remaining_pending or remaining_expected != expected or not (out / "completed.json").exists():
                raise ValueError("Run did not complete every frozen case")
            staged_report = report.with_name("." + report.name + ".staging-" + attempt)
            state.update(state="summarizing", report_staging_name=staged_report.name)
            atomic_json(checkpoint, state)
            summarize(out, staged_report)
            state["state"] = "verifying"
            atomic_json(checkpoint, state)
            verification = verify(out)
            if verification.get("status") != "passed":
                raise ValueError("Independent verification did not pass")
            for name in ["freeze.json", "plans.json", "verification.json"]:
                (staged_report / name).write_bytes((out / name).read_bytes())
            atomic_json(staged_report / ".reproduction.json", {
                "schema": SCHEMA, "freeze_sha256": frozen_hash, "completed_utc": stamp(),
                "automation_sha256": sha(__file__), "files": inventory(staged_report)})
            validate_report(staged_report, frozen_hash)
            # Locks protect cooperating writers; recheck immediately before moving old content.
            validate_report(report, frozen_hash)
            if report.exists():
                archive = report.with_name("." + report.name + ".archive-" + attempt)
                os.rename(report, archive)
                state["previous_report_archive"] = archive.name
            os.rename(staged_report, report)
            state.update(state="complete", finished_utc=stamp(), completed_cases=len(expected),
                         verification_sha256=sha(out / "verification.json"))
            atomic_json(checkpoint, state)
            print("Reproduction complete: every frozen case retained, report published, independent verification passed.", flush=True)
            return state
        except BaseException as error:
            state["failed_stage"] = state["state"]
            state.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         finished_utc=stamp(), error=repr(error))
            if checkpoint is not None:
                atomic_json(checkpoint, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/legacy_scores")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/reproduction")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/reproduction")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly archive and retry failed cases; investigate their errors first")
    args = parser.parse_args()
    try:
        reproduce(args.data, args.out, args.report, args.retry_failed)
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are retained; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print("Reproduction stopped: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
