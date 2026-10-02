"""Freeze the plan, fit calibrators, then score frozen evaluation rows."""
import importlib.metadata
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .calibrators import fit, predict, select_ridge
from .data import audit_bundle, load_case, plan_budgets
from .io import digest, read, write, now
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def freeze(bundle, out, config_path=None):
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new run directory: " + str(out))
    if out.is_relative_to(bundle):
        raise ValueError("Run outputs must be outside the input bundle")
    config_path = Path(config_path or ROOT / "protocols/numerical_v2.json")
    config = read(config_path)
    manifest, audit = audit_bundle(bundle)
    if len(manifest["cases"]) != config["expected_source_cases"]:
        raise ValueError("Input case count differs from the declared protocol")
    seeds = {case["seed"] for case in manifest["cases"]}
    expected = {(seed, rep, weight) for seed in seeds
        for rep in ["maest", "mert_v0", "mert_v1"] for weight in ["ambient_balanced", "unweighted"]}
    if {(c["seed"], c["representation"], c["weighting"]) for c in manifest["cases"]} != expected:
        raise ValueError("Missing representation/weighting design cell")
    plans = {}
    for case in manifest["cases"]:
        key = str(case["seed"])
        if key not in plans:
            data = load_case(bundle, case)
            plans[key] = plan_budgets(data["cal_artists"], case["seed"], config)
            for plan in plans[key]:
                plan["track_ids"] = data["cal_ids"][plan["indices"]].tolist()
    out.mkdir(parents=True)
    write(out / "config.json", config)
    write(out / "plans.json", plans)
    write(out / "audit.json", audit)
    shutil.copy2(ROOT / "protocols/numerical_v2.md", out / "protocol.md")
    sources = sorted((ROOT / "music_calibration_portable").rglob("*.py"))
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = digest(source)
    write(out / "freeze.json", {
        "created_utc": now(), "stage": "Before new calibration fits and scoring; historical outcomes known",
        "bundle_relative_to_run": os.path.relpath(bundle, out),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "bundle_metadata_hashes": {p.relative_to(bundle).as_posix(): digest(p)
            for p in sorted(bundle.rglob("*.json"))},
        "source_hashes": source_hashes,
        "local_hashes": {name: digest(out / name) for name in
                         ["config.json", "plans.json", "audit.json", "protocol.md"]},
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]}})
    print(f"Frozen {len(manifest['cases'])} source cases and {len(plans)} paired budget/fold plans.", flush=True)


def verify_frozen(out):
    out = Path(out).resolve()
    record = read(out / "freeze.json")
    expected_sources = set(record["source_hashes"])
    current_sources = {p.relative_to(ROOT).as_posix() for p in (ROOT / "music_calibration_portable").rglob("*.py")}
    saved_sources = {p.relative_to(out / "source").as_posix() for p in (out / "source/music_calibration_portable").rglob("*.py")}
    if current_sources != expected_sources or saved_sources != expected_sources:
        raise ValueError("Research source inventory changed after freezing")
    bundle = (out / record["bundle_relative_to_run"]).resolve()
    if digest(bundle / "manifest.json") != record["bundle_manifest_sha256"]:
        raise ValueError("Input manifest changed after freezing")
    for name, expected in record["bundle_metadata_hashes"].items():
        if digest(bundle / name) != expected:
            raise ValueError("Input metadata changed: " + name)
    for name, expected in record["source_hashes"].items():
        if digest(ROOT / name) != expected or digest(out / "source" / name) != expected:
            raise ValueError("Research source changed after freezing: " + name)
    for name, expected in record["local_hashes"].items():
        if digest(out / name) != expected:
            raise ValueError("Frozen plan changed: " + name)
    manifest, _ = audit_bundle(bundle)
    return bundle, manifest, read(out / "config.json"), read(out / "plans.json")


def fit_methods(cal_logits, cal_y, folds, offsets, config):
    """Evaluation labels/logits deliberately absent from the fitting API."""
    records = {}
    for method in config["methods"]:
        records[method] = []
        for label in range(4):
            if method == "prior_offset":
                record = {"method": "prior_offset", "offset": float(offsets[label]),
                          "success": True, "fallback": False}
            elif method == "ridge_platt":
                record = select_ridge(cal_logits[:, label], cal_y[:, label], np.asarray(folds),
                    penalties=config["ridge_penalties"],
                    tie_tolerance=config["selection_tie_tolerance"])
            else:
                record = fit(cal_logits[:, label], cal_y[:, label], method)
            records[method].append(record)
    return records


def predict_methods(logits, records):
    return {method: np.column_stack([
        expit(logits[:, j] + record["offset"]) if method == "prior_offset"
        else predict(logits[:, j], record) for j, record in enumerate(fitted)])
        for method, fitted in records.items()}


def run(out):
    out = Path(out).resolve()
    bundle, manifest, config, plans = verify_frozen(out)
    completed = []
    with threadpool_limits(limits=1):
        for case in manifest["cases"]:
            data = load_case(bundle, case)
            for budget_index, plan in enumerate(plans[str(case["seed"]) ]):
                destination = out / "cases" / case["id"] / f"budget_{budget_index}"
                if destination.exists():
                    status = read(destination / "status.json")
                    if status["state"] != "complete":
                        raise ValueError("Incomplete run retained; use a new output directory")
                    for name, sha in status["hashes"].items():
                        if digest(destination / name) != sha:
                            raise ValueError("Completed case changed")
                    completed.append(destination.relative_to(out).as_posix())
                    continue
                destination.mkdir(parents=True)
                write(destination / "status.json", {"state": "running", "started_utc": now()})
                try:
                    indices = np.asarray(plan["indices"])
                    if data["cal_ids"][indices].tolist() != plan["track_ids"]:
                        raise ValueError("Budget rows no longer match frozen plan")
                    fitted = fit_methods(data["cal_logits"][indices], data["cal_y"][indices],
                                         plan["folds"], data["offsets"], config)
                    # Save parameters before applying them to evaluation rows.
                    write(destination / "calibrators.json", fitted)
                    probabilities = predict_methods(data["eval_logits"], fitted)
                    control_errors = {}
                    for method in ["raw", "prior_offset", "platt", "temperature"]:
                        if plan["fraction"] == 1 or method in ["raw", "prior_offset"]:
                            error = float(np.max(np.abs(probabilities[method] - data["reference_" + method])))
                            control_errors[method] = error
                            if method in ["raw", "prior_offset"] and error > config["historical_prediction_tolerance"]:
                                raise ValueError(f"Historical {method} control mismatch: {error}")
                    np.savez_compressed(destination / "predictions.npz", ids=data["eval_ids"],
                        artists=data["eval_artists"], y=data["eval_y"], **probabilities)
                    write(destination / "metrics.json", {method: metrics(data["eval_y"], p)
                                                         for method, p in probabilities.items()})
                    write(destination / "case.json", {**case, "budget_index": budget_index,
                        "fraction": plan["fraction"], "calibration_artists": plan["n_artists"],
                        "calibration_tracks": plan["n_tracks"], "historical_control_max_error": control_errors})
                    write(destination / "status.json", {"state": "complete", "completed_utc": now(),
                        "hashes": {name: digest(destination / name) for name in
                                   ["calibrators.json", "predictions.npz", "metrics.json", "case.json"]}})
                    completed.append(destination.relative_to(out).as_posix())
                except Exception as error:
                    write(destination / "status.json", {"state": "failed", "error": repr(error), "time": now()})
                    raise
            print(f"Completed {len(completed)}/{len(manifest['cases']) * len(config['budgets'])} cases: {case['id']}", flush=True)
    write(out / "completed.json", {"state": "complete", "completed_utc": now(), "cases": completed,
        "status_hashes": {path: digest(out / path / "status.json") for path in completed}})
