#!/usr/bin/env python3
"""Frozen repeated-subset study using the byte-identical budget_v1 engine."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music_calibration import LABELS
from music_calibration.data import audit_bundle
from music_calibration.experiment import freeze as freeze_child
from music_calibration.experiment import run as run_child
from music_calibration.experiment import verify_frozen
from music_calibration.io import digest, inside, now, read, write
from music_calibration.report import summarize as report_child
from music_calibration.report import plot as plot_cells
from music_calibration.verify import verify as verify_child


def _source_files():
    paths = [ROOT / "scripts/stability.py", ROOT / "protocols/stability_v1.md",
             ROOT / "protocols/stability_v1.json", ROOT / "protocols/budget_v1.md",
             ROOT / "protocols/budget_v1.json"]
    paths += sorted((ROOT / "music_calibration").rglob("*.py"))
    paths += sorted(p for p in (ROOT / "reports/budget_v1").rglob("*") if p.is_file())
    return sorted(set(paths))


def _validate_config(study, base):
    seeds = study["budget_seeds"]
    if (len(seeds) != 5 or len(set(seeds)) != 5 or
            any(type(seed) is not int or seed < 0 for seed in seeds) or
            base["budget_seed"] in seeds):
        raise ValueError("Require five distinct new nonnegative budget seeds")
    if (base["budgets"] != [0.25, 0.5, 1.0] or base["inner_folds"] != 3 or
            base["inner_seed"] != 2026100102 or
            study["expected_source_cases"] != base["expected_source_cases"]):
        raise ValueError("Study requires the unchanged budget_v1 design")
    if study["only_changed_child_config_field"] != "budget_seed":
        raise ValueError("Only budget_seed may vary")


def check_child_plans(plans_by_draw, original_plans, expected_artists=94):
    """Check all label-free sampling controls before the first new fit."""
    if len(plans_by_draw) != 5:
        raise ValueError("Expected five child plans")
    seeds = set(original_plans)
    if any(set(plans) != seeds for plans in plans_by_draw):
        raise ValueError("Child outer-split inventories differ")
    distinct = 0
    for seed in sorted(seeds):
        full = original_plans[seed][2]
        if full["fraction"] != 1.0 or full["n_artists"] != expected_artists:
            raise ValueError("Unexpected original calibration pool")
        for plans in plans_by_draw:
            if len(plans[seed]) != 3 or plans[seed][2] != full:
                raise ValueError("Full-budget plan or fold assignment changed")
        for budget_index in [0, 1]:
            groups = [tuple(plans[seed][budget_index]["track_ids"]) for plans in plans_by_draw]
            if len(set(groups)) != 5:
                raise ValueError("A repeated partial subset must be retained as a protocol failure; no redraw")
            distinct += len(set(groups))
        distinct += 1
    return {"outer_splits": len(seeds), "distinct_split_budget_subsets": distinct,
            "full_budget_plans_identical_to_original": True,
            "partial_draws_distinct": True}


def freeze_study(data, out, config_path=None):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new study directory: " + str(out))
    if out.is_relative_to(data):
        raise ValueError("Study outputs must be outside the input bundle")
    config_path = Path(config_path or ROOT / "protocols/stability_v1.json").resolve()
    study, base = read(config_path), read(ROOT / "protocols/budget_v1.json")
    _validate_config(study, base)
    manifest, audit = audit_bundle(data)
    if (audit["cases"] != study["expected_source_cases"] or
            audit["outer_splits"] != study["expected_outer_splits"]):
        raise ValueError("Source design does not match the frozen study")
    original_freeze = read(ROOT / "reports/budget_v1/freeze.json")
    # The original freeze serialized parsed config with io.write, rather than
    # copying its source formatting. Reconstruct those exact receipt-bound bytes.
    config_sha = hashlib.sha256((json.dumps(base, indent=2, allow_nan=False) + "\n").encode("utf-8")).hexdigest()
    if config_sha != original_freeze["local_hashes"]["config.json"]:
        raise ValueError("Original budget_v1 config/protocol changed: protocols/budget_v1.json")
    if digest(ROOT / "protocols/budget_v1.md") != original_freeze["local_hashes"]["protocol.md"]:
        raise ValueError("Original budget_v1 config/protocol changed: protocols/budget_v1.md")
    for name, expected in original_freeze["source_hashes"].items():
        if digest(inside(ROOT, name)) != expected:
            raise ValueError("Original budget_v1 engine changed: " + name)
    if digest(data / "manifest.json") != original_freeze["bundle_manifest_sha256"]:
        raise ValueError("Expected exactly the original score bundle")
    out.mkdir(parents=True)
    write(out / "config.json", study)
    write(out / "base_config.json", base)
    children, plans = [], []
    # All five child freezes precede any fitting function call.
    for draw, budget_seed in enumerate(study["budget_seeds"]):
        config = {**base, "budget_seed": budget_seed}
        config_name = f"child_configs/draw_{draw:02d}.json"
        write(out / config_name, config)
        name = f"children/draw_{draw:02d}"
        freeze_child(data, out / name, config_path=out / config_name)
        plans.append(read(out / name / "plans.json"))
        children.append({"draw": draw, "budget_seed": budget_seed, "path": name,
                         "config_path": config_name,
                         "freeze_sha256": digest(out / name / "freeze.json")})
    plan_audit = check_child_plans(plans, read(ROOT / "reports/budget_v1/plans.json"),
                                   study["expected_calibration_artists"])
    distinct = plan_audit["distinct_split_budget_subsets"] * 6
    executions = len(children) * len(manifest["cases"]) * len(base["budgets"])
    if (distinct != study["expected_distinct_subset_cases"] or
            executions != study["expected_executed_cases"]):
        raise ValueError("Execution or distinct-subset count mismatch")
    write(out / "plan_audit.json", {**plan_audit, "executed_cases": executions,
                                     "distinct_subset_cases": distinct})
    source_hashes = {}
    for source in _source_files():
        relative = source.relative_to(ROOT).as_posix()
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = digest(source)
    names = ["config.json", "base_config.json", "plan_audit.json"]
    names += [child["config_path"] for child in children]
    write(out / "study_freeze.json", {
        "created_utc": now(), "stage": "All five plans frozen before any additional fitting",
        "data_relative_to_study": os.path.relpath(data, out),
        "data_manifest_sha256": digest(data / "manifest.json"),
        "source_hashes": source_hashes,
        "local_hashes": {name: digest(out / name) for name in names},
        "children": children, "original_draw_included": False,
        "executed_cases": executions, "distinct_subset_cases": distinct})
    print(f"Frozen five complete child plans: {executions} executions, {distinct} distinct subset cases.", flush=True)


def verify_study_freeze(out):
    out = Path(out).resolve()
    record = read(out / "study_freeze.json")
    current = {p.relative_to(ROOT).as_posix() for p in _source_files()}
    if current != set(record["source_hashes"]):
        raise ValueError("Study execution-source inventory changed")
    for name, expected in record["source_hashes"].items():
        if digest(inside(ROOT, name)) != expected or digest(inside(out / "source", name)) != expected:
            raise ValueError("Frozen study source or original artifacts changed: " + name)
    for name, expected in record["local_hashes"].items():
        if digest(inside(out, name)) != expected:
            raise ValueError("Study plan/configuration changed: " + name)
    data = inside(out, record["data_relative_to_study"]) if not record["data_relative_to_study"].startswith("..") else (out / record["data_relative_to_study"]).resolve()
    if digest(data / "manifest.json") != record["data_manifest_sha256"]:
        raise ValueError("Study input manifest changed")
    study, base = read(out / "config.json"), read(out / "base_config.json")
    _validate_config(study, base)
    if len(record["children"]) != 5:
        raise ValueError("Study child inventory changed")
    plans = []
    for index, child in enumerate(record["children"]):
        path = inside(out, child["path"])
        if child["draw"] != index or child["budget_seed"] != study["budget_seeds"][index]:
            raise ValueError("Child draw identity changed")
        if digest(path / "freeze.json") != child["freeze_sha256"]:
            raise ValueError("Child freeze receipt changed")
        actual_data, _, config, child_plans = verify_frozen(path)
        if actual_data != data or config != {**base, "budget_seed": child["budget_seed"]}:
            raise ValueError("A child changed more than budget_seed")
        plans.append(child_plans)
    check_child_plans(plans, read(ROOT / "reports/budget_v1/plans.json"),
                      study["expected_calibration_artists"])
    return record, study, base


def run_study(out):
    out = Path(out).resolve()
    record, _, _ = verify_study_freeze(out)
    completions = []
    try:
        for child in record["children"]:
            path = inside(out, child["path"])
            print(f"Starting draw {child['draw'] + 1}/5 (budget seed {child['budget_seed']}).", flush=True)
            write(out / "progress.json", {"state": "running", "draw": child["draw"], "time": now()})
            run_child(path)
            report_child(path, out / "child_reports" / f"draw_{child['draw']:02d}")
            verification = verify_child(path)
            if verification["status"] != "passed" or not verification["summary_verified"]:
                raise ValueError("A child did not pass independent verification")
            names = ["completed.json", "verification.json", "summary.json"]
            completions.append({"draw": child["draw"], "path": child["path"],
                "hashes": {name: digest(path / name) for name in names},
                "verification_checks": verification["check_count"]})
        write(out / "run_completed.json", {"state": "complete", "completed_utc": now(),
              "study_freeze_sha256": digest(out / "study_freeze.json"), "children": completions})
        write(out / "progress.json", {"state": "complete", "completed_children": 5, "time": now()})
    except BaseException as error:
        write(out / "progress.json", {"state": "failed_or_interrupted", "error": repr(error),
                                      "completed_children": len(completions), "time": now()})
        raise


def _contrast(values, tolerance):
    values = np.asarray(values, dtype=float)
    wins, losses = int(np.sum(values < -tolerance)), int(np.sum(values > tolerance))
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            "range": float(values.max() - values.min()), "wins": wins,
            "ties": int(len(values) - wins - losses), "losses": losses,
            "sign_flip": bool(wins and losses)}


def aggregate_rows(rows, draw_count=5, tolerance=1e-12):
    """Equal-weight outer-split summaries; repeated full-budget endpoints count once."""
    by_split = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ["seed", "representation", "weighting", "fraction", "method"])
        by_split[key].append(row)
    split_rows = []
    for key, values in sorted(by_split.items()):
        if len(values) != draw_count or {row["draw"] for row in values} != set(range(draw_count)):
            raise ValueError("Missing or duplicate draw in an aggregate cell")
        values = sorted(values, key=lambda row: row["draw"])
        if key[3] == 1.0:
            for name in ["macro_brier", "macro_log_loss", "delta_raw", "delta_platt"]:
                if any(row[name] != values[0][name] for row in values[1:]):
                    raise ValueError("Repeated full-budget values are not identical")
            values = values[:1]
        row = dict(zip(["seed", "representation", "weighting", "fraction", "method"], key))
        row.update(draws=len(values), mean_brier=float(np.mean([v["macro_brier"] for v in values])),
                   mean_log_loss=float(np.mean([v["macro_log_loss"] for v in values])),
                   fallback_labels=sum(v["fallback_labels"] for v in values))
        for contrast in ["raw", "platt"]:
            row.update({f"{contrast}_{name}": value for name, value in
                        _contrast([v["delta_" + contrast] for v in values], tolerance).items()})
        split_rows.append(row)
    groups = defaultdict(list)
    for row in split_rows:
        groups[tuple(row[name] for name in ["representation", "weighting", "fraction", "method"])].append(row)
    cells = []
    for key, values in sorted(groups.items()):
        row = dict(zip(["representation", "weighting", "fraction", "method"], key))
        row.update(outer_splits=len(values), draws_per_split=values[0]["draws"],
                   evaluated_distinct_cases=sum(v["draws"] for v in values),
                   mean_brier=float(np.mean([v["mean_brier"] for v in values])),
                   mean_log_loss=float(np.mean([v["mean_log_loss"] for v in values])),
                   fallback_labels=sum(v["fallback_labels"] for v in values))
        for contrast in ["raw", "platt"]:
            means = [v[f"{contrast}_mean"] for v in values]
            ranges = [v[f"{contrast}_range"] for v in values]
            row.update({f"mean_delta_{contrast}": float(np.mean(means)),
                f"median_split_mean_delta_{contrast}": float(np.median(means)),
                f"min_split_mean_delta_{contrast}": float(min(means)),
                f"max_split_mean_delta_{contrast}": float(max(means)),
                f"mean_within_split_range_{contrast}": float(np.mean(ranges)),
                f"max_within_split_range_{contrast}": float(max(ranges)),
                f"sign_flip_splits_{contrast}": sum(v[f"{contrast}_sign_flip"] for v in values),
                f"all_draws_improve_splits_{contrast}": sum(v[f"{contrast}_wins"] == v["draws"] for v in values),
                f"all_draws_deteriorate_splits_{contrast}": sum(v[f"{contrast}_losses"] == v["draws"] for v in values)})
        cells.append(row)
    return split_rows, cells


def _csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty report table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_results(out):
    out = Path(out).resolve()
    record, study, base = verify_study_freeze(out)
    completion = read(out / "run_completed.json")
    if (completion["state"] != "complete" or len(completion["children"]) != 5 or
            completion["study_freeze_sha256"] != digest(out / "study_freeze.json")):
        raise ValueError("The complete study receipt is missing or changed")
    rows, selection_rows, full_hashes = [], [], {}
    total_cases = 0
    for child, finished in zip(record["children"], completion["children"]):
        path = inside(out, child["path"])
        if finished["path"] != child["path"] or finished["draw"] != child["draw"]:
            raise ValueError("Completed child identities changed")
        for name, expected in finished["hashes"].items():
            if digest(inside(path, name)) != expected:
                raise ValueError("Child completion or verification changed")
        completed = read(path / "completed.json")
        verification = read(path / "verification.json")
        if verification["status"] != "passed" or not verification["summary_verified"]:
            raise ValueError("A child lacks a passing numerical verification")
        if len(completed["cases"]) != study["expected_source_cases"] * 3:
            raise ValueError("Incomplete child case inventory")
        for relative in completed["cases"]:
            destination = inside(path, relative)
            if digest(destination / "status.json") != completed["status_hashes"][relative]:
                raise ValueError("Child case status changed")
            status = read(destination / "status.json")
            if status["state"] != "complete":
                raise ValueError("Incomplete case cannot be summarized")
            for name, expected in status["hashes"].items():
                if digest(inside(destination, name)) != expected:
                    raise ValueError("Child result changed: " + relative + "/" + name)
            case = read(destination / "case.json")
            scores = read(destination / "metrics.json")
            fitted = read(destination / "calibrators.json")
            total_cases += 1
            if case["fraction"] == 1.0:
                key = (case["seed"], case["representation"], case["weighting"])
                hashes = {name: status["hashes"][name] for name in
                          ["predictions.npz", "metrics.json", "calibrators.json"]}
                if key in full_hashes and hashes != full_hashes[key]:
                    raise ValueError("Repeated full-budget prediction/fit/metric hashes differ")
                full_hashes[key] = hashes
            identity = {name: case[name] for name in ["seed", "representation", "weighting", "fraction"]}
            for method in base["methods"]:
                rows.append({"draw": child["draw"], "budget_seed": child["budget_seed"],
                    **identity, "method": method, "calibration_artists": case["calibration_artists"],
                    "calibration_tracks": case["calibration_tracks"],
                    "macro_brier": scores[method]["macro_brier"],
                    "macro_log_loss": scores[method]["macro_log_loss"],
                    "delta_raw": scores[method]["macro_brier"] - scores["raw"]["macro_brier"],
                    "delta_platt": scores[method]["macro_brier"] - scores["platt"]["macro_brier"],
                    "fallback_labels": sum(int(r["fallback"]) for r in fitted[method]),
                    "is_duplicate_full_endpoint": case["fraction"] == 1.0 and child["draw"] > 0})
            for label, fitted_label in zip(LABELS, fitted["ridge_platt"]):
                selection_rows.append({"draw": child["draw"], **identity, "label": label,
                    "selected_penalty": fitted_label["selected_penalty"],
                    "inner_fallback_count": fitted_label["inner_fallback_count"],
                    "selected_inner_fallback_count": fitted_label["selected_inner_fallback_count"],
                    "final_fit_fallback": fitted_label["final_fit_fallback"]})
    if total_cases != study["expected_executed_cases"] or len(full_hashes) != study["expected_source_cases"]:
        raise ValueError("Study execution or full-control count mismatch")
    with (ROOT / "reports/budget_v1/case_metrics.csv").open(encoding="utf-8", newline="") as stream:
        historical = {(int(r["seed"]), r["representation"], r["weighting"], r["method"]): r
                      for r in csv.DictReader(stream) if float(r["fraction"]) == 1.0}
    error = 0.0
    for row in rows:
        if row["fraction"] == 1.0 and row["draw"] == 0:
            old = historical[(row["seed"], row["representation"], row["weighting"], row["method"])]
            error = max(error, *(abs(row[name] - float(old[name])) for name in ["macro_brier", "macro_log_loss"]))
    if error > study["historical_metric_tolerance"]:
        raise ValueError("Full-budget metrics do not reproduce historical results")
    controls = {"full_budget_identical_prediction_fit_metric_hashes": True,
        "full_budget_source_cases": len(full_hashes), "historical_full_max_absolute_error": error,
        "all_children_independently_verified": True,
        "child_verification_checks": sum(c["verification_checks"] for c in completion["children"])}
    return rows, selection_rows, controls, study


def summarize_study(out, report):
    out, report = Path(out).resolve(), Path(report).resolve()
    if report.exists():
        raise FileExistsError("Use a new report directory; existing results are preserved")
    if report.is_relative_to(ROOT / "data") or report.is_relative_to(ROOT / "reports/budget_v1"):
        raise ValueError("Report must not be written into original data or reports")
    rows, selections, controls, study = collect_results(out)
    split_rows, cells = aggregate_rows(rows, tolerance=study["comparison_tolerance"])
    grouped = defaultdict(list)
    for row in selections:
        grouped[tuple(row[n] for n in ["seed", "representation", "weighting", "fraction", "label"])].append(row)
    selection_summary = []
    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["draw"])
        if key[3] == 1.0:
            values = values[:1]
        penalties = [str(v["selected_penalty"]) for v in values]
        row = dict(zip(["seed", "representation", "weighting", "fraction", "label"], key))
        row.update(draws=len(values), distinct_strengths=len(set(penalties)),
            identity_draws=penalties.count("identity"), selected_strengths=json.dumps(penalties),
            inner_fallback_count=sum(v["inner_fallback_count"] for v in values),
            selected_inner_fallback_count=sum(v["selected_inner_fallback_count"] for v in values),
            final_fit_fallbacks=sum(int(v["final_fit_fallback"]) for v in values))
        selection_summary.append(row)
    summary = {"experiment": "stability_v1", "scope": study["scope"],
        "executed_cases": study["expected_executed_cases"],
        "distinct_subset_cases": study["expected_distinct_subset_cases"],
        "historical_draw_included": False, "full_endpoints_counted_once": True,
        "cells": cells, "controls": controls,
        "distinct_case_fallback_labels": sum(r["fallback_labels"] for r in rows if not r["is_duplicate_full_endpoint"]),
        "distinct_ridge_inner_fallbacks": sum(r["inner_fallback_count"] for r in selection_summary),
        "distinct_selected_inner_fallbacks": sum(r["selected_inner_fallback_count"] for r in selection_summary),
        "partial_label_cells_with_varying_ridge_strength": sum(r["distinct_strengths"] > 1 for r in selection_summary if r["fraction"] < 1),
        "partial_label_cells": sum(r["fraction"] < 1 for r in selection_summary)}
    report.mkdir(parents=True)
    write(report / "summary.json", summary)
    write(report / "controls.json", controls)
    _csv(report / "case_metrics.csv", rows)
    _csv(report / "within_split.csv", split_rows)
    _csv(report / "summary_cells.csv", cells)
    _csv(report / "ridge_selections.csv", selections)
    _csv(report / "ridge_stability.csv", selection_summary)
    plot_cells(cells, report)
    for filename in ["study_freeze.json", "config.json", "plan_audit.json", "run_completed.json"]:
        shutil.copy2(out / filename, report / filename)
    lines = ["# Calibration subset and selection stability", "",
        "Five new calibration-artist draws reuse the unchanged budget_v1 engine and all 120 source cases.",
        "The study executes 1,800 cases, covering 1,320 distinct source/subset conditions.",
        "Full-budget outputs are repeated numerical controls and are aggregated once per outer split.",
        "The original historical partial-budget draw is excluded from these summaries.", "",
        "![Mean Brier changes across repeated calibration subsets](budget_brier.png)", "",
        "Curves average five partial-budget draws within each of 20 overlapping outer splits.",
        "The common full-budget endpoint is counted once. These curves are not confidence intervals.", "",
        "All five child runs passed independent numerical verification. This is not new-data confirmation.",
        "Within-split ranges and sign changes describe five draws under a fixed fold-assignment procedure.",
        "Selected artists and resulting fold memberships vary together; their effects are not separated.",
        "Twenty outer splits overlap, and no confidence interval or independent-replicate test is constructed.", "",
        "## Ridge calibration compared with raw and ordinary Platt", "",
        "| Representation | Weighting | Budget | Mean change vs raw | Mean change vs Platt | Raw sign-flip splits / 20 |", 
        "|---|---|---:|---:|---:|---:|"]
    for cell in cells:
        if cell["method"] == "ridge_platt":
            lines.append(f"| {cell['representation']} | {cell['weighting']} | {100 * cell['fraction']:.0f}% | "
                f"{cell['mean_delta_raw']:+.6f} | {cell['mean_delta_platt']:+.6f} | {cell['sign_flip_splits_raw']} |")
    lines += ["", "Negative Brier changes are improvements. They are not accuracy changes or proof of reliability in every probability bin.",
        "The full-budget sign-flip count is zero by construction because its endpoint is identical across draws.",
        "All seven methods, individual draw outcomes and within-split ranges are retained in the CSV tables.",
        "Ridge choices and explicit fallbacks are recorded separately. No best-performing draw is selected.", "",
        "Missing source tags, historical selection, artist aliases and unknown encoder-pretraining exposure remain limitations.",
        "The released music application and original budget_v1 experiment are unchanged.", ""]
    (report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write(report / "report_receipt.json", {"created_utc": now(),
        "study_freeze_sha256": digest(out / "study_freeze.json"),
        "run_completed_sha256": digest(out / "run_completed.json"),
        "artifact_hashes": {p.name: digest(p) for p in sorted(report.iterdir()) if p.is_file()}})
    print(f"Saved {len(cells)} summary cells, {len(split_rows)} within-split rows and {len(rows)} executed method rows.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "summarize", "all"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/legacy_scores")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command in ["summarize", "all"] and args.report is None:
        parser.error("--report is required for summarize/all")
    if args.command in ["freeze", "all"]:
        if args.command == "all" and args.out.exists():
            verify_study_freeze(args.out)
        else:
            freeze_study(args.data, args.out)
    if args.command in ["run", "all"]:
        run_study(args.out)
    if args.command in ["summarize", "all"]:
        summarize_study(args.out, args.report)


if __name__ == "__main__":
    main()
