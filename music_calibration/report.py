"""Report every predefined comparison; overlapping splits are descriptive."""
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from . import LABELS
from .io import digest, read, write


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out, report):
    out, report = Path(out).resolve(), Path(report).resolve()
    from .experiment import verify_frozen
    verify_frozen(out)
    completed = read(out / "completed.json")
    if completed["state"] != "complete":
        raise ValueError("Only complete runs can be summarized")
    report.mkdir(parents=True, exist_ok=True)
    rows, label_rows, reliability, fallback_rows, selections = [], [], [], [], []
    groups = defaultdict(list)
    for relative in completed["cases"]:
        directory = out / relative
        status = read(directory / "status.json")
        if digest(directory / "status.json") != completed["status_hashes"][relative]:
            raise ValueError("Status changed")
        for name, expected in status["hashes"].items():
            if digest(directory / name) != expected:
                raise ValueError("Case changed")
        case, all_metrics, fitted = [read(directory / name) for name in
                                    ["case.json", "metrics.json", "calibrators.json"]]
        for method, scores in all_metrics.items():
            row = {key: case[key] for key in ["seed", "representation", "weighting", "fraction"]}
            row.update(method=method, calibration_artists=case["calibration_artists"],
                       calibration_tracks=case["calibration_tracks"],
                       macro_brier=scores["macro_brier"], macro_log_loss=scores["macro_log_loss"],
                       delta_raw=scores["macro_brier"] - all_metrics["raw"]["macro_brier"],
                       delta_platt=scores["macro_brier"] - all_metrics["platt"]["macro_brier"],
                       fallback_labels=sum(int(r.get("fallback", False)) for r in fitted[method]),
                       successful_labels=sum(int(r["success"]) for r in fitted[method]),
                       bound_hit_labels=sum(int(r.get("bound_hit", False)) for r in fitted[method]))
            rows.append(row)
            groups[tuple(row[k] for k in ["representation", "weighting", "fraction", "method"])].append(row)
            for label, name in enumerate(LABELS):
                label_rows.append({**{k: row[k] for k in ["seed", "representation", "weighting", "fraction", "method"]},
                    "label": name, "brier": scores["label_brier"][label],
                    "delta_raw": scores["label_brier"][label] - all_metrics["raw"]["label_brier"][label],
                    "log_loss": scores["label_log_loss"][label],
                    "average_precision": scores["label_average_precision"][label],
                    "ece_5": scores["label_ece_5"][label], "ece_10": scores["label_ece_10"][label]})
                rec = fitted[method][label]
                if rec.get("fallback", False):
                    fallback_rows.append({"case": relative, "method": method, "label": name,
                                          "stage": "final", "reason": rec.get("reason")})
                if method == "ridge_platt":
                    selections.append({"case": relative, "representation": case["representation"],
                        "weighting": case["weighting"], "fraction": case["fraction"], "label": name,
                        "selected_penalty": rec["selected_penalty"],
                        "inner_fallback_count": rec["inner_fallback_count"],
                        "selected_inner_fallback_count": rec["selected_inner_fallback_count"],
                        "final_fit_fallback": rec["final_fit_fallback"]})
                # A declared single example split is sufficient for readable reliability tables;
                # all other tables remain in case metrics for exact reproduction.
                if case["seed"] == 2026092800 and case["fraction"] == 1:
                    for n_bins in [5, 10]:
                        for item in scores[f"reliability_{n_bins}"][label]:
                            reliability.append({"seed": case["seed"], "representation": case["representation"],
                                "weighting": case["weighting"], "method": method, "label": name,
                                "bins": n_bins, **item})
    cells = []
    for key, values in sorted(groups.items()):
        deltas = np.array([r["delta_raw"] for r in values])
        platt = np.array([r["delta_platt"] for r in values])
        cell = dict(zip(["representation", "weighting", "fraction", "method"], key))
        cell.update(n=len(values), mean_brier=float(np.mean([r["macro_brier"] for r in values])),
            mean_delta_raw=float(deltas.mean()), median_delta_raw=float(np.median(deltas)),
            min_delta_raw=float(deltas.min()), max_delta_raw=float(deltas.max()),
            wins=int((deltas < -1e-12).sum()), ties=int((abs(deltas) <= 1e-12).sum()),
            losses=int((deltas > 1e-12).sum()), mean_delta_platt=float(platt.mean()),
            wins_vs_platt=int((platt < -1e-12).sum()),
            final_fallback_labels=sum(r["fallback_labels"] for r in values),
            successful_labels=sum(r["successful_labels"] for r in values),
            bound_hit_labels=sum(r["bound_hit_labels"] for r in values))
        cells.append(cell)
    summary = {"scope": "Exploratory, previously observed data; 20 overlapping splits per cell, not independent datasets",
        "cases": len(completed["cases"]), "method_cases": len(rows), "cells": cells,
        "final_fallback_labels": len(fallback_rows),
        "ridge_inner_fallbacks": sum(r["inner_fallback_count"] for r in selections),
        "ridge_selected_inner_fallbacks": sum(r["selected_inner_fallback_count"] for r in selections),
        "ridge_selection_counts": dict(Counter(str(r["selected_penalty"]) for r in selections))}
    write(out / "summary.json", summary)
    write(report / "summary.json", summary)
    write(report / "fallbacks.json", fallback_rows)
    write_csv(report / "case_metrics.csv", rows)
    write_csv(report / "label_metrics.csv", label_rows)
    write_csv(report / "summary_cells.csv", cells)
    write_csv(report / "ridge_selections.csv", selections)
    if reliability:
        write_csv(report / "example_reliability.csv", reliability)
    plot(cells, report)
    lines = ["# Limited-data calibration benchmark", "",
        "All 360 planned cases are retained: 20 historical artist splits × 3 representations ×",
        "2 training-weight settings × 3 calibration budgets. Seven probability treatments",
        "are compared on the same evaluation rows within each split. The 1,206 songs /",
        "469 artists have been used in earlier research; these results are exploratory.", "",
        "![Mean Brier changes](budget_brier.png)", "",
        "Negative changes mean lower probability error than raw scores. Each plotted mean",
        "uses 20 overlapping historical splits; the lines are not confidence intervals.", "",
        "## Full calibration budget", "",
        "| Representation | Ambient weighting | Method | Mean Brier | Change vs raw | Better / 20 |", "|---|---|---|---:|---:|---:|"]
    for cell in cells:
        if cell["fraction"] == 1:
            lines.append(f"| {cell['representation']} | {cell['weighting']} | {cell['method']} | "
                f"{cell['mean_brier']:.6f} | {cell['mean_delta_raw']:+.6f} | {cell['wins']} |")
    lines += ["", "## Completeness and limitations", "",
        f"Final label-fit fallbacks: {summary['final_fallback_labels']}. Ridge inner-fit fallbacks: "
        f"{summary['ridge_inner_fallbacks']} (selected candidates: {summary['ridge_selected_inner_fallbacks']}).",
        "All policy scores include declared fallback behavior; successful-fit counts and bound hits",
        "are in `summary_cells.csv`. `ridge_selections.csv` reports each selection and fallback count.", "",
        "The paired budgets use one nested artist draw per outer split. This does not estimate",
        "within-split subset-sampling variability. These are source tags, not human-adjudicated",
        "genre truth; missing tags, artist aliases and encoder pretraining overlap remain limitations.",
        "Lower Brier is not proof of calibration in every probability interval. Isotonic ties",
        "may affect average precision. No classifier replacement, fresh confirmation, or publication",
        "novelty is claimed. The v1.2 application is unchanged.", "",
        "## Reproduction", "",
        "See the repository README and frozen protocol. All 25%, 50% and 100% budget cells",
        "and label metrics are in the adjacent CSV/JSON files. Numerical verification is",
        "recorded separately in `verification.json`. Independent numerical reconstruction",
        "checks implementation consistency; it is not independent scientific replication.", ""]
    (report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reported {len(cells)} cells; {len(rows)} method/case results.", flush=True)


def plot(cells, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["prior_offset", "temperature", "platt", "beta", "isotonic", "ridge_platt"]
    colors = ["#6a737d", "#d08928", "#1b75bb", "#7c4d9e", "#b1494b", "#238b65"]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(10.6, 6), sharex=True, sharey=True)
        for row, weighting in enumerate(["ambient_balanced", "unweighted"]):
            for col, representation in enumerate(["maest", "mert_v0", "mert_v1"]):
                ax = axes[row, col]
                ax.axhline(0, color="#222222", ls="--", lw=1)
                for method, color in zip(methods, colors):
                    chosen = sorted([c for c in cells if c["weighting"] == weighting
                                     and c["representation"] == representation and c["method"] == method],
                                    key=lambda c: c["fraction"])
                    ax.plot([c["fraction"] * 100 for c in chosen],
                            [c["mean_delta_raw"] for c in chosen], marker="o", ms=3.8,
                            lw=1.4, color=color, label=method)
                ax.set_title(representation.upper().replace("_", " "))
                ax.set_xticks([25, 50, 100])
                ax.grid(axis="y", alpha=.18)
                if col == 0:
                    ax.set_ylabel(weighting.replace("_", " ") + "\nMean Brier change")
                if row == 1:
                    ax.set_xlabel("Calibration artists retained (%)")
        fig.suptitle("Probability calibration with limited artist data", fontsize=14)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6,
                   frameon=False, bbox_to_anchor=(.5, .025))
        fig.text(.5, .005, "Negative is better than raw • 20 overlapping historical splits • Descriptive means",
                 ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=[0, .10, 1, .96])
        fig.savefig(report / "budget_brier.png", dpi=200)
        fig.savefig(report / "budget_brier.pdf")
        plt.close(fig)
