"""Probability quality and descriptive fixed-bin reliability diagnostics."""
import numpy as np
from sklearn.metrics import average_precision_score


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if (y.shape != p.shape or y.ndim != 2 or y.shape[1] != 4 or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()):
        raise ValueError("Expected aligned four-label binary targets and probabilities")
    brier = np.mean((p - y) ** 2, axis=0)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    log_loss = -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped), axis=0)
    result = {"macro_brier": float(brier.mean()), "label_brier": brier.tolist(),
            "macro_log_loss": float(log_loss.mean()), "label_log_loss": log_loss.tolist(),
            "label_average_precision": [float(average_precision_score(y[:, j], p[:, j]))
                if len(np.unique(y[:, j])) == 2 else None for j in range(4)],
            "label_mean_probability": p.mean(axis=0).tolist(),
            "label_positive_fraction": y.mean(axis=0).tolist()}
    for n_bins in [5, 10]:
        reliability = []
        for label in range(4):
            bins = np.minimum((p[:, label] * n_bins).astype(int), n_bins - 1)
            rows = []
            for index in range(n_bins):
                mask = bins == index
                rows.append({"bin": index, "count": int(mask.sum()),
                             "mean_probability": float(p[mask, label].mean()) if mask.any() else None,
                             "positive_fraction": float(y[mask, label].mean()) if mask.any() else None})
            reliability.append(rows)
        result[f"reliability_{n_bins}"] = reliability
        result[f"label_ece_{n_bins}"] = [sum(
            r["count"] / len(y) * abs(r["mean_probability"] - r["positive_fraction"])
            for r in rows if r["count"]) for rows in reliability]
    return result
