"""Validate portable score bundles without importing the original application."""
from pathlib import Path
import numpy as np
from . import LABELS
from .io import digest, inside, read


def load_case(root, case):
    path = inside(root, case["path"])
    if digest(path) != case["sha256"]:
        raise ValueError("Score bundle checksum mismatch: " + case["id"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for part in ["fit", "cal", "eval"]:
        ids, artists, y = [arrays[part + "_" + key] for key in ["ids", "artists", "y"]]
        if (ids.ndim != 1 or artists.shape != ids.shape or y.shape != (len(ids), 4)
                or len(ids) == 0 or len(set(ids)) != len(ids) or not np.isin(y, [0, 1]).all()
                or ids.dtype.kind not in "US" or artists.dtype.kind not in "US"):
            raise ValueError("Invalid IDs/groups/targets: " + case["id"])
        if part != "fit":
            z = arrays[part + "_logits"]
            if z.shape != y.shape or not np.isfinite(z).all():
                raise ValueError("Invalid logits")
    for left, right in [("fit", "cal"), ("fit", "eval"), ("cal", "eval")]:
        for key in ["ids", "artists"]:
            if set(arrays[left + "_" + key]) & set(arrays[right + "_" + key]):
                raise ValueError("Artist or track leakage: " + case["id"])
    offsets = arrays["offsets"]
    if offsets.shape != (4,) or not np.isfinite(offsets).all():
        raise ValueError("Invalid class-weight offsets")
    expected = np.zeros(4)
    if case["weighting"] == "ambient_balanced":
        n = arrays["fit_y"].sum(axis=0)[2]
        if not 0 < n < len(arrays["fit_y"]):
            raise ValueError("Ambient fit labels require both classes")
        expected[2] = np.log(n / (len(arrays["fit_y"]) - n))
    elif case["weighting"] != "unweighted":
        raise ValueError("Unknown weighting")
    if not np.allclose(offsets, expected, atol=1e-14, rtol=0):
        raise ValueError("Class-weight offsets do not match fit labels")
    for method in ["raw", "platt", "temperature", "prior_offset"]:
        p = arrays["reference_" + method]
        if (p.shape != arrays["eval_y"].shape or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()):
            raise ValueError("Invalid historical control")
    return arrays


def audit_bundle(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["labels"] != list(LABELS):
        raise ValueError("Unsupported bundle schema/labels")
    for name, expected in manifest["provenance_hashes"].items():
        if digest(inside(root, name)) != expected:
            raise ValueError("Bundle provenance checksum mismatch: " + name)
    cases = manifest["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Missing or duplicate cases")
    anchors, tracks, artists = {}, {}, set()
    for case in cases:
        if (not isinstance(case["seed"], int) or case["seed"] < 0
                or case["representation"] not in ["maest", "mert_v0", "mert_v1"]
                or case["weighting"] not in ["ambient_balanced", "unweighted"]
                or case["id"] != f"{case['seed']}/{case['representation']}/{case['weighting']}"):
            raise ValueError("Invalid case identity")
        arrays = load_case(root, case)
        anchor = anchors.setdefault(case["seed"], arrays)
        for part in ["fit", "cal", "eval"]:
            for key in ["ids", "artists", "y"]:
                if not np.array_equal(arrays[part + "_" + key], anchor[part + "_" + key]):
                    raise ValueError("Paired cases have different split rows")
            for i, artist, y in zip(arrays[part + "_ids"], arrays[part + "_artists"], arrays[part + "_y"]):
                target = (str(artist), tuple(map(int, y)))
                if str(i) in tracks and tracks[str(i)] != target:
                    raise ValueError("Track/artist/target alignment changed across cases")
                tracks[str(i)] = target
                artists.add(str(artist))
    return manifest, {"cases": len(cases), "outer_splits": len(anchors),
                      "unique_tracks": len(tracks), "unique_artists": len(artists),
                      "all_hashes_and_artist_boundaries_valid": True}


def plan_budgets(artists, seed, config):
    budgets, folds_count = config["budgets"], config["inner_folds"]
    if (not budgets or any(not np.isfinite(f) or not 0 < f <= 1 for f in budgets)
            or sorted(set(budgets)) != budgets or budgets[-1] != 1
            or isinstance(folds_count, bool) or not isinstance(folds_count, int) or folds_count < 2):
        raise ValueError("Invalid budget fractions or fold count")
    artists = np.asarray(artists)
    groups = np.unique(artists)
    rng = np.random.default_rng(np.random.SeedSequence([seed, config["budget_seed"]]))
    ordering = rng.permutation(groups)
    plans = []
    for budget_index, fraction in enumerate(config["budgets"]):
        selected = ordering[:int(np.ceil(fraction * len(groups)))]
        if len(selected) < config["inner_folds"]:
            raise ValueError("Too few artists for declared folds")
        indices = np.flatnonzero(np.isin(artists, selected))
        rng = np.random.default_rng(np.random.SeedSequence([seed, config["inner_seed"], budget_index]))
        folded = rng.permutation(np.sort(selected))
        mapping = {str(a): int(i % config["inner_folds"]) for i, a in enumerate(folded)}
        folds = np.array([mapping[str(a)] for a in artists[indices]])
        plans.append({"fraction": fraction, "artist_ids": sorted(map(str, selected)),
                      "indices": indices.tolist(), "folds": folds.tolist(),
                      "n_artists": len(selected), "n_tracks": len(indices)})
    return plans
