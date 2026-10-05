"""Reconstruct the published fresh-cohort results, optionally refitting calibrators.

The input is the public score/parameter bundle, never the application/audio tree.
This is numerical reproduction of an existing study, not a new experiment.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['raw', 'prior_offset', 'temperature', 'platt', 'beta', 'isotonic', 'ridge_platt']
LABELS = ['electronic', 'pop', 'ambient', 'rock']
TOL = 2e-12
REFIT_TOL = 1e-8


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, tol=TOL):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message + ': shape/finite')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    require(error <= tol, message + f': maximum absolute error {error}')
    return error


def compare_tree(a, b, message):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), message + ': keys')
        return max((compare_tree(a[k], b[k], message + '/' + str(k)) for k in a), default=0.)
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), message + ': list length')
        return max((compare_tree(x, y, message) for x, y in zip(a, b)), default=0.)
    if a is None or isinstance(a, (str, bool)):
        require(a == b, message + ': value')
        return 0.
    return close(a, b, message)


def checked_path(root, name):
    root = Path(root).resolve()
    candidate = (root / name).resolve()
    require(not Path(name).is_absolute() and candidate.is_relative_to(root), 'Manifest path outside repository')
    return candidate


def verify_integrity(root):
    manifest = read(root / 'data/fresh_confirmation_v1/manifest.json')
    require(manifest['schema_version'] == 1 and manifest['source_case_count'] == 440, 'Unexpected bundle version/design')
    for name, entry in manifest['exports'].items():
        path = checked_path(root, name)
        require(sha(path) == entry['export_sha256'], 'Public artifact changed: ' + name)
        if entry['operation'].startswith('gzip with mtime 0'):
            require(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == entry['original_sha256'], 'Decompressed original hash differs: ' + name)
        elif entry['operation'] == 'exact bytes':
            require(entry['original_sha256'] == entry['export_sha256'], 'Exact-copy hash identity differs: ' + name)
    for name, expected in manifest['numerical_source_hashes'].items():
        require(sha(checked_path(root, name)) == expected, 'Frozen numerical source changed: ' + name)
    return manifest


def validate_arrays(cal, ev, labels, config):
    require('y' not in ev, 'Evaluation score file contains labels')
    require(np.array_equal(ev['ids'], labels['ids']), 'Evaluation label row order differs')
    require(not set(cal['ids']) & set(ev['ids']), 'Calibration/evaluation track overlap')
    require(not set(cal['artists']) & set(ev['artists']), 'Calibration/evaluation artist overlap')
    for name, data, targets in [('calibration', cal, cal['y']), ('evaluation', ev, labels['y'])]:
        n = len(data['ids'])
        require(data['ids'].ndim == 1 and data['artists'].shape == (n,), name + ': identities')
        require(data['ids'].dtype.kind in 'US' and data['artists'].dtype.kind in 'US', name + ': string identities')
        require(len(set(data['ids'])) == n and n > 0, name + ': unique IDs')
        require(data['logits'].shape == (20, 2, n, 4) and np.isfinite(data['logits']).all(), name + ': logits')
        require(targets.shape == (n, 4) and np.isin(targets, [0, 1]).all(), name + ': labels')
        require(data['head_seeds'].tolist() == config['head_seeds'], name + ': head inventory')
        require(data['weightings'].tolist() == config['weightings'], name + ': weighting inventory')
        require(data['offsets'].shape == (20, 2, 4) and np.isfinite(data['offsets']).all(), name + ': offsets')
    require(np.array_equal(cal['offsets'], ev['offsets']), 'Head offsets differ')
    require(config['labels'] == LABELS and config['methods'] == METHODS, 'Unexpected labels/methods')
    require(np.all(cal['offsets'][:, 0] == 0) and np.all(cal['offsets'][:, 1, [0, 1, 3]] == 0), 'Offset affects an unintended label')


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z, dtype=float)))


def independent_predict(z, record):
    method = record.get('effective_method', record['method'])
    if method == 'raw':
        return sigmoid(z)
    if method == 'prior_offset':
        return sigmoid(z + record['offset'])
    if method == 'isotonic':
        return np.interp(z, record['x_thresholds'], record['y_thresholds'])
    if method == 'beta':
        u = -record['a'] * np.logaddexp(0., -z) + record['b'] * np.logaddexp(0., z) + record['intercept']
    else:
        u = record['slope'] * z + record['intercept']
    return sigmoid(u)


def verify_oof(z, y, folds, record, tolerance):
    errors, scores = [], []
    for candidate in record['oof_scores']:
        penalty = candidate['penalty']
        oof = np.empty(len(z))
        if penalty == 'identity':
            oof = sigmoid(z)
        else:
            require({int(r['fold']) for r in candidate['fold_fits']} == set(map(int, np.unique(folds))), 'OOF folds missing')
            for fold in candidate['fold_fits']:
                val = folds == fold['fold']
                require(fold['validation_n'] == int(val.sum()) and fold['train_n'] == int((~val).sum()), 'OOF count mismatch')
                oof[val] = independent_predict(z[val], fold['fit'])
        errors.append(close(oof, record['oof_predictions'][str(penalty)], 'OOF probabilities'))
        score = float(np.mean((oof-y)**2))
        errors.append(close(score, candidate['brier'], 'OOF Brier'))
        scores.append((penalty, score))
    minimum = min(s for _, s in scores)
    best = max([p for p, s in scores if s <= minimum + tolerance], key=lambda p: math.inf if p == 'identity' else p)
    require(best == record['selected_penalty'], 'Selected ridge penalty differs')
    require(record['selection']['uses_evaluation_data'] is False, 'OOF marked evaluation data used')
    return max(errors, default=0.)


def bootstrap(values, artists, seed, reps):
    groups = sorted(set(artists.tolist()))
    sums = np.asarray([values[artists == g].sum() for g in groups])
    counts = np.asarray([(artists == g).sum() for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(reps, len(groups)))
    results = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return results, np.quantile(results, [.025, .975])


def numerical_counts(value):
    result = {'single_class_fallbacks': 0, 'optimized_fits': 0, 'bound_hits': 0}
    def visit(v):
        if isinstance(v, dict):
            if v.get('fallback'):
                require(v.get('reason') == 'single_class_calibration', 'Undeclared fallback')
                result['single_class_fallbacks'] += 1
            if 'optimizer_success' in v:
                require(v['optimizer_success'], 'Numerical fit did not converge')
                result['optimized_fits'] += 1
            result['bound_hits'] += bool(v.get('bound_hit'))
            for child in v.values(): visit(child)
        elif isinstance(v, list):
            for child in v: visit(child)
    visit(value)
    return result


def verify(root=ROOT, refit=False):
    root = Path(root).resolve()
    manifest = verify_integrity(root)
    sys.path.insert(0, str(root))
    from music_calibration_portable.experiment import fit_methods, predict_methods
    from music_calibration_portable.metrics import metrics
    data = root / 'data/fresh_confirmation_v1'
    report = root / 'reports/fresh_confirmation_v1'
    config = read(root / 'research/fresh_confirmation_v1/config.json')
    cal, ev, labels = (load(data / n) for n in ['calibration_scores.npz', 'evaluation_inputs.npz', 'evaluation_labels.npz'])
    validate_arrays(cal, ev, labels, config)
    y, artists = labels['y'], ev['artists']
    selected = read(data / 'selected_manifest.json')['tracks']
    lookup = {r['track_id']: r for r in selected}
    excluded = read(data / 'exclusions.json')
    require(not set(lookup) & set(excluded['tracks']), 'Previously exposed track selected')
    require(not {r['artist_id'] for r in selected} & set(excluded['artists']), 'Previously exposed artist selected')
    for role, scores, target in [('calibration', cal, cal['y']), ('evaluation', ev, y)]:
        require(len(scores['ids']) == manifest[role + '_tracks'], 'Track count differs')
        require(len(set(scores['artists'])) == manifest[role + '_artists'], 'Artist count differs')
        for tid, artist, actual_y in zip(scores['ids'], scores['artists'], target):
            row = lookup[str(tid)]
            require(row['role'] == role and row['artist_id'] == artist, 'Role or artist differs')
            expected_y = [int(bool(set(row['tags']) & set(config['ontology'][label]))) for label in LABELS]
            require(np.array_equal(actual_y, expected_y), 'Broad labels differ from source tags')
    plans_list = read(data / 'calibration_plans.json')['draws']
    require([d['seed'] for d in plans_list] == config['draw_seeds'], 'Draw seeds differ')
    plans = {(draw['draw'], plan['n_artists']): plan for draw in plans_list for plan in draw['budgets']}
    full = config['budgets'][-1]
    for draw in plans_list:
        require([p['n_artists'] for p in draw['budgets']] == config['budgets'], 'Budget inventory differs')
        require(plans[(draw['draw'], full)] == plans[(1, full)], 'Full-budget copies differ')
    expected = {f"draw_{d:02d}/budget_{b}/{h}/{w}" for d, b in plans if b != full or d == 1 for h in config['head_seeds'] for w in config['weightings']}
    cases = read(data / 'fitted_calibrators.json.gz')['cases']
    require(len(cases) == 440 and {c['key'] for c in cases} == expected, 'Case inventory differs')
    reference_metrics = {r['key']: r for r in read(report / 'case_metrics.json.gz')}
    require(set(reference_metrics) == expected, 'Reference case inventory differs')
    index = {str(t): i for i, t in enumerate(cal['ids'])}
    max_independent = max_oof = max_metrics = max_refit = 0.
    primary, cells, method_metrics = [], {}, {}
    raw_controls, paired, info_rows, diagnostics = {}, {}, [], []
    with threadpool_limits(limits=1):
        for number, case in enumerate(cases, 1):
            info, records = case['case'], case['calibrators']
            info_rows.append(info)
            plan = plans[(info['draw'], info['nominal_budget'])]
            retained = [(t, f) for t, f in zip(plan['track_ids'], plan['folds']) if t in index]
            require(info['actual_track_ids'] == [t for t, _ in retained] and info['actual_folds'] == [f for _, f in retained], 'Subset/folds differ')
            indices = np.asarray([index[t] for t, _ in retained]); folds = np.asarray([f for _, f in retained])
            for artist in set(cal['artists'][indices]):
                require(len(set(folds[cal['artists'][indices] == artist])) == 1, 'Artist split across inner folds')
            h = config['head_seeds'].index(info['head_seed']); w = config['weightings'].index(info['weighting'])
            require(set(records) == set(METHODS), 'Method inventory differs')
            pred = predict_methods(ev['logits'][h, w], records)
            diagnostics.append(numerical_counts(records))
            require(diagnostics[-1] == info['diagnostics'], 'Numerical diagnostic counts differ')
            for method in METHODS:
                independent = np.column_stack([independent_predict(ev['logits'][h, w, :, j], r) for j, r in enumerate(records[method])])
                max_independent = max(max_independent, close(independent, pred[method], 'Independent probability reconstruction'))
                require(pred[method].shape == (637, 4) and np.isfinite(pred[method]).all() and np.all((pred[method] >= 0) & (pred[method] <= 1)), 'Invalid probabilities')
            for j, record in enumerate(records['ridge_platt']):
                max_oof = max(max_oof, verify_oof(cal['logits'][h, w, indices, j], cal['y'][indices, j], folds, record, config['selection_tie_tolerance']))
            if refit:
                fresh = fit_methods(cal['logits'][h, w, indices], cal['y'][indices], folds, cal['offsets'][h, w], config)
                require(numerical_counts(fresh) == diagnostics[-1], 'Refit diagnostic counts differ')
                fresh_pred = predict_methods(ev['logits'][h, w], fresh)
                for method in METHODS:
                    max_refit = max(max_refit, close(fresh_pred[method], pred[method], 'Refitted probabilities', REFIT_TOL))
                for a, b in zip(fresh['ridge_platt'], records['ridge_platt']):
                    require(a['selected_penalty'] == b['selected_penalty'], 'Refit selected penalty differs')
                    for flag in ['final_fit_fallback', 'inner_fallback_count']:
                        require(a[flag] == b[flag], 'Refit fallback differs')
                    for old, new in zip(b['oof_scores'], a['oof_scores']):
                        require(old['penalty'] == new['penalty'], 'Refit OOF inventory differs')
                        max_refit = max(max_refit, close(a['oof_predictions'][str(new['penalty'])], b['oof_predictions'][str(old['penalty'])], 'Refit OOF probabilities', REFIT_TOL))
            for m in ['raw', 'prior_offset']:
                anchor = raw_controls.setdefault((h, w, m), pred[m])
                close(pred[m], anchor, 'Control changed across subsets', 0.)
            pair_key = (info['draw'], info['nominal_budget'], h)
            if pair_key in paired:
                other = paired.pop(pair_key)
                for m in METHODS: close(pred[m][:, [0, 1, 3]], other[m][:, [0, 1, 3]], 'Nonambient pair differs', 0.)
            else:
                paired[pair_key] = pred
            computed = {m: metrics(y, pred[m]) for m in METHODS}
            max_metrics = max(max_metrics, compare_tree(reference_metrics[case['key']]['metrics'], computed, 'Per-case metrics'))
            key = (info['weighting'], info['nominal_budget'])
            cell = cells.setdefault(key, {m: [] for m in METHODS})
            method_metrics.setdefault(key, []).append(computed)
            losses = {m: np.mean((pred[m]-y)**2, axis=1) for m in METHODS}
            for m in METHODS: cell[m].append(losses[m])
            if info['weighting'] == 'unweighted' and info['nominal_budget'] in config['primary']['budgets']:
                primary.append(losses['ridge_platt']-losses['platt'])
            if number % 40 == 0: print(f"Checked {number}/440 cases" + (' including refit' if refit else ''), flush=True)
    require(not paired and len(primary) == 200, 'Pair or primary inventory differs')
    d = np.mean(primary, axis=0)
    stored_boot = load(report / 'primary_artist_bootstrap.npz')
    close(d, stored_boot['per_track_difference'], 'Primary per-track difference')
    reps, ci = bootstrap(d, artists, config['bootstrap_seed'], config['bootstrap_reps'])
    primary_result = read(report / 'primary.json')
    close(reps, stored_boot['estimates'], 'Primary bootstrap draws')
    close(ci, primary_result['ci95'], 'Primary interval')
    close(d.mean(), primary_result['estimate'], 'Primary estimate')
    groups = set(artists)
    checks = {'calibration_artist_retention': len(set(cal['artists'])) >= math.ceil(.8*config['planned_calibration_artists']),
              'evaluation_artist_retention': len(groups) >= max(config['minimum_evaluation_artists'], math.ceil(.8*config['planned_evaluation_artists'])),
              'evaluation_track_retention': len(artists) >= math.ceil(.8*config['planned_evaluation_tracks']),
              'every_calibration_subset_retained': all(r['actual_n_artists'] >= math.ceil(.8*r['nominal_budget']) for r in info_rows)}
    for j, label in enumerate(LABELS):
        positive = sum(np.any(y[artists == g, j]) for g in groups)
        checks[label + '_artist_support'] = bool(positive >= config['minimum_positive_artists'] and len(groups)-positive >= config['minimum_negative_artists'])
    require(checks == read(report / 'coverage.json')['checks'], 'Coverage checks differ')
    conclusion = ('incomplete_evidence' if not all(checks.values()) else 'supports_prespecified_mean_improvement' if ci[1] < 0 else 'supports_mean_deterioration' if ci[0] > 0 else 'inconclusive')
    require(primary_result['conclusion'] == conclusion, 'Primary conclusion differs')
    with (report / 'summary_cells.csv').open(newline='', encoding='utf-8') as f: summary = list(csv.DictReader(f))
    for row in summary:
        key = (row['weighting'], int(row['nominal_budget'])); method = row['method']
        close(np.mean(cells[key][method]), float(row['macro_brier']), 'Summary Brier')
        close(np.mean([r[method]['macro_log_loss'] for r in method_metrics[key]]), float(row['macro_log_loss']), 'Summary log loss')
        for j, label in enumerate(LABELS):
            for suffix, field in [('brier', 'label_brier'), ('average_precision', 'label_average_precision'), ('ece_5', 'label_ece_5'), ('ece_10', 'label_ece_10')]:
                close(np.mean([r[method][field][j] for r in method_metrics[key]]), float(row[label + '_' + suffix]), 'Summary ' + suffix)
    for row in read(report / 'secondary_contrasts.json'):
        cell = cells[(row['weighting'], row['nominal_budget'])]
        comparator = row['contrast'].removeprefix('ridge_platt minus ')
        difference = np.mean(cell['ridge_platt'], axis=0) - np.mean(cell[comparator], axis=0)
        _, interval = bootstrap(difference, artists, config['bootstrap_seed'], config['bootstrap_reps'])
        close(difference.mean(), row['estimate'], 'Secondary estimate')
        close(interval, row['ci95'], 'Secondary interval')
    return {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': sha(data / 'manifest.json'), 'verifier_sha256': sha(Path(__file__)),
            'cases': len(cases), 'probabilities_reconstructed': len(cases)*len(METHODS)*len(y)*4,
            'ridge_oof_selections_checked': len(cases)*4, 'primary_conditions': len(primary),
            'max_independent_probability_error': max_independent, 'max_oof_error': max_oof,
            'max_case_metric_error': max_metrics, 'refit': refit, 'max_refit_probability_or_oof_error': max_refit if refit else None,
            'primary_estimate': float(d.mean()), 'primary_ci95': ci.tolist(), 'conclusion': conclusion,
            'versions': {n: importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','threadpoolctl']},
            'scope': 'Public score-level numerical reproduction; chronology/audio/features are represented by historical receipts, not independently re-audited by this command.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Also refit all 440 cases from frozen calibration scores and folds')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/fresh_confirmation_verification.json')
    args = parser.parse_args()
    out = args.out.resolve()
    for protected in ['data', 'research', 'reports', 'music_calibration_portable', 'scripts', 'paper']:
        require(not out.is_relative_to(ROOT / protected), 'Output must not overwrite published inputs')
    require(not out.exists(), 'Use a new receipt path; existing results are not overwritten')
    result = verify(ROOT, refit=args.refit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
