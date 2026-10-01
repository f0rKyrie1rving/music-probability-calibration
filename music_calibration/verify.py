"""Independent reconstruction of frozen score experiments, never a replication claim.

This module intentionally imports no experiment, fitting, prediction, data-loader
or metric implementation from this package. Only JSON, NumPy and public metric
primitives are used to check the saved numerical records.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['electronic', 'pop', 'ambient', 'rock']


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    _require(p.is_relative_to(root), 'Relative path escapes its artifact root')
    return p


def _require(condition, description):
    if not condition:
        raise ValueError(description)


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=np.float64)))


def _map(z, record):
    """Reconstruct a saved probability function without invoking its producer."""
    z = np.asarray(z, dtype=np.float64)
    _require(np.isfinite(z).all(), 'Nonfinite input logits')
    method = record.get('effective_method', record['method'])
    if record['method'] == 'prior_offset':
        _require(np.isfinite(record['offset']), 'Nonfinite prior offset')
        u = z + record['offset']
    elif method == 'raw':
        u = z
    elif method == 'isotonic':
        x, y = np.asarray(record['x_thresholds']), np.asarray(record['y_thresholds'])
        _require(x.ndim == 1 and len(x) > 0 and x.shape == y.shape and
                 np.isfinite(x).all() and np.isfinite(y).all() and
                 (np.diff(x) > 0).all() and (np.diff(y) >= 0).all() and
                 ((y >= 0) & (y <= 1)).all(), 'Invalid isotonic map')
        return np.interp(z, x, y)
    elif method == 'beta':
        a, b, c = record['a'], record['b'], record['intercept']
        _require(np.isfinite([a, b, c]).all() and .001 <= a <= 100 and .001 <= b <= 100
                 and -20 <= c <= 20, 'Invalid beta map')
        u = -a * np.logaddexp(0, -z) + b * np.logaddexp(0, z) + c
    elif method in ('temperature', 'platt', 'ridge_platt'):
        a, b = record['slope'], record['intercept']
        _require(np.isfinite([a, b]).all() and .001 <= a <= 100 and -20 <= b <= 20,
                 'Invalid sigmoid map')
        if method == 'temperature':
            _require(b == 0 and np.isclose(record['temperature'], 1 / a, atol=1e-12, rtol=0),
                     'Invalid temperature')
        u = a * z + b
    else:
        raise ValueError('Unknown effective method: ' + method)
    return _sigmoid(u)


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.max_error = 0.0

    def truth(self, condition, category, description):
        _require(condition, description)
        self.counts[category] += 1

    def close(self, actual, expected, category, description, tolerance=1e-11):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.truth(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                   category, description + ': invalid shape or finite values')
        error = float(np.max(np.abs(a - b))) if a.size else 0.
        self.max_error = max(self.max_error, error)
        self.truth(error <= tolerance, category, description + f': error {error}')

    def equal(self, actual, expected, category, description):
        self.truth(np.array_equal(actual, expected), category, description)

    def nested(self, actual, expected, category, description):
        if isinstance(expected, dict):
            self.truth(isinstance(actual, dict) and set(actual) == set(expected), category,
                       description + ': schema differs')
            for key, value in expected.items():
                self.nested(actual[key], value, category, description + '/' + str(key))
        elif isinstance(expected, list):
            self.truth(isinstance(actual, list) and len(actual) == len(expected), category,
                       description + ': length differs')
            for i, value in enumerate(expected):
                self.nested(actual[i], value, category, description + '/' + str(i))
        elif isinstance(expected, float):
            self.close(actual, expected, category, description)
        else:
            self.truth(type(actual) is type(expected) and actual == expected, category, description)

    def hashes(self, root, hashes, category):
        for name, expected in hashes.items():
            self.truth(_sha(_path(root, name)) == expected, category, 'Hash changed: ' + name)


def _metrics(y, p):
    """Independent reduction and reliability-bin reconstruction."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    _require(y.shape == p.shape and y.ndim == 2 and y.shape[1] == 4 and len(y) and
             np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
             'Invalid predictions for scoring')
    brier = np.sum(np.square(p - y), axis=0) / len(y)
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(safe) + (1 - y) * np.log1p(-safe)).sum(axis=0) / len(y)
    results = {'macro_brier': float(sum(brier) / 4), 'label_brier': brier.tolist(),
               'macro_log_loss': float(sum(loss) / 4), 'label_log_loss': loss.tolist(),
               'label_average_precision': [float(average_precision_score(y[:, j], p[:, j]))
                    if np.unique(y[:, j]).size == 2 else None for j in range(4)],
               'label_mean_probability': (p.sum(axis=0) / len(y)).tolist(),
               'label_positive_fraction': (y.sum(axis=0) / len(y)).tolist()}
    for count in (5, 10):
        reliability, ece = [], []
        for j in range(4):
            assignment = np.floor(p[:, j] * count).astype(np.int64)
            assignment[p[:, j] == 1] = count - 1
            rows, error = [], 0.
            for index in range(count):
                values, targets = p[assignment == index, j], y[assignment == index, j]
                n = len(values)
                mean, rate = (float(values.sum() / n), float(targets.sum() / n)) if n else (None, None)
                rows.append({'bin': index, 'count': n, 'mean_probability': mean, 'positive_fraction': rate})
                if n:
                    error += n * abs(mean - rate) / len(y)
            reliability.append(rows)
            ece.append(error)
        results['reliability_' + str(count)] = reliability
        results['label_ece_' + str(count)] = ece
    return results


def _fit_record(check, z, y, record, description):
    """Check fitted objectives against precisely the supplied training rows."""
    method = record['method']
    _require(record['success'] == (not record['fallback']), description + ': inconsistent success/fallback')
    if method == 'prior_offset':
        return
    if record['fallback']:
        check.truth(record['effective_method'] == 'raw' and record['reason'] in
                    ['single_class_calibration', 'optimizer_nonconvergence', 'nonfinite_fit'],
                    'fallbacks', description + ': invalid fallback')
        if record['reason'] == 'single_class_calibration':
            check.truth(np.unique(y).size == 1, 'fallbacks', description + ': class support exists')
        elif record['reason'] == 'optimizer_nonconvergence':
            check.truth(not record['optimizer_success'], 'fallbacks', description + ': optimizer succeeded')
        return
    if record['effective_method'] == 'raw':
        check.truth(record['slope'] == 1 and record['intercept'] == 0,
                    'parameters', description + ': identity changed')
        return
    check.truth(np.unique(y).size == 2, 'parameters', description + ': absent outcome')
    if method == 'isotonic':
        _map(z, record)
        return
    if method == 'beta':
        u = (-record['a'] * np.logaddexp(0, -z) +
             record['b'] * np.logaddexp(0, z) + record['intercept'])
    else:
        u = record['slope'] * z + record['intercept']
    nll = float((np.logaddexp(0, u) - y * u).mean())
    penalty = record['penalty'] * ((record['slope'] - 1) ** 2 + record['intercept'] ** 2) if method == 'ridge_platt' else 0.
    check.close(record['calibration_log_loss'], nll, 'fitting_objectives', description + ': log loss')
    check.close(record['objective'], nll + penalty, 'fitting_objectives', description + ': objective')
    check.close(record['initial_objective'], float((np.logaddexp(0, z) - y * z).mean()),
                'fitting_objectives', description + ': initial objective')
    check.truth(record['optimizer_success'] and record['success'], 'fitting_objectives', description + ': failed fit')


def _ridge(check, z, y, folds, record, config):
    expected_grid = ['identity'] + config['ridge_penalties']
    rows = record['oof_scores']
    check.truth([row['penalty'] for row in rows] == expected_grid, 'selection', 'Ridge candidate grid differs')
    expected_keys = {'identity'} | {str(p) for p in config['ridge_penalties']}
    check.truth(set(record['oof_predictions']) == expected_keys, 'selection', 'OOF candidates missing')
    scores, inner_fallbacks = [], 0
    for row in rows:
        penalty = row['penalty']
        if penalty == 'identity':
            reconstructed = _sigmoid(z)
            check.truth(row['fold_fits'] == [] and row['inner_fallback_count'] == 0,
                        'selection', 'Identity unexpectedly fitted')
            fallback_count = 0
        else:
            reconstructed = np.empty(len(y), dtype=float)
            fits = row['fold_fits']
            check.equal([fit['fold'] for fit in fits], np.unique(folds), 'selection', 'Ridge fold records differ')
            fallback_count = 0
            for fitted in fits:
                valid = folds == fitted['fold']
                check.truth(fitted['train_n'] == int((~valid).sum()) and
                            fitted['validation_n'] == int(valid.sum()), 'selection', 'OOF train/validation count changed')
                parameters = fitted['fit']
                check.truth(parameters['method'] == 'ridge_platt' and parameters['penalty'] == penalty,
                            'selection', 'OOF fit penalty differs')
                _fit_record(check, z[~valid], y[~valid], parameters, 'Ridge inner training objective')
                reconstructed[valid] = _map(z[valid], parameters)
                fallback_count += int(parameters['fallback'])
            check.truth(fallback_count == row['inner_fallback_count'], 'fallbacks', 'Inner fallback count differs')
        stored = record['oof_predictions'][str(penalty)]
        check.close(stored, reconstructed, 'oof_predictions', 'OOF predictions do not reconstruct')
        brier = float(np.square(reconstructed - y).mean())
        check.close(row['brier'], brier, 'selection', 'OOF candidate score differs')
        scores.append((penalty, brier, fallback_count))
        inner_fallbacks += fallback_count
    best = min(row[1] for row in scores)
    tied = [row for row in scores if row[1] <= best + config['selection_tie_tolerance']]
    chosen = max(tied, key=lambda row: np.inf if row[0] == 'identity' else row[0])
    check.truth(record['selected_penalty'] == chosen[0], 'selection', 'Wrong strongest/identity tie selection')
    if chosen[0] == 'identity':
        check.truth(record['effective_method'] == 'raw' and not record['fallback'], 'selection', 'Selected identity changed')
    else:
        check.truth(record['penalty'] == chosen[0], 'selection', 'Final ridge penalty differs')
    check.truth(record['inner_fallback_count'] == inner_fallbacks and
                record['selected_inner_fallback_count'] == chosen[2] and
                record['final_fit_fallback'] == record['fallback'], 'fallbacks', 'Policy fallback totals differ')
    selection = record['selection']
    check.truth(selection['criterion'] == 'track_weighted_oof_brier' and
                selection['tie_rule'] == 'identity_then_strongest_penalty' and
                selection['tie_tolerance'] == config['selection_tie_tolerance'] and
                selection['fold_count'] == len(np.unique(folds)) and
                selection['uses_evaluation_data'] is False, 'selection', 'Selection contract differs')
    check.close(selection['minimum_oof_brier'], best, 'selection', 'Minimum OOF score differs')
    check.close(selection['selected_oof_brier'], chosen[1], 'selection', 'Selected OOF score differs')


def _budget_plans(check, plans, artists, ids, seed, config):
    unique = np.unique(artists)
    order = np.random.default_rng(np.random.SeedSequence([seed, config['budget_seed']])).permutation(unique)
    check.truth(len(plans) == len(config['budgets']), 'plans', 'Missing budget plan')
    previous = set()
    for index, (plan, fraction) in enumerate(zip(plans, config['budgets'])):
        selected = order[:int(np.ceil(fraction * len(unique)))]
        rows = np.flatnonzero(np.isin(artists, selected))
        random = np.random.default_rng(np.random.SeedSequence([seed, config['inner_seed'], index]))
        folded = random.permutation(np.sort(selected))
        mapping = {str(artist): i % config['inner_folds'] for i, artist in enumerate(folded)}
        folds = [mapping[str(artist)] for artist in artists[rows]]
        expected = {'fraction': fraction, 'artist_ids': sorted(map(str, selected)),
                    'indices': rows.tolist(), 'folds': folds, 'n_artists': len(selected),
                    'n_tracks': len(rows), 'track_ids': ids[rows].tolist()}
        check.nested(plan, expected, 'plans', 'Frozen artist budget/fold plan')
        check.truth(previous.issubset(set(selected)), 'plans', 'Artist budgets are not nested')
        check.truth(len(np.unique(folds)) == config['inner_folds'], 'plans', 'Fold missing')
        previous = set(selected)


def _verify(out):
    check = Checks()
    frozen = _read(out / 'freeze.json')
    bundle = (out / frozen['bundle_relative_to_run']).resolve()
    check.truth(_sha(bundle / 'manifest.json') == frozen['bundle_manifest_sha256'], 'hashes', 'Input manifest changed')
    check.hashes(bundle, frozen['bundle_metadata_hashes'], 'hashes')
    check.hashes(out, frozen['local_hashes'], 'hashes')
    check.hashes(ROOT, frozen['source_hashes'], 'hashes')
    check.hashes(out / 'source', frozen['source_hashes'], 'hashes')
    for name, expected in frozen['versions'].items():
        check.truth(importlib.metadata.version(name) == expected, 'runtime', 'Numerical runtime changed: ' + name)
    manifest, config, plans = [_read(path) for path in
                              [bundle / 'manifest.json', out / 'config.json', out / 'plans.json']]
    check.hashes(bundle, manifest['provenance_hashes'], 'hashes')
    check.truth(manifest['schema_version'] == 1 and manifest['labels'] == LABELS, 'inputs', 'Wrong score schema')
    cases = manifest['cases']
    check.truth(len(cases) == config['expected_source_cases'] and len({c['id'] for c in cases}) == len(cases),
                'completeness', 'Input cases missing or duplicated')
    check.truth(len(set(config['methods'])) == len(config['methods']), 'completeness', 'Duplicate method')
    expected_paths = [f"cases/{case['id']}/budget_{i}" for case in cases for i in range(len(config['budgets']))]
    completed = _read(out / 'completed.json')
    check.truth(completed['state'] == 'complete' and set(completed['cases']) == set(expected_paths) and
                len(completed['cases']) == len(expected_paths) and set(completed['status_hashes']) == set(expected_paths),
                'completeness', 'Completed case list does not match the entire protocol')
    actual_paths = {p.parent.relative_to(out).as_posix() for p in (out / 'cases').rglob('status.json')}
    check.truth(actual_paths == set(expected_paths), 'completeness', 'Unexpected or missing case directory')
    check.truth(set(plans) == {str(c['seed']) for c in cases}, 'plans', 'Wrong set of outer seeds')
    tracks, anchors, weighting_pairs = {}, {}, {}
    loaded, fallback_count = {}, 0
    for case in cases:
        path = _path(bundle, case['path'])
        check.truth(_sha(path) == case['sha256'], 'hashes', 'Source score hash differs: ' + case['id'])
        data = _arrays(path)
        for part in ('fit', 'cal', 'eval'):
            ids, artists, y = [data[part + '_' + key] for key in ('ids', 'artists', 'y')]
            check.truth(ids.ndim == 1 and ids.dtype.kind in 'US' and artists.shape == ids.shape and
                        artists.dtype.kind in 'US' and y.shape == (len(ids), 4) and len(set(ids)) == len(ids) and
                        len(ids) > 0 and np.isin(y, [0, 1]).all(), 'alignment', 'Invalid source rows')
            for track, artist, target in zip(ids, artists, y):
                value = (str(artist), tuple(map(int, target)))
                check.truth(str(track) not in tracks or tracks[str(track)] == value,
                            'alignment', 'Track artist/target changed across cases')
                tracks[str(track)] = value
            if part != 'fit':
                check.truth(data[part + '_logits'].shape == y.shape and np.isfinite(data[part + '_logits']).all(),
                            'alignment', 'Invalid source logits')
        for a, b in [('fit', 'cal'), ('fit', 'eval'), ('cal', 'eval')]:
            for key in ('ids', 'artists'):
                check.truth(not set(data[a + '_' + key]) & set(data[b + '_' + key]),
                            'artist_boundaries', 'Source track/artist leakage')
        anchor = anchors.setdefault(case['seed'], data)
        for part in ('fit', 'cal', 'eval'):
            for key in ('ids', 'artists', 'y'):
                check.equal(data[part + '_' + key], anchor[part + '_' + key], 'alignment', 'Unpaired split rows')
        _budget_plans(check, plans[str(case['seed'])], data['cal_artists'], data['cal_ids'], case['seed'], config)
        offsets = np.zeros(4)
        if case['weighting'] == 'ambient_balanced':
            positives = data['fit_y'][:, 2].sum()
            check.truth(0 < positives < len(data['fit_y']), 'offsets', 'Degenerate fit prevalence')
            offsets[2] = np.log(positives / (len(data['fit_y']) - positives))
        else:
            check.truth(case['weighting'] == 'unweighted', 'offsets', 'Unknown weighting')
        check.close(data['offsets'], offsets, 'offsets', 'Offset did not use fit labels', tolerance=1e-14)
        paired = weighting_pairs.setdefault((case['seed'], case['representation']), {})
        paired[case['weighting']] = data
        for budget_index, plan in enumerate(plans[str(case['seed'])]):
            relative = f"cases/{case['id']}/budget_{budget_index}"
            directory = _path(out, relative)
            check.truth(_sha(directory / 'status.json') == completed['status_hashes'][relative],
                        'hashes', 'Case status hash differs')
            status = _read(directory / 'status.json')
            check.truth(status['state'] == 'complete' and set(status['hashes']) ==
                        {'calibrators.json', 'predictions.npz', 'metrics.json', 'case.json'},
                        'completeness', 'Incomplete case status')
            check.hashes(directory, status['hashes'], 'hashes')
            fitted, saved_metrics, case_record = [_read(directory / name) for name in
                                                ('calibrators.json', 'metrics.json', 'case.json')]
            predictions = _arrays(directory / 'predictions.npz')
            check.truth(set(fitted) == set(saved_metrics) == set(config['methods']) and
                        set(predictions) == set(config['methods']) | {'ids', 'artists', 'y'},
                        'completeness', 'Missing/extra methods')
            for key in ('ids', 'artists', 'y'):
                check.equal(predictions[key], data['eval_' + key], 'alignment', 'Evaluation row mismatch')
            ci, folds = np.asarray(plan['indices']), np.asarray(plan['folds'])
            historical_errors = {}
            for method in config['methods']:
                check.truth(len(fitted[method]) == 4, 'parameters', 'Missing label calibrator')
                reconstructed = []
                for label, record in enumerate(fitted[method]):
                    check.truth(record['method'] == method, 'parameters', 'Wrong label method')
                    _fit_record(check, data['cal_logits'][ci, label], data['cal_y'][ci, label], record,
                                relative + '/' + method + '/' + str(label))
                    if method == 'prior_offset':
                        check.close(record['offset'], offsets[label], 'offsets', 'Saved offset changed')
                    if method == 'ridge_platt':
                        _ridge(check, data['cal_logits'][ci, label], data['cal_y'][ci, label], folds, record, config)
                    reconstructed.append(_map(data['eval_logits'][:, label], record))
                    fallback_count += int(record['fallback'])
                p = np.column_stack(reconstructed)
                check.close(predictions[method], p, 'predictions', 'Saved predictions do not reconstruct')
                check.nested(saved_metrics[method], _metrics(data['eval_y'], predictions[method]),
                             'metrics', 'Probability metrics and reliability')
                if method in ('raw', 'prior_offset') or (plan['fraction'] == 1 and method in ('platt', 'temperature')):
                    reference = data['reference_' + method]
                    check.close(predictions[method], reference, 'historical_controls', 'Historical control differs',
                                tolerance=config['historical_prediction_tolerance'])
                    historical_errors[method] = float(np.max(np.abs(predictions[method] - reference)))
            check.nested(case_record, {**case, 'budget_index': budget_index, 'fraction': plan['fraction'],
                         'calibration_artists': plan['n_artists'], 'calibration_tracks': plan['n_tracks'],
                         'historical_control_max_error': historical_errors}, 'case_metadata', 'Case metadata differs')
            loaded[(case['seed'], case['representation'], case['weighting'], budget_index)] = {
                'predictions': predictions, 'metrics': saved_metrics, 'fitted': fitted,
                'case': case_record, 'relative_path': relative}
    for (seed, representation), pair in weighting_pairs.items():
        check.truth(set(pair) == {'ambient_balanced', 'unweighted'}, 'negative_controls', 'Unpaired weighting')
        for part in ('cal', 'eval'):
            check.equal(pair['ambient_balanced'][part + '_logits'][:, [0, 1, 3]],
                        pair['unweighted'][part + '_logits'][:, [0, 1, 3]],
                        'negative_controls', 'Unchanged-label logits differ between weightings')
        for i in range(len(config['budgets'])):
            for method in config['methods']:
                check.equal(loaded[(seed, representation, 'ambient_balanced', i)]['predictions'][method][:, [0, 1, 3]],
                            loaded[(seed, representation, 'unweighted', i)]['predictions'][method][:, [0, 1, 3]],
                            'negative_controls', 'Unchanged-label predictions differ between weightings')
    if (out / 'summary.json').exists():
        _summary(check, _read(out / 'summary.json'), loaded, config)
    return {'status': 'passed', 'cases': len(expected_paths), 'source_cases': len(cases),
            'unique_tracks': len(tracks), 'unique_artists': len({v[0] for v in tracks.values()}),
            'check_count': sum(check.counts.values()), 'check_categories': dict(sorted(check.counts.items())),
            'maximum_absolute_comparison_error': check.max_error, 'label_fit_fallbacks': fallback_count,
            'summary_verified': (out / 'summary.json').exists(), 'errors': []}


def _summary(check, summary, loaded, config):
    groups, fallback_total, ridge_inner, ridge_selected = {}, 0, 0, 0
    selection_counts = Counter()
    for entry in loaded.values():
        case, metrics, fitted = entry['case'], entry['metrics'], entry['fitted']
        for method in config['methods']:
            key = (case['representation'], case['weighting'], case['fraction'], method)
            score = metrics[method]['macro_brier']
            records = fitted[method]
            fallback = sum(int(r['fallback']) for r in records)
            groups.setdefault(key, []).append((score, score - metrics['raw']['macro_brier'],
                score - metrics['platt']['macro_brier'], fallback,
                sum(int(r['success']) for r in records),
                sum(int(r.get('bound_hit', False)) for r in records)))
            fallback_total += fallback
            if method == 'ridge_platt':
                for record in records:
                    ridge_inner += record['inner_fallback_count']
                    ridge_selected += record['selected_inner_fallback_count']
                    selection_counts[str(record['selected_penalty'])] += 1
    cells = []
    for key, records in sorted(groups.items()):
        array = np.asarray(records)
        delta, platt = array[:, 1], array[:, 2]
        ordered = sorted(delta.tolist())
        half = len(ordered) // 2
        median = ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2
        cell = dict(zip(['representation', 'weighting', 'fraction', 'method'], key))
        cell.update(n=len(records), mean_brier=float(array[:, 0].sum() / len(records)),
                    mean_delta_raw=float(delta.sum() / len(records)), median_delta_raw=float(median),
                    min_delta_raw=float(min(delta)), max_delta_raw=float(max(delta)),
                    wins=sum(v < -1e-12 for v in delta.tolist()),
                    ties=sum(abs(v) <= 1e-12 for v in delta.tolist()),
                    losses=sum(v > 1e-12 for v in delta.tolist()),
                    mean_delta_platt=float(platt.sum() / len(records)),
                    wins_vs_platt=sum(v < -1e-12 for v in platt.tolist()),
                    final_fallback_labels=int(array[:, 3].sum()),
                    successful_labels=int(array[:, 4].sum()), bound_hit_labels=int(array[:, 5].sum()))
        cells.append(cell)
    expected = {'scope': 'Exploratory, previously observed data; 20 overlapping splits per cell, not independent datasets',
                'cases': len(loaded), 'method_cases': len(loaded) * len(config['methods']), 'cells': cells,
                'final_fallback_labels': fallback_total, 'ridge_inner_fallbacks': ridge_inner,
                'ridge_selected_inner_fallbacks': ridge_selected, 'ridge_selection_counts': dict(selection_counts)}
    check.nested(summary, expected, 'summary', 'Descriptive summary reconstruction')


def verify(out):
    out = Path(out).resolve()
    try:
        result = _verify(out)
    except Exception as error:
        result = {'status': 'failed', 'errors': [repr(error)],
                  'scope': 'Numerical and artifact verification, not independent scientific replication.'}
        (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        raise
    result.update({'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'verifier_sha256': _sha(__file__),
                   'independence': 'No package fitting, prediction, metric, experiment or data-loader functions imported.',
                   'scope': 'Numerical and artifact verification, not independent scientific replication.'})
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result
