#!/usr/bin/env python3
"""Export aligned portable probabilities and strictly compare six numerical-v2 runs.

Every evaluation probability and every candidate OOF probability is compared at
absolute tolerance 1e-8, with no relative tolerance or diagnostic exceptions.
Model selections, fallback decisions, reasons and row/fold alignment are exact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

_SPEC = importlib.util.spec_from_file_location('strict_report_comparison', Path(__file__).with_name('compare_reports.py'))
_reports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_reports)

RUNS = ['budget_original', 'draw_01', 'draw_02', 'draw_03', 'draw_04', 'draw_05']
METHODS = ['raw', 'prior_offset', 'temperature', 'platt', 'beta', 'isotonic', 'ridge_platt']
LABELS = ['electronic', 'pop', 'ambient', 'rock']
CANDIDATES = ['identity', '0.0', '0.001', '0.01', '0.1', '1.0']
ATOL = 1e-8
INDEX_FILE = 'portable_index.json'
ARRAY_FILE = 'portable_fits.npz'
ARRAY_KEYS = {'eval_probabilities', 'oof_probabilities'}
RECORD_KEYS = {'id', 'eval_ids_sha256', 'cal_ids_sha256', 'eval_targets_sha256', 'cal_folds_sha256',
               'eval_start', 'eval_stop', 'oof', 'oof_scores', 'selected_penalty', 'final_fallback',
               'final_fallback_reason', 'candidate_fold_fallbacks'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _safe_case(path):
    parts = Path(path).parts
    require(isinstance(path, str) and len(parts) == 5 and parts[0] == 'cases'
            and not Path(path).is_absolute() and all(part not in ('.', '..') for part in parts)
            and re.fullmatch(r'budget_[0-9]+', parts[-1]) is not None,
            'Unsupported case path: ' + str(path))
    return path


def _fallback(value, reason, context):
    require(type(value) is bool, context + ': fallback must be boolean')
    require((reason is None if not value else isinstance(reason, str) and bool(reason)),
            context + ': fallback reason must match fallback decision')
    require(not value or reason == 'single_class_calibration',
            context + ': numerical-failure fallback is prohibited in numerical-v2')


def _probabilities(values, context):
    require(values.dtype == np.dtype('float64') and values.ndim == 1,
            context + ': expected a flat float64 probability array')
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
            context + ': nonfinite or out-of-range probability')


def export_run_signatures(runs: dict[str, Path], destination: Path):
    """Export completed saved runs; never access audio, feature caches or refit."""
    destination = Path(destination)
    require(set(runs) == set(RUNS), 'Exactly the six declared run names are required')
    destination.mkdir(parents=True, exist_ok=True)
    require(not (destination / INDEX_FILE).exists() and not (destination / ARRAY_FILE).exists(),
            'Portable signatures already exist; refusing to overwrite')
    benchmark = {'schema_version': 1, 'runs': RUNS}
    if (destination / 'benchmark.json').exists():
        require(_reports._json(destination / 'benchmark.json') == benchmark, 'Benchmark run contract changed')
    records, eval_chunks, oof_chunks = [], [], []
    eval_offset = oof_offset = 0
    source_hashes = {}
    for name in RUNS:
        root = Path(runs[name])
        completed = _reports._json(root / 'completed.json')
        config = _reports._json(root / 'config.json')
        plans = _reports._json(root / 'plans.json')
        require(completed.get('state') == 'complete', name + ': run is incomplete')
        cases = completed.get('cases')
        require(isinstance(cases, list) and len(cases) == len(set(cases))
                and len(cases) == config['expected_source_cases'] * len(config['budgets']), name + ': incomplete case inventory')
        require(config['methods'] == METHODS and [str(x) for x in config['ridge_penalties']] == CANDIDATES[1:]
                and config['inner_folds'] == 3, name + ': unsupported method/grid/fold contract')
        require(set(completed.get('status_hashes', {})) == set(cases), name + ': missing status inventory')
        for filename in ['completed.json', 'config.json', 'plans.json']:
            source_hashes[name + '/' + filename] = (root / filename, _reports._sha(root / filename))
        for case_path in sorted(cases):
            _safe_case(case_path)
            case_root = root / case_path
            status_path = case_root / 'status.json'
            require(_reports._sha(status_path) == completed['status_hashes'][case_path], name + '/' + case_path + ': changed status')
            status = _reports._json(status_path)
            require(status.get('state') == 'complete' and set(status.get('hashes', {})) ==
                    {'case.json', 'calibrators.json', 'predictions.npz', 'metrics.json'}, 'Incomplete case artifacts')
            for filename, expected in status['hashes'].items():
                require(_reports._sha(case_root / filename) == expected, name + '/' + case_path + ': changed ' + filename)
                source_hashes[name + '/' + case_path + '/' + filename] = (case_root / filename, expected)
            source_hashes[name + '/' + case_path + '/status.json'] = (status_path, completed['status_hashes'][case_path])
            case = _reports._json(case_root / 'case.json')
            plan = plans[str(case['seed'])][case['budget_index']]
            require(case_path == 'cases/' + case['id'] + '/budget_' + str(case['budget_index']), 'Case identity does not match path')
            require(plan['n_tracks'] == case['calibration_tracks'] == len(plan['track_ids']) == len(plan['folds']), 'Calibration alignment changed')
            require(len(set(plan['track_ids'])) == len(plan['track_ids']), 'Duplicate calibration IDs')
            fitted = _reports._json(case_root / 'calibrators.json')
            require(set(fitted) == set(METHODS), 'Calibrator method inventory changed')
            with np.load(case_root / 'predictions.npz', allow_pickle=False) as archive:
                require(set(archive.files) == {'ids', 'artists', 'y'} | set(METHODS)
                        and len(archive.files) == len(set(archive.files)), 'Prediction array inventory changed')
                ids, y = archive['ids'], archive['y']
                require(ids.ndim == 1 and len(ids) > 0 and len(set(ids.tolist())) == len(ids), 'Invalid evaluation IDs')
                require(y.shape == (len(ids), len(LABELS)) and np.isin(y, [0, 1]).all(), 'Invalid evaluation labels')
                alignment = dict(eval_ids_sha256=canonical_hash(ids.tolist()), cal_ids_sha256=canonical_hash(plan['track_ids']),
                                 eval_targets_sha256=canonical_hash(y.tolist()), cal_folds_sha256=canonical_hash(plan['folds']))
                for method in METHODS:
                    matrix = archive[method]
                    require(matrix.dtype == np.dtype('float64') and matrix.shape == y.shape, 'Prediction shape/dtype changed')
                    require(len(fitted[method]) == len(LABELS), 'Calibrator label count changed')
                    for label_index, label in enumerate(LABELS):
                        fit = fitted[method][label_index]
                        values = matrix[:, label_index].copy()
                        _probabilities(values, method + '/' + label)
                        record = {'id': name + '/' + case_path + '/' + method + '/' + label, **alignment,
                            'eval_start': eval_offset, 'eval_stop': eval_offset + len(values), 'oof': [], 'oof_scores': [],
                            'selected_penalty': None, 'final_fallback': fit['fallback'],
                            'final_fallback_reason': fit.get('reason'), 'candidate_fold_fallbacks': []}
                        _fallback(record['final_fallback'], record['final_fallback_reason'], record['id'])
                        eval_offset += len(values)
                        eval_chunks.append(values)
                        if method == 'ridge_platt':
                            record['selected_penalty'] = str(fit['selected_penalty'])
                            require([str(candidate['penalty']) for candidate in fit['oof_scores']] == CANDIDATES
                                    and set(fit['oof_predictions']) == set(CANDIDATES), 'OOF candidate inventory changed')
                            for candidate in fit['oof_scores']:
                                key = str(candidate['penalty'])
                                values = np.asarray(fit['oof_predictions'][key], dtype=np.float64)
                                _probabilities(values, record['id'] + '/OOF/' + key)
                                require(len(values) == len(plan['track_ids']), 'OOF row alignment changed')
                                record['oof'].append({'candidate': key, 'start': oof_offset, 'stop': oof_offset + len(values)})
                                record['oof_scores'].append({'candidate': key, 'brier': candidate['brier']})
                                oof_offset += len(values)
                                oof_chunks.append(values)
                                for fold in candidate['fold_fits']:
                                    fold_fit = fold['fit']
                                    record['candidate_fold_fallbacks'].append({'candidate': key, 'fold': fold['fold'],
                                        'fallback': fold_fit['fallback'], 'reason': fold_fit.get('reason')})
                        records.append(record)
    arrays = {'eval_probabilities': np.concatenate(eval_chunks),
              'oof_probabilities': np.concatenate(oof_chunks) if oof_chunks else np.array([], dtype=np.float64)}
    index = {'schema_version': 1, 'records': records}
    _validate_index(index, arrays, benchmark)
    for path, expected in source_hashes.values():
        require(_reports._sha(path) == expected, 'Source changed during signature export: ' + path.name)
    with tempfile.NamedTemporaryFile(dir=destination, prefix='.portable-', suffix='.npz', delete=False) as temporary:
        array_temporary = Path(temporary.name)
    try:
        np.savez_compressed(array_temporary, **arrays)
        array_temporary.replace(destination / ARRAY_FILE)
        _reports._atomic_json(destination / INDEX_FILE, index)
        if not (destination / 'benchmark.json').exists():
            _reports._atomic_json(destination / 'benchmark.json', benchmark)
    finally:
        array_temporary.unlink(missing_ok=True)
    return {'runs': len(RUNS), 'records': len(records), 'eval_probabilities': eval_offset, 'oof_probabilities': oof_offset,
            'portable_index_sha256': _reports._sha(destination / INDEX_FILE),
            'portable_fits_sha256': _reports._sha(destination / ARRAY_FILE)}


def _validate_benchmark(benchmark):
    require(isinstance(benchmark, dict) and benchmark.keys() == {'schema_version', 'runs'}
            and type(benchmark['schema_version']) is int and benchmark['schema_version'] == 1
            and benchmark['runs'] == RUNS, 'Unsupported benchmark schema/run order')


def _validate_index(index, arrays, benchmark):
    _validate_benchmark(benchmark)
    require(isinstance(index, dict) and index.keys() == {'schema_version', 'records'} and type(index['schema_version']) is int
            and index['schema_version'] == 1 and isinstance(index['records'], list) and bool(index['records']), 'Unsupported index schema')
    require(set(arrays) == ARRAY_KEYS, 'Portable NPZ keys changed')
    for key, values in arrays.items():
        _probabilities(values, key)
    eval_cursor = oof_cursor = 0
    seen, previous = set(), None
    runs_seen = set()
    alignment_by_case = {}
    for record in index['records']:
        require(isinstance(record, dict) and record.keys() == RECORD_KEYS, 'Unexpected fit record schema')
        identifier = record['id']
        require(isinstance(identifier, str) and identifier not in seen, 'Duplicate/invalid fit ID')
        seen.add(identifier)
        parts = identifier.split('/')
        require(len(parts) == 8 and parts[0] in RUNS and parts[-2] in METHODS and parts[-1] in LABELS, 'Invalid fit ID')
        _safe_case('/'.join(parts[1:6]))
        order = (RUNS.index(parts[0]), '/'.join(parts[1:6]), METHODS.index(parts[-2]), LABELS.index(parts[-1]))
        require(previous is None or order > previous, 'Fit records reordered')
        previous = order
        runs_seen.add(parts[0])
        for key in ['eval_ids_sha256', 'cal_ids_sha256', 'eval_targets_sha256', 'cal_folds_sha256']:
            require(isinstance(record[key], str) and re.fullmatch('[0-9a-f]{64}', record[key]) is not None, 'Invalid alignment hash')
        alignment = tuple(record[key] for key in ['eval_ids_sha256', 'cal_ids_sha256', 'eval_targets_sha256', 'cal_folds_sha256'])
        case_key = '/'.join(parts[:6])
        require(alignment_by_case.setdefault(case_key, alignment) == alignment, 'Case alignment differs between fits')
        require(type(record['eval_start']) is int and type(record['eval_stop']) is int
                and record['eval_start'] == eval_cursor and record['eval_stop'] > eval_cursor, 'Evaluation slice gap/overlap/empty')
        eval_cursor = record['eval_stop']
        _fallback(record['final_fallback'], record['final_fallback_reason'], identifier)
        for key in ['oof', 'oof_scores', 'candidate_fold_fallbacks']:
            require(isinstance(record[key], list), 'Expected record list: ' + key)
        if parts[-2] != 'ridge_platt':
            require(record['selected_penalty'] is None and not record['oof'] and not record['oof_scores']
                    and not record['candidate_fold_fallbacks'], 'Non-ridge record has selection/OOF data')
            continue
        require(record['selected_penalty'] in CANDIDATES, 'Unsupported selected penalty')
        require(len(record['oof']) == len(CANDIDATES) == len(record['oof_scores']), 'Missing/extra OOF candidates')
        oof_length = None
        for candidate, score, name in zip(record['oof'], record['oof_scores'], CANDIDATES):
            require(isinstance(candidate, dict) and candidate.keys() == {'candidate', 'start', 'stop'}
                    and candidate['candidate'] == name, 'OOF candidate identity/order changed')
            require(type(candidate['start']) is int and type(candidate['stop']) is int
                    and candidate['start'] == oof_cursor and candidate['stop'] > oof_cursor, 'OOF slice gap/overlap/empty')
            length = candidate['stop'] - candidate['start']
            require(oof_length is None or length == oof_length, 'OOF lengths differ across candidates')
            oof_length, oof_cursor = length, candidate['stop']
            require(isinstance(score, dict) and score.keys() == {'candidate', 'brier'} and score['candidate'] == name
                    and type(score['brier']) is float and math.isfinite(score['brier']) and 0 <= score['brier'] <= 1, 'Malformed OOF score')
        expected_folds = [(candidate, fold) for candidate in CANDIDATES[1:] for fold in range(3)]
        require(len(record['candidate_fold_fallbacks']) == len(expected_folds), 'Missing candidate fold records')
        for fold, expected in zip(record['candidate_fold_fallbacks'], expected_folds):
            require(isinstance(fold, dict) and fold.keys() == {'candidate', 'fold', 'fallback', 'reason'}
                    and (fold['candidate'], fold['fold']) == expected and type(fold['fold']) is int, 'Candidate fold identity/order changed')
            _fallback(fold['fallback'], fold['reason'], identifier + '/fold')
    require(runs_seen == set(RUNS), 'Index does not contain all six runs')
    require(eval_cursor == arrays['eval_probabilities'].size and oof_cursor == arrays['oof_probabilities'].size,
            'Unindexed or truncated probability values')


def _load(root):
    benchmark = _reports._json(root / 'benchmark.json')
    index = _reports._json(root / INDEX_FILE)
    with np.load(root / ARRAY_FILE, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'Duplicate NPZ members')
        arrays = {name: archive[name] for name in archive.files}
    _validate_index(index, arrays, benchmark)
    return benchmark, index, arrays


def compare_portable_runs(reference, actual, out, reports_only=False):
    reference, actual, out = Path(reference).resolve(), Path(actual).resolve(), Path(out).resolve()
    require(not out.is_relative_to(reference), 'Receipt must not be inside reference results')
    require(type(reports_only) is bool, 'reports_only must be boolean')
    input_names = ['benchmark.json'] + ([] if reports_only else [INDEX_FILE, ARRAY_FILE]) + [run + '/' + name for run in RUNS for name in _reports.FILES]
    require(out not in {(actual / name).resolve() for name in input_names}, 'Receipt cannot overwrite comparison input')
    receipt = {'status': 'failed', 'absolute_tolerance': ATOL, 'relative_tolerance': 0.0, 'reports_only': reports_only,
        'scope': 'Strict cross-platform numerical comparison of six runs, all evaluation and candidate OOF probabilities, selected penalties and fallback decisions; no diagnostic exceptions.',
        'reference_sha256': {}, 'actual_sha256': {}, 'reports': {}, 'errors': []}
    if reports_only:
        receipt['scope'] = 'Strict numerical report comparison of six runs only; probability signatures were not compared. No diagnostic exceptions.'
    try:
        for name in input_names:
            receipt['reference_sha256'][name] = _reports._sha(reference / name)
            receipt['actual_sha256'][name] = _reports._sha(actual / name)
        for root in [reference, actual]:
            _validate_benchmark(_reports._json(root / 'benchmark.json'))
        if not reports_only:
            lb, li, la = _load(reference)
            rb, ri, ra = _load(actual)
            require(lb == rb, 'Benchmark mismatch')
            require(len(li['records']) == len(ri['records']), 'Fit record count changed')
            index_comparison = _reports.Comparison(ATOL)
            index_comparison.json(li, ri, INDEX_FILE)
            receipt['index_numeric_comparisons'] = index_comparison.numeric_comparisons
            receipt['index_maximum_absolute_error'] = index_comparison.maximum_absolute_error
            receipt['probabilities'] = {}
            for key in sorted(ARRAY_KEYS):
                left, right = la[key], ra[key]
                require(left.shape == right.shape, key + ': shape changed')
                maximum, worst = 0.0, None
                for start in range(0, len(left), 1_000_000):
                    error = np.abs(left[start:start + 1_000_000] - right[start:start + 1_000_000])
                    local = int(error.argmax())
                    if float(error[local]) > maximum:
                        maximum, worst = float(error[local]), start + local
                receipt['probabilities'][key] = {'values_compared': len(left), 'maximum_absolute_error': maximum,
                                               'worst_flat_index': worst}
                if worst is not None:
                    for record in li['records']:
                        if key == 'eval_probabilities' and record['eval_start'] <= worst < record['eval_stop']:
                            receipt['probabilities'][key]['worst_fit_id'] = record['id']
                            break
                        if key == 'oof_probabilities':
                            candidate = next((x for x in record['oof'] if x['start'] <= worst < x['stop']), None)
                            if candidate is not None:
                                receipt['probabilities'][key].update(worst_fit_id=record['id'], worst_candidate=candidate['candidate'])
                                break
                require(maximum <= ATOL, key + ': probability drift exceeds 1e-8')
        for run in RUNS:
            receipt['reports'][run] = _reports.compare_reports(reference / run, actual / run, atol=ATOL)
        if not reports_only:
            # Completeness is bound to each report's published method-case inventory.
            expected_ids = []
            for run in RUNS:
                rows = _reports._csv(reference / run / 'case_metrics.csv', _reports.CSV_SCHEMAS['case_metrics.csv'])
                fractions = sorted({row['fraction'] for row in rows}, key=float)
                for row in rows:
                    case = f"cases/{row['seed']}/{row['representation']}/{row['weighting']}/budget_{fractions.index(row['fraction'])}"
                    expected_ids.extend(run + '/' + case + '/' + row['method'] + '/' + label for label in LABELS)
            require(len(expected_ids) == len(set(expected_ids)) == len(li['records'])
                    and set(expected_ids) == {record['id'] for record in li['records']}, 'Probability fits do not cover all published method/label cases')
        for name in input_names:
            require(_reports._sha(reference / name) == receipt['reference_sha256'][name]
                    and _reports._sha(actual / name) == receipt['actual_sha256'][name], 'Input changed during comparison: ' + name)
        receipt.update(status='passed', runs=len(RUNS))
        if not reports_only:
            receipt['fit_records'] = len(li['records'])
    except (OSError, ValueError, TypeError, KeyError) as error:
        receipt['errors'] = [str(error)]
        raise
    finally:
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['comparator_sha256'] = _reports._sha(Path(__file__))
        receipt['report_comparator_sha256'] = _reports._sha(Path(__file__).with_name('compare_reports.py'))
        _reports._atomic_json(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reports-only', action='store_true', help='Compare the six published report sets only; no probability-level reproduction claim')
    args = parser.parse_args()
    try:
        result = compare_portable_runs(args.reference, args.actual, args.out, reports_only=args.reports_only)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('Portable comparison failed: ' + str(error))
        return 1
    if args.reports_only:
        print(f"Report-only comparison passed: {result['runs']} runs; probability signatures were not compared.")
    else:
        print(f"Portable comparison passed: {result['runs']} runs, {result['fit_records']} fits; all eval/OOF probabilities and reports within 1e-8, selections and fallbacks exact.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
