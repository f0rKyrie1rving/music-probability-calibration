#!/usr/bin/env python3
"""Retain paired v1-to-v2 changes on six identical exposed-data plans.

This post-fit diagnostic does not require v2 to match v1, does not refit, and
makes no claim about accuracy on new music. Both improvements and deterioration
are retained. Repeated full-budget endpoints count once in unique summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np

_SPEC = importlib.util.spec_from_file_location('audit_report_io', Path(__file__).with_name('compare_reports.py'))
_io = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_io)
LABELS = ['electronic', 'pop', 'ambient', 'rock']
PROBABILITY_TOLERANCE = 1e-8
BRIER_TOLERANCE = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _runs(legacy_budget, legacy_stability, portable):
    old_paths = [Path(legacy_budget)] + sorted((Path(legacy_stability) / 'children').glob('*'))
    new_paths = sorted((Path(portable) / 'children').glob('*'))
    def by_seed(paths):
        result = {}
        for path in paths:
            if not path.is_dir():
                continue
            config = _io._json(path / 'config.json')
            seed = config['budget_seed']
            require(type(seed) is int and seed not in result, 'Duplicate or invalid budget seed')
            result[seed] = (path, config)
        return result
    old, new = by_seed(old_paths), by_seed(new_paths)
    expected = set(_io._json(Path(legacy_stability) / 'config.json')['budget_seeds'])
    expected.add(_io._json(Path(legacy_budget) / 'config.json')['budget_seed'])
    require(len(expected) == 6 and old.keys() == new.keys() == expected, 'Need exactly six completed runs matched by config budget_seed')
    return [(seed, old[seed], new[seed]) for seed in sorted(expected)]


def _policy(records):
    selections, fallback = {}, {}
    for method, fits in records.items():
        require(len(fits) == len(LABELS), 'Fit label inventory mismatch')
        for label, fit in zip(LABELS, fits):
            require(type(fit['fallback']) is bool, 'Malformed final fallback')
            fallback[method + '/' + label + '/final'] = {'fallback': fit['fallback'], 'reason': fit.get('reason')}
            if method != 'ridge_platt':
                continue
            selections[label] = str(fit['selected_penalty'])
            require(fit['final_fit_fallback'] == fit['fallback'], 'Inconsistent ridge final fallback')
            for candidate in fit['oof_scores']:
                for fold in candidate['fold_fits']:
                    key = 'ridge_platt/' + label + '/candidate_' + str(candidate['penalty']) + '/fold_' + str(fold['fold'])
                    require(key not in fallback and type(fold['fit']['fallback']) is bool, 'Duplicate or malformed candidate fold')
                    fallback[key] = {'fallback': fold['fit']['fallback'], 'reason': fold['fit'].get('reason')}
    return selections, fallback


def _summary(rows):
    deltas = [row['delta_v2_minus_v1'] for row in rows]
    return {'n': len(rows), 'mean_v1_brier': statistics.mean(row['v1_brier'] for row in rows),
            'mean_v2_brier': statistics.mean(row['v2_brier'] for row in rows),
            'mean_delta_v2_minus_v1': statistics.mean(deltas), 'minimum_delta': min(deltas), 'maximum_delta': max(deltas),
            'improved': sum(value < -BRIER_TOLERANCE for value in deltas),
            'deteriorated': sum(value > BRIER_TOLERANCE for value in deltas),
            'ties': sum(abs(value) <= BRIER_TOLERANCE for value in deltas)}


def audit(legacy_budget, legacy_stability, portable, out):
    roots = [Path(value).resolve() for value in [legacy_budget, legacy_stability, portable]]
    out = Path(out).resolve()
    require(not any(out.is_relative_to(root) for root in roots), 'Output must not modify input run trees')
    result = {'status': 'error', 'scope': 'Post-fit paired numerical-v1 versus numerical-v2 diagnostic on the same six exposed-data plans. No equality requirement, method selection, new-data accuracy claim or independent-sample inference.',
              'probability_change_tolerance': PROBABILITY_TOLERANCE, 'brier_direction_tolerance': BRIER_TOLERANCE,
              'input_runs': [], 'changed_cases': [], 'changed_selections': [], 'changed_fallback_locations': [], 'errors': []}
    tracked = {}
    def track(path, alias):
        digest = _io._sha(path)
        require(alias not in tracked, 'Duplicate source inventory alias')
        tracked[alias] = (path, digest)
        return digest
    try:
        pairs = _runs(*roots)
        all_rows, unique_rows = [], []
        executed_cases = changed_cases = changed_labels = probability_values = changed_values = 0
        maximum_error = 0.0
        full_endpoints = {}
        run_counts = []
        for seed, (old_root, old_config), (new_root, new_config) in pairs:
            fixed_keys = ['expected_source_cases', 'budgets', 'budget_seed', 'inner_seed', 'inner_folds', 'methods', 'ridge_penalties', 'identity_candidate']
            require(all(old_config[key] == new_config[key] for key in fixed_keys), 'Statistical design changed beyond solver')
            methods = old_config['methods']
            require(len(methods) == len(set(methods)) and 'ridge_platt' in methods, 'Invalid method inventory')
            metadata = {}
            for side, root in [('v1', old_root), ('v2', new_root)]:
                alias = side + '/budget_seed_' + str(seed)
                metadata[side] = {name: track(root / name, alias + '/' + name)
                                  for name in ['config.json', 'plans.json', 'freeze.json', 'completed.json']}
            old_plans, new_plans = _io._json(old_root / 'plans.json'), _io._json(new_root / 'plans.json')
            require(old_plans == new_plans, 'Calibration IDs/folds/plans changed for matched budget seed')
            old_freeze, new_freeze = _io._json(old_root / 'freeze.json'), _io._json(new_root / 'freeze.json')
            require(old_freeze['bundle_manifest_sha256'] == new_freeze['bundle_manifest_sha256']
                    and old_freeze['bundle_metadata_hashes'] == new_freeze['bundle_metadata_hashes'], 'Input score bundle changed')
            completions = [_io._json(root / 'completed.json') for root in [old_root, new_root]]
            for completed in completions:
                require(completed['state'] == 'complete' and len(completed['cases']) == len(set(completed['cases']))
                        == old_config['expected_source_cases'] * len(old_config['budgets']), 'Incomplete execution inventory')
                require(set(completed['status_hashes']) == set(completed['cases']), 'Incomplete status inventory')
            require(set(completions[0]['cases']) == set(completions[1]['cases']), 'Case inventory changed')
            result['input_runs'].append({'budget_seed': seed, 'metadata_sha256': metadata,
                'bundle_manifest_sha256': old_freeze['bundle_manifest_sha256'], 'cases': len(completions[0]['cases'])})
            run_changed = run_selected = run_fallbacks = 0
            for case_path in sorted(completions[0]['cases']):
                require(not Path(case_path).is_absolute() and len(Path(case_path).parts) == 5
                        and Path(case_path).parts[0] == 'cases' and '..' not in Path(case_path).parts, 'Unsafe case path')
                case_objects, fitted, predictions, saved_metrics = [], [], [], []
                for side, root, completed in zip(['v1', 'v2'], [old_root, new_root], completions):
                    directory = root / case_path
                    alias = side + '/budget_seed_' + str(seed) + '/' + case_path
                    require(track(directory / 'status.json', alias + '/status.json') == completed['status_hashes'][case_path], 'Status hash mismatch')
                    status = _io._json(directory / 'status.json')
                    require(status['state'] == 'complete' and set(status['hashes']) == {'case.json', 'calibrators.json', 'predictions.npz', 'metrics.json'}, 'Incomplete case artifact inventory')
                    for name, expected in status['hashes'].items():
                        require(track(directory / name, alias + '/' + name) == expected, 'Artifact hash mismatch: ' + alias + '/' + name)
                    case_objects.append(_io._json(directory / 'case.json'))
                    fitted.append(_io._json(directory / 'calibrators.json'))
                    saved_metrics.append(_io._json(directory / 'metrics.json'))
                    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
                        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == {'ids', 'artists', 'y'} | set(methods), 'Prediction inventory mismatch')
                        predictions.append({key: archive[key] for key in archive.files})
                case_keys = ['id', 'seed', 'representation', 'weighting', 'path', 'sha256', 'budget_index', 'fraction', 'calibration_artists', 'calibration_tracks']
                require(all(case_objects[0][key] == case_objects[1][key] for key in case_keys), 'Classifier/source/case identity mismatch')
                case = case_objects[0]
                require(case_path == 'cases/' + case['id'] + '/budget_' + str(case['budget_index']), 'Case/path identity mismatch')
                for key in ['ids', 'artists', 'y']:
                    require(np.array_equal(predictions[0][key], predictions[1][key]), 'Evaluation rows/artists/targets changed')
                y = predictions[0]['y']
                require(y.shape == (len(predictions[0]['ids']), len(LABELS)) and np.isin(y, [0, 1]).all(), 'Invalid evaluation targets')
                require(all(set(records) == set(methods) for records in fitted), 'Fitted method inventory changed')
                old_selections, old_fallbacks = _policy(fitted[0])
                new_selections, new_fallbacks = _policy(fitted[1])
                require(old_fallbacks.keys() == new_fallbacks.keys(), 'Candidate/fold policy inventory changed')
                case_id = 'budget_seed_' + str(seed) + '/' + case_path
                selection_changes = []
                for label in LABELS:
                    if old_selections[label] != new_selections[label]:
                        index = LABELS.index(label)
                        details = {'case': case_id, 'label': label, 'v1_selected': old_selections[label], 'v2_selected': new_selections[label], 'candidate_scores': {}}
                        for side, records in [('v1', fitted[0]), ('v2', fitted[1])]:
                            details['candidate_scores'][side] = [{key: candidate[key] for key in ['penalty', 'brier', 'inner_fallback_count']}
                                                                 for candidate in records['ridge_platt'][index]['oof_scores']]
                        result['changed_selections'].append(details)
                        selection_changes.append(label)
                fallback_changes = [key for key in old_fallbacks if old_fallbacks[key] != new_fallbacks[key]]
                result['changed_fallback_locations'].extend({'case': case_id, 'location': key, 'v1': old_fallbacks[key], 'v2': new_fallbacks[key]} for key in fallback_changes)
                per_method, case_values = [], 0
                for method in methods:
                    old_p, new_p = predictions[0][method], predictions[1][method]
                    require(old_p.shape == new_p.shape == y.shape and old_p.dtype == new_p.dtype == np.dtype('float64'), 'Probability shape/dtype mismatch')
                    require(np.isfinite(old_p).all() and np.isfinite(new_p).all() and ((old_p >= 0) & (old_p <= 1)).all()
                            and ((new_p >= 0) & (new_p <= 1)).all(), 'Invalid probability')
                    error = np.abs(new_p - old_p)
                    count = int((error > PROBABILITY_TOLERANCE).sum())
                    maximum = float(error.max())
                    old_brier, new_brier = float(np.mean((old_p - y) ** 2)), float(np.mean((new_p - y) ** 2))
                    require(abs(old_brier - saved_metrics[0][method]['macro_brier']) <= 1e-12
                            and abs(new_brier - saved_metrics[1][method]['macro_brier']) <= 1e-12, 'Saved macro Brier inconsistent with probabilities')
                    row = {'method': method, 'v1_brier': old_brier, 'v2_brier': new_brier,
                           'delta_v2_minus_v1': new_brier - old_brier,
                           'maximum_probability_difference': maximum, 'changed_probability_values': count}
                    per_method.append(row)
                    all_rows.append({**row, 'representation': case['representation'], 'weighting': case['weighting'], 'fraction': case['fraction']})
                    probability_values += old_p.size
                    changed_values += count
                    case_values += count
                    maximum_error = max(maximum_error, maximum)
                full_key = case_path
                unique = case['fraction'] != 1.0 or full_key not in full_endpoints
                if case['fraction'] == 1.0:
                    plan = old_plans[str(case['seed'])][case['budget_index']]
                    endpoint = {'alignment': canonical_hash({'track_ids': plan['track_ids'], 'folds': plan['folds'], 'eval_ids': predictions[0]['ids'].tolist(), 'eval_y': y.tolist()}),
                        'v1_policy': canonical_hash([old_selections, old_fallbacks]), 'v2_policy': canonical_hash([new_selections, new_fallbacks]),
                        'v1_probabilities': {m: hashlib.sha256(predictions[0][m].tobytes()).hexdigest() for m in methods},
                        'v2_probabilities': {m: hashlib.sha256(predictions[1][m].tobytes()).hexdigest() for m in methods}}
                    require(full_key not in full_endpoints or full_endpoints[full_key] == endpoint, 'Repeated full endpoint changed; refusing deduplication')
                    full_endpoints[full_key] = endpoint
                if unique:
                    unique_rows.extend({**row, 'representation': case['representation'], 'weighting': case['weighting'], 'fraction': case['fraction']} for row in per_method)
                changed_brier_methods = [row['method'] for row in per_method if abs(row['delta_v2_minus_v1']) > BRIER_TOLERANCE]
                is_changed = bool(case_values or selection_changes or fallback_changes or changed_brier_methods)
                if is_changed:
                    result['changed_cases'].append({'case': case_id, 'fraction': case['fraction'], 'counted_in_unique_summary': unique,
                        'selected_labels_changed': selection_changes, 'fallback_locations_changed': len(fallback_changes),
                        'changed_brier_methods': changed_brier_methods, 'methods': per_method})
                executed_cases += 1
                changed_cases += int(is_changed)
                changed_labels += len(selection_changes)
                run_changed += int(is_changed)
                run_selected += len(selection_changes)
                run_fallbacks += len(fallback_changes)
            run_counts.append({'budget_seed': seed, 'executed_cases': len(completions[0]['cases']), 'changed_cases': run_changed,
                               'changed_label_choices': run_selected, 'changed_fallback_locations': run_fallbacks})
        def grouped(rows, keys):
            groups = defaultdict(list)
            for row in rows:
                groups[tuple(row[key] for key in keys)].append(row)
            return [{**dict(zip(keys, key)), **_summary(values)} for key, values in sorted(groups.items())]
        result['executed_summary_by_method'] = grouped(all_rows, ['method'])
        result['unique_summary_by_method'] = grouped(unique_rows, ['method'])
        result['unique_summary_cells'] = grouped(unique_rows, ['representation', 'weighting', 'fraction', 'method'])
        result['counts'] = {'matched_budget_seeds': len(pairs), 'executed_cases': executed_cases,
            'unique_source_subset_cases': len(unique_rows) // len(methods), 'changed_executed_cases': changed_cases,
            'changed_label_choices': changed_labels, 'changed_fallback_locations': len(result['changed_fallback_locations']),
            'evaluation_probability_values_compared': probability_values, 'changed_probability_values': changed_values}
        result['run_counts'] = run_counts
        result['maximum_probability_difference'] = maximum_error
        result['deduplication'] = 'All six run executions are retained in raw counts. Equal-input/equal-output full-budget endpoints are verified by exact probability and policy hashes, then counted once per source case in unique summaries. Partial budgets retain all six matched draws. Splits and songs overlap; summaries are descriptive.'
        for path, expected in tracked.values():
            require(_io._sha(path) == expected, 'Input changed during audit: ' + path.name)
        result['verified_artifact_inventory_sha256'] = canonical_hash({key: digest for key, (_, digest) in tracked.items()})
        result['verified_artifact_files'] = len(tracked)
        result['status'] = 'diagnosed'
    except (OSError, ValueError, TypeError, KeyError) as error:
        message = str(error)
        for root, alias in zip(roots, ['[legacy-budget]', '[legacy-stability]', '[portable]']):
            message = message.replace(str(root), alias)
        result['errors'] = [message]
        raise
    finally:
        result['created_utc'] = datetime.now(timezone.utc).isoformat()
        result['audit_script_sha256'] = _io._sha(Path(__file__))
        _io._atomic_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-budget', type=Path, required=True)
    parser.add_argument('--legacy-stability', type=Path, required=True)
    parser.add_argument('--portable', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.legacy_budget, args.legacy_stability, args.portable, args.out)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('Numerical change audit failed: ' + str(error))
        return 1
    print(f"Numerical changes retained: {result['counts']['changed_executed_cases']}/{result['counts']['executed_cases']} cases, "
          f"{result['counts']['changed_label_choices']} label choices; {result['counts']['unique_source_subset_cases']} unique source/subset cases. This is not a v1-equality or new-data accuracy gate.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
