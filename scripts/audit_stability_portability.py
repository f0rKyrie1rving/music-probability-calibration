#!/usr/bin/env python3
"""Enumerate stability report differences without weakening the acceptance gate.

Exit 0 means matched within 1e-8; exit 2 means well-formed but not reproduced;
exit 1 means missing, malformed or incomplete evidence. No fitting is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys

_SPEC = importlib.util.spec_from_file_location('report_schema_helpers', Path(__file__).with_name('compare_reports.py'))
_helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_helpers)

FILES = ('summary.json', 'controls.json', 'case_metrics.csv', 'within_split.csv',
         'summary_cells.csv', 'ridge_selections.csv', 'ridge_stability.csv', 'config.json')
CONTEXT_FILES = ('study_freeze.json',)
ATOL = 1e-8
METHODS = {'raw', 'prior_offset', 'temperature', 'platt', 'beta', 'isotonic', 'ridge_platt'}
REPRESENTATIONS = {'maest', 'mert_v0', 'mert_v1'}
WEIGHTINGS = {'ambient_balanced', 'unweighted'}
FRACTIONS = {'0.25', '0.5', '1.0'}
LABELS = {'electronic', 'pop', 'ambient', 'rock'}
SUMMARY_KEYS = {'experiment', 'scope', 'executed_cases', 'distinct_subset_cases', 'historical_draw_included',
                'full_endpoints_counted_once', 'cells', 'controls', 'distinct_case_fallback_labels',
                'distinct_ridge_inner_fallbacks', 'distinct_selected_inner_fallbacks',
                'partial_label_cells_with_varying_ridge_strength', 'partial_label_cells'}
CONTROL_KEYS = {'full_budget_identical_prediction_fit_metric_hashes', 'full_budget_source_cases',
                'historical_full_max_absolute_error', 'all_children_independently_verified', 'child_verification_checks'}
CONFIG_KEYS = {'schema_version', 'experiment', 'base_experiment', 'budget_seeds', 'expected_source_cases',
               'expected_outer_splits', 'expected_calibration_artists', 'expected_executed_cases',
               'expected_distinct_subset_cases', 'comparison_tolerance', 'historical_metric_tolerance',
               'only_changed_child_config_field', 'full_budget_rule', 'historical_partial_budget_rule', 'scope'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row, keys):
    return tuple(row[key] for key in keys)


def load_report(root):
    result = {}
    for name in FILES + CONTEXT_FILES:
        path = root / name
        require(path.is_file(), 'Missing report: ' + name)
        if name.endswith('.csv'):
            result[name] = _helpers._csv(path, _helpers.STABILITY_SCHEMAS[name])
        else:
            result[name] = _helpers._json(path)
    config = result['config.json']
    require(isinstance(config, dict) and config.keys() == CONFIG_KEYS, 'Unexpected stability config schema')
    require(config['experiment'] == 'stability_v1' and config['base_experiment'] == 'budget_v1', 'Unexpected experiment')
    for key in ['expected_source_cases', 'expected_outer_splits', 'expected_calibration_artists',
                'expected_executed_cases', 'expected_distinct_subset_cases']:
        require(type(config[key]) is int and config[key] > 0, 'Invalid config count: ' + key)
    seeds = config['budget_seeds']
    require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) and len(seeds) > 0
            and all(type(seed) is int and seed >= 0 for seed in seeds), 'Invalid budget seeds')
    source_count, draws = config['expected_source_cases'], len(seeds)
    require(source_count == config['expected_outer_splits'] * len(REPRESENTATIONS) * len(WEIGHTINGS), 'Source Cartesian count')
    require(config['expected_executed_cases'] == source_count * draws * len(FRACTIONS), 'Executed config count')
    require(config['expected_distinct_subset_cases'] == source_count * (2 * draws + 1), 'Distinct config count')
    expected_counts = {'case_metrics.csv': source_count * draws * 3 * len(METHODS),
        'within_split.csv': source_count * 3 * len(METHODS),
        'summary_cells.csv': len(REPRESENTATIONS) * len(WEIGHTINGS) * 3 * len(METHODS),
        'ridge_selections.csv': source_count * draws * 3 * len(LABELS),
        'ridge_stability.csv': source_count * 3 * len(LABELS)}
    source_seeds = {row['seed'] for row in result['case_metrics.csv']}
    require(len(source_seeds) == config['expected_outer_splits'], 'Outer seed inventory')
    for name, expected_count in expected_counts.items():
        records = result[name]
        require(len(records) == expected_count, name + ': incomplete or extra row count')
        keys = _helpers.STABILITY_SCHEMAS[name][4]
        domains = {'seed': source_seeds, 'representation': REPRESENTATIONS, 'weighting': WEIGHTINGS,
                   'fraction': FRACTIONS, 'method': METHODS, 'label': LABELS, 'draw': set(range(draws))}
        for row in records:
            require(all(row[key] in domains[key] for key in keys), name + ': unknown identity value')
            if 'draw' in row and 'budget_seed' in row:
                require(row['budget_seed'] == seeds[row['draw']], name + ': budget seed mapping')
            if 'is_duplicate_full_endpoint' in row:
                require(row['is_duplicate_full_endpoint'] == str(row['fraction'] == '1.0' and row['draw'] > 0), name + ': duplicate endpoint flag')
            if 'calibration_artists' in row:
                require(row['calibration_artists'] == math.ceil(config['expected_calibration_artists'] * float(row['fraction'])), name + ': artist count')
            if 'draws' in row:
                require(row['draws'] == (1 if row['fraction'] == '1.0' else draws), name + ': draw count')
            if 'outer_splits' in row:
                expected_draws = 1 if row['fraction'] == '1.0' else draws
                require(row['outer_splits'] == len(source_seeds) and row['draws_per_split'] == expected_draws
                        and row['evaluated_distinct_cases'] == len(source_seeds) * expected_draws, name + ': cell counts')
            allowed_strengths = {'identity'} | set(_helpers.BUDGET_RIDGE_PENALTIES)
            if 'selected_penalty' in row:
                require(row['selected_penalty'] in allowed_strengths, name + ': invalid selected penalty')
            if 'selected_strengths' in row:
                strengths = json.loads(row['selected_strengths'])
                require(isinstance(strengths, list) and len(strengths) == row['draws']
                        and all(type(value) is str and value in allowed_strengths for value in strengths),
                        name + ': malformed selected strength sequence')
    summary, controls = result['summary.json'], result['controls.json']
    require(isinstance(summary, dict) and summary.keys() == SUMMARY_KEYS, 'Unexpected summary schema')
    require(isinstance(controls, dict) and controls.keys() == CONTROL_KEYS, 'Unexpected controls schema')
    _helpers._unique_json_rows(summary, 'summary.json')
    require(len(summary['cells']) == expected_counts['summary_cells.csv'], 'Summary JSON cell count')
    cell_keys = _helpers.STABILITY_SCHEMAS['summary_cells.csv'][0]
    for cell, row in zip(summary['cells'], result['summary_cells.csv']):
        require(cell.keys() == set(cell_keys), 'Unexpected summary cell schema')
        require(identity(cell, ('representation', 'weighting', 'method')) == identity(row, ('representation', 'weighting', 'method'))
                and str(cell['fraction']) == row['fraction'], 'JSON/CSV cell identity alignment')
    freeze = result['study_freeze.json']
    require(isinstance(freeze, dict) and isinstance(freeze.get('source_hashes'), dict)
            and isinstance(freeze.get('local_hashes'), dict), 'Missing freeze context')
    require(freeze['local_hashes'].get('config.json') == _helpers._sha(root / 'config.json'), 'Config not bound to freeze')
    return result


def audit(reference, actual, out):
    reference, actual, out = Path(reference).resolve(), Path(actual).resolve(), Path(out).resolve()
    require(not out.is_relative_to(reference), 'Audit receipt cannot be written inside reference reports')
    require(out not in {(actual / name).resolve() for name in FILES + CONTEXT_FILES}, 'Audit receipt cannot overwrite inputs')
    receipt = {'status': 'error', 'absolute_tolerance': ATOL, 'relative_tolerance': 0.0,
        'scope': 'Diagnostic enumeration of report differences, not an acceptance override or fresh-data validation.',
        'reference_sha256': {}, 'actual_sha256': {}, 'differences': [], 'errors': []}
    stats = {}
    try:
        for name in FILES + CONTEXT_FILES:
            receipt['reference_sha256'][name] = _helpers._sha(reference / name)
            receipt['actual_sha256'][name] = _helpers._sha(actual / name)
        left, right = load_report(reference), load_report(actual)
        for name in FILES:
            stats[name] = {'scalar_comparisons': 0, 'numeric_roundoff_within_tolerance': 0,
                           'maximum_absolute_error': 0.0, 'differences': 0}

        def compare(a, b, filename, location, row_identity=None, field=None):
            stat = stats[filename]
            if isinstance(a, dict):
                require(isinstance(b, dict) and a.keys() == b.keys(), location + ': JSON schema mismatch')
                for key in a:
                    compare(a[key], b[key], filename, location + '/' + key, row_identity, key)
                return
            if isinstance(a, list):
                require(isinstance(b, list) and len(a) == len(b), location + ': JSON array length mismatch')
                for index, (x, y) in enumerate(zip(a, b)):
                    ident = row_identity
                    if filename == 'summary.json' and field == 'cells':
                        keys = ('representation', 'weighting', 'fraction', 'method')
                        require(identity(x, keys) == identity(y, keys), location + ': cell identities/order differ')
                        ident = {key: x[key] for key in keys}
                    compare(x, y, filename, location + '/' + str(index), ident, field)
                return
            require(type(a) is type(b), location + ': scalar type changed')
            stat['scalar_comparisons'] += 1
            error = None
            if type(a) is float:
                require(math.isfinite(a) and math.isfinite(b), location + ': nonfinite number')
                error = abs(a - b)
                stat['maximum_absolute_error'] = max(stat['maximum_absolute_error'], error)
                # Configuration is an experimental identity, not a measurement.
                tolerance = 0.0 if filename == 'config.json' or field == 'fraction' else ATOL
                mismatch = error > tolerance
                if error and not mismatch:
                    stat['numeric_roundoff_within_tolerance'] += 1
            else:
                mismatch = a != b
            if mismatch:
                item = {'file': filename, 'location': location, 'field': field,
                        'kind': 'numeric' if error is not None else 'discrete', 'reference': a, 'actual': b}
                if row_identity is not None:
                    item['identity'] = row_identity
                if error is not None:
                    item['absolute_error'] = error
                receipt['differences'].append(item)
                stat['differences'] += 1

        for name in FILES:
            if name.endswith('.csv'):
                keys = _helpers.STABILITY_SCHEMAS[name][4]
                require(len(left[name]) == len(right[name]), name + ': row count differs')
                for index, (a, b) in enumerate(zip(left[name], right[name]), 2):
                    require(identity(a, keys) == identity(b, keys), name + ': row identities/order differ')
                    compare(a, b, name, name + ':' + str(index), {key: a[key] for key in keys})
            else:
                compare(left[name], right[name], name, name)
        receipt['configuration_context'] = {side: {
            'config': reports['config.json'],
            'data_manifest_sha256': reports['study_freeze.json'].get('data_manifest_sha256'),
            'source_hashes': reports['study_freeze.json']['source_hashes'],
            'study_freeze_sha256': receipt[side + '_sha256']['study_freeze.json']}
            for side, reports in [('reference', left), ('actual', right)]}
        contexts = receipt['configuration_context']
        context_match = all(contexts['reference'][field] == contexts['actual'][field]
                            for field in ['data_manifest_sha256', 'source_hashes'])
        receipt['same_data_and_source_hashes'] = context_match
        context_differences = []
        if contexts['reference']['data_manifest_sha256'] != contexts['actual']['data_manifest_sha256']:
            context_differences.append({'field': 'data_manifest_sha256',
                'reference': contexts['reference']['data_manifest_sha256'],
                'actual': contexts['actual']['data_manifest_sha256']})
        for name in sorted(contexts['reference']['source_hashes'].keys() | contexts['actual']['source_hashes'].keys()):
            a, b = contexts['reference']['source_hashes'].get(name), contexts['actual']['source_hashes'].get(name)
            if a != b:
                context_differences.append({'field': 'source_hashes/' + name, 'reference': a, 'actual': b})
        receipt['context_differences'] = context_differences
        tol = left['config.json']['comparison_tolerance']
        def sign(value):
            return 'improves' if value < -tol else 'deteriorates' if value > tol else 'tie'
        ridge = []
        for a, b in zip(left['summary_cells.csv'], right['summary_cells.csv']):
            if a['method'] != 'ridge_platt':
                continue
            row = {key: a[key] for key in ['representation', 'weighting', 'fraction']}
            for side, record in [('reference', a), ('actual', b)]:
                row[side] = {contrast: {'mean_brier_delta': record['mean_delta_' + contrast],
                                       'direction': sign(record['mean_delta_' + contrast])}
                             for contrast in ['raw', 'platt']}
            row['direction_changed'] = any(row['reference'][contrast]['direction'] != row['actual'][contrast]['direction']
                                           for contrast in ['raw', 'platt'])
            ridge.append(row)
        require(len(ridge) == 18, 'Ridge interpretation table incomplete')
        receipt['ridge_vs_raw_and_platt'] = ridge
        receipt['ridge_direction_changed_cells'] = sum(row['direction_changed'] for row in ridge)
        receipt['ridge_sign_tolerance'] = tol
        for name in FILES + CONTEXT_FILES:
            require(_helpers._sha(reference / name) == receipt['reference_sha256'][name]
                    and _helpers._sha(actual / name) == receipt['actual_sha256'][name], 'Input changed during audit: ' + name)
        receipt['per_file'] = stats
        receipt['difference_count'] = len(receipt['differences'])
        receipt['difference_kinds'] = dict(Counter(row['kind'] for row in receipt['differences']))
        receipt['status'] = 'matched' if not receipt['differences'] and context_match else 'not_reproduced'
        receipt['maximum_absolute_error'] = max(item['maximum_absolute_error'] for item in stats.values())
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as error:
        receipt['errors'] = [str(error)]
        raise
    finally:
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['audit_script_sha256'] = _helpers._sha(Path(__file__))
        receipt['schema_helper_sha256'] = _helpers._sha(Path(__file__).with_name('compare_reports.py'))
        _helpers._atomic_json(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.reference, args.actual, args.out)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as error:
        print('Portability evidence error: ' + str(error), file=sys.stderr)
        return 1
    print(f"Portability diagnostic: {result['status']}; {result['difference_count']} report differences; "
          f"{len(result['context_differences'])} data/source context differences; "
          f"maximum numeric error {result['maximum_absolute_error']:.12g}; "
          f"ridge direction changes in {result['ridge_direction_changed_cells']}/18 cells.")
    print('This diagnostic does not override the strict reproduction gate.')
    return 0 if result['status'] == 'matched' else 2


if __name__ == '__main__':
    raise SystemExit(main())
