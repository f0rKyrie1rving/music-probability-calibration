#!/usr/bin/env python3
"""Independent standard-library aggregation audit; does not import study code."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if len(header) != len(set(header)):
            raise ValueError('Duplicate CSV columns: ' + path.name)
        result = list(reader)
        if any(None in row or None in row.values() for row in result):
            raise ValueError('Malformed CSV: ' + path.name)
        return result


def historical_full_control(keyed, historical_rows, source_ids, methods, tolerance, claimed):
    """Reconstruct the historical full-budget control without using study code."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid frozen historical metric tolerance')
    historical = {}
    for row in historical_rows:
        fraction = float(row['fraction'])
        if not math.isfinite(fraction):
            raise ValueError('Nonfinite historical budget')
        if fraction != 1.0:
            continue
        key = (row['seed'], row['representation'], row['weighting'], row['method'])
        if key in historical:
            raise ValueError('Duplicate historical full-budget row')
        historical[key] = row
    expected = {(*source, method) for source in source_ids for method in methods}
    if set(historical) != expected:
        raise ValueError('Historical full-budget Cartesian coverage differs')
    error = 0.0
    for key in sorted(expected):
        current = keyed[(0, *key[:3], 1.0, key[3])]
        previous = historical[key]
        for name in ['macro_brier', 'macro_log_loss']:
            actual, reference = float(current[name]), float(previous[name])
            if not math.isfinite(actual) or not math.isfinite(reference):
                raise ValueError('Nonfinite historical control metric')
            error = max(error, abs(actual - reference))
    if error > tolerance:
        raise ValueError('Historical full-budget metric difference exceeds frozen tolerance')
    if (type(claimed) not in (int, float) or not math.isfinite(claimed)
            or claimed < 0 or abs(claimed - error) > 1e-12):
        raise ValueError('Historical full-budget control receipt does not match reconstructed difference')
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-report', type=Path, default=Path(__file__).resolve().parents[1] / 'reports/budget_v1',
                        help='Original published budget_v1 report used for the frozen historical control')
    args = parser.parse_args()
    report = args.report
    names = ['case_metrics.csv', 'within_split.csv', 'summary_cells.csv', 'summary.json',
             'ridge_selections.csv', 'ridge_stability.csv', 'config.json', 'controls.json',
             'run_completed.json', 'study_freeze.json', 'report_receipt.json']
    inputs = {name: report / name for name in names}
    inputs.update(base_config=args.base_config, score_manifest=args.manifest,
                  historical_case_metrics=args.base_report / 'case_metrics.csv')
    hashes = {name: sha(path) for name, path in inputs.items()}
    counts = defaultdict(int)
    max_error = 0.0

    def check(condition, context, category='structure'):
        counts[category] += 1
        if not condition:
            raise ValueError(context)

    def equal(actual, expected, context, category='aggregation'):
        nonlocal max_error
        counts[category] += 1
        if isinstance(expected, bool):
            check(type(actual) is bool and actual == expected, context, category)
        elif isinstance(expected, int):
            check(type(actual) is int and actual == expected, context, category)
        elif isinstance(expected, float):
            check(type(actual) in (int, float) and math.isfinite(actual), context, category)
            error = abs(actual - expected)
            max_error = max(max_error, error)
            check(error <= 1e-12, context + ': float mismatch', category)
        elif isinstance(expected, dict):
            check(isinstance(actual, dict) and actual.keys() == expected.keys(), context + ': fields', category)
            for key, value in expected.items():
                equal(actual[key], value, context + '/' + key, category)
        elif isinstance(expected, list):
            check(isinstance(actual, list) and len(actual) == len(expected), context + ': length', category)
            for i, (a, b) in enumerate(zip(actual, expected)):
                equal(a, b, context + '/' + str(i), category)
        else:
            check(type(actual) is type(expected) and actual == expected, context, category)

    def csv_equal(actual, expected, context):
        check(actual.keys() == expected.keys(), context + ': fields')
        for key, value in expected.items():
            raw = actual[key]
            if isinstance(value, bool):
                check(raw in ('True', 'False'), context + '/' + key + ': bool format')
                parsed = raw == 'True'
            elif isinstance(value, int):
                check(raw == str(int(raw)), context + '/' + key + ': integer format')
                parsed = int(raw)
            elif isinstance(value, float):
                parsed = float(raw)
            else:
                parsed = raw
            equal(parsed, value, context + '/' + key)

    study, base, manifest = read(inputs['config.json']), read(args.base_config), read(args.manifest)
    methods, fractions, labels = base['methods'], base['budgets'], manifest['labels']
    n_draws = len(study['budget_seeds'])
    source_ids = {(str(x['seed']), x['representation'], x['weighting']) for x in manifest['cases']}
    check(len(source_ids) == len(manifest['cases']) == study['expected_source_cases'], 'source count')
    outer_seeds = {x[0] for x in source_ids}
    check(len(outer_seeds) == study['expected_outer_splits'], 'outer split count')
    check(set(fractions) == {0.25, 0.5, 1.0}, 'frozen budget set')
    receipt = read(inputs['report_receipt.json'])
    for name in names:
        if name != 'report_receipt.json':
            check(receipt['artifact_hashes'][name] == hashes[name], 'Receipt mismatch: ' + name, 'provenance')
    check(receipt['study_freeze_sha256'] == hashes['study_freeze.json'], 'study freeze receipt', 'provenance')
    check(receipt['run_completed_sha256'] == hashes['run_completed.json'], 'completion receipt', 'provenance')
    frozen_historical_hash = read(inputs['study_freeze.json'])['source_hashes']['reports/budget_v1/case_metrics.csv']
    check(hashes['historical_case_metrics'] == frozen_historical_hash, 'Frozen historical report changed', 'provenance')

    raw_rows = rows(inputs['case_metrics.csv'])
    keyed = {}
    expected_case_keys = {(draw, *source, fraction, method)
                         for draw, source, fraction, method in itertools.product(range(n_draws), source_ids, fractions, methods)}
    for row in raw_rows:
        key = (int(row['draw']), row['seed'], row['representation'], row['weighting'], float(row['fraction']), row['method'])
        check(key not in keyed, 'duplicate case row')
        for name in ['macro_brier', 'macro_log_loss', 'delta_raw', 'delta_platt']:
            row[name] = float(row[name])
            check(math.isfinite(row[name]), 'nonfinite case metric')
        for name in ['draw', 'budget_seed', 'calibration_artists', 'calibration_tracks', 'fallback_labels']:
            row[name] = int(row[name])
        row['fraction'] = float(row['fraction'])
        check(row['budget_seed'] == study['budget_seeds'][row['draw']], 'budget seed mapping')
        check(row['calibration_artists'] == math.ceil(study['expected_calibration_artists'] * row['fraction']), 'artist budget size')
        check(row['is_duplicate_full_endpoint'] == str(row['fraction'] == 1.0 and row['draw'] > 0), 'full endpoint duplicate flag')
        check(0 <= row['macro_brier'] <= 1 and row['macro_log_loss'] >= 0, 'valid proper scores')
        check(0 <= row['fallback_labels'] <= len(labels), 'valid fallback label count')
        keyed[key] = row
    check(keyed.keys() == expected_case_keys, 'case Cartesian coverage')
    for key, row in keyed.items():
        for contrast in ['raw', 'platt']:
            reference = keyed[(*key[:-1], contrast)]
            equal(row['delta_' + contrast], row['macro_brier'] - reference['macro_brier'], 'paired delta ' + contrast, 'paired_deltas')
            for name in ['calibration_artists', 'calibration_tracks']:
                equal(row[name], reference[name], 'same calibration sample', 'paired_deltas')
        if key[4] == 1.0:
            reference = keyed[(0, *key[1:])]
            for field in ['macro_brier', 'macro_log_loss', 'delta_raw', 'delta_platt', 'fallback_labels', 'calibration_tracks']:
                check(row[field] == reference[field], 'repeated full endpoint differs', 'full_endpoint_control')
        if row['method'] in ['raw', 'prior_offset']:
            reference = keyed[(0, *key[1:4], 1.0, key[-1])]
            for field in ['macro_brier', 'macro_log_loss']:
                check(row[field] == reference[field], 'uncalibrated control changes with budget', 'unchanged_controls')

    distinct = [row for key, row in keyed.items() if key[4] != 1.0 or key[0] == 0]
    executed_cases = len(keyed) // len(methods)
    distinct_cases = len(distinct) // len(methods)
    check(executed_cases == study['expected_executed_cases'], 'executed case count')
    check(distinct_cases == study['expected_distinct_subset_cases'], 'distinct case count')
    groups = defaultdict(list)
    for row in distinct:
        groups[(row['seed'], row['representation'], row['weighting'], row['fraction'], row['method'])].append(row)
    reconstructed_splits = []
    tol = study['comparison_tolerance']
    for key, group in sorted(groups.items()):
        expected_draws = 1 if key[3] == 1.0 else n_draws
        check(len(group) == expected_draws, 'within split draw count')
        result = dict(zip(['seed', 'representation', 'weighting', 'fraction', 'method'], key))
        result.update(draws=len(group), mean_brier=statistics.mean(row['macro_brier'] for row in group),
                      mean_log_loss=statistics.mean(row['macro_log_loss'] for row in group),
                      fallback_labels=sum(row['fallback_labels'] for row in group))
        for contrast in ['raw', 'platt']:
            delta = [row['macro_brier'] - keyed[(row['draw'], row['seed'], row['representation'], row['weighting'], row['fraction'], contrast)]['macro_brier'] for row in group]
            wins = sum(x < -tol for x in delta)
            losses = sum(x > tol for x in delta)
            result.update({contrast + '_' + name: value for name, value in {
                'mean': statistics.mean(delta), 'median': statistics.median(delta), 'min': min(delta),
                'max': max(delta), 'range': max(delta) - min(delta), 'wins': wins,
                'ties': len(delta) - wins - losses, 'losses': losses, 'sign_flip': wins > 0 and losses > 0}.items()})
        reconstructed_splits.append(result)
    split_rows = rows(inputs['within_split.csv'])
    check(len(split_rows) == len(reconstructed_splits), 'within-split table length')
    for index, (actual, expected) in enumerate(zip(split_rows, reconstructed_splits)):
        csv_equal(actual, expected, 'within_split row ' + str(index))

    cells = defaultdict(list)
    for row in reconstructed_splits:
        cells[(row['representation'], row['weighting'], row['fraction'], row['method'])].append(row)
    reconstructed_cells = []
    for key, group in sorted(cells.items()):
        check(len(group) == len(outer_seeds), 'outer splits per cell')
        result = dict(zip(['representation', 'weighting', 'fraction', 'method'], key))
        result.update(outer_splits=len(group), draws_per_split=group[0]['draws'],
                      evaluated_distinct_cases=sum(row['draws'] for row in group),
                      mean_brier=statistics.mean(row['mean_brier'] for row in group),
                      mean_log_loss=statistics.mean(row['mean_log_loss'] for row in group),
                      fallback_labels=sum(row['fallback_labels'] for row in group))
        for contrast in ['raw', 'platt']:
            delta = [row[contrast + '_mean'] for row in group]
            ranges = [row[contrast + '_range'] for row in group]
            result.update({name + '_' + contrast: value for name, value in {
                'mean_delta': statistics.mean(delta), 'median_split_mean_delta': statistics.median(delta),
                'min_split_mean_delta': min(delta), 'max_split_mean_delta': max(delta),
                'mean_within_split_range': statistics.mean(ranges), 'max_within_split_range': max(ranges),
                'sign_flip_splits': sum(row[contrast + '_sign_flip'] for row in group),
                'all_draws_improve_splits': sum(row[contrast + '_wins'] == row['draws'] for row in group),
                'all_draws_deteriorate_splits': sum(row[contrast + '_losses'] == row['draws'] for row in group)}.items()})
        reconstructed_cells.append(result)
    cell_rows = rows(inputs['summary_cells.csv'])
    check(len(cell_rows) == len(reconstructed_cells), 'cell table length')
    for index, (actual, expected) in enumerate(zip(cell_rows, reconstructed_cells)):
        csv_equal(actual, expected, 'summary cell ' + str(index))

    selected = rows(inputs['ridge_selections.csv'])
    selected_keys = set()
    ridge_groups = defaultdict(list)
    all_selected = {}
    for row in selected:
        draw, fraction = int(row['draw']), float(row['fraction'])
        key = (draw, row['seed'], row['representation'], row['weighting'], fraction, row['label'])
        check(key not in selected_keys, 'duplicate ridge selection')
        selected_keys.add(key)
        all_selected[key] = row
        if fraction != 1.0 or draw == 0:
            ridge_groups[key[1:]].append(row)
    expected_ridge_keys = {(draw, *source, fraction, label)
                          for draw, source, fraction, label in itertools.product(range(n_draws), source_ids, fractions, labels)}
    check(selected_keys == expected_ridge_keys, 'ridge Cartesian coverage')
    for key, row in all_selected.items():
        if key[4] == 1.0:
            reference = all_selected[(0, *key[1:])]
            for name in ['selected_penalty', 'inner_fallback_count', 'selected_inner_fallback_count', 'final_fit_fallback']:
                check(row[name] == reference[name], 'repeated full ridge selection differs', 'full_endpoint_control')
    ridge_expected = []
    for key, group in sorted(ridge_groups.items()):
        group = sorted(group, key=lambda row: int(row['draw']))
        values = [row['selected_penalty'] for row in group]
        check(len(group) == (1 if key[3] == 1.0 else n_draws), 'ridge draw count')
        check(all(row['final_fit_fallback'] in ['True', 'False'] for row in group), 'ridge bool format')
        result = dict(zip(['seed', 'representation', 'weighting', 'fraction', 'label'], key))
        result.update(draws=len(group), distinct_strengths=len(set(values)), identity_draws=values.count('identity'),
                      selected_strengths=json.dumps(values),
                      inner_fallback_count=sum(int(row['inner_fallback_count']) for row in group),
                      selected_inner_fallback_count=sum(int(row['selected_inner_fallback_count']) for row in group),
                      final_fit_fallbacks=sum(row['final_fit_fallback'] == 'True' for row in group))
        ridge_expected.append(result)
    ridge_rows = rows(inputs['ridge_stability.csv'])
    check(len(ridge_rows) == len(ridge_expected), 'ridge table length')
    for index, (actual, expected) in enumerate(zip(ridge_rows, ridge_expected)):
        csv_equal(actual, expected, 'ridge stability row ' + str(index))

    summary, controls, completed = read(inputs['summary.json']), read(inputs['controls.json']), read(inputs['run_completed.json'])
    check(completed['state'] == 'complete' and len(completed['children']) == n_draws, 'completed child count', 'provenance')
    historical_error = historical_full_control(keyed, rows(inputs['historical_case_metrics']),
        source_ids, methods, study['historical_metric_tolerance'], controls['historical_full_max_absolute_error'])
    expected_controls = dict(full_budget_identical_prediction_fit_metric_hashes=True,
                             full_budget_source_cases=len(source_ids), historical_full_max_absolute_error=historical_error,
                             all_children_independently_verified=True,
                             child_verification_checks=sum(child['verification_checks'] for child in completed['children']))
    equal(controls, expected_controls, 'control receipt')
    expected_summary = dict(experiment=study['experiment'], scope=study['scope'], executed_cases=executed_cases,
                            distinct_subset_cases=distinct_cases, historical_draw_included=False,
                            full_endpoints_counted_once=True, cells=reconstructed_cells, controls=controls,
                            distinct_case_fallback_labels=sum(row['fallback_labels'] for row in distinct),
                            distinct_ridge_inner_fallbacks=sum(row['inner_fallback_count'] for row in ridge_expected),
                            distinct_selected_inner_fallbacks=sum(row['selected_inner_fallback_count'] for row in ridge_expected),
                            partial_label_cells_with_varying_ridge_strength=sum(row['fraction'] != 1.0 and row['distinct_strengths'] > 1 for row in ridge_expected),
                            partial_label_cells=sum(row['fraction'] != 1.0 for row in ridge_expected))
    equal(summary, expected_summary, 'summary.json')
    for name, path in inputs.items():
        check(sha(path) == hashes[name], 'Input changed during audit: ' + name, 'provenance')
    out = report / 'independent_aggregation_check.json'
    record = dict(schema='independent-stability-aggregation-check-v1', status='passed',
                  created_utc=datetime.now(timezone.utc).isoformat(), helper_sha256=sha(Path(__file__)),
                  scope='Independent standard-library reconstruction from published case metrics and ridge selections; no study implementation imported. This audits numerical aggregation, not new-data confirmation or independent rerunning of fitted predictions.',
                  absolute_tolerance=1e-12, max_absolute_error=max_error,
                  historical_full_control=dict(reconstructed_max_absolute_error=historical_error,
                                               frozen_metric_tolerance=study['historical_metric_tolerance']),
                  checks=sum(counts.values()), checks_by_category=dict(counts), input_sha256=hashes,
                  rows=dict(case_metrics=len(raw_rows), distinct_method_cases=len(distinct), within_split=len(split_rows),
                            summary_cells=len(cell_rows), ridge_selections=len(selected), ridge_stability=len(ridge_rows)),
                  full_endpoint=dict(executed_source_cases=n_draws * len(source_ids), counted_source_cases=len(source_ids),
                                     duplicate_method_rows_removed=len(raw_rows) - len(distinct)),
                  reconstructed_totals={key: expected_summary[key] for key in ['executed_cases', 'distinct_subset_cases', 'distinct_case_fallback_labels', 'distinct_ridge_inner_fallbacks', 'distinct_selected_inner_fallbacks', 'partial_label_cells_with_varying_ridge_strength', 'partial_label_cells']})
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(out)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
