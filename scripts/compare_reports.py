#!/usr/bin/env python3
"""Check regenerated numerical reports against the committed benchmark claims.

Floating-point measurements allow a declared absolute tolerance. Experimental
identities, selected candidates, counts, row order and schema must match exactly.
Figures and timestamp-bearing execution receipts are deliberately outside scope.
An explicit budget-only option can disclose unselected-candidate inner-fallback
count differences after all selected/final outcomes and numerical guards pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

FILES = ('summary.json', 'summary_cells.csv', 'case_metrics.csv', 'label_metrics.csv',
         'ridge_selections.csv', 'example_reliability.csv', 'fallbacks.json')
# Bounds apply only to the original budget_v1 schema enabled by the explicit
# diagnostic option. protocols/budget_v1.json declares five fitted penalties and
# three inner folds. Identity is evaluated directly and has no optimizer fits.
BUDGET_RIDGE_PENALTIES = frozenset({'0.0', '0.001', '0.01', '0.1', '1.0'})
BUDGET_INNER_FOLDS = 3
# Each tuple is (ordered columns, integer-count columns, continuous numeric columns,
# nullable continuous columns, unique row-identity columns). Unlisted columns are
# exact categorical strings, including budgets and selected penalty-grid values.
CSV_SCHEMAS = {
    'summary_cells.csv': (
        'representation weighting fraction method n mean_brier mean_delta_raw median_delta_raw min_delta_raw max_delta_raw wins ties losses mean_delta_platt wins_vs_platt final_fallback_labels successful_labels bound_hit_labels'.split(),
        'n wins ties losses wins_vs_platt final_fallback_labels successful_labels bound_hit_labels'.split(),
        'mean_brier mean_delta_raw median_delta_raw min_delta_raw max_delta_raw mean_delta_platt'.split(), [],
        'representation weighting fraction method'.split()),
    'case_metrics.csv': (
        'seed representation weighting fraction method calibration_artists calibration_tracks macro_brier macro_log_loss delta_raw delta_platt fallback_labels successful_labels bound_hit_labels'.split(),
        'seed calibration_artists calibration_tracks fallback_labels successful_labels bound_hit_labels'.split(),
        'macro_brier macro_log_loss delta_raw delta_platt'.split(), [],
        'seed representation weighting fraction method'.split()),
    'label_metrics.csv': (
        'seed representation weighting fraction method label brier delta_raw log_loss average_precision ece_5 ece_10'.split(),
        ['seed'], 'brier delta_raw log_loss average_precision ece_5 ece_10'.split(), ['average_precision'],
        'seed representation weighting fraction method label'.split()),
    'ridge_selections.csv': (
        'case representation weighting fraction label selected_penalty inner_fallback_count selected_inner_fallback_count final_fit_fallback'.split(),
        'inner_fallback_count selected_inner_fallback_count'.split(), [], [], ['case', 'label']),
    'example_reliability.csv': (
        'seed representation weighting method label bins bin count mean_probability positive_fraction'.split(),
        'seed bins bin count'.split(), ['mean_probability', 'positive_fraction'],
        ['mean_probability', 'positive_fraction'],
        'seed representation weighting method label bins bin'.split()),
}


# The stability extension retains the same file names for some richer tables.
# Explicit alternative schemas prevent guessing whether a numeric field is a
# measurement, an experimental identifier, or a discrete count.
STABILITY_SCHEMAS = {
    'case_metrics.csv': (
        'draw budget_seed seed representation weighting fraction method calibration_artists calibration_tracks macro_brier macro_log_loss delta_raw delta_platt fallback_labels is_duplicate_full_endpoint'.split(),
        'draw budget_seed seed calibration_artists calibration_tracks fallback_labels'.split(),
        'macro_brier macro_log_loss delta_raw delta_platt'.split(), [],
        'draw seed representation weighting fraction method'.split()),
    'ridge_selections.csv': (
        'draw seed representation weighting fraction label selected_penalty inner_fallback_count selected_inner_fallback_count final_fit_fallback'.split(),
        'draw seed inner_fallback_count selected_inner_fallback_count'.split(), [], [],
        'draw seed representation weighting fraction label'.split()),
    'ridge_stability.csv': (
        'seed representation weighting fraction label draws distinct_strengths identity_draws selected_strengths inner_fallback_count selected_inner_fallback_count final_fit_fallbacks'.split(),
        'seed draws distinct_strengths identity_draws inner_fallback_count selected_inner_fallback_count final_fit_fallbacks'.split(), [], [],
        'seed representation weighting fraction label'.split()),
}
_within_columns = 'seed representation weighting fraction method draws mean_brier mean_log_loss fallback_labels'.split()
_within_counts = 'seed draws fallback_labels'.split()
_within_numbers = ['mean_brier', 'mean_log_loss']
_cell_columns = 'representation weighting fraction method outer_splits draws_per_split evaluated_distinct_cases mean_brier mean_log_loss fallback_labels'.split()
_cell_counts = 'outer_splits draws_per_split evaluated_distinct_cases fallback_labels'.split()
_cell_numbers = ['mean_brier', 'mean_log_loss']
for _contrast in ['raw', 'platt']:
    _within_columns += [f'{_contrast}_{suffix}' for suffix in ['mean', 'median', 'min', 'max', 'range', 'wins', 'ties', 'losses', 'sign_flip']]
    _within_counts += [f'{_contrast}_{suffix}' for suffix in ['wins', 'ties', 'losses']]
    _within_numbers += [f'{_contrast}_{suffix}' for suffix in ['mean', 'median', 'min', 'max', 'range']]
    _new_numbers = [f'{prefix}_{_contrast}' for prefix in ['mean_delta', 'median_split_mean_delta',
        'min_split_mean_delta', 'max_split_mean_delta', 'mean_within_split_range', 'max_within_split_range']]
    _new_counts = [f'{prefix}_{_contrast}' for prefix in ['sign_flip_splits', 'all_draws_improve_splits', 'all_draws_deteriorate_splits']]
    _cell_columns += _new_numbers + _new_counts
    _cell_numbers += _new_numbers
    _cell_counts += _new_counts
STABILITY_SCHEMAS['within_split.csv'] = (_within_columns, _within_counts, _within_numbers, [],
    'seed representation weighting fraction method'.split())
STABILITY_SCHEMAS['summary_cells.csv'] = (_cell_columns, _cell_counts, _cell_numbers, [],
    'representation weighting fraction method'.split())
ALLOWED_FILES = set(FILES) | set(STABILITY_SCHEMAS) | {'controls.json'}
BOOL_COLUMNS = {'final_fit_fallback', 'is_duplicate_full_endpoint', 'raw_sign_flip', 'platt_sign_flip'}


def _schema(path):
    with path.open(encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream, strict=True), None)
    variants = [mapping[path.name] for mapping in [CSV_SCHEMAS, STABILITY_SCHEMAS] if path.name in mapping]
    match = [schema for schema in variants if schema[0] == header]
    _require(len(match) == 1, path.name + ': missing, duplicated, reordered or unexpected columns')
    return match[0]


class RegressionError(ValueError):
    """The report has changed beyond the declared numerical tolerance."""


def _require(condition, message):
    if not condition:
        raise RegressionError(message)


def _sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f'{path.name}: duplicate JSON key {key!r}')
            result[key] = value
        return result

    def invalid(value):
        raise RegressionError(f'{path.name}: nonfinite JSON number {value}')

    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=invalid)


class Comparison:
    def __init__(self, tolerance, allow_unselected_fallback_drift=False):
        self.tolerance = tolerance
        self.allow_unselected_fallback_drift = allow_unselected_fallback_drift
        self.diagnostic_differences = []
        self.numeric_comparisons = 0
        self.exact_comparisons = 0
        self.maximum_absolute_error = 0.

    def numeric(self, reference, actual, location, *, exact=False):
        _require(math.isfinite(reference) and math.isfinite(actual), location + ': nonfinite measurement')
        error = abs(reference - actual)
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.numeric_comparisons += 1
        threshold = 0. if exact else self.tolerance
        _require(error <= threshold,
                 f'{location}: numeric drift {actual!r} versus {reference!r} (absolute error {error:.12g}, allowed {threshold:.12g})')

    def exact(self, reference, actual, location):
        self.exact_comparisons += 1
        _require(type(reference) is type(actual) and actual == reference,
                 f'{location}: exact value/type changed: {actual!r} versus {reference!r}')

    def diagnostic(self, reference, actual, location, identity=None):
        # These differences remain provisional until every ordinary report
        # comparison and the diagnostic-total consistency checks have passed.
        _require(type(reference) is int and type(actual) is int and reference >= 0 and actual >= 0,
                 location + ': diagnostic count must remain a nonnegative integer')
        if actual == reference:
            self.exact(reference, actual, location)
        else:
            difference = {'location': location, 'reference': reference, 'actual': actual,
                          'delta': actual - reference}
            if identity is not None:
                difference['identity'] = identity
            self.diagnostic_differences.append(difference)

    def json(self, reference, actual, location, key=None):
        if self.allow_unselected_fallback_drift and location == 'summary.json/ridge_inner_fallbacks':
            self.diagnostic(reference, actual, location)
        elif isinstance(reference, dict):
            _require(isinstance(actual, dict) and actual.keys() == reference.keys(),
                     location + ': JSON keys changed or missing')
            for child_key, value in reference.items():
                self.json(value, actual[child_key], location + '/' + child_key, child_key)
        elif isinstance(reference, list):
            _require(isinstance(actual, list) and len(actual) == len(reference),
                     location + ': JSON array length changed')
            for index, value in enumerate(reference):
                self.json(value, actual[index], f'{location}/{index}', key)
        elif type(reference) is float:
            _require(type(actual) is float, location + ': floating-point JSON type changed')
            self.numeric(reference, actual, location, exact=key in {'fraction', 'selected_penalty'})
        else:
            # In particular, int/boolean counts cannot become floats or strings.
            self.exact(reference, actual, location)


def _unique_json_rows(value, filename):
    if filename == 'controls.json':
        _require(isinstance(value, dict), 'controls.json: expected an object')
        return
    if filename == 'summary.json':
        _require(isinstance(value, dict) and isinstance(value.get('cells'), list),
                 'summary.json: expected summary object with cells')
        rows, keys = value['cells'], ('representation', 'weighting', 'fraction', 'method')
    else:
        _require(isinstance(value, list), 'fallbacks.json: expected a list')
        rows, keys = value, ('case', 'method', 'label', 'stage')
    identities = set()
    for i, row in enumerate(rows):
        _require(isinstance(row, dict) and all(k in row for k in keys),
                 f'{filename}: row {i} lacks identity fields')
        identity = tuple(row[k] for k in keys)
        _require(all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in identity),
                 f'{filename}: invalid row identity')
        _require(identity not in identities, f'{filename}: duplicated row identity {identity}')
        identities.add(identity)


def _csv(path, schema):
    columns, integers, numbers, nullable, keys = schema
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        _require(header == columns, f'{path.name}: missing, duplicated, reordered or unexpected columns')
        records, identities = [], set()
        for line, values in enumerate(reader, start=2):
            _require(len(values) == len(columns), f'{path.name}:{line}: missing or extra fields')
            row = {}
            for column, text in zip(columns, values):
                location = f'{path.name}:{line}/{column}'
                if column in integers:
                    _require(re.fullmatch(r'0|[1-9][0-9]*', text) is not None,
                             location + ': count/identifier must be a nonnegative integer')
                    row[column] = int(text)
                elif column in numbers:
                    if text == '' and column in nullable:
                        row[column] = None
                    else:
                        try:
                            row[column] = float(text)
                        except ValueError:
                            raise RegressionError(location + ': invalid measurement') from None
                        _require(math.isfinite(row[column]), location + ': nonfinite measurement')
                else:
                    _require(text != '', location + ': missing category')
                    row[column] = text
            for boolean in BOOL_COLUMNS & set(row):
                _require(row[boolean] in ('True', 'False'),
                         f'{path.name}:{line}: invalid boolean {boolean}')
            identity = tuple(row[k] for k in keys)
            _require(identity not in identities, f'{path.name}:{line}: duplicated row identity {identity}')
            identities.add(identity)
            records.append(row)
        _require(bool(records), path.name + ': report is empty')
    return records


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _diagnostic_totals(summary, selections, case_metrics, side):
    """Check that the only relaxed total is exactly the sum of recorded rows."""
    inner = sum(row['inner_fallback_count'] for row in selections)
    selected = sum(row['selected_inner_fallback_count'] for row in selections)
    final = sum(row['final_fit_fallback'] == 'True' for row in selections)
    for row in selections:
        fitted_selection = row['selected_penalty'] != 'identity'
        _require(not fitted_selection or row['selected_penalty'] in BUDGET_RIDGE_PENALTIES,
                 side + ': candidate is outside the original budget_v1 penalty grid')
        selected_limit = BUDGET_INNER_FOLDS if fitted_selection else 0
        _require(row['selected_inner_fallback_count'] <= selected_limit,
                 side + ': selected inner fallback count exceeds its possible fold fits')
        _require(row['inner_fallback_count'] <= len(BUDGET_RIDGE_PENALTIES) * BUDGET_INNER_FOLDS,
                 side + ': inner fallback count exceeds the original budget_v1 fit count')
        _require(row['selected_inner_fallback_count'] <= row['inner_fallback_count'],
                 side + ': selected inner fallback count exceeds the total')
        unselected_limit = (len(BUDGET_RIDGE_PENALTIES) - int(fitted_selection)) * BUDGET_INNER_FOLDS
        _require(row['inner_fallback_count'] - row['selected_inner_fallback_count'] <= unselected_limit,
                 side + ': unselected inner fallback count exceeds its possible fold fits')
    for key, total in [('ridge_inner_fallbacks', inner), ('ridge_selected_inner_fallbacks', selected)]:
        _require(type(summary.get(key)) is int and summary[key] == total,
                 side + ': ' + key + ' is inconsistent with ridge_selections.csv')
    _require(summary.get('ridge_selection_counts') == dict(Counter(row['selected_penalty'] for row in selections)),
             side + ': selected candidate totals are inconsistent')
    _require(len({row['case'] for row in selections}) == summary.get('cases'),
             side + ': ridge source-case count is inconsistent')
    _require(len(case_metrics) == summary.get('method_cases'), side + ': method-case count is inconsistent')
    _require(sum(row['fallback_labels'] for row in case_metrics) == summary.get('final_fallback_labels'),
             side + ': final fallback label total is inconsistent')
    _require(final == sum(row['fallback_labels'] for row in case_metrics if row['method'] == 'ridge_platt'),
             side + ': final ridge fallback count is inconsistent')
    return {'inner_fallbacks': inner, 'selected_inner_fallbacks': selected,
            'unselected_inner_fallbacks': inner - selected, 'final_ridge_fallbacks': final}


def compare_reports(reference, actual, out=None, atol=1e-8, files=None, allow_unselected_fallback_drift=False):
    reference, actual = Path(reference).resolve(), Path(actual).resolve()
    out = Path(out).resolve() if out is not None else None
    files = tuple(FILES if files is None else files)
    _require(bool(files) and len(set(files)) == len(files) and
             all(isinstance(name, str) and name in ALLOWED_FILES and Path(name).name == name for name in files),
             'Choose distinct supported report basenames; paths outside the report root are forbidden')
    _require(isinstance(atol, (int, float)) and not isinstance(atol, bool) and math.isfinite(atol) and atol >= 0,
             'Absolute tolerance must be finite and nonnegative')
    _require(type(allow_unselected_fallback_drift) is bool, 'Diagnostic drift flag must be boolean')
    _require(not allow_unselected_fallback_drift or set(files) == set(FILES),
             'Unselected fallback drift mode requires all seven original budget reports')
    if out is not None:
        _require(not out.is_relative_to(reference), 'Receipt cannot overwrite or be written inside the reference reports')
        _require(out not in {(actual / name).resolve() for name in files}, 'Receipt cannot overwrite an actual report')
    comparison = Comparison(float(atol), allow_unselected_fallback_drift)
    receipt = {'status': 'running', 'absolute_tolerance': float(atol), 'relative_tolerance': 0.,
               'scope': 'Numerical regression against committed report values; not fresh-data or scientific validation.',
               'compared_files': list(files), 'reference_sha256': {}, 'actual_sha256': {},
               'row_counts': {}, 'errors': [],
               'unselected_fallback_drift': {'allowed': allow_unselected_fallback_drift,
                   'accepted': False, 'differences': comparison.diagnostic_differences}}
    parsed = {}
    try:
        for name in files:
            rp, ap = reference / name, actual / name
            _require(rp.is_file(), 'Missing reference report: ' + name)
            _require(ap.is_file(), 'Missing actual report: ' + name)
            receipt['reference_sha256'][name], receipt['actual_sha256'][name] = _sha(rp), _sha(ap)
            if name.endswith('.json'):
                expected, observed = _json(rp), _json(ap)
                _unique_json_rows(expected, name)
                _unique_json_rows(observed, name)
                comparison.json(expected, observed, name)
                receipt['row_counts'][name] = len(expected['cells']) if name == 'summary.json' else len(expected)
            else:
                schema = _schema(rp)
                _require(not allow_unselected_fallback_drift or schema == CSV_SCHEMAS[name],
                         'Unselected fallback drift mode only supports original budget report schemas')
                expected, observed = _csv(rp, schema), _csv(ap, schema)
                _require(len(expected) == len(observed), name + ': row count changed')
                for index, (left, right) in enumerate(zip(expected, observed), start=2):
                    for column in schema[0]:
                        location = f'{name}:{index}/{column}'
                        if allow_unselected_fallback_drift and name == 'ridge_selections.csv' and column == 'inner_fallback_count':
                            comparison.diagnostic(left[column], right[column], location,
                                                  {key: left[key] for key in schema[4]})
                        elif column in schema[2] and left[column] is not None and right[column] is not None:
                            comparison.numeric(left[column], right[column], location)
                        else:
                            comparison.exact(left[column], right[column], location)
                receipt['row_counts'][name] = len(expected)
            parsed[name] = (expected, observed)
        if allow_unselected_fallback_drift:
            totals = {}
            for index, side in enumerate(['reference', 'actual']):
                totals[side] = _diagnostic_totals(parsed['summary.json'][index],
                    parsed['ridge_selections.csv'][index], parsed['case_metrics.csv'][index], side)
            receipt['unselected_fallback_drift']['totals'] = totals
            receipt['unselected_fallback_drift']['guard'] = (
                'All seven reports checked; selected penalties, selected inner fallbacks, final fallbacks, '
                'other counts and categories match exactly; output metrics match within the declared '
                'absolute tolerance. Both inner-fallback totals match their complete row inventories; '
                'per-row counts respect the original five fitted penalties and three-fold limits, '
                'with no optimizer fits for identity.')
        # A report being concurrently rewritten is not a stable regression input.
        for name in files:
            _require(_sha(reference / name) == receipt['reference_sha256'][name] and
                     _sha(actual / name) == receipt['actual_sha256'][name],
                     'Report changed during comparison: ' + name)
        receipt['status'] = 'passed'
        receipt['unselected_fallback_drift']['accepted'] = bool(comparison.diagnostic_differences)
    except (OSError, ValueError, csv.Error, TypeError) as error:
        receipt['status'] = 'failed'
        receipt['errors'] = [str(error)]
        raise
    finally:
        receipt.update(numeric_comparisons=comparison.numeric_comparisons,
                       exact_comparisons=comparison.exact_comparisons,
                       maximum_absolute_error=comparison.maximum_absolute_error,
                       completed_utc=datetime.now(timezone.utc).isoformat())
        if out is not None:
            _atomic_json(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--files', nargs='+', default=list(FILES),
                        help='Override the default seven reports with supported report basenames')
    parser.add_argument('--allow-unselected-fallback-drift', action='store_true',
                        help='Explicitly disclose and allow only unselected ridge inner-fallback count differences; all seven budget reports and consistency guards are required')
    args = parser.parse_args()
    try:
        result = compare_reports(args.reference, args.actual, args.out, args.atol, files=args.files,
                                 allow_unselected_fallback_drift=args.allow_unselected_fallback_drift)
    except (OSError, ValueError, csv.Error, TypeError) as error:
        print('Report regression failed: ' + str(error), file=sys.stderr)
        return 1
    print(f"Report regression passed: {len(result['compared_files'])} files, {result['numeric_comparisons']} numeric "
          f"comparisons, maximum absolute error {result['maximum_absolute_error']:.12g}.")
    drift = result['unselected_fallback_drift']
    if drift['differences']:
        print(f"Explicitly allowed diagnostic differences: {len(drift['differences'])}; this is not an exact diagnostic match.")
        for difference in drift['differences']:
            print('Unselected fallback diagnostic drift: ' + json.dumps(difference, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
