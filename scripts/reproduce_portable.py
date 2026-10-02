"""Freeze and reproduce all six historical sampling plans with numerical v2."""
import argparse
from contextlib import ExitStack
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from music_calibration_portable import experiment
from music_calibration_portable.io import digest, now, read, write
from music_calibration_portable.report import summarize
from music_calibration_portable.verify import verify
from reproduce import Lock, validate_paths, inventory

NAMES = ['budget_original'] + [f'draw_{i:02d}' for i in range(1, 6)]
SEEDS = [2026100101] + list(range(2026100201, 2026100206))


def sources():
    return sorted([* (ROOT / 'music_calibration_portable').rglob('*.py'),
                   Path(__file__).resolve(), ROOT / 'protocols/numerical_v2.json',
                   ROOT / 'protocols/numerical_v2.md', ROOT / 'protocols/numerical_study_v2.json',
                   ROOT / 'scripts/compare_portable_runs.py', ROOT / 'scripts/compare_reports.py',
                   ROOT / 'scripts/reproduce.py', ROOT / 'protocols/budget_v1.json',
                   ROOT / 'reports/budget_v1/freeze.json', ROOT / 'reports/budget_v1/plans.json'])


def freeze(data, out):
    if out.exists():
        raise FileExistsError('Use a fresh run root or a complete existing freeze')
    base = read(ROOT / 'protocols/numerical_v2.json')
    legacy = read(ROOT / 'protocols/budget_v1.json')
    study = read(ROOT / 'protocols/numerical_study_v2.json')
    if study['run_names'] != NAMES or study['budget_seeds'] != SEEDS or study['expected_executed_cases'] != 2160:
        raise ValueError('Unexpected numerical study inventory')
    changed = {key for key in base.keys() | legacy.keys() if base.get(key) != legacy.get(key)}
    if changed != {'experiment', 'scope', 'numerical_solver'}:
        raise ValueError('Only solver/version metadata may change in the base configuration')
    if digest(data / 'manifest.json') != read(ROOT / 'reports/budget_v1/freeze.json')['bundle_manifest_sha256']:
        raise ValueError('This numerical study requires exactly the historical score bundle')
    out.mkdir(parents=True)
    children = []
    for name, seed in zip(NAMES, SEEDS):
        config = out / 'configurations' / (name + '.json')
        write(config, {**base, 'budget_seed': seed})
        child = out / 'children' / name
        experiment.freeze(data, child, config)
        children.append({'name': name, 'seed': seed, 'freeze_sha256': digest(child / 'freeze.json')})
    plans = [read(out / 'children' / name / 'plans.json') for name in NAMES]
    if plans[0] != read(ROOT / 'reports/budget_v1/plans.json'):
        raise ValueError('Original sampling plan changed')
    distinct = sum(len({tuple(plan[seed][index]['track_ids']) for plan in plans})
                   for seed in plans[0] for index in range(3)) * 6
    if distinct != study['expected_distinct_subset_cases']:
        raise ValueError('Unexpected number of distinct calibration subsets; no redraw permitted')
    paths = sources()
    for path in paths:
        target = out / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(out / 'portable_freeze.json', {
        'schema_version': 1, 'created_utc': now(),
        'stage': 'All six plans and numerical sources frozen before fitting',
        'source_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        'children': children, 'expected_executions': 2160,
        'distinct_source_subset_cases': distinct,
        'runtime': {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()},
        'data_manifest_sha256': digest(data / 'manifest.json')})


def check_freeze(data, out):
    frozen = read(out / 'portable_freeze.json')
    current = {p.relative_to(ROOT).as_posix() for p in sources()}
    if current != set(frozen['source_hashes']):
        raise ValueError('Frozen numerical source inventory changed')
    for name, expected in frozen['source_hashes'].items():
        if digest(ROOT / name) != expected or digest(out / 'source' / name) != expected:
            raise ValueError('Frozen numerical source changed: ' + name)
    if frozen['data_manifest_sha256'] != digest(data / 'manifest.json'):
        raise ValueError('Input manifest changed')
    if frozen['runtime']['python'] != platform.python_version():
        raise ValueError('Frozen Python version changed')
    if (frozen['runtime']['system'] != platform.system() or
            frozen['runtime']['machine'] != platform.machine()):
        raise ValueError('Frozen operating system or architecture changed; create a new run directory '
                         'for a different OS or architecture')
    if [(x['name'], x['seed']) for x in frozen['children']] != list(zip(NAMES, SEEDS)):
        raise ValueError('Six-run inventory changed')
    full_plans = None
    for item in frozen['children']:
        child = out / 'children' / item['name']
        if digest(child / 'freeze.json') != item['freeze_sha256']:
            raise ValueError('Child freeze changed')
        for package, expected in read(child / 'freeze.json')['versions'].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError('Frozen numerical dependency changed: ' + package)
        actual_data, _, config, plans = experiment.verify_frozen(child)
        if actual_data != data or config['budget_seed'] != item['seed']:
            raise ValueError('Child input/configuration mismatch')
        full = {seed: rows[2] for seed, rows in plans.items()}
        if full_plans is not None and full != full_plans:
            raise ValueError('Full-budget plans differ across repeated draws')
        full_plans = full
    return frozen


def recover_running(child):
    """Only parent-owned runs reach here, under both output locks."""
    _, manifest, config, _ = experiment.verify_frozen(child)
    expected = {f"cases/{case['id']}/budget_{index}" for case in manifest['cases']
                for index in range(len(config['budgets']))}
    statuses = sorted((child / 'cases').rglob('status.json'))
    for path in statuses:
        if path.parent.relative_to(child).as_posix() not in expected:
            raise ValueError('Unplanned case is preserved: ' + str(path))
    for path in statuses:
        record = read(path)
        if record['state'] == 'complete':
            continue  # The engine checks every completed artifact before reuse.
        if record['state'] != 'running':
            raise ValueError('Failed or unrecognized case retained for investigation: ' + str(path))
        relative = path.parent.relative_to(child)
        hashes = inventory(path.parent)
        destination = child / 'recovery' / uuid.uuid4().hex
        destination.mkdir(parents=True)
        shutil.move(str(path.parent), str(destination / 'case'))
        write(destination / 'recovery.json', {'case': relative.as_posix(),
              'reason': 'Interrupted owned case retained and refitted from its frozen plan',
              'created_utc': now(), 'original_sha256': hashes})


def numerical_controls(children):
    numerical_failures, fitted_count, single_class, repeated = [], 0, 0, 0
    max_gradient = max_step = 0.0
    full = {}
    for name, child in children.items():
        completed = read(child / 'completed.json')
        for relative in completed['cases']:
            path = child / relative
            case, records = read(path / 'case.json'), read(path / 'calibrators.json')
            if case['fraction'] == 1.0:
                hashes = {key: digest(path / key) for key in ['calibrators.json', 'predictions.npz', 'metrics.json']}
                if relative in full:
                    if full[relative] != hashes:
                        raise ValueError('Same-platform full-budget repeat changed: ' + relative)
                    repeated += 1
                else:
                    full[relative] = hashes
            for method, labels in records.items():
                for label, record in enumerate(labels):
                    fits = [('final', record)]
                    if method == 'ridge_platt':
                        fits += [(f"candidate/{row['penalty']}/fold/{fold['fold']}", fold['fit'])
                                 for row in record['oof_scores'] for fold in row['fold_fits']]
                    for stage, fitted in fits:
                        if fitted.get('fallback'):
                            if fitted['reason'] == 'single_class_calibration':
                                single_class += 1
                            else:
                                numerical_failures.append(f'{name}/{relative}/{method}/{label}/{stage}')
                        if 'solver' in fitted:
                            fitted_count += 1
                            max_gradient = max(max_gradient, fitted['projected_gradient'])
                            max_step = max(max_step, fitted['relative_parameter_step'])
    return {'numerical_failure_count': len(numerical_failures), 'numerical_failures': numerical_failures,
            'optimizer_fits': fitted_count, 'single_class_fits': single_class,
            'maximum_projected_gradient': max_gradient, 'maximum_relative_parameter_step': max_step,
            'identical_full_budget_repeats': repeated, 'full_budget_source_cases': len(full)}


def check_report(report, frozen_hash):
    marker, receipt = report / '.portable.json', report / 'reproduction_receipt.json'
    if (not marker.is_file() or read(marker) != {'freeze_sha256': frozen_hash} or
            not receipt.is_file() or read(receipt).get('status') != 'passed'):
        raise ValueError('Report is not a complete result owned by this frozen run')
    actual = inventory(report)
    actual.pop('reproduction_receipt.json')
    if actual != read(receipt)['report_sha256']:
        raise ValueError('Published report changed; preserving it without overwrite')


def reproduce(data, out, report):
    data, out, report = validate_paths(data, out, report)
    with ExitStack() as locks:
        for path in sorted([out, report]):
            locks.enter_context(Lock(path))
        if not out.exists():
            if report.exists():
                raise FileExistsError('Existing report cannot be adopted by a new run')
            freeze(data, out)
        frozen = check_freeze(data, out)
        frozen_hash = digest(out / 'portable_freeze.json')
        if report.exists():
            check_report(report, frozen_hash)
        stage = report.with_name('.' + report.name + '.staging-' + uuid.uuid4().hex)
        stage.mkdir(parents=True, exist_ok=False)
        write(stage / '.portable.json', {'freeze_sha256': frozen_hash})
        children = {name: out / 'children' / name for name in NAMES}
        for name, child in children.items():
            print('Numerical v2: ' + name, flush=True)
            recover_running(child)
            experiment.run(child)
            summarize(child, stage / name)
            result = verify(child)
            if result['status'] != 'passed' or not result['summary_verified']:
                raise ValueError('Independent reconstruction failed: ' + name)
            for filename in ['freeze.json', 'verification.json', 'config.json']:
                shutil.copy2(child / filename, stage / name / filename)
        check_freeze(data, out)
        control = numerical_controls(children)
        write(stage / 'numerical_controls.json', control)
        if control['numerical_failure_count'] or control['identical_full_budget_repeats'] != 600:
            raise ValueError('Numerical convergence/repetition acceptance failed')
        from compare_portable_runs import export_run_signatures
        export_run_signatures(children, stage)
        shutil.copy2(out / 'portable_freeze.json', stage / 'portable_freeze.json')
        write(stage / 'reproduction_receipt.json', {'schema_version': 1, 'status': 'passed',
              'completed_utc': now(), 'scope': 'One-platform numerical v2 reproduction; cross-platform comparison is a separate required check.',
              'freeze_sha256': frozen_hash, 'cases': frozen['expected_executions'],
              'report_sha256': inventory(stage)})
        if report.exists():
            check_report(report, frozen_hash)
            report.rename(report.with_name('.' + report.name + '.archive-' + uuid.uuid4().hex))
        stage.rename(report)
        print('All 2,160 numerical-v2 cases independently verified; cross-platform comparison still required.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/legacy_scores')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/portable_reproduction')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/portable_reproduction')
    args = parser.parse_args()
    reproduce(args.data, args.out, args.report)
