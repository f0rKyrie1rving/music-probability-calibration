#!/usr/bin/env python3
"""Export verified historical scores without importing or changing the old project.

This is a one-time, local migration tool. The resulting score bundle is sufficient
for calibration experiments without audio, encoder weights or feature caches.
It does not make the already observed historical data an untouched test set.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np

LABELS = ['electronic', 'pop', 'ambient', 'rock']
RUN = Path('outputs/calibration_extension/20260927_v2')
EXCLUSIONS = Path('docs/application_candidate_validation/future_exclusions.json')
FEATURES = {
    'maest': ('outputs/maest_hf/features', (1206, 2304)),
    'mert_v0': ('outputs/expanded/mert_layers', (1206, 13, 768)),
    'mert_v1': ('outputs/mert_v1/mert_layers', (1206, 13, 768)),
}
PROVENANCE = {
    'maest': {'model_plan_sha256': 'experiments/maest_hf_plan.json',
              'encoder_manifest_sha256': 'models/mtg-upf-maest-519l/SOURCES.json',
              'extractor_source_sha256': 'maest_hf_features.py',
              'driver_source_sha256': 'extract_expanded_maest_hf.py'},
    'mert_v0': {'model_plan_sha256': 'experiments/expanded_model_plan.json',
                'encoder_manifest_sha256': 'models/mert-v0-public/SOURCES.json',
                'base_extractor_sha256': 'mert_features.py',
                'layer_extractor_sha256': 'mert_layer_features.py',
                'driver_source_sha256': 'extract_expanded_mert.py'},
    'mert_v1': {'model_plan_sha256': 'experiments/mert_v1_plan_v2.json',
                'encoder_manifest_sha256': 'models/mert-v1-95m/SOURCES.json',
                'extractor_source_sha256': 'mert_v1_features.py',
                'driver_source_sha256': 'extract_expanded_mert_v1.py'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def checked_path(base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), 'Non-relative provenance path')
    return path


def check_hashes(base, hashes):
    for name, expected in hashes.items():
        require(sha256(checked_path(base, name)) == expected, f'Changed input: {name}')


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -np.asarray(logits, dtype=np.float64)))


def close(actual, expected, message, tolerance=2e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message)
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(error <= tolerance, f'{message}: maximum error {error}')
    return error


def check_split(split, ids, artists, y):
    """Reject row duplication, omissions, artist leakage and degenerate targets."""
    indices = {}
    for role in ('fit', 'calibration', 'evaluation'):
        index = np.asarray(split['indices'][role])
        require(index.ndim == 1 and index.dtype.kind in 'iu' and len(index) > 0,
                f'Invalid {role} indices')
        require(np.all((index >= 0) & (index < len(ids))), 'Out-of-range split index')
        require(len(set(index.tolist())) == len(index), 'Repeated split row')
        require(np.all((y[index].sum(0) > 0) & (y[index].sum(0) < len(index))),
                'Single-class split label')
        indices[role] = index
    require(sorted(np.concatenate(list(indices.values())).tolist()) == list(range(len(ids))),
            'Split does not partition the development pool exactly once')
    roles = list(indices)
    for j, role in enumerate(roles):
        for other in roles[j + 1:]:
            require(not set(artists[indices[role]]) & set(artists[indices[other]]),
                    f'Artist leakage between {role} and {other}')
    return indices


def check_rows(saved, index, ids, artists, y):
    for name, reference in [('ids', ids[index]), ('artists', artists[index]), ('y', y[index])]:
        require(np.array_equal(saved[name], reference), f'Misaligned saved {name}')
    require(saved['logits'].shape == (len(index), len(LABELS)) and
            np.isfinite(saved['logits']).all(), 'Invalid saved logits')


def derived_offsets(heads, fit_y, weighting):
    require(np.asarray(heads['n_fit']).shape == () and int(heads['n_fit']) == len(fit_y),
            'Saved head fit count differs from fit IDs')
    require(np.array_equal(heads['positive_fit'], fit_y.sum(0)),
            'Saved head positive counts differ from fit labels')
    offsets = np.zeros(4, dtype=np.float64)
    if weighting == 'ambient_balanced':
        count = int(fit_y[:, 2].sum())
        require(0 < count < len(fit_y), 'Invalid fitted ambient prevalence')
        offsets[2] = np.log(count / (len(fit_y) - count))
    else:
        require(weighting == 'unweighted', 'Unknown weighting')
    return offsets


def export(legacy_root, destination):
    root, out = Path(legacy_root).resolve(), Path(destination).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite bundle: {out}')
    # All audit work below is read-only; write the bundle only after it passes.
    run = root / RUN
    frozen = read(run / 'freeze.json')
    check_hashes(root, frozen['input_hashes'])
    check_hashes(root, frozen['source_hashes'])
    check_hashes(run / 'source', frozen['source_hashes'])
    check_hashes(run, frozen['local_hashes'])
    verification = read(run / 'verification.json')
    require(verification['status'] == 'passed' and not verification['partial'] and
            verification['completed_runs'] == verification['planned_runs'] == 120 and
            not verification['errors'], 'Legacy run verification is incomplete')
    config, splits = read(run / 'config.json'), read(run / 'splits.json')
    require(config['representations'] == list(FEATURES) and
            config['weightings'] == ['ambient_balanced', 'unweighted'] and
            len(config['seeds']) == 20 and len(set(config['seeds'])) == 20 and
            [split['seed'] for split in splits] == config['seeds'], 'Unexpected case plan')

    plan = read(root / 'experiments/final_holdout_plan.json')
    metadata = read(root / 'data/expanded_manifest.json')
    require(plan['labels'] == metadata['labels'] == LABELS, 'Label order differs')
    require(sha256(root / 'data/expanded_manifest.json') == plan['expanded_manifest_sha256'],
            'Development manifest differs from original plan')
    by_id = {row['track_id']: row for row in metadata['tracks']}
    require(len(by_id) == len(metadata['tracks']), 'Duplicate manifest IDs')
    rows = [by_id[track] for track in plan['fit_ids']]
    ids = np.asarray([row['track_id'] for row in rows])
    artists = np.asarray([row['artist_id'] for row in rows])
    y = np.asarray([[int(bool(set(row['tags']) & set(plan['ontology'][label])))
                     for label in LABELS] for row in rows], dtype=np.int64)
    require(len(rows) == len(set(ids)) == 1206 and len(set(artists)) == 469 and
            all(row['phase_role'] in ('train', 'development_validation') for row in rows),
            'Unexpected development cohort')
    # Manifest targets are historical exact tags. The final experiment deliberately
    # derives broader targets (e.g. punkrock -> rock) from this frozen ontology.
    literals = {}
    for node in ast.parse((root / 'expanded_model.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('LABELS', 'ONTOLOGY'):
                    literals[target.id] = ast.literal_eval(node.value)
    require(literals == {'LABELS': LABELS, 'ONTOLOGY': plan['ontology']},
            'Frozen ontology differs from original target-definition source')
    require(y.sum(0).tolist() == [435, 261, 244, 220], 'Changed broad target counts')
    historical_target_differences = int((y != np.asarray([row['targets'] for row in rows])).any(axis=1).sum())

    features, feature_audit = {}, {}
    for representation, (prefix, shape) in FEATURES.items():
        array_path = root / (prefix + '.npy')
        meta = read(root / (prefix + '.json'))
        require(meta['ids'] == ids.tolist() and meta['test_tracks'] == 0 and
                meta['feature_sha256'] == frozen['input_hashes'][prefix + '.npy'] and
                meta['expanded_manifest_sha256'] == plan['expanded_manifest_sha256'],
                f'Changed feature provenance: {representation}')
        for key, path in PROVENANCE[representation].items():
            require(meta[key] == frozen['input_hashes'][path], f'Broken provenance: {path}')
        values = np.load(array_path, mmap_mode='r', allow_pickle=False)
        require(values.shape == shape and values.dtype == np.float32 and np.isfinite(values).all(),
                f'Invalid feature cache: {representation}')
        features[representation] = np.asarray(values if representation == 'maest'
                                               else values[:, 1:].mean(axis=1), dtype=np.float64)
        feature_audit[representation] = {'path': prefix + '.npy', 'sha256': meta['feature_sha256'],
            'source_shape': list(shape), 'classifier_shape': list(features[representation].shape),
            'pooling': 'original MAEST' if representation == 'maest' else 'float32 mean of layers 1..12',
            'ids_aligned': True, 'finite': True}

    exclusions = read(root / EXCLUSIONS)
    require(set(ids).issubset(exclusions['tracks']) and set(artists).issubset(exclusions['artists']),
            'Latest exclusions do not cover all exported historical examples')
    original_exclusions = root / 'outputs/application_candidate_validation/20260928_v1/future_exclusions.json'
    if original_exclusions.exists():
        require(read(original_exclusions) == exclusions, 'Published exclusions differ from source receipt')
    prepared, case_provenance, maximum_error = [], [], 0.0
    for split in splits:
        indices = check_split(split, ids, artists, y)
        fi, ci, ei = [indices[role] for role in ('fit', 'calibration', 'evaluation')]
        for representation in config['representations']:
            for weighting in config['weightings']:
                case_id = f"{split['seed']}/{representation}/{weighting}"
                relative = RUN / 'runs' / case_id
                folder = root / relative
                status = read(folder / 'status.json')
                require(status['state'] == 'complete', f'Incomplete case: {case_id}')
                required = {'heads.npz', 'calibrators.json', 'calibration.npz', 'evaluation.npz', 'metrics.json'}
                require(required.issubset(status['hashes']), f'Incomplete hashes: {case_id}')
                check_hashes(folder, status['hashes'])
                head = load_npz(folder / 'heads.npz')
                cal, evaluation = [load_npz(folder / name) for name in ('calibration.npz', 'evaluation.npz')]
                check_rows(cal, ci, ids, artists, y)
                check_rows(evaluation, ei, ids, artists, y)
                fitted = read(folder / 'calibrators.json')
                offsets = derived_offsets(head, y[fi], weighting)
                maximum_error = max(maximum_error, close(fitted['offsets'], offsets, 'Prior correction mismatch'))
                for indices_, saved in [(ci, cal), (ei, evaluation)]:
                    x = features[representation][indices_]
                    require(np.all(head['scale'] > 0), 'Nonpositive feature scaling')
                    reconstructed = ((x - head['mean']) / head['scale']) @ head['coef'].T + head['intercept']
                    maximum_error = max(maximum_error, close(saved['logits'], reconstructed,
                                                            f'Feature/head logit reconstruction: {case_id}'))
                references = {'raw': sigmoid(evaluation['logits']),
                              'prior_offset': sigmoid(evaluation['logits'] + offsets)}
                for method in ('sigmoid', 'temperature'):
                    parameters = fitted['calibrators'][method]
                    require(len(parameters) == 4 and all(p['success'] for p in parameters), 'Failed calibrator')
                    references[method] = np.column_stack([
                        sigmoid(p['slope'] * evaluation['logits'][:, j] + p['intercept'])
                        for j, p in enumerate(parameters)])
                for method, reconstructed in references.items():
                    maximum_error = max(maximum_error, close(evaluation[method], reconstructed,
                                                            f'Saved reference reconstruction: {case_id}/{method}'))
                arrays = {'fit_ids': ids[fi], 'fit_artists': artists[fi], 'fit_y': y[fi],
                          'cal_ids': ids[ci], 'cal_artists': artists[ci], 'cal_y': y[ci], 'cal_logits': cal['logits'],
                          'eval_ids': ids[ei], 'eval_artists': artists[ei], 'eval_y': y[ei], 'eval_logits': evaluation['logits'],
                          'offsets': offsets, 'reference_raw': evaluation['raw'], 'reference_platt': evaluation['sigmoid'],
                          'reference_temperature': evaluation['temperature'], 'reference_prior_offset': evaluation['prior_offset']}
                record = {'id': case_id, 'seed': split['seed'], 'representation': representation,
                          'weighting': weighting, 'path': 'cases/' + case_id.replace('/', '_') + '.npz'}
                prepared.append((record, arrays))
                case_provenance.append({'id': case_id, 'path': relative.as_posix(),
                    'status_sha256': sha256(folder / 'status.json'), 'files_sha256': status['hashes']})
    require(len(prepared) == 120, 'Incomplete export')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=root, text=True).strip()
    source = {'repository': url, 'commit': commit, 'run_path': RUN.as_posix(),
              'run_freeze_sha256': sha256(run / 'freeze.json'), 'original_freeze_base_commit': frozen['git_commit'],
              'note': 'Original freeze preceded the research commit; source hashes identify the exact executed code.'}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cases').mkdir()
    (out / 'provenance').mkdir()
    records = []
    for record, arrays in prepared:
        np.savez_compressed(out / record['path'], **arrays)
        records.append({**record, 'sha256': sha256(out / record['path'])})
    for filename in ('freeze.json', 'config.json', 'protocol.md', 'splits.json', 'verification.json', 'audit.json'):
        shutil.copyfile(run / filename, out / 'provenance' / ('legacy_' + filename))
    shutil.copyfile(root / EXCLUSIONS, out / 'provenance/future_exclusions.json')
    attribution_fields = ('track_id', 'artist_id', 'title', 'artist', 'track_url', 'license_url',
                          'license_code', 'attribution_verbatim')
    save(out / 'provenance/track_attribution.json', {
        'source_dataset': 'MTG-Jamendo', 'source_url': 'https://github.com/MTG/mtg-jamendo-dataset',
        'scope': 'Inherited source metadata and per-track attribution; no audio is included. '
                 'Software licensing does not relicense source audio or metadata.',
        'tracks': [{key: row[key] for key in attribution_fields} for row in rows]})
    save(out / 'provenance/export_audit.json', {
        'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(), 'source': source,
        'exporter_sha256': sha256(__file__), 'feature_audit': feature_audit,
        'cohort': {'tracks': len(ids), 'artists': len(set(artists)), 'positive_labels': y.sum(0).tolist(),
                   'label_definition': 'Frozen broad ontology applied to original tags, not historical exact-tag targets',
                   'ontology': plan['ontology'], 'historical_exact_tag_target_differences': historical_target_differences},
        'cases': case_provenance, 'maximum_absolute_reconstruction_error': maximum_error,
        'checks': ['All original frozen input/source/snapshot hashes match',
                   'Every case is complete and its saved file hashes match',
                   'Original IDs, artists and ontology-derived labels align in every case',
                   'All representations and weightings use the same split for each seed',
                   'Fit, calibration and evaluation artists are disjoint within each seed',
                   'Logits reconstruct from the original feature caches and fitted heads',
                   'Prior offsets derive only from verified classifier-fit targets',
                   'All four saved reference probabilities reconstruct numerically'],
        'future_exclusions': {'source_path': EXCLUSIONS.as_posix(), 'sha256': sha256(root / EXCLUSIONS),
                              'tracks': len(exclusions['tracks']), 'artists': len(exclusions['artists'])},
        'excluded_assets': ['audio', 'encoder weights', 'classifier weights', 'full feature caches'],
        'scope': 'Migration and numerical provenance verification, not independent scientific validation.'})
    hashes = {p.relative_to(out).as_posix(): sha256(p) for p in sorted((out / 'provenance').iterdir())}
    manifest = {'schema_version': 1, 'labels': LABELS, 'cases': records, 'source': source,
                'exposure': 'retrospective/development, already observed',
                'provenance_hashes': hashes,
                'independence_note': '20 overlapping splits of one 1206-track cohort; 120 cases are not independent datasets.'}
    save(out / 'manifest.json', manifest)
    return {'cases': len(records), 'tracks': len(ids), 'artists': len(set(artists)),
            'maximum_absolute_reconstruction_error': maximum_error,
            'manifest_sha256': sha256(out / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.legacy_root, args.out), indent=2))


if __name__ == '__main__':
    main()
