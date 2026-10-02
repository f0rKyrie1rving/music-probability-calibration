"""Portable pipeline corruption guards and independent optimality verification."""
import copy
import importlib.util
import math
import platform
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from music_calibration_portable.calibrators import fit
from music_calibration_portable.io import digest, read, write
from music_calibration_portable.verify import Checks, _fit_record, _optimality
from scripts import reproduce as original_wrapper

_SPEC = importlib.util.spec_from_file_location('portable_pipeline_for_tests',
    Path(__file__).resolve().parents[1] / 'scripts/reproduce_portable.py')
pipeline = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {'reproduce': original_wrapper}):
    _SPEC.loader.exec_module(pipeline)


class IndependentOptimalityTests(unittest.TestCase):
    def test_bound_active_solution_has_nonzero_gradient_but_valid_kkt(self):
        z, y = np.ones(4), np.array([0., 1., 0., 1.])
        result = fit(z, y, 'temperature')
        self.assertTrue(result['success'], result)
        self.assertEqual(result['slope'], .001)
        self.assertGreater(result['gradient'][0], 0.)
        self.assertEqual(result['projected_gradient'], 0.)
        checks = Checks()
        _fit_record(checks, z, y, result, 'active lower bound')
        self.assertGreater(checks.counts['optimality'], 0)

    def test_forged_success_with_correct_gradient_but_failed_kkt_is_rejected(self):
        # This exact one-dimensional gradient is known without the fit implementation.
        gradient = 1 / (1 + math.exp(-1.)) - .5
        forged = {'method': 'temperature', 'slope': 1., 'intercept': 0., 'success': True,
                  'solver': 'bounded-logistic-newton-v1', 'gradient': [gradient],
                  'projected_gradient': gradient, 'convex_gap_bound': gradient * (1 - .001),
                  'relative_parameter_step': 0.}
        with self.assertRaisesRegex(ValueError, 'KKT condition failed'):
            _optimality(Checks(), np.ones(4), np.array([0., 1., 0., 1.]), forged, 'forged convergence')

    def test_saved_wrong_gradient_cannot_pass_independent_reconstruction(self):
        z, y = np.array([-2., -1., -.1, .3, 1., 2.]), np.array([0, 1, 0, 1, 0, 1])
        result = fit(z, y, 'ridge_platt', penalty=.1)
        result['gradient'][0] += 1e-4
        with self.assertRaisesRegex(ValueError, 'gradient'):
            _optimality(Checks(), z, y, result, 'tampered gradient')

    def test_unconverged_parameter_step_is_rejected_even_with_small_gradient(self):
        z, y = np.ones(4), np.array([0., 1., 0., 1.])
        result = fit(z, y, 'temperature')
        result['relative_parameter_step'] = 1e-5
        with self.assertRaisesRegex(ValueError, 'parameter step not converged'):
            _optimality(Checks(), z, y, result, 'unfinished optimizer')

    def test_single_class_remains_explicit_fallback(self):
        z, y = np.array([-1., 0., 1.]), np.ones(3)
        result = fit(z, y, 'ridge_platt', penalty=.01)
        self.assertFalse(result['success'])
        self.assertTrue(result['fallback'])
        self.assertEqual(result['reason'], 'single_class_calibration')
        self.assertNotIn('solver', result)
        _fit_record(Checks(), z, y, result, 'single-class policy')


class PortablePipelineGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.project = self.root / 'project'
        self.data, self.out, self.report = self.root / 'data', self.root / 'run', self.root / 'report'
        self.source = self.project / 'mock/source.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# frozen solver fixture\n')
        write(self.data / 'manifest.json', {'cases': []})
        saved_source = self.out / 'source/mock/source.py'
        saved_source.parent.mkdir(parents=True)
        saved_source.write_bytes(self.source.read_bytes())
        children = []
        for name, seed in zip(pipeline.NAMES, pipeline.SEEDS):
            child = self.out / 'children' / name
            write(child / 'freeze.json', {'versions': {}})
            children.append({'name': name, 'seed': seed, 'freeze_sha256': digest(child / 'freeze.json')})
        self.frozen = {'source_hashes': {'mock/source.py': digest(self.source)},
                       'data_manifest_sha256': digest(self.data / 'manifest.json'),
                       'children': children, 'expected_executions': 2160,
                       'runtime': {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()}}
        write(self.out / 'portable_freeze.json', self.frozen)
        self.manifest = {'cases': [{'id': '1/maest/unweighted', 'seed': 1}]}
        self.plans = {'1': [{'fraction': .25}, {'fraction': .5}, {'fraction': 1.}]}
        def verified(child):
            index = pipeline.NAMES.index(child.name)
            return self.data, self.manifest, {'budget_seed': pipeline.SEEDS[index], 'budgets': [.25, .5, 1.]}, copy.deepcopy(self.plans)
        self.addCleanup(self.temp.cleanup)
        for item in [patch.object(pipeline, 'ROOT', self.project),
                     patch.object(pipeline, 'sources', return_value=[self.source]),
                     patch.object(pipeline.experiment, 'verify_frozen', side_effect=verified)]:
            item.start()
            self.addCleanup(item.stop)

    def test_frozen_live_or_snapshot_source_change_stops_before_fitting(self):
        for path in [self.source, self.out / 'source/mock/source.py']:
            original = path.read_bytes()
            path.write_bytes(original + b'# changed\n')
            with self.subTest(path=path), patch.object(pipeline.experiment, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    pipeline.reproduce(self.data, self.out, self.report)
                run.assert_not_called()
                self.assertFalse(self.report.exists())
            path.write_bytes(original)

    def test_environment_version_change_stops_before_recovery_or_fitting(self):
        changed = copy.deepcopy(self.frozen)
        changed['runtime']['python'] = '0.0.invalid'
        write(self.out / 'portable_freeze.json', changed)
        with patch.object(pipeline.experiment, 'run') as run, patch.object(pipeline, 'recover_running') as recovery:
            with self.assertRaisesRegex(ValueError, 'Python version changed'):
                pipeline.reproduce(self.data, self.out, self.report)
            run.assert_not_called()
            recovery.assert_not_called()

    def test_system_or_architecture_change_requires_new_run_before_recovery_or_fitting(self):
        for field in ['system', 'machine']:
            with self.subTest(field=field), patch.object(pipeline.platform, field, return_value='different-platform'), \
                    patch.object(pipeline.experiment, 'run') as run, patch.object(pipeline, 'recover_running') as recovery:
                with self.assertRaisesRegex(ValueError, 'create a new run directory.*different OS or architecture'):
                    pipeline.reproduce(self.data, self.out, self.report)
                run.assert_not_called()
                recovery.assert_not_called()
                self.assertFalse(self.report.exists())

    def test_changed_child_freeze_and_parent_inventory_are_rejected(self):
        child = self.out / 'children' / pipeline.NAMES[-1] / 'freeze.json'
        original = child.read_bytes()
        child.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Child freeze changed'):
            pipeline.check_freeze(self.data, self.out)
        child.write_bytes(original)
        changed = copy.deepcopy(self.frozen)
        changed['children'].pop()
        write(self.out / 'portable_freeze.json', changed)
        with self.assertRaisesRegex(ValueError, 'Six-run inventory changed'):
            pipeline.check_freeze(self.data, self.out)

    def test_unrelated_existing_report_is_not_adopted_or_overwritten(self):
        self.report.mkdir()
        sentinel = self.report / 'notes.txt'
        sentinel.write_text('important unrelated notes')
        with patch.object(pipeline.experiment, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'not.*owned'):
                pipeline.reproduce(self.data, self.out, self.report)
            run.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'important unrelated notes')
        self.assertFalse((self.report / '.portable.json').exists())

    def test_changed_owned_report_is_preserved_before_any_case_execution(self):
        write(self.report / '.portable.json', {'freeze_sha256': digest(self.out / 'portable_freeze.json')})
        write(self.report / 'summary.json', {'result': 'original'})
        write(self.report / 'reproduction_receipt.json', {'status': 'passed',
            'freeze_sha256': digest(self.out / 'portable_freeze.json'),
            'report_sha256': original_wrapper.inventory(self.report)})
        write(self.report / 'summary.json', {'result': 'user edit'})
        with patch.object(pipeline.experiment, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'changed'):
                pipeline.reproduce(self.data, self.out, self.report)
            run.assert_not_called()
        self.assertEqual(read(self.report / 'summary.json'), {'result': 'user edit'})

    def test_unplanned_running_case_is_preserved_not_omitted_from_results(self):
        child = self.out / 'children' / pipeline.NAMES[0]
        status = child / 'cases/999/unknown/unweighted/budget_0/status.json'
        write(status, {'state': 'running'})
        before = status.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Unplanned case'):
            pipeline.recover_running(child)
        self.assertEqual(status.read_bytes(), before)
        self.assertFalse((child / 'recovery').exists())

    def test_interrupted_planned_case_keeps_exact_original_bytes_when_archived(self):
        child = self.out / 'children' / pipeline.NAMES[0]
        original = child / 'cases/1/maest/unweighted/budget_0'
        write(original / 'status.json', {'state': 'running', 'phase': 'fit interrupted'})
        (original / 'partial.bin').write_bytes(b'original partial evidence')
        before = original_wrapper.inventory(original)
        pipeline.recover_running(child)
        self.assertFalse(original.exists())
        recovered = list((child / 'recovery').glob('*/recovery.json'))
        self.assertEqual(len(recovered), 1)
        record = read(recovered[0])
        self.assertEqual(record['original_sha256'], before)
        self.assertEqual(original_wrapper.inventory(recovered[0].parent / 'case'), before)

    def test_failed_owned_case_is_retained_instead_of_silently_retried(self):
        child = self.out / 'children' / pipeline.NAMES[0]
        status = child / 'cases/1/maest/unweighted/budget_0/status.json'
        write(status, {'state': 'failed', 'error': 'real numerical bug'})
        before = status.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Failed|failed'):
            pipeline.recover_running(child)
        self.assertEqual(status.read_bytes(), before)
        self.assertFalse((child / 'recovery').exists())


if __name__ == '__main__':
    unittest.main()
