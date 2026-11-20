"""Pipeline checks; the supplied solver implementations are not modified."""

from contextlib import redirect_stdout
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .base_experiment import (METHODS, MODELS, aggregate, generate_corruption,
                              generate_system, run_experiment, validate_system)


class BaseExperimentTests(unittest.TestCase):
    def test_data_generation_and_support(self):
        A, x, b = generate_system(42, 0)
        for model in MODELS:
            support, rho = generate_corruption(b, 42, 0, model)
            validate_system(A, x, b, support, rho, model)
        A_again, x_again, b_again = generate_system(42, 0)
        for a, other in zip((A, x, b), (A_again, x_again, b_again)):
            np.testing.assert_array_equal(a, other)
        self.assertFalse(np.array_equal(A, generate_system(42, 1)[0]))

    def test_statistics_use_trial_axis(self):
        errors = np.array([[[1., 0.2], [1., 0.4]],
                           [[1., 0.6], [1., 0.8]],
                           [[1., 1.0], [1., 1.2]]])
        stats = aggregate(errors, np.array([0.1, 0.3, 0.9]))
        np.testing.assert_allclose(stats['median'], [[1., 0.6], [1., 0.8]])
        np.testing.assert_allclose(stats['q25'], [[1., 0.4], [1., 0.6]])
        np.testing.assert_allclose(stats['q75'], [[1., 0.8], [1., 1.0]])
        self.assertAlmostEqual(float(stats['ls_median']), 0.3)

    def test_saved_runs_reproduce_and_match_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = (Path(directory) / name for name in ('first', 'second'))
            with redirect_stdout(io.StringIO()):
                run_experiment(first, trials=1, iterations=8, seed=42)
                run_experiment(second, trials=1, iterations=8, seed=42)
            for file in first.glob('*.npz'):
                with np.load(file) as a, np.load(second / file.name) as b:
                    self.assertEqual(a.files, b.files)
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            with np.load(first / 'system_00.npz') as system:
                for model in MODELS:
                    with np.load(first / f'{model}_trial_00.npz') as trial:
                        np.testing.assert_allclose(
                            trial['b_corrupted'], system['b_clean'] + trial['rho'])
                        # Verify the LS reference independently through normal-equation residuals.
                        residual = system['A'] @ trial['x_ls'] - trial['b_corrupted']
                        np.testing.assert_allclose(system['A'].T @ residual, 0., atol=1e-11)
                    with np.load(first / f'{model}_results.npz') as result:
                        self.assertEqual(result['errors'].shape, (1, len(METHODS), 8))
                        np.testing.assert_allclose(result['errors'][:, :, 0], 1.)
                        self.assertTrue(np.isfinite(result['errors']).all())
                        np.testing.assert_array_equal(result['iterations'], np.arange(8))
            with self.assertRaises(FileExistsError):
                run_experiment(first, trials=1, iterations=8, seed=42)


if __name__ == '__main__':
    unittest.main()
