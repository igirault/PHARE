#
# Config-level validation tests for the period-based dump cadence options
# (write_niter_period / write_time_period on diagnostics, niter_period / time_period on
# restarts). These only construct diagnostics/restart options (never run the simulator), so
# they are cheap and need no cpp module / MPI / HighFive.
#

import unittest
import numpy as np
import pyphare.pharein as ph

baseArgs = dict(
    boundary_types="periodic",
    cells=np.array([20]),
    dl=0.3,
    time_step=0.001,
    time_step_nbr=10,
)


class DiagnosticsCadenceValidation(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None
        ph.Simulation(**baseArgs)

    def tearDown(self):
        ph.global_vars.sim = None

    def test_rejects_fractional_write_niter_period(self):
        with self.assertRaises(RuntimeError):
            ph.ElectromagDiagnostics(quantity="B", write_niter_period=2.9)

    def test_accepts_integer_valued_float_write_niter_period(self):
        diag = ph.ElectromagDiagnostics(quantity="B", write_niter_period=3.0)
        self.assertEqual(diag.write_niter_period, 3)

    def test_rejects_non_positive_write_niter_period(self):
        with self.assertRaises(RuntimeError):
            ph.ElectromagDiagnostics(quantity="B", write_niter_period=0)

    def test_write_niter_period_and_write_timestamps_are_mutually_exclusive(self):
        with self.assertRaises(RuntimeError):
            ph.ElectromagDiagnostics(
                quantity="B", write_niter_period=2, write_timestamps=np.array([0.0])
            )

    def test_write_niter_period_and_write_time_period_are_mutually_exclusive(self):
        with self.assertRaises(RuntimeError):
            ph.ElectromagDiagnostics(
                quantity="B", write_niter_period=2, write_time_period=0.1
            )


class RestartsCadenceValidation(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None

    def tearDown(self):
        ph.global_vars.sim = None

    def _simulation(self, **restart_options):
        return ph.Simulation(restart_options=restart_options, **baseArgs)

    def test_rejects_fractional_niter_period(self):
        with self.assertRaises(RuntimeError):
            self._simulation(mode="overwrite", niter_period=2.9)

    def test_accepts_integer_valued_float_niter_period(self):
        sim = self._simulation(mode="overwrite", niter_period=3.0)
        self.assertEqual(sim.restart_options["write_niter_period"], 3)

    def test_rejects_non_positive_niter_period(self):
        with self.assertRaises(RuntimeError):
            self._simulation(mode="overwrite", niter_period=0)

    def test_niter_period_and_timestamps_are_mutually_exclusive(self):
        with self.assertRaises(RuntimeError):
            self._simulation(mode="overwrite", niter_period=2, timestamps=[0.0])

    def test_niter_period_and_time_period_are_mutually_exclusive(self):
        with self.assertRaises(RuntimeError):
            self._simulation(mode="overwrite", niter_period=2, time_period=0.1)


if __name__ == "__main__":
    unittest.main()
