#
# Config-level validation tests for the time_step_type option (constant vs adaptive).
# These only construct ph.Simulation (pharein) and never run the simulator, so they are cheap
# and need no cpp module / MPI / HighFive.
#

import unittest
import numpy as np
import pyphare.pharein as ph

# minimal valid geometry; time parameters are supplied per-test
baseArgs = dict(
    boundary_types="periodic",
    cells=np.array([20]),
    dl=0.3,
)


# pure pharein config validation: never runs the simulator, so no MPI / cpp module needed
class TimeStepValidation(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None

    def tearDown(self):
        ph.global_vars.sim = None

    # ---- constant (default, backward compatible) -------------------------------------------

    def test_constant_is_the_default(self):
        sim = ph.Simulation(time_step=0.001, time_step_nbr=10, **baseArgs)
        self.assertEqual(sim.time_step_type, "constant")
        self.assertEqual(sim.time_step, 0.001)
        self.assertEqual(sim.time_step_nbr, 10)

    def test_constant_explicit(self):
        sim = ph.Simulation(
            time_step_type="constant", time_step=0.001, final_time=1.0, **baseArgs
        )
        self.assertEqual(sim.time_step_type, "constant")
        self.assertEqual(sim.time_step, 0.001)

    # ---- adaptive --------------------------------------------------------------------------

    def test_adaptive_accepts_final_time_and_cfl(self):
        sim = ph.Simulation(
            time_step_type="adaptive",
            time_step_cfl=0.4,
            final_time=1.0,
            **baseArgs,
        )
        self.assertEqual(sim.time_step_type, "adaptive")
        self.assertEqual(sim.time_step_cfl, 0.4)
        self.assertEqual(sim.final_time, 1.0)
        # with adaptive dt these are unknown ahead of the run
        self.assertIsNone(sim.time_step)
        self.assertIsNone(sim.time_step_nbr)

    def test_adaptive_fourier_defaults_to_cfl(self):
        sim = ph.Simulation(
            time_step_type="adaptive", time_step_cfl=0.4, final_time=1.0, **baseArgs
        )
        self.assertEqual(sim.time_step_fourier, 0.4)

    def test_adaptive_fourier_explicit(self):
        sim = ph.Simulation(
            time_step_type="adaptive",
            time_step_cfl=0.4,
            time_step_fourier=0.2,
            final_time=1.0,
            **baseArgs,
        )
        self.assertEqual(sim.time_step_fourier, 0.2)

    def test_adaptive_requires_cfl(self):
        with self.assertRaises(ValueError):
            ph.Simulation(time_step_type="adaptive", final_time=1.0, **baseArgs)

    def test_adaptive_requires_final_time(self):
        with self.assertRaises(ValueError):
            ph.Simulation(time_step_type="adaptive", time_step_cfl=0.4, **baseArgs)

    def test_adaptive_rejects_time_step(self):
        with self.assertRaises(ValueError):
            ph.Simulation(
                time_step_type="adaptive",
                time_step_cfl=0.4,
                final_time=1.0,
                time_step=0.001,
                **baseArgs,
            )

    def test_adaptive_rejects_time_step_nbr(self):
        with self.assertRaises(ValueError):
            ph.Simulation(
                time_step_type="adaptive",
                time_step_cfl=0.4,
                final_time=1.0,
                time_step_nbr=10,
                **baseArgs,
            )

    def test_adaptive_rejects_non_positive_cfl(self):
        with self.assertRaises(ValueError):
            ph.Simulation(
                time_step_type="adaptive",
                time_step_cfl=0.0,
                final_time=1.0,
                **baseArgs,
            )

    # ---- unknown keyword -------------------------------------------------------------------

    def test_unknown_time_step_type_raises(self):
        with self.assertRaises(ValueError):
            ph.Simulation(
                time_step_type="variable",  # common mistake: the option is "adaptive"
                time_step_cfl=0.4,
                final_time=1.0,
                **baseArgs,
            )


if __name__ == "__main__":
    unittest.main()
