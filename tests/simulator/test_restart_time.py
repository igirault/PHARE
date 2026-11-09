#!/usr/bin/env python3

import unittest
from pathlib import Path

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator import simulator
from pyphare.simulator.simulator import Simulator, startMPI
from tests.simulator import SimulatorTest

GAMMA = 5.0 / 3.0
V = (0.3, -0.2, 0.1)
B = (0.5, 0.4, 0.3)
P = 0.2

OUT = "phare_outputs/restart_time"
CONSTANT = dict(time_step=0.1, time_step_nbr=6)
ADAPTIVE = dict(time_step={"mode": "adaptive", "cfl_wave": 0.5}, final_time=1.0)


def uniform_model():
    def const(value):
        return lambda x, y: value + 0.0 * x

    ph.MHDModel(
        density=const(1.0),
        vx=const(V[0]),
        vy=const(V[1]),
        vz=const(V[2]),
        bx=const(B[0]),
        by=const(B[1]),
        bz=const(B[2]),
        p=const(P),
    )


class RestartTimeTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None
        super().tearDown()

    def out(self):
        out = f"{OUT}/{self._testMethodName}"
        if out not in self.diag_dirs:
            self.register_diag_dir_for_cleanup(out)
        return out

    def run_recording(self, restart_options, diag_dir=None, diags=False, **time_kwargs):
        out = self.out()
        ph.global_vars.sim = None
        ph.Simulation(
            interp_order=None,
            model_options=["MHDModel"],
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
            mhd_timestepper="TVDRK3",
            hall=False,
            eta=0.0,
            nu=0.0,
            gamma=GAMMA,
            max_mhd_level=1,
            cells=(20, 10),
            dl=(0.3, 0.2),
            restart_options=dict(
                dir=f"{out}/restarts", mode="overwrite", **restart_options
            ),
            diag_options={
                "format": "phareh5",
                "options": {"dir": f"{out}/{diag_dir or 'diags'}", "mode": "overwrite"},
            },
            **time_kwargs,
        )
        uniform_model()
        if diags:
            final_time = time_kwargs["final_time"]
            ph.ElectromagDiagnostics(quantity="B", write_timestamps=[final_time])
            for quantity in ("rho", "V", "P"):
                ph.MHDDiagnostics(quantity=quantity, write_timestamps=[final_time])

        simulator.exit_on_exception = False
        try:
            self.simulator = Simulator(ph.global_vars.sim).initialize()
            times = [self.simulator.currentTime()]
            end_time = self.simulator.cpp_sim.endTime()
            while self.simulator.currentTime() < end_time:
                self.simulator.advance()
                times.append(self.simulator.currentTime())
        finally:
            simulator.exit_on_exception = True
            if self.simulator is not None:
                self.simulator.reset()
            self.simulator = None
        return times

    def assert_h5_trees_equal(self, dir0, dir1):
        from pyphare import cpp

        if cpp.mpi_rank() > 0:
            return

        import h5py

        def datasets(path):
            found = {}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    found[name] = obj[()]

            with h5py.File(path, "r") as f:
                f.visititems(visit)
            return found

        names = sorted(p.name for p in Path(dir0).glob("*.h5"))
        self.assertGreater(len(names), 0)
        for name in names:
            d0 = datasets(Path(dir0) / name)
            d1 = datasets(Path(dir1) / name)
            self.assertGreater(len(d0), 0)
            self.assertEqual(sorted(d0), sorted(d1))
            for key, value in d0.items():
                np.testing.assert_equal(value, d1[key])

    def test_constant_restart_reproduces_times(self):
        tA = self.run_recording(dict(timestamps=[0.3]), **CONSTANT)
        tB = self.run_recording(dict(restart_time=0.3), time_step=0.1, time_step_nbr=3)
        self.assertEqual(tB, tA[3:])

    def test_restart_time_rounding_to_folder(self):
        tA = self.run_recording(dict(timestamps=[0.3]), **CONSTANT)
        tB = self.run_recording(
            dict(restart_time=0.300004), time_step=0.1, time_step_nbr=3
        )
        self.assertEqual(tB, tA[3:])

    def test_constant_restart_with_new_dt_reanchors(self):
        tA = self.run_recording(dict(timestamps=[0.3]), **CONSTANT)
        tB = self.run_recording(dict(restart_time=0.3), time_step=0.05, time_step_nbr=4)
        self.assertEqual(tB, [tA[3] + float(k) * 0.05 for k in range(5)])

    def test_adaptive_to_constant_restart_reanchors(self):
        tA = self.run_recording(dict(timestamps=[0.5]), **ADAPTIVE)
        tB = self.run_recording(
            dict(restart_time="auto"), time_step=0.01, time_step_nbr=5
        )
        self.assertIn(tB[0], tA)
        self.assertEqual(tB, [tB[0] + float(k) * 0.01 for k in range(6)])

    def test_adaptive_restart_reproduces_run(self):
        tA = self.run_recording(
            dict(timestamps=[0.5]), diag_dir="diag_A", diags=True, **ADAPTIVE
        )
        tB = self.run_recording(
            dict(restart_time="auto"), diag_dir="diag_B", diags=True, **ADAPTIVE
        )
        self.assertIn(tB[0], tA)
        self.assertEqual(tB, tA[tA.index(tB[0]) :])
        self.assert_h5_trees_equal(f"{self.out()}/diag_A", f"{self.out()}/diag_B")

    def test_adaptive_restart_lands_on_final_time(self):
        restart_time = 63.0 / 64.0
        self.run_recording(
            dict(timestamps=[restart_time]),
            time_step=restart_time / 32,
            time_step_nbr=32,
        )
        for cfl in (0.507, 0.173, 0.365):
            with self.subTest(cfl=cfl):
                tB = self.run_recording(
                    dict(restart_time=restart_time),
                    time_step={"mode": "adaptive", "cfl_wave": cfl},
                    final_time=2.0,
                )
                steps = np.diff(tB)
                self.assertEqual(tB[0], restart_time)
                self.assertEqual(tB[-1], 2.0)
                self.assertGreater(steps.min(), 1e-6 * steps.max())

    def test_restart_with_zero_steps(self):
        tA = self.run_recording(dict(timestamps=[0.3]), **CONSTANT)
        tB = self.run_recording(dict(restart_time=0.3), time_step=0.1, time_step_nbr=0)
        self.assertEqual(tB, [tA[3]])

    def test_adaptive_restart_near_final_time(self):
        tA = self.run_recording(dict(timestamps=[1.0]), **ADAPTIVE)
        tB = self.run_recording(dict(restart_time="auto"), **ADAPTIVE)
        self.assertIn(tB[0], tA)
        self.assertEqual(tB, tA[tA.index(tB[0]) :])
        self.assertEqual(tB[-1], 1.0)

    def test_same_dt_chain_keeps_stored_anchor(self):
        tA = self.run_recording(dict(timestamps=[0.3]), time_step=0.1, time_step_nbr=3)
        tB = self.run_recording(
            dict(restart_time=0.3, timestamps=[tA[3] + 4.0 * 0.05]),
            time_step=0.05,
            time_step_nbr=6,
        )
        tC = self.run_recording(
            dict(restart_time="auto"), time_step=0.05, time_step_nbr=2
        )
        self.assertEqual(tC, tB[4:7])

    def to_old_format(self, restart_time, clock_offset=0.0):
        from pyphare import cpp
        from pyphare.cpp import cpp_etc_lib

        if cpp.mpi_rank() == 0:
            import h5py

            folder = cpp_etc_lib().restart_path_for_time(
                f"{self.out()}/restarts", restart_time
            )
            record = (
                "time_step_mode",
                "time_step",
                "time_anchor_time",
                "time_anchor_step",
            )
            for path in Path(folder).rglob("*"):
                if path.is_file() and h5py.is_hdf5(path):
                    with h5py.File(path, "a") as f:
                        if "phare" in f:
                            for name in record:
                                if name in f["phare"].attrs:
                                    del f["phare"].attrs[name]
                        clock = "TimeRefinementIntegrator/d_integrator_time"
                        if clock in f:
                            f[clock][...] = f[clock][()] + clock_offset

        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()

    def test_old_format_restart_on_the_grid_keeps_original_anchor(self):
        tA = self.run_recording(dict(timestamps=[0.5]), time_step=0.1, time_step_nbr=11)
        self.to_old_format(0.5, clock_offset=1e-12)
        tB = self.run_recording(dict(restart_time=0.5), time_step=0.1, time_step_nbr=6)
        self.assertEqual(tB[0], tA[5] + 1e-12)
        self.assertEqual(tB[1:], tA[6:])

    def test_old_format_chain_with_changed_dt_reanchors(self):
        tA = self.run_recording(dict(timestamps=[0.3]), time_step=0.1, time_step_nbr=3)
        tB = self.run_recording(
            dict(restart_time=0.3, timestamps=[tA[3] + 4.0 * 0.05]),
            time_step=0.05,
            time_step_nbr=6,
        )
        self.to_old_format(0.5)
        tC = self.run_recording(
            dict(restart_time="auto"), time_step=0.05, time_step_nbr=2
        )
        self.assertEqual(tC, [tB[4] + float(k) * 0.05 for k in range(3)])


if __name__ == "__main__":
    startMPI()
    unittest.main()
