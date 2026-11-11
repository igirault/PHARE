import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

ALPHA = np.pi / 6
COS, SIN = np.cos(ALPHA), np.sin(ALPHA)
N = 32
LO, HI = N // 4, 3 * N // 4 - 1
DL = ((1.0 / N) / COS, (1.0 / N) / SIN)
WIDTH = 0.005
NSTEPS = 12
DT = 0.2 / (4.0 / DL[0] + 4.0 / DL[1])
DIAG_DIR = "phare_outputs/mhd_prolongation/sharp_jump"


def inner(x, y):
    s = np.mod(x * COS + y * SIN, 1.0)
    return 0.5 * (np.tanh((s - 0.25) / WIDTH) - np.tanh((s - 0.75) / WIDTH))


def tangential(x, y):
    return 1.0 - 2.0 * inner(x, y)


def run_simulation():
    ph.global_vars.sim = None
    timestamps = DT * np.arange(NSTEPS + 1)
    ph.Simulation(
        smallest_patch_size=8,
        time_step=DT,
        final_time=NSTEPS * DT,
        cells=(N, N),
        dl=DL,
        nesting_buffer=1,
        refinement="boxes",
        refinement_boxes={"L0": {"B0": [[LO, LO], [HI, HI]]}},
        max_mhd_level=2,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=False,
        model_options=["MHDModel"],
        diag_options={
            "format": "phareh5",
            "options": {"dir": DIAG_DIR, "mode": "overwrite"},
        },
    )
    ph.MHDModel(
        density=lambda x, y: 1.0 - 0.875 * inner(x, y),
        vx=lambda x, y: 0 * x,
        vy=lambda x, y: 0 * x,
        vz=lambda x, y: 0 * x,
        bx=lambda x, y: 0.75 * COS - tangential(x, y) * SIN,
        by=lambda x, y: 0.75 * SIN + tangential(x, y) * COS,
        bz=lambda x, y: 0 * x,
        p=lambda x, y: 1.0 - 0.9 * inner(x, y),
    )
    ph.MHDDiagnostics(quantity="P", write_timestamps=timestamps)
    simulator = Simulator(ph.global_vars.sim)
    simulator.run().reset()
    del simulator
    ph.global_vars.sim = None
    return Run(DIAG_DIR), timestamps


def own(data):
    return tuple(slice(g, g + n) for g, n in zip(data.ghosts_nbr, data.box.shape))


class SharpJumpRefinementTest(unittest.TestCase):
    def test_refining_a_sharp_jump_keeps_the_pressure_positive(self):
        run, timestamps = run_simulation()
        for time in timestamps:
            hierarchy = run.GetMHDP(time, all_primal=False)
            levels = hierarchy.levels(time)
            self.assertEqual(len(levels), 2)
            for level in levels.values():
                for patch in level.patches:
                    data = patch.patch_datas["mhdP"]
                    pressure = np.asarray(data.dataset[:])[own(data)]
                    self.assertTrue(np.isfinite(pressure).all(), f"t={time}")
                    self.assertGreater(pressure.min(), 0.0, f"t={time}")


if __name__ == "__main__":
    startMPI()
    unittest.main()
