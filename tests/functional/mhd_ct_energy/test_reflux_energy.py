import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

ALPHA = np.pi / 6
COS, SIN = np.cos(ALPHA), np.sin(ALPHA)
GAMMA = 5.0 / 3.0
P0 = 0.1
DB = 0.01
N = 32
LO, HI = N // 4, 3 * N // 4 - 1
DL = ((1.0 / N) / COS, (1.0 / N) / SIN)
FINAL = 0.25
NSTEPS = 33
DIAG_DIR = "phare_outputs/mhd_ct_energy/reflux_energy"


def phase(x, y):
    return 2 * np.pi * (x * COS + y * SIN)


def run_simulation(name, amr, reflux_energy="conservative"):
    ph.global_vars.sim = None
    levels = (
        dict(
            refinement="boxes",
            refinement_boxes={"L0": {"B0": [[LO, LO], [HI, HI]]}},
            max_mhd_level=2,
        )
        if amr
        else dict(refinement="tagging", max_mhd_level=1, max_nbr_levels=1)
    )
    ph.Simulation(
        smallest_patch_size=8,
        time_step=FINAL / NSTEPS,
        final_time=FINAL,
        cells=(N, N),
        dl=DL,
        nesting_buffer=1,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=False,
        model_options=["MHDModel"],
        reflux_energy=reflux_energy,
        diag_options={
            "format": "phareh5",
            "options": {"dir": f"{DIAG_DIR}/{name}", "mode": "overwrite"},
        },
        **levels,
    )
    ph.MHDModel(
        density=lambda x, y: 1.0 + 0 * x,
        vx=lambda x, y: -DB * np.sin(phase(x, y)) * SIN,
        vy=lambda x, y: DB * np.sin(phase(x, y)) * COS,
        vz=lambda x, y: DB * np.cos(phase(x, y)),
        bx=lambda x, y: COS - DB * np.sin(phase(x, y)) * SIN,
        by=lambda x, y: SIN + DB * np.sin(phase(x, y)) * COS,
        bz=lambda x, y: DB * np.cos(phase(x, y)),
        p=lambda x, y: P0 + 0 * x,
    )
    for quantity in ("P", "Etot"):
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=[0.0, FINAL])
    simulator = Simulator(ph.global_vars.sim)
    simulator.run().reset()
    del simulator
    ph.global_vars.sim = None
    return Run(f"{DIAG_DIR}/{name}")


def own(data):
    return tuple(slice(g, g + n) for g, n in zip(data.ghosts_nbr, data.box.shape))


def coarse_pressure(run):
    out = np.full((N, N), np.nan)
    for patch in run.GetMHDP(FINAL, all_primal=False).level(0, FINAL).patches:
        data = patch.patch_datas["mhdP"]
        lower, upper = data.box.lower, data.box.upper
        out[lower[0] : upper[0] + 1, lower[1] : upper[1] + 1] = np.asarray(
            data.dataset[:]
        )[own(data)]
    return out


def first_coarse_row():
    i = np.arange(N)
    distance = np.maximum(np.maximum(LO - i, i - HI), 0)
    return np.maximum(distance[:, None], distance[None, :]) == 1


def composite_energy(run, time):
    total = 0.0
    for ilvl, level in run.GetMHDEtot(time, all_primal=False).levels(time).items():
        volume = np.prod(np.asarray(DL) / 2**ilvl)
        for patch in level.patches:
            data = patch.patch_datas["mhdEtot"]
            values = np.asarray(data.dataset[:], dtype=float)[own(data)]
            if ilvl == 0:
                i = np.arange(data.box.lower[0], data.box.upper[0] + 1)
                j = np.arange(data.box.lower[1], data.box.upper[1] + 1)
                covered = ((i >= LO) & (i <= HI))[:, None] & ((j >= LO) & (j <= HI))[
                    None, :
                ]
                values = np.where(covered, 0.0, values)
            total += values.sum() * volume
    return total


class RefluxEnergyTest(unittest.TestCase):
    def test_consistent_reflux_removes_the_coarse_fine_pressure_error(self):
        uniform = coarse_pressure(run_simulation("uniform", amr=False))
        conservative = run_simulation("conservative", amr=True)
        consistent = run_simulation("consistent", amr=True, reflux_energy="consistent")

        row = first_coarse_row()
        excess = {}
        for name, run in (("conservative", conservative), ("consistent", consistent)):
            pressure = coarse_pressure(run)
            self.assertTrue(np.isfinite(pressure).all())
            excess[name] = float(np.sqrt(np.mean((pressure - uniform)[row] ** 2)))
        print(f"first coarse row pressure excess {excess}", flush=True)
        self.assertLess(excess["consistent"], 0.35 * excess["conservative"])

        e0 = composite_energy(conservative, 0.0)
        for run in (conservative, consistent):
            drift = abs(composite_energy(run, FINAL) - e0) / e0
            self.assertLess(drift, 1e-6)


if __name__ == "__main__":
    startMPI()
    unittest.main()
