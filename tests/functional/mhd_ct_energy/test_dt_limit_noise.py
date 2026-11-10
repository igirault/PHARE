import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.hierarchy import hierarchy_from
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

GAMMA = 5.0 / 3.0
CELLS = (32, 32)
DL = (0.1, 0.4)
L = (CELLS[0] * DL[0], CELLS[1] * DL[1])
FINAL = 15.0
AMP = 1e-4
DIAG_DIR = "phare_outputs/mhd_ct_energy/dt_limit_noise"

BASE = dict(rho=1.0, vx=0.0, vy=0.0, vz=0.0, bx=0.0, by=0.0, bz=1.0, P=0.25 / GAMMA)


def noise_modes():
    rng = np.random.default_rng(12345)
    modes = [(13, 15, 0.0)]
    for _ in range(16):
        mx, my = rng.integers(1, 16, size=2)
        modes.append((int(mx), int(my), float(rng.uniform(0, 2 * np.pi))))
    return modes


MODES = noise_modes()


def perturbed(name, index):
    def fn(x, y):
        noise = sum(
            np.cos(2 * np.pi * (mx * x / L[0] + my * y / L[1]) + phi + index)
            for mx, my, phi in MODES
        )
        return BASE[name] + AMP * noise

    return fn


def run_simulation():
    ph.global_vars.sim = None
    sim = ph.Simulation(
        model_options=["MHDModel"],
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=False,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        max_mhd_level=1,
        time_step={"mode": "adaptive", "cfl": 1.0},
        final_time=FINAL,
        cells=CELLS,
        dl=DL,
        diag_options={
            "format": "phareh5",
            "options": {"dir": DIAG_DIR, "mode": "overwrite"},
        },
    )
    names = ("rho", "vx", "vy", "vz", "bx", "by", "bz", "P")
    fields = {name: perturbed(name, i) for i, name in enumerate(names)}
    ph.MHDModel(
        density=fields["rho"],
        vx=fields["vx"],
        vy=fields["vy"],
        vz=fields["vz"],
        bx=fields["bx"],
        by=fields["by"],
        bz=fields["bz"],
        p=fields["P"],
    )
    timestamps = [0.0, FINAL]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="rho", write_timestamps=timestamps)
    simulator = Simulator(sim)
    simulator.run().reset()
    del simulator
    ph.global_vars.sim = None


def mean_square_deviation(hierarchy, time, name, base):
    values = {}
    for level in hierarchy.levels(time).values():
        for patch in level.patches:
            data = patch.patch_datas[name]
            x, y = data.meshgrid()
            field = np.asarray(data.dataset[:])
            inside = (x >= -1e-9) & (x < L[0] - 1e-9) & (y >= -1e-9) & (y < L[1] - 1e-9)
            for xi, yi, vi in zip(x[inside], y[inside], field[inside]):
                values[(round(xi, 6), round(yi, 6))] = vi
    deviation = np.array(list(values.values())) - base
    return float(np.mean(deviation**2))


def perturbation_rms(run, time):
    rho = run.GetMHDrho(time, all_primal=False)
    b = run.GetB(time, all_primal=False)
    total = mean_square_deviation(rho, time, "mhdRho", BASE["rho"])
    for name, key in (("Bx", "bx"), ("By", "by"), ("Bz", "bz")):
        total += mean_square_deviation(b, time, name, BASE[key])
    return np.sqrt(total)


class TimeStepLimitNoiseTest(unittest.TestCase):
    def test_noise_does_not_grow_at_the_limit(self):
        run_simulation()
        run = Run(DIAG_DIR)
        last = float(hierarchy_from(h5_filename=f"{DIAG_DIR}/mhd_rho.h5").times()[-1])
        rms0 = perturbation_rms(run, 0.0)
        rms1 = perturbation_rms(run, last)
        print(f"perturbation rms t=0 {rms0:.4e}  t={last} {rms1:.4e}", flush=True)
        self.assertGreaterEqual(last, 0.9 * FINAL)
        self.assertTrue(np.isfinite(rms1))
        self.assertLessEqual(rms1, rms0)


if __name__ == "__main__":
    startMPI()
    unittest.main()
