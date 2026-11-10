import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

CELLS = (81, 23)
DL = (1.0, 4.0)
L = (CELLS[0] * DL[0], CELLS[1] * DL[1])
MODE = (1, 2)
K = (2 * np.pi * MODE[0] / L[0], 2 * np.pi * MODE[1] / L[1])
DT = 0.5
FINAL = 1000.0
EPS = 1e-3
DIAG_DIR = "phare_outputs/mhd_ct_energy/wenoz_mode_decay"

BASE = dict(rho=1.0, vx=0.0, vy=1.0, vz=0.0, bx=0.5, by=np.sqrt(3) / 2, bz=0.0, P=0.6)
AMPLITUDES = dict(
    rho=0.314309961006 + 0.046738661809j,
    vx=-0.019957672270 + 0.261806285923j,
    vy=0.372015051603 - 0.094003205221j,
    vz=0.0,
    bx=0.176430907862 - 0.239009944395j,
    by=-0.101427069334 + 0.137402672215j,
    bz=0.0,
    P=0.311507992006 + 0.050311597542j,
)

GETTERS = (
    ("GetMHDrho", ("mhdRho",)),
    ("GetMHDrhoV", ("mhdRhoVx", "mhdRhoVy", "mhdRhoVz")),
    ("GetB", ("Bx", "By", "Bz")),
    ("GetMHDEtot", ("mhdEtot",)),
)


def perturbed(name):
    def fn(x, y):
        phase = np.exp(1j * (K[0] * x + K[1] * y))
        return BASE[name] + 2 * EPS * np.real(AMPLITUDES[name] * phase)

    return fn


def run_simulation():
    ph.global_vars.sim = None
    sim = ph.Simulation(
        model_options=["MHDModel"],
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=False,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        max_mhd_level=1,
        time_step=DT,
        final_time=FINAL,
        cells=CELLS,
        dl=DL,
        diag_options={
            "format": "phareh5",
            "options": {"dir": DIAG_DIR, "mode": "overwrite"},
        },
    )
    ph.MHDModel(
        density=perturbed("rho"),
        vx=perturbed("vx"),
        vy=perturbed("vy"),
        vz=perturbed("vz"),
        bx=perturbed("bx"),
        by=perturbed("by"),
        bz=perturbed("bz"),
        p=perturbed("P"),
    )
    timestamps = [0.0, FINAL]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ("rho", "rhoV", "Etot"):
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)
    simulator = Simulator(sim)
    simulator.run().reset()
    del simulator
    ph.global_vars.sim = None


def projection(hierarchy, time, name, k, length):
    values = {}
    for level in hierarchy.levels(time).values():
        for patch in level.patches:
            data = patch.patch_datas[name]
            x, y = data.meshgrid()
            field = np.asarray(data.dataset[:])
            inside = (
                (x >= -1e-9)
                & (x < length[0] - 1e-9)
                & (y >= -1e-9)
                & (y < length[1] - 1e-9)
            )
            for xi, yi, vi in zip(x[inside], y[inside], field[inside]):
                values[(round(xi, 6), round(yi, 6))] = vi
    points = np.array(list(values.keys()))
    field = np.array(list(values.values()))
    phase = np.exp(-1j * (k[0] * points[:, 0] + k[1] * points[:, 1]))
    return np.sum(field * phase) / len(field)


def mode_amplitudes(run, time, k, length):
    out = []
    for getter, names in GETTERS:
        hierarchy = getattr(run, getter)(time, all_primal=False)
        out += [projection(hierarchy, time, name, k, length) for name in names]
    return np.array(out)


class WenozModeDecayTest(unittest.TestCase):
    def test_wenoz_mode_decays(self):
        run_simulation()
        run = Run(DIAG_DIR)
        a0 = mode_amplitudes(run, 0.0, K, L)
        a1 = mode_amplitudes(run, FINAL, K, L)
        ratio = np.linalg.norm(a1) / np.linalg.norm(a0)
        print(f"WENOZ mode amplitude ratio t={FINAL}/t=0: {ratio:.4f}", flush=True)
        self.assertLess(ratio, 0.9)


if __name__ == "__main__":
    startMPI()
    unittest.main()
