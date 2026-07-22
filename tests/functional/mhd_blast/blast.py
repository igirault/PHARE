#!/usr/bin/env python3
import os

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

# Wu & Shu 2018 (DOI 10.1137/18M1168042), Example 4.4 (Balsara & Spicer 1999).
# PHARE is Heaviside-Lorentz (mu0=1): magnetic pressure = B**2/2, beta = 2 p / B**2.
# The paper's Ba = 100/sqrt(4 pi) is only its CGS->normalized rescaling; expressed
# natively here as Ba = sqrt(2 pa / beta), which reproduces the paper's beta exactly.
CASES = {
    "reference": dict(pe=1e3, pa=0.1, beta=2.5, tmax=0.01, dt=2.0e-5),
    "blast1": dict(pe=1e3, pa=0.1, beta=2.51e-4, tmax=0.01, dt=2.0e-6),
    "blast2": dict(pe=1e4, pa=0.1, beta=2.51e-6, tmax=0.001, dt=2.0e-7),
}

RHO = 1.0
GAMMA = 1.4
R0 = 0.1
XC, YC = 0.5, 0.5


def config(label):
    case = CASES[label]
    pe, pa, beta = case["pe"], case["pa"], case["beta"]
    tmax, dt = case["tmax"], case["dt"]

    Ba = np.sqrt(2.0 * pa / beta)  # native field from target beta

    diag_dir = f"phare_outputs/blast/{label}"
    n_out = 5
    timestamps = np.arange(0, tmax + dt, tmax / n_out)

    cells = (320, 320)
    dl = (1.0 / cells[0], 1.0 / cells[1])

    sim = ph.Simulation(
        time_step=dt,
        final_time=tmax,
        cells=cells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=False,
        res=False,
        hyper_res=False,
        model_options=["MHDModel"],
    )

    def r(x, y):
        return np.sqrt((x - XC) ** 2 + (y - YC) ** 2)

    def density(x, y):
        return RHO + 0.0 * x

    def vx(x, y):
        return 0.0 * x

    def vy(x, y):
        return 0.0 * x

    def vz(x, y):
        return 0.0 * x

    def bx(x, y):
        return Ba + 0.0 * x

    def by(x, y):
        return 0.0 * x

    def bz(x, y):
        return 0.0 * x

    def p(x, y):
        return np.where(r(x, y) < R0, pe, pa)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def run_case(label):
    ph.global_vars.sim = None
    Ba = np.sqrt(2.0 * CASES[label]["pa"] / CASES[label]["beta"])
    if cpp.mpi_rank() == 0:
        print(f"[blast] case={label} beta={CASES[label]['beta']:.3e} Ba={Ba:.4f}", flush=True)
    sim = config(label)
    Simulator(sim).run().reset()
    ph.global_vars.sim = None
    if cpp.mpi_rank() == 0:
        print(f"[blast] case={label} completed to t={CASES[label]['tmax']}", flush=True)
