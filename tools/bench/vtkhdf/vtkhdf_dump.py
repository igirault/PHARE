import os
import time

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

cells = tuple(int(c) for c in os.environ.get("E2E_CELLS", "128,128,128").split(","))
largest = int(os.environ.get("E2E_LARGEST", "32"))
ndumps = int(os.environ.get("E2E_DUMPS", "4"))
diag_dir = os.environ["E2E_DIR"]
dl = (0.4, 0.4, 0.4)
time_step = 0.002
timestamps = [k * time_step for k in range(1, ndumps + 1)]


def config():
    L = 0.5
    sim = ph.Simulation(
        time_step=time_step,
        final_time=time_step * (ndumps + 1),
        dl=dl,
        cells=cells,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        nesting_buffer=1,
        largest_patch_size=largest,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
    )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def bx(x, y, z):
        Ly = sim.simulation_domain()[1]
        return -1 + 2.0 * (S(y, Ly * 0.25, L) - S(y, Ly * 0.75, L))

    def zero(x, y, z):
        return 0.0 * x

    def b2(x, y, z):
        return bx(x, y, z) ** 2

    def p(x, y, z):
        return 1.0 - b2(x, y, z) / 2.0 + 0.5

    def density(x, y, z):
        return 0.5 + 0.0 * x

    ph.MHDModel(
        density=density, vx=zero, vy=zero, vz=zero, bx=bx, by=zero, bz=zero, p=p
    )
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "rhoV", "Etot", "V"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)
    return sim


def main():
    startMPI()
    rank = cpp.mpi_rank()
    t0 = time.time()
    sim = Simulator(config(), auto_dump=False)
    sim.initialize()
    cpp.mpi_barrier()
    if rank == 0:
        print(
            f"E2E init {time.time() - t0:.1f}s ranks={cpp.mpi_size()} cells={cells}",
            flush=True,
        )
    for k, ts in enumerate(timestamps):
        cpp.mpi_barrier()
        a = time.time()
        sim.cpp_sim.dump_diagnostics(timestamp=ts, timestep=time_step)
        cpp.mpi_barrier()
        if rank == 0:
            print(f"E2E dump {k} {time.time() - a:.3f}s", flush=True)
    sim.reset()
    ph.global_vars.sim = None


if __name__ == "__main__":
    main()
