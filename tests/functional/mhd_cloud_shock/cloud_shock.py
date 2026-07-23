#!/usr/bin/env python3
#
# mhd_cloud_shock
#
# Cloud-shock interaction problem (Toth 2000, 10.1006/jcph.2000.6519, S6.5),
# originally introduced by Dai & Woodward, 1998, 10.1006/jcph.1998.5944.
#
# Domain [0,1]x[0,1]. A discontinuity at x=0.6 separates a left MHD state
# (high pressure, at rest) from a right state (low pressure, moving in -x at
# highly super-fast-magnetosonic speed). A dense circular cloud sits at
# (0.8, 0.5), radius 0.15, embedded in the right state. Because the right
# state carries vx=-11.2536 into the domain from x=1, and the corresponding
# fast magnetosonic speed there is ~1.52 (Bx=0 so vf = sqrt(cs^2+va^2)), the
# flow at xupper is super-fast-magnetosonic: a fixed/inflow boundary
# sits on the right. xlower/ylower/yupper use "open"
# (zero-gradient) condition.
#
# At the time when this case is committed (23/07/2025), the case is found
# not to work in following conditions:
# - ideal MHD with WENOZ, no matter the resolution between dx = 1/50 and 1/800:
#   Is observed to fails on NaNs around t ~ 3.0 - 3.5.
# - ideal MHD with WENO3. Fails right before final time.
# - Hall MHD, no matter the reconstruction (Linear+VanLeer, WENOZ).


import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from pyphare import cpp
import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "1"  # turn on scope timing

ph.NO_GUI()


@dataclass
class State:
    gamma: float
    rho: float
    p: float
    vx: float
    bx: float
    by: float
    bz: float
    b: float = field(init=False)
    cs: float = field(init=False)
    va: float = field(init=False)
    cf_fast: float = field(init=False)

    def __post_init__(self):
        b = np.linalg.norm([self.bx, self.by, self.bz])
        cs = np.sqrt(self.gamma * self.p / self.rho)
        va = b / np.sqrt(self.rho)
        cos_theta = (self.bx / b) if b > 0 else 0.0
        common = np.sqrt((cs**2 + va**2) ** 2 - 4.0 * cs**2 * va**2 * cos_theta**2)
        self.b, self.cs, self.va = b, cs, va
        self.cf_fast = np.sqrt((cs**2 + va**2 + common) / 2)


gamma = 5.0 / 3.0
XC = 0.6
CLOUD_X, CLOUD_Y, CLOUD_R = 0.8, 0.5, 0.15
CLOUD_RHO = 10.0

LEFT = State(gamma, 3.86859, 167.345, 0.0, 0.0, 2.1826182, -2.1826182)
RIGHT = State(gamma, 1.0, 1.0, -11.2536, 0.0, 0.56418958, 0.56418958)

ncells = (50, 50) 
domain_size = (1.0, 1.0)
dl = (domain_size[0] / ncells[0], domain_size[1] / ncells[1])


def whistler_speed(rho, bdotb, dx):
    """ Whistler wave speed at grid spacing dx, same formula as PHARE's own. """
    inv_mesh_size = 1.0 / dx
    vw = np.sqrt(1.0 + 0.25 * inv_mesh_size**2) + 0.5 * inv_mesh_size
    return np.sqrt(bdotb) * vw / rho


def _bdotb(state):
    return state.bx**2 + state.by**2 + state.bz**2


HALL = False

cfl = 0.8
dx_min = min(dl)
LEFT_CW = whistler_speed(LEFT.rho, _bdotb(LEFT), dx_min) if HALL else 0.0
RIGHT_CW = whistler_speed(RIGHT.rho, _bdotb(RIGHT), dx_min) if HALL else 0.0
max_speed = max(
    abs(LEFT.vx) + LEFT.cf_fast + LEFT_CW, abs(RIGHT.vx) + RIGHT.cf_fast + RIGHT_CW
)
time_step = cfl * dx_min / max_speed
final_time = 0.06
time_step_nbr = int(np.ceil(final_time / time_step))
timestamps = np.arange(0, time_step_nbr + 1, 10) * time_step
diag_dir = "phare_outputs/mhd_cloud_shock"


def print_case_info():
    things_to_print = {
        "left state": LEFT,
        "right state": RIGHT,
        "left whistler speed": LEFT_CW,
        "right whistler speed": RIGHT_CW,
        "max_speed": max_speed,
        "dt": time_step,
    }
    if cpp.mpi_rank() == 0:
        for key, thing in things_to_print.items():
            print(f"{key} = {thing}")


def config():
    sim = ph.Simulation(
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=ncells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=3,
        max_nbr_levels=3,
        nesting_buffer=1,
        smallest_patch_size=15,
        eta=0.0,
        nu=0.0,
        hyper_mode="spatial",
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite", "allow_emergency_dumps": True},
        },
        strict=True,
        gamma=gamma,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        interp_order=1,
        mhd_timestepper="TVDRK2",
        hall=HALL,
        res=False,
        hyper_res=False,
        model_options=["MHDModel"],
        boundary_types=("physical", "physical"),
        boundary_conditions={
            "xlower": {"type": "open"},
            "ylower": {"type": "open"},
            "yupper": {"type": "open"},
            "xupper": {
                "type": "super-magnetofast-inflow",
                "data": {
                    "velocity": [RIGHT.vx, 0.0, 0.0],
                    "density": RIGHT.rho,
                    "pressure": RIGHT.p,
                    "B": [RIGHT.bx, RIGHT.by, RIGHT.bz],
                },
            },
        },
    )

    def density(x, y):
        rho = np.where(x < XC, LEFT.rho, RIGHT.rho)
        r = np.sqrt((x - CLOUD_X) ** 2 + (y - CLOUD_Y) ** 2)
        return np.where(r < CLOUD_R, CLOUD_RHO, rho)

    def vx(x, y):
        return np.where(x < XC, LEFT.vx, RIGHT.vx)

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return 0.0

    def by(x, y):
        return np.where(x < XC, LEFT.by, RIGHT.by)

    def bz(x, y):
        return np.where(x < XC, LEFT.bz, RIGHT.bz)

    def p(x, y):
        return np.where(x < XC, LEFT.p, RIGHT.p)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)

    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def plot_file_for_qty(plot_dir, qty, time):
    return f"{plot_dir}/cloud_shock_{qty}_t{time}.png"


def plot(diag_dir, plot_dir):
    run = Run(diag_dir)
    for time in timestamps:
        run.GetDivB(time).plot(
            filename=plot_file_for_qty(plot_dir, "divb", time),
            plot_patches=True,
            vmin=-1e-11,
            vmax=1e-11,
        )
        run.GetRanks(time).plot(
            filename=plot_file_for_qty(plot_dir, "Ranks", time), plot_patches=True
        )
        run.GetMHDrho(time).plot(
            filename=plot_file_for_qty(plot_dir, "rho", time), plot_patches=True
        )
        for c in ["x", "y", "z"]:
            run.GetMHDV(time).plot(
                filename=plot_file_for_qty(plot_dir, f"v{c}", time),
                plot_patches=True,
                qty=f"{c}",
            )
            run.GetB(time).plot(
                filename=plot_file_for_qty(plot_dir, f"b{c}", time),
                plot_patches=True,
                qty=f"{c}",
            )
        run.GetMHDP(time).plot(
            filename=plot_file_for_qty(plot_dir, "p", time), plot_patches=True
        )
        if HALL:
            run.GetJ(time).plot(
                filename=plot_file_for_qty(plot_dir, "jz", time),
                qty="z",
                plot_patches=True,
            )


def count_nans(diag_dir):
    """Total NaN count over every dumped field (all times, levels, patches,
    ghosts included). Reads the h5 files directly: the pharesee reconstruction
    pads with NaN and would give false positives.
    """
    import h5py

    n = 0
    for fname in ("EM_B.h5", "mhd_rho.h5", "mhd_V.h5", "mhd_P.h5"):
        path = Path(diag_dir) / fname
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:

            def visit(_, obj):
                nonlocal n
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
                    n += int(np.isnan(obj[...]).sum())

            f.visititems(visit)
    return n


def assert_inflow_holds_right_state(diag_dir):
    """Checks that the xupper super-magnetofast-inflow boundary kept
    the rightmost column of cells pinned at the prescribed right state
    (density, vx) at every dump, on L0.
    """
    if cpp.mpi_rank() != 0:
        return

    diag_path = Path(diag_dir)
    rho_path = diag_path / "mhd_rho.h5"
    v_path = diag_path / "mhd_V.h5"
    if not rho_path.exists() or not v_path.exists():
        return

    import h5py

    x_upper_index = ncells[0] - 1  # last global level-0 cell index in x

    def last_interior_column(grp, dataset_name):
        if int(grp.attrs["upper"][0]) != x_upper_index:
            return None
        ds = grp[dataset_name]
        ghosts = int(ds.attrs["ghosts"])
        nx = int(grp.attrs["nbrCells"][0])
        col = ghosts + nx - 1
        return np.asarray(ds[col, :])

    checked_any = False
    with h5py.File(rho_path, "r") as frho, h5py.File(v_path, "r") as fv:
        for t in timestamps:
            tkey = f"t/{t:.10f}"
            lvl0_rho = frho.get(f"{tkey}/pl0")
            lvl0_v = fv.get(f"{tkey}/pl0")
            if lvl0_rho is None or lvl0_v is None:
                continue

            rhos = [
                col
                for grp in lvl0_rho.values()
                if (col := last_interior_column(grp, "rho")) is not None
            ]
            vxs = [
                col
                for grp in lvl0_v.values()
                if (col := last_interior_column(grp, "V_x")) is not None
            ]

            if not rhos or not vxs:
                continue

            rho_right = np.concatenate(rhos)
            vx_right = np.concatenate(vxs)

            assert np.all(np.isfinite(rho_right)), (
                f"non-finite density at t={t} on the xupper boundary column"
            )
            assert np.all(np.isfinite(vx_right)), (
                f"non-finite vx at t={t} on the xupper boundary column"
            )
            assert np.allclose(rho_right, RIGHT.rho, rtol=5e-2), (
                f"inflow density at t={t} drifted from the prescribed right "
                f"state: got range [{rho_right.min()},{rho_right.max()}], "
                f"expected ~{RIGHT.rho}"
            )
            assert np.allclose(vx_right, RIGHT.vx, rtol=5e-2), (
                f"inflow vx at t={t} drifted from the prescribed right state: "
                f"got range [{vx_right.min()},{vx_right.max()}], expected ~{RIGHT.vx}"
            )
            checked_any = True

    assert checked_any, "no timestamps were available to check the inflow boundary"


def assert_divb_null(diag_dir, tol=1e-11):
    """
    Checks that divB is null.
    """
    if cpp.mpi_rank() != 0:
        return

    diag_path = Path(diag_dir)
    if not diag_path.exists():
        return

    run = Run(diag_dir)
    checked_any = False
    for t in timestamps:
        try:
            divb_f = run.GetDivB(t)
        except Exception:
            continue

        levels = divb_f.patch_levels[0]
        if not levels:
            continue

        maxdiv = 0.0
        for _ilvl, lvl in levels.items():
            for patch in lvl.patches:
                vals = np.asarray(patch.patch_datas["value"].dataset[:])
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    maxdiv = max(maxdiv, float(np.abs(finite).max()))

        assert maxdiv < tol, f"divB={maxdiv} exceeds tol={tol} at t={t}"
        checked_any = True

    assert checked_any, "no timestamps were available to check divB"


class CloudShockTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super(CloudShockTest, self).__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super(CloudShockTest, self).tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        self.register_diag_dir_for_cleanup(diag_dir)
        Simulator(config()).run().reset()
        print_case_info()
        try:
            if cpp.mpi_rank() == 0:
                self.assertEqual(
                    count_nans(diag_dir),
                    0,
                    "NaN found in dumped fields: inflow/outflow BC or regrid "
                    "ghost fill likely broken",
                )
                plot_dir = Path(f"{diag_dir}_plots") / str(cpp.mpi_size())
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot(diag_dir, plot_dir)
            assert_inflow_holds_right_state(diag_dir)
            assert_divb_null(diag_dir)
        except Exception:  # allow case to exit in case of Exception
            import signal
            import traceback

            traceback.print_exc()
            os.killpg(os.getpgrp(), signal.SIGTERM)
            raise
        cpp.mpi_barrier()
        return self


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    startMPI()
    CloudShockTest().test_run().tearDown()
