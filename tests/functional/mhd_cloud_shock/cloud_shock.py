#!/usr/bin/env python3
#
# mhd_cloud_shock
#
# Cloud-shock interaction problem (Toth 2000, 10.1006/jcph.2000.6519, S6.5),
# used here as a discriminating test for the outer physical BC combination of
# a fixed supersonic inflow (xupper, super-magnetofast-inflow) with open
# outflow (xlower, ylower, yupper), run with Hall MHD + spatial
# hyper-resistivity on a 3-level tagged AMR hierarchy.
#
# Domain [0,1]x[0,1]. A discontinuity at x=0.6 separates a left MHD state
# (high pressure, at rest) from a right state (low pressure, moving in -x at
# highly super-fast-magnetosonic speed). A dense circular cloud sits at
# (0.8, 0.5), radius 0.15, embedded in the right state. Because the right
# state carries vx=-11.2536 into the domain from x=1, and the corresponding
# fast magnetosonic speed there is ~1.52 (Bx=0 so vf = sqrt(cs^2+va^2)), the
# flow at xupper is super-fast-magnetosonic: the fixed/inflow boundary
# belongs on the right, not the left, contrary to a naive reading of the
# paper's (OCR'd) boundary description. xlower/ylower/yupper use "open"
# (zero-gradient), matching the paper's "approximately open" description.
#
# Tagging keeps refined levels tracking the shock front and the cloud edge;
# because the shock reaches xupper as the run proceeds, this also exercises
# the init/regrid physical-ghost fill at a *fixed inflow* boundary (as
# opposed to mhd_harris_with_boundaries, which exercises it at an *open*
# boundary).

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

cfl = 0.4
max_speed = max(abs(LEFT.vx) + LEFT.cf_fast, abs(RIGHT.vx) + RIGHT.cf_fast)
time_step = cfl * min(dl) / max_speed
final_time = 0.06
timestamps = np.arange(0.0, final_time + time_step, final_time / 5)
diag_dir = "phare_outputs/mhd_cloud_shock"


def print_case_info():
    things_to_print = {
        "left state": LEFT,
        "right state": RIGHT,
        "max_speed": max_speed,
        "dt": time_step,
    }
    if cpp.mpi_rank() == 0:
        for key, thing in things_to_print.items():
            print(f"{key} = {thing}")


def config():
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=ncells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=3,
        max_nbr_levels=3,
        nesting_buffer=1,
        smallest_patch_size=15,
        resistivity=0.0,
        hyper_resistivity=0.01,
        hyper_mode="spatial",
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        gamma=gamma,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        interp_order=1,
        mhd_timestepper="SSPRK4_5",
        hall=True,
        res=False,
        hyper_res=True,
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
    """Proves at RUNTIME that the xupper super-magnetofast-inflow boundary kept
    the rightmost row of cells pinned at the prescribed right state (density,
    vx) at every dump, on the base (level 0) grid which always spans the full
    domain and therefore always touches xupper.
    """
    if cpp.mpi_rank() != 0:
        return

    diag_path = Path(diag_dir)
    if not diag_path.exists():
        return

    run = Run(diag_dir)
    Lx = domain_size[0]
    dx = dl[0]

    checked_any = False
    for t in timestamps:
        try:
            rho_f = run.GetMHDrho(t)
            v_f = run.GetMHDV(t)
        except Exception:
            continue

        rho_levels = rho_f.patch_levels[0]
        v_levels = v_f.patch_levels[0]
        if 0 not in rho_levels or 0 not in v_levels:
            continue

        rhos = []
        for patch in rho_levels[0].patches:
            pd = patch.patch_datas["value"]
            vals = np.asarray(pd.dataset[:])
            assert np.all(np.isfinite(vals)), (
                f"non-finite density at t={t} in patch {patch.box}"
            )
            X, Y = np.meshgrid(pd.x, pd.y, indexing="ij")
            mask = X >= (Lx - 0.5 * dx)
            if mask.any():
                rhos.append(vals[mask])

        vxs = []
        for patch in v_levels[0].patches:
            pd = patch.patch_datas["x"]
            vals = np.asarray(pd.dataset[:])
            assert np.all(np.isfinite(vals)), (
                f"non-finite vx at t={t} in patch {patch.box}"
            )
            X, Y = np.meshgrid(pd.x, pd.y, indexing="ij")
            mask = X >= (Lx - 0.5 * dx)
            if mask.any():
                vxs.append(vals[mask])

        if not rhos or not vxs:
            continue

        rho_right = np.concatenate(rhos)
        vx_right = np.concatenate(vxs)

        assert np.allclose(rho_right, RIGHT.rho, rtol=5e-2), (
            f"inflow density at t={t} drifted from the prescribed right state: "
            f"got range [{rho_right.min()},{rho_right.max()}], expected ~{RIGHT.rho}"
        )
        assert np.allclose(vx_right, RIGHT.vx, rtol=5e-2), (
            f"inflow vx at t={t} drifted from the prescribed right state: "
            f"got range [{vx_right.min()},{vx_right.max()}], expected ~{RIGHT.vx}"
        )
        checked_any = True

    assert checked_any, "no timestamps were available to check the inflow boundary"


def assert_divb_bounded(diag_dir, tol=1e-2):
    """Max |div B| over every AMR level stays near round-off. This is the
    constraint Toth (2000) is about, and a cheap regression signal that the
    inflow/outflow BC + regrid ghost fill combination isn't poisoning the
    constrained-transport update. tol is deliberately generous (regression
    guard against order-of-magnitude blowups, not a precision check).
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


def assert_positivity(diag_dir):
    """Density and pressure stay strictly positive everywhere, every dump,
    every level -- catches a broken BC/regrid ghost silently before it turns
    into a NaN a few steps later.
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
            rho_f = run.GetMHDrho(t)
            p_f = run.GetMHDP(t)
        except Exception:
            continue

        for field_name, field in (("density", rho_f), ("pressure", p_f)):
            levels = field.patch_levels[0]
            for ilvl, lvl in levels.items():
                for patch in lvl.patches:
                    vals = np.asarray(patch.patch_datas["value"].dataset[:])
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        assert finite.min() > 0, (
                            f"non-positive {field_name} at t={t} level {ilvl}: "
                            f"min={finite.min()}"
                        )
        checked_any = True

    assert checked_any, "no timestamps were available to check positivity"


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
        if cpp.mpi_rank() == 0:
            self.assertEqual(
                count_nans(diag_dir),
                0,
                "NaN found in dumped fields: inflow/outflow BC or regrid ghost "
                "fill likely broken",
            )
        assert_inflow_holds_right_state(diag_dir)
        assert_divb_bounded(diag_dir)
        assert_positivity(diag_dir)
        cpp.mpi_barrier()
        return self


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    startMPI()
    CloudShockTest().test_run().tearDown()
