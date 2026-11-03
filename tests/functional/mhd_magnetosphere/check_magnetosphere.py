import glob
import os
import re
import sys

import magnetosphere_2d as run
import numpy as np

from pyphare.pharesee.hierarchy import hierarchy_from

small = 1e-30


def level_arrays(h5, key, time):
    hier = hierarchy_from(h5_filename=h5, times=time)
    arrays = {}
    for ilvl, level in hier.levels(time).items():
        shape = tuple(int(c) * 2**ilvl for c in run.cells)
        global_array = np.full(shape, np.nan)
        for patch in level.patches:
            pd = patch.patch_datas[key]
            g = pd.ghosts_nbr
            interior = pd.dataset[:][tuple(slice(n, -n) for n in g)]
            lo = patch.box.lower
            global_array[
                lo[0] : lo[0] + interior.shape[0], lo[1] : lo[1] + interior.shape[1]
            ] = interior
        arrays[ilvl] = global_array
    return arrays


def radius_map(ilvl):
    dx = run.dl / 2**ilvl
    nx, ny = (int(c) * 2**ilvl for c in run.cells)
    x = (np.arange(nx) + 0.5) * dx[0]
    y = (np.arange(ny) + 0.5) * dx[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.hypot(X - run.center[0], Y - run.center[1]), x, dx


def dumped_times(h5):
    import h5py

    with h5py.File(h5, "r") as f:
        return sorted(float(t) for t in f["t"])


def check(diag_dir):
    failures = []
    path = lambda name: os.path.join(diag_dir, name)
    times = dumped_times(path("mhd_rho.h5"))
    print(
        f"{len(times)} dumps, last at t={times[-1]:.1f} (final_time {run.final_time:.1f})"
    )
    if times[-1] < 0.999 * run.final_time:
        failures.append(f"run stopped at t={times[-1]:.1f} before final_time")

    for t in times:
        for h5, key in (("mhd_rho.h5", "mhdRho"), ("mhd_P.h5", "mhdP")):
            for ilvl, a in level_arrays(path(h5), key, t).items():
                covered = np.isfinite(a)
                if not covered.any():
                    failures.append(f"{key} L{ilvl} t={t:.1f}: empty level")
                r, _, dx = radius_map(ilvl)
                if key == "mhdRho":
                    deep = covered & (r < run.ib_radius - 5 * dx.max())
                    if deep.any() and not np.allclose(a[deep], run.rho_in, rtol=1e-6):
                        failures.append(f"safe-state rho not held L{ilvl} t={t:.1f}")
        rhov = [level_arrays(path("mhd_rhoV.h5"), f"mhdRhoV{c}", t) for c in "xyz"]
        for ilvl in rhov[0]:
            r, _, dx = radius_map(ilvl)
            deep = np.isfinite(rhov[0][ilvl]) & (r < run.ib_radius - 5 * dx.max())
            for c, comp in zip("xyz", rhov):
                if deep.any() and np.abs(comp[ilvl][deep]).max() > 1e-6:
                    failures.append(f"rhoV{c} not zero in body L{ilvl} t={t:.1f}")

    for t in times:
        rho0 = level_arrays(path("mhd_rho.h5"), "mhdRho", t)[0]
        P0 = level_arrays(path("mhd_P.h5"), "mhdP", t)[0]
        if not (np.isfinite(rho0).all() and np.isfinite(P0).all()):
            failures.append(f"non-finite L0 values at t={t:.1f}")

    last = times[-1]
    vx0 = level_arrays(path("mhd_V.h5"), "mhdVx", last)[0]
    P0 = level_arrays(path("mhd_P.h5"), "mhdP", last)[0]
    r0, x, dx = radius_map(0)
    j = int(run.center[1] / dx[1])
    sunward = x < run.center[0] - run.ib_radius
    slowed = np.where(sunward & (vx0[:, j] < 0.5 * run.u_wind))[0]
    if slowed.size == 0:
        failures.append("no bow shock: the wind is not slowed upstream of the body")
    else:
        x_shock = x[slowed.min()]
        standoff = (run.center[0] - x_shock) / run.re
        upstream_ok = np.allclose(
            vx0[: max(slowed.min() - 3, 1), j], run.u_wind, rtol=0.1
        )
        print(
            f"bow shock: Vx < u_wind/2 from {standoff:.2f} R_E upstream of the center"
        )
        print(f"undisturbed wind upstream of the shock: {upstream_ok}")
        if not (run.ib_radius / run.re < standoff < 60) or not upstream_ok:
            failures.append(
                f"bow shock not where expected (standoff {standoff:.2f} R_E)"
            )

    floored = P0 <= 1.0001e-12
    if floored.any():
        print(
            f"L0 floored cells at t={last:.1f}: {int(floored.sum())}, "
            f"r/R in [{r0[floored].min() / run.ib_radius:.2f}, {r0[floored].max() / run.ib_radius:.2f}]"
        )
    else:
        print(f"L0 floored cells at t={last:.1f}: 0")

    hier = hierarchy_from(h5_filename=path("EM_B.h5"), times=last)
    for ilvl, level in hier.levels(last).items():
        dxl = run.dl / 2**ilvl
        worst = {"all": 0.0, "outside body": 0.0}
        for patch in level.patches:
            comps = []
            for c in "xyz":
                pd = patch.patch_datas[f"B{c}"]
                g = pd.ghosts_nbr
                comps.append(pd.dataset[:][tuple(slice(n, -n) for n in g)])
            bx, by, bz = comps
            div = np.diff(bx, axis=0) / dxl[0] + np.diff(by, axis=1) / dxl[1]
            bmag = np.sqrt(
                (0.5 * (bx[1:, :] + bx[:-1, :])) ** 2
                + (0.5 * (by[:, 1:] + by[:, :-1])) ** 2
                + bz[: div.shape[0], : div.shape[1]] ** 2
            )
            ratio = np.abs(div) * dxl.min() / (bmag + small)
            lo = patch.box.lower
            xc = (lo[0] + np.arange(div.shape[0]) + 0.5) * dxl[0]
            yc = (lo[1] + np.arange(div.shape[1]) + 0.5) * dxl[1]
            X, Y = np.meshgrid(xc, yc, indexing="ij")
            outside = (
                np.hypot(X - run.center[0], Y - run.center[1])
                > run.ib_radius + 3 * dxl.max()
            )
            worst["all"] = max(worst["all"], float(np.nanmax(ratio)))
            if outside.any():
                worst["outside body"] = max(
                    worst["outside body"], float(np.nanmax(ratio[outside]))
                )
        print(
            f"L{ilvl}: max |divB| dx/|B| = {worst['all']:.3e} (outside body: {worst['outside body']:.3e})"
        )

    clips = {"density": 0, "pressure": 0}
    for f in glob.glob(os.path.join(".log", "*.out")):
        with open(f) as fh:
            for line in fh:
                m = re.search(
                    r"PHARE_CLIPS level=\d+ density=(\d+) pressure=(\d+)", line
                )
                if m:
                    clips["density"] += int(m.group(1))
                    clips["pressure"] += int(m.group(2))
    print(f"clips (all ranks, all calls): {clips}")

    for f in failures:
        print("FAIL:", f)
    print("RESULT:", "FAIL" if failures else "PASS")
    return not failures


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else run.diag_dir) else 1)
