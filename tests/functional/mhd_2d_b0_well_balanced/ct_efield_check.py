#!/usr/bin/env python3
"""
Probe the constrained-transport EMF in FLOW (V != 0), where the rest well-balanced
test is blind (it only exercises V = 0, so every V-dependent EMF term is untested).

Ideal Ohm's law: E = -V x B. Two regimes:

  uniform B0 + uniform V0 + B1 = 0
      -> exact steady state, E must equal the uniform constant -V0 x B0 EVERYWHERE,
         to machine precision. No gradients => no upwind diffusion, no truncation.
         A clean pass/fail on the CT motional term + coefficient wiring.

  dipole  B0 + uniform V0 + B1 = 0
      -> not steady, but at/near t0  E ~ -V0 x B0(x). Residual R = E + V x B
         (using dumped E, total B, V) localizes any gradient-handling error.

Electric-field ghost cells are not necessarily filled, so all metrics are reported
both over the full patch and over the strict interior (border stripped).

Usage:
  phare-run python ct_efield_check.py <dir> --v0 <V0> [--b0 bx by bz]   # uniform
  phare-run python ct_efield_check.py <dir> --v0 <V0> --dipole          # dipole residual
"""

import argparse
import numpy as np
from pyphare.pharesee.run import Run


def comp_of(name):
    n = name.lower()
    for ax in ("x", "y", "z"):
        if n.endswith(ax) or f"_{ax}" in n or f"e{ax}" in n:
            return ax
    return None


def stat(arr, nstrip):
    a = np.abs(arr)
    full = float(a.max()) if a.size else 0.0
    if arr.ndim >= 1 and all(s > 2 * nstrip for s in arr.shape):
        sl = tuple(slice(nstrip, -nstrip) for _ in range(arr.ndim))
        inner = float(np.abs(arr[sl]).max())
    else:
        inner = full
    return full, inner


def uniform_check(path, V0, B0, nstrip):
    # E = -V x B ; V=(V0,0,0) => E=(0, -V0*Bz... ) ; general cross product:
    Vx = V0
    Bx, By, Bz = B0
    # V x B with V=(Vx,0,0): (0*Bz-0*By, 0*Bx-Vx*Bz, Vx*By-0*Bx) = (0, -Vx*Bz, Vx*By)
    E_expected = {"x": 0.0, "y": Vx * Bz, "z": -Vx * By}  # E = -(VxB)
    r = Run(path)
    t = float(r.times("EM_E")[-1])
    h = r._get_hier_for(t, "EM_E")
    print(f"[uniform] dir={path}  t={t:.4e}  V0={V0}  B0={B0}")
    print(f"  expected E = (Ex,Ey,Ez) = ({E_expected['x']}, {E_expected['y']}, {E_expected['z']})")
    worst_inner = 0.0
    for patch in h.level(0, t).patches:
        for name, pd in patch.patch_datas.items():
            ax = comp_of(name)
            if ax is None:
                continue
            d = np.asarray(pd.dataset[:])
            dev = d - E_expected[ax]
            full, inner = stat(dev, nstrip)
            worst_inner = max(worst_inner, inner)
            print(f"  {name:>8}: shape={tuple(d.shape)}  max|E-expected| full={full:.3e}"
                  f"  interior(strip {nstrip})={inner:.3e}")
    verdict = "PASS (CT motional exact in uniform flow)" if worst_inner < 1e-11 \
        else "FAIL (CT motional/coefficient bug in uniform flow)"
    print(f"  => worst interior deviation = {worst_inner:.3e}  -> {verdict}")
    return worst_inner


def dipole_residual(path, V0, nstrip):
    # R = E + V x B, all interpolated to primal nodes (all_primal=True).
    r = Run(path)
    t = float(r.times("EM_E")[-1])
    E = r.GetE(t, all_primal=True)
    B = r.GetB(t, all_primal=True)
    V = r.GetMHDV(t, all_primal=True) if hasattr(r, "GetMHDV") else r._get_hier_for(t, "mhd_V")
    print(f"[dipole] dir={path}  t={t:.4e}  V0={V0}  (residual R = E + VxB)")

    def comp(hier, qty, ax):
        for patch in hier.level(0, t).patches:
            for name, pd in patch.patch_datas.items():
                if comp_of(name) == ax:
                    yield name, np.asarray(pd.dataset[:])

    # crude single-patch assumption for the probe; report per-component residual
    def grab(hier, ax):
        out = None
        for _, arr in comp(hier, None, ax):
            out = arr
        return out

    Ex, Ey, Ez = (grab(E, a) for a in "xyz")
    Bx, By, Bz = (grab(B, a) for a in "xyz")
    Vx, Vy, Vz = (grab(V, a) for a in "xyz")
    if any(a is None for a in (Ex, Bx, Vx)):
        print("  could not align E/B/V components on a common grid; inspect raw dumps")
        return
    # broadcast to common shape if needed
    shp = Ez.shape
    def fit(a):
        return a if a.shape == shp else a[tuple(slice(0, s) for s in shp)]
    Bx, By, Bz, Vx, Vy, Vz = map(fit, (Bx, By, Bz, Vx, Vy, Vz))
    VxB = (Vy * Bz - Vz * By, Vz * Bx - Vx * Bz, Vx * By - Vy * Bx)
    Rz = Ez + VxB[2]
    Rx = fit(Ex) + VxB[0]
    Ry = fit(Ey) + VxB[1]
    for nm, R in (("Rx", Rx), ("Ry", Ry), ("Rz", Rz)):
        full, inner = stat(R, nstrip)
        print(f"  {nm}: max|R| full={full:.3e}  interior(strip {nstrip})={inner:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--v0", type=float, required=True)
    ap.add_argument("--b0", type=float, nargs=3, default=[0.0, 0.3, 0.0])
    ap.add_argument("--dipole", action="store_true")
    ap.add_argument("--strip", type=int, default=6)
    a = ap.parse_args()
    if a.dipole:
        dipole_residual(a.dir, a.v0, a.strip)
    else:
        uniform_check(a.dir, a.v0, a.b0, a.strip)
