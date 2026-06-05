#!/usr/bin/env python3
"""
Analyze the B0 well-balanced test output(s).

For each phare_outputs_* directory, prints max|V| and max|B1| versus time. The exact
solution is rest, so these are pure measures of the spurious forcing produced by the
B0 handling in the flux.

Usage:
    phare-run python analyze.py <dir1> [<dir2> ...]

If exactly two dirs are given AND one looks like the dipole run and the other the
uniform run, also prints the control ratio max|V|_uniform / max|V|_dipole, which
should be ~1e-8 or smaller if the bending is caused by the B0 *gradient* handling.
"""

import sys
import numpy as np
from pyphare.pharesee.run import Run


def _max_abs_over_hierarchy(hier, time):
    """Max |value| across all components/patches of a raw stored hierarchy."""
    m = 0.0
    for patch in hier.level(0, time).patches:
        for pd in patch.patch_datas.values():
            m = max(m, float(np.max(np.abs(pd.dataset[:]))))
    return m


def _series(path):
    r = Run(path)
    times = r.times("mhd_V")
    max_v, max_b1 = [], []
    for t in times:
        tf = float(t)
        vh = r._get_hier_for(tf, "mhd_V")
        max_v.append(_max_abs_over_hierarchy(vh, tf))
        try:
            b1h = r._get_hier_for(tf, "EM_B1")
            max_b1.append(_max_abs_over_hierarchy(b1h, tf))
        except Exception:
            max_b1.append(float("nan"))
    return np.array([float(t) for t in times]), np.array(max_v), np.array(max_b1)


def main(dirs):
    summary = {}
    for path in dirs:
        t, mv, mb1 = _series(path)
        summary[path] = (t, mv, mb1)
        print(f"\n=== {path} ===")
        print(f"{'time':>12}  {'max|V|':>14}  {'max|B1|':>14}")
        for ti, vi, bi in zip(t, mv, mb1):
            print(f"{ti:12.5e}  {vi:14.6e}  {bi:14.6e}")
        print(f"final max|V| = {mv[-1]:.6e}   peak max|V| = {mv.max():.6e}")

    # control ratio if we can identify a dipole and a uniform run
    dip = next((p for p in dirs if "dipole" in p), None)
    uni = next((p for p in dirs if "uniform" in p), None)
    if dip and uni and dip != uni:
        vdip = summary[dip][1].max()
        vuni = summary[uni][1].max()
        ratio = vuni / vdip if vdip > 0 else float("inf")
        print("\n=== control: uniform vs dipole B0 ===")
        print(f"peak max|V| dipole  = {vdip:.6e}")
        print(f"peak max|V| uniform = {vuni:.6e}")
        print(f"ratio uniform/dipole = {ratio:.3e}")
        verdict = "CONFIRMED" if (vdip > 1e3 * max(vuni, 1e-300)) else "INCONCLUSIVE"
        print(
            f"[{verdict}] spurious motion is driven by the B0 gradient "
            f"(dipole nonzero, uniform ~machine-zero)"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
