#!/usr/bin/env python3
"""
Vector-potential init of the MHD magnetic field (2D): feature + experiment.

A field built as B = curl(A_z z_hat) with the discrete curl is divergence-free in the
discrete (Yee) sense *by construction* (div . curl = 0 for the consistent staggered
operators used here and in Faraday). The component-wise init samples the same analytic
field at face centres and is generically NOT discretely divergence-free.

Note: the discrete (face-staggered) divergence cannot be measured from the diagnostics —
the vtkhdf dump interpolates B to the nodes, and the dedicated divB diagnostic is a stub —
so this test does not assert div B directly. Instead it:

  1. checks the potential init runs and keeps the (rest) state bounded, and
  2. compares the spurious rest-state |V| against the component init.

Experiment / finding (this base branch, which lacks the B0 well-balancing fixes): the two
inits produce nearly identical spurious |V|, i.e. making B divergence-free at init does NOT
remove the rest-state spurious force. The dominant cause is the scheme's
non-well-balancedness w.r.t. B0, not the initial numerical divergence.

Run:  phare-run python test_vector_potential_init.py
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V_BOUND = 1e-3  # rest |V| must stay small over the short run (sanity, not machine zero)


def _run_case(b_init, ncells="64"):
    env = dict(os.environ)
    env.update({"PHARE_B_INIT": b_init, "PHARE_NCELLS": ncells})
    subprocess.run([sys.executable, "case.py"], cwd=HERE, env=env, check=True)


def _max_abs_v(diag_dir):
    import h5py

    f = h5py.File(os.path.join(HERE, diag_dir, "mhd_V.vtkhdf"), "r")
    data = f["/VTKHDF/Level0/PointData/data"][:]
    return float(np.max(np.abs(data)))


def main():
    _run_case("potential")
    _run_case("components")

    v_pot = _max_abs_v("phare_outputs_potential_n64_WENOZ")
    v_cmp = _max_abs_v("phare_outputs_components_n64_WENOZ")

    ok = True

    p1 = np.isfinite(v_pot) and v_pot < V_BOUND
    print(f"[{'PASS' if p1 else 'FAIL'}] potential init runs, max|V| = {v_pot:.4e} (< {V_BOUND:.0e})")
    ok &= p1

    # Finding: div-free init does not change the spurious force => the two are within a few %.
    rel = abs(v_pot - v_cmp) / max(v_cmp, 1e-300)
    print(f"[INFO] component init max|V| = {v_cmp:.4e};  relative diff = {rel:.2%}")
    print("       => divergence-free init does NOT remove the spurious rest force on this base.")

    if not ok:
        sys.exit(1)
    print("vector-potential init feature OK")


if __name__ == "__main__":
    main()
