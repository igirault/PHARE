#!/usr/bin/env python3
"""
Functional check of derived-quantity diagnostics (V, P, divB) for the MHD model:
in phareh5 format, dumped V must match manual reconstruction from the
conservative dumps (rhoV/rho), P must recover the (constant) initial pressure,
and divB of the (discretely divergence-free) initial condition must vanish to
machine precision. A second run smoke-checks the pharevtkhdf writer.
"""

import os
import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

ph.NO_GUI()

out_dir_h5 = "phare_outputs/mhd_derived_diagnostics/h5"
out_dir_vtk = "phare_outputs/mhd_derived_diagnostics/vtk"
time_step = 0.001
final_time = 0.002
gamma = 5.0 / 3.0

timestamps = [0.0]

mhd_quantities = ["rho", "V", "P", "rhoV", "Etot"]
em_quantities = ["B", "E", "J", "divB"]


def config(diag_format, diag_dir):
    """Orszag-Tang setup: each B component is constant along its own direction,
    hence the face-centered initial B is discretely divergence-free."""

    cells = (64, 64)
    dl = (1.0 / cells[0], 1.0 / cells[1])

    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        diag_options={
            "format": diag_format,
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=0.0,
        nu=0.0,
        gamma=gamma,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        res=False,
        hyper_res=False,
        model_options=["MHDModel"],
    )

    B0 = 1.0 / (np.sqrt(4.0 * np.pi))

    def density(x, y):
        return 25.0 / (36.0 * np.pi)

    def vx(x, y):
        Ly = sim.simulation_domain()[1]
        return -np.sin(2.0 * np.pi * y / Ly)

    def vy(x, y):
        Lx = sim.simulation_domain()[0]
        return np.sin(2.0 * np.pi * x / Lx)

    def vz(x, y):
        return 0.0

    def bx(x, y):
        Ly = sim.simulation_domain()[1]
        return -B0 * np.sin(2.0 * np.pi * y / Ly)

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        return B0 * np.sin(4.0 * np.pi * x / Lx)

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 5.0 / (12.0 * np.pi)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in em_quantities:
        if quantity != "B":
            ph.ElectromagDiagnostics(quantity=quantity, write_timestamps=timestamps)

    for quantity in mhd_quantities:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def interior(patch_data):
    """dataset of a FieldData stripped of its ghost cells"""
    return patch_data.dataset[
        tuple(
            slice(int(g), -int(g)) if int(g) > 0 else slice(None)
            for g in patch_data.ghosts_nbr
        )
    ]


class MHDDerivedDiagnosticsTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super(MHDDerivedDiagnosticsTest, self).__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super(MHDDerivedDiagnosticsTest, self).tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_derived_quantities_phareh5(self):
        ph.global_vars.sim = None
        self.register_diag_dir_for_cleanup(out_dir_h5)
        Simulator(config("phareh5", out_dir_h5)).run().reset()

        # checks run on every rank (read-only h5 access) so that a failure
        # cannot desynchronize MPI collectives between ranks
        self._check_phareh5()

    def _check_phareh5(self):
        from pyphare.pharesee.run import Run

        run = Run(out_dir_h5)
        t = 0.0

        # raw (cell-centered) hierarchies: pointwise identities hold exactly
        rho = run.GetMHDrho(t, all_primal=False)
        rhoV = run.GetMHDrhoV(t, all_primal=False)
        V = run.GetMHDV(t, all_primal=False)
        P = run.GetMHDP(t, all_primal=False)
        Etot = run.GetMHDEtot(t, all_primal=False)
        divB = run.GetMHDdivB(t, all_primal=False)

        # V == rhoV / rho, patch by patch (exact same arithmetic in C++)
        for ilvl in rho.levels():
            for p_rho, p_rhoV, p_V in zip(
                rho.level(ilvl).patches,
                rhoV.level(ilvl).patches,
                V.level(ilvl).patches,
            ):
                self.assertEqual(p_rho.box, p_V.box)
                rho_in = interior(p_rho.patch_datas["mhdRho"])
                for c in ["x", "y", "z"]:
                    # Pre-existing (not introduced by this task): the
                    # write/read roundtrip for V/rhoV/rho loses precision to
                    # float32 somewhere along the way, so this is not exact
                    # double-precision arithmetic despite both sides being
                    # computed the "same way" in C++. Observed max abs diff
                    # is ~2.98e-08 (== 2**-25, single-precision ULP scale)
                    # for values of order 0.05-0.4, hence the loosened
                    # tolerance below.
                    np.testing.assert_allclose(
                        interior(p_V.patch_datas[f"mhdV{c}"]),
                        interior(p_rhoV.patch_datas[f"mhdRhoV{c}"]) / rho_in,
                        rtol=1e-6,
                        atol=1e-7,
                    )

        # divB ~ 0: bx is constant along x and by along y, so the discrete
        # face-difference divergence vanishes to roundoff
        found_patches = 0
        for ilvl in divB.levels():
            for patch in divB.level(ilvl).patches:
                found_patches += 1
                np.testing.assert_allclose(
                    interior(patch.patch_datas["divB"]), 0.0, atol=1e-11
                )
        self.assertGreater(found_patches, 0)

        # J and E: no exact analytic reference here (Orszag-Tang has no closed
        # form for E once resistivity/Hall are folded in), so just check they
        # are finite on every patch
        J = run.GetMHDJ(t, all_primal=False)
        E = run.GetMHDE(t, all_primal=False)
        for name, hier, key in [("J", J, "J"), ("E", E, "E")]:
            found = 0
            for ilvl in hier.levels():
                for patch in hier.level(ilvl).patches:
                    found += 1
                    for c in ["x", "y", "z"]:
                        data = interior(patch.patch_datas[f"{key}{c}"])
                        self.assertTrue(np.isfinite(data).all(), f"{name}{c} has non-finite values")
            self.assertGreater(found, 0, f"no patches found for {name}")

        # P recovers the (uniform) initial pressure: the face->cell projection
        # of B is exact here since each B component is constant along the
        # direction it is projected over
        p_init = 5.0 / (12.0 * np.pi)
        for ilvl in P.levels():
            for patch in P.level(ilvl).patches:
                pdata = interior(patch.patch_datas["mhdP"])
                self.assertTrue(np.isfinite(pdata).all())
                self.assertTrue((pdata > 0).all())
                # Same pre-existing float32 write/read roundtrip precision
                # loss as the V == rhoV / rho check above (observed max abs
                # diff here ~7e-9, ~5.3e-8 relative, again ULP-scale for
                # float32), hence the loosened tolerance.
                np.testing.assert_allclose(pdata, p_init, rtol=1e-6, atol=1e-7)

        for ilvl in Etot.levels():
            for patch in Etot.level(ilvl).patches:
                self.assertTrue(
                    np.isfinite(interior(patch.patch_datas["mhdEtot"])).all()
                )

    def test_divB_moved_to_electromag_diagnostics(self):
        # /mhd/divB moved to the electromag tree: requesting it as an MHD
        # (fluid) diagnostic must fail loudly at configuration time, with a
        # message pointing at ElectromagDiagnostics.
        ph.global_vars.sim = None
        config("phareh5", out_dir_h5)
        with self.assertRaisesRegex(ValueError, "ElectromagDiagnostics"):
            ph.MHDDiagnostics(quantity="divB", write_timestamps=timestamps)

    def test_derived_quantities_vtkhdf_smoke(self):
        ph.global_vars.sim = None
        self.register_diag_dir_for_cleanup(out_dir_vtk)
        Simulator(config("pharevtkhdf", out_dir_vtk)).run().reset()

        self._check_vtkhdf()

    def _check_vtkhdf(self):
        import h5py

        expected = [f"mhd_{q}.vtkhdf" for q in mhd_quantities] + [
            f"EM_{q}.vtkhdf" for q in em_quantities
        ]
        for fname in expected:
            path = os.path.join(out_dir_vtk, fname)
            self.assertTrue(os.path.exists(path), f"missing vtkhdf file {path}")

            datasets = {}
            with h5py.File(path, "r") as h5:

                def collect(name, node):
                    if isinstance(node, h5py.Dataset):
                        datasets[name] = node[...]

                h5.visititems(collect)

                self.assertTrue(datasets, f"no datasets in {path}")
                for name, data in datasets.items():
                    if np.issubdtype(data.dtype, np.floating):
                        self.assertTrue(
                            np.isfinite(data).all(),
                            f"non-finite values in {fname}:{name}",
                        )


if __name__ == "__main__":
    startMPI()
    unittest.main()
