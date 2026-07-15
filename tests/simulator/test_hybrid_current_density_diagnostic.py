#!/usr/bin/env python3
"""
Functional check that the Hybrid model's EM_J diagnostic (current density,
recomputed via core::Ampere from B) writes finite values end-to-end.
"""

import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

ph.NO_GUI()

out_dir = "phare_outputs/hybrid_current_density_diagnostic"
timestamps = [0.0]


def config():
    cells = 40
    dl = 0.2

    sim = ph.Simulation(
        time_step=0.001,
        final_time=0.002,
        cells=cells,
        dl=dl,
        boundary_types="periodic",
        diag_options={
            "format": "phareh5",
            "options": {"dir": out_dir, "mode": "overwrite"},
        },
    )

    def density(x):
        return 1.0

    def by(x):
        return 0.1 * np.sin(2 * np.pi * x / sim.simulation_domain()[0])

    def bz(x):
        return 0.1 * np.cos(2 * np.pi * x / sim.simulation_domain()[0])

    def bx(x):
        return 1.0

    def v(x):
        return 0.0

    def vth(x):
        return 0.01

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "mass": 1,
            "charge": 1,
            "density": density,
            "vbulkx": v,
            "vbulky": v,
            "vbulkz": v,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": 100,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="J", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="divB", write_timestamps=timestamps)

    return sim


def interior(patch_data):
    return patch_data.dataset[
        tuple(
            slice(int(g), -int(g)) if int(g) > 0 else slice(None)
            for g in patch_data.ghosts_nbr
        )
    ]


class HybridCurrentDensityDiagnosticTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super(HybridCurrentDensityDiagnosticTest, self).__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super(HybridCurrentDensityDiagnosticTest, self).tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_em_j_is_finite(self):
        ph.global_vars.sim = None
        self.register_diag_dir_for_cleanup(out_dir)
        Simulator(config()).run().reset()

        from pyphare.pharesee.run import Run

        run = Run(out_dir)
        hier = run._get_hierarchy(0.0, "EM_J.h5")

        found_patches = 0
        for ilvl in hier.levels():
            for patch in hier.level(ilvl).patches:
                found_patches += 1
                for c in ["x", "y", "z"]:
                    data = interior(patch.patch_datas[f"J{c}"])
                    self.assertTrue(np.isfinite(data).all(), f"J{c} has non-finite values")
        self.assertGreater(found_patches, 0)

    def test_em_divb_is_small(self):
        # divB is now a real Hybrid electromag diagnostic (Yee div of the
        # face-centered B). For the smooth, divergence-free init field it must
        # be ~0 to discrete precision everywhere in the interior.
        ph.global_vars.sim = None
        self.register_diag_dir_for_cleanup(out_dir)
        Simulator(config()).run().reset()

        from pyphare.pharesee.run import Run

        run = Run(out_dir)
        hier = run._get_hierarchy(0.0, "EM_divB.h5")

        found_patches = 0
        for ilvl in hier.levels():
            for patch in hier.level(ilvl).patches:
                found_patches += 1
                data = interior(patch.patch_datas["divB"])
                self.assertTrue(np.isfinite(data).all(), "divB has non-finite values")
                self.assertTrue(np.abs(data).max() < 1e-10, "divB is not ~0")
        self.assertGreater(found_patches, 0)


if __name__ == "__main__":
    startMPI()
    unittest.main()
