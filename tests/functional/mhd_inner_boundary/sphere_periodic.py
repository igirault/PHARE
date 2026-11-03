import argparse
import os
import sys

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.hierarchy import hierarchy_from
from pyphare.simulator.simulator import Simulator, startMPI
from tests.simulator import SimulatorTest

ph.NO_GUI()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-field", action="store_true")
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--under-resolved", action="store_true")
    args, _ = parser.parse_known_args(argv)
    args.external_field = args.external_field or args.levels > 1
    return args


ARGS = parse_args(sys.argv[1:])

cells = (64, 64)
dl = (0.15625, 0.15625)
domain_size = (cells[0] * dl[0], cells[1] * dl[1])
center = (domain_size[0] / 2, domain_size[1] / 2)
radius = 3.0 * dl[0] if ARGS.under_resolved else 1.0

body_density = 1.5
body_pressure = 0.8
safe_density = 2.0
safe_pressure = 0.5

time_step = 1e-3
time_step_nbr = 50
final_time = time_step * time_step_nbr
restart_time = time_step * (time_step_nbr // 2)

variant = "_".join(
    name
    for name, on in (
        ("dipole", ARGS.external_field),
        (f"L{ARGS.levels}", ARGS.levels > 1),
        ("restart", ARGS.restart),
        ("under_resolved", ARGS.under_resolved),
    )
    if on
)
diag_dir = "phare_outputs/mhd_inner_boundary_sphere_periodic" + (
    f"_{variant}" if variant else ""
)
restart_dir = diag_dir + "_checkpoints"


def config(restart=None):
    kwargs = {}
    if ARGS.external_field:
        kwargs["external_field"] = {
            "type": "dipole",
            "position": center,
            "moment": (0.0, 2.0),
            "radius": radius - 2 * dl[0],
        }
    if ARGS.levels > 1:
        kwargs["tagging_threshold"] = 0.1
    if restart is not None:
        kwargs["restart_options"] = restart

    reconstruction, limiter, hall = (
        ("WENOZ", "None", True) if ARGS.under_resolved else ("Linear", "VanLeer", False)
    )

    sim = ph.Simulation(
        smallest_patch_size=8,
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=ARGS.levels,
        max_nbr_levels=ARGS.levels,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=hall,
        model_options=["MHDModel"],
        inner_boundary={
            "name": "sphere",
            "shape": "sphere",
            "center": center,
            "radius": radius,
            "condition_type": "fixed-state",
            "density": body_density,
            "pressure": body_pressure,
            "inactive_safe_state": {
                "density": safe_density,
                "pressure": safe_pressure,
            },
        },
        **kwargs,
    )

    def density(x, y):
        return 1.0 + 0.0 * x

    def vx(x, y):
        return 0.5 + 0.0 * x

    def zero(x, y):
        return 0.0 * x

    def bz(x, y):
        return 1e-2 + 0.0 * x

    def p(x, y):
        return 1.0 + 0.0 * x

    ph.MHDModel(
        density=density, vx=vx, vy=zero, vz=zero, b1x=zero, b1y=zero, b1z=bz, p=p
    )

    timestamps = [0.0, restart_time, final_time]
    for quantity in ["rho", "P", "rhoV"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


class MHDInnerBoundarySphereTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super().tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        self.register_diag_dir_for_cleanup(diag_dir)
        if ARGS.restart:
            self.register_diag_dir_for_cleanup(restart_dir)
            Simulator(
                config(
                    {
                        "dir": restart_dir,
                        "mode": "overwrite",
                        "timestamps": [restart_time],
                    }
                )
            ).run().reset()
            ph.global_vars.sim = None
            Simulator(
                config(
                    {
                        "dir": restart_dir,
                        "mode": "overwrite",
                        "restart_time": restart_time,
                    }
                )
            ).run().reset()
        else:
            Simulator(config()).run().reset()

        error = None
        if cpp.mpi_rank() == 0:
            try:
                self._check_hierarchy()
            except (AssertionError, KeyError, OSError, ValueError) as e:
                error = e
        from mpi4py import MPI

        if MPI.COMM_WORLD.bcast(error is not None, root=0):
            raise error if error is not None else RuntimeError("check failed on rank 0")
        return self

    def _interior(self, pd):
        ghosts = pd.ghosts_nbr
        data = pd.dataset[:][tuple(slice(g, -g) for g in ghosts)]
        x = pd.x[ghosts[0] : -ghosts[0]]
        y = pd.y[ghosts[1] : -ghosts[1]]
        X, Y = np.meshgrid(x, y, indexing="ij")
        return data, np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2), pd.dl

    def _check_hierarchy(self):
        rho_hier = hierarchy_from(
            h5_filename=os.path.join(diag_dir, "mhd_rho.h5"), times=final_time
        )
        p_hier = hierarchy_from(
            h5_filename=os.path.join(diag_dir, "mhd_P.h5"), times=final_time
        )
        rhov_hier = hierarchy_from(
            h5_filename=os.path.join(diag_dir, "mhd_rhoV.h5"), times=final_time
        )

        if ARGS.levels > 1:
            self.assertGreater(len(rho_hier.levels(final_time)), 1, "no refined level")

        checked_body = 0
        for ilvl, level in rho_hier.levels(final_time).items():
            for patch in level.patches:
                data, r, dlv = self._interior(patch.patch_datas["mhdRho"])
                self.assertTrue(np.all(np.isfinite(data)), f"rho not finite on L{ilvl}")
                self.assertTrue(np.all(data > 0.0), f"rho <= 0 on L{ilvl}")
                deep = r < radius - 5 * max(dlv)
                checked_body += deep.sum()
                np.testing.assert_allclose(data[deep], safe_density, rtol=1e-12)

        for ilvl, level in p_hier.levels(final_time).items():
            for patch in level.patches:
                data, r, dlv = self._interior(patch.patch_datas["mhdP"])
                self.assertTrue(np.all(np.isfinite(data)), f"P not finite on L{ilvl}")
                fluid = r > radius + 3 * max(dlv)
                self.assertTrue(np.all(data[fluid] > 0.0), f"P <= 0 on L{ilvl}")

        for ilvl, level in rhov_hier.levels(final_time).items():
            for patch in level.patches:
                for c in "xyz":
                    data, r, dlv = self._interior(patch.patch_datas[f"mhdRhoV{c}"])
                    self.assertTrue(
                        np.all(np.isfinite(data)), f"rhoV not finite on L{ilvl}"
                    )
                    deep = r < radius - 5 * max(dlv)
                    np.testing.assert_allclose(data[deep], 0.0, atol=1e-12)

        if not ARGS.under_resolved:
            self.assertGreater(checked_body, 0, "no in-body cells were checked")


def main():
    startMPI()
    MHDInnerBoundarySphereTest().test_run().tearDown()


if __name__ == "__main__":
    main()
