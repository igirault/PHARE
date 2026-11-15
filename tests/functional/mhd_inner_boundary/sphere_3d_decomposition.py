import os

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.hierarchy import hierarchy_from
from pyphare.simulator.simulator import Simulator, startMPI
from tests.simulator import SimulatorTest

ph.NO_GUI()

cells = (40, 40, 40)
dl = (1.0, 1.0, 1.0)
center = (20.3, 20.1, 19.2)
radius = 7.0

time_step = 1.0
time_step_nbr = 25
final_time = time_step * time_step_nbr

diag_dir = "phare_outputs/mhd_inner_boundary_sphere_3d_decomposition"


def config(out_dir, largest_patch_size):
    sim = ph.Simulation(
        smallest_patch_size=8,
        largest_patch_size=largest_patch_size,
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": out_dir, "mode": "overwrite"},
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=False,
        model_options=["MHDModel"],
        inner_boundary={
            "name": "sphere",
            "shape": "sphere",
            "center": center,
            "radius": radius,
            "condition_type": "fixed-state",
            "density": 1.0,
            "pressure": 0.1,
            "inactive_safe_state": {"density": 1.0, "pressure": 0.1},
        },
        domain_boundaries={
            side: {"type": "open"}
            for side in ("xlower", "xupper", "ylower", "yupper", "zlower", "zupper")
        },
    )

    def one(x, y, z):
        return 1.0 + 0.0 * x

    def zero(x, y, z):
        return 0.0 * x

    def vx(x, y, z):
        return 0.1 + 0.0 * x

    def bz(x, y, z):
        return 0.1 + 0.0 * x

    def p(x, y, z):
        return 0.1 + 0.0 * x

    ph.MHDModel(density=one, vx=vx, vy=zero, vz=zero, b1x=zero, b1y=zero, b1z=bz, p=p)

    for quantity in ["rho", "P", "rhoV"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=[final_time])

    return sim


class MHDInnerBoundarySphere3DDecompositionTest(SimulatorTest):
    def tearDown(self):
        super().tearDown()
        ph.global_vars.sim = None

    def test_run(self):
        runs = []
        for largest in (cells[0], 10):
            out_dir = f"{diag_dir}_largest{largest}"
            self.register_diag_dir_for_cleanup(out_dir)
            ph.global_vars.sim = None
            Simulator(config(out_dir, largest)).run().reset()
            runs.append(out_dir)
        ph.global_vars.sim = None

        if cpp.mpi_rank() == 0:
            one, many = runs
            for h5, keys in (
                ("mhd_rho.h5", ["mhdRho"]),
                ("mhd_P.h5", ["mhdP"]),
                ("mhd_rhoV.h5", ["mhdRhoVx", "mhdRhoVy", "mhdRhoVz"]),
            ):
                for key in keys:
                    a = self._level0(os.path.join(one, h5), key)
                    b = self._level0(os.path.join(many, h5), key)
                    scale = np.abs(a).max()
                    np.testing.assert_allclose(
                        b, a, rtol=1e-6, atol=1e-6 * scale, err_msg=key
                    )
        return self

    def _level0(self, h5, key):
        hier = hierarchy_from(h5_filename=h5, times=final_time)
        out = np.full(cells, np.nan)
        for patch in hier.levels(final_time)[0].patches:
            pd = patch.patch_datas[key]
            g = pd.ghosts_nbr
            data = pd.dataset[:][tuple(slice(n, -n) for n in g)]
            lo = patch.box.lower
            out[
                lo[0] : lo[0] + data.shape[0],
                lo[1] : lo[1] + data.shape[1],
                lo[2] : lo[2] + data.shape[2],
            ] = data
        self.assertTrue(np.all(np.isfinite(out)), f"{key} not finite or not covered")
        return out


def main():
    startMPI()
    MHDInnerBoundarySphere3DDecompositionTest().test_run().tearDown()


if __name__ == "__main__":
    main()
