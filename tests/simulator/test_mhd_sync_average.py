import os
import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.hierarchy import hierarchy_from
from pyphare.simulator.simulator import Simulator, startMPI
from tests.simulator import SimulatorTest

ph.NO_GUI()

cells = (32, 32)
dl = (0.1, 0.1)
center = (1.6, 1.6)
radius = 0.35
time_step = 1e-3
time_step_nbr = 10
timestamps = [time_step * i for i in range(time_step_nbr + 1)]
pressure_floor = 1e-3
refinement_box = ((8, 8), (23, 23))


def config(diag_dir, with_body):
    kwargs = {}
    if with_body:
        kwargs["inner_boundary"] = {
            "name": "body",
            "shape": "sphere",
            "center": center,
            "radius": radius,
            "condition_type": "fixed-state",
            "density": 2.0,
            "pressure": 0.5,
        }

    sim = ph.Simulation(
        smallest_patch_size=8,
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=cells,
        dl=dl,
        refinement="boxes",
        refinement_boxes={"L0": {"B0": refinement_box}},
        max_mhd_level=2,
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
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
        pressure_floor=pressure_floor,
        **kwargs,
    )

    def blob(x, y):
        return ((x - 1.2) ** 2 + (y - 1.2) ** 2) < 0.15**2

    def density(x, y):
        return 1.0 + 0.2 * np.sin(2 * np.pi * x / 3.2)

    def vx(x, y):
        return 0.5 + 0.0 * x

    def vy(x, y):
        return 0.2 + 0.0 * x

    def vz(x, y):
        return 0.0 * x

    def bx(x, y):
        return 0.1 + 0.0 * x

    def by(x, y):
        return 0.2 + 0.0 * x

    def bz(x, y):
        return 0.3 + 0.0 * x

    def p(x, y):
        return np.where(blob(x, y), 0.5 * pressure_floor, 1.0)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    for quantity in ["rho", "rhoV", "Etot", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def level_arrays(h5, key, time):
    hier = hierarchy_from(h5_filename=h5, times=time)
    arrays = {"fp32": False}
    for ilvl, level in hier.levels(time).items():
        shape = tuple(c * 2**ilvl for c in cells)
        global_array = np.full(shape, np.nan)
        for patch in level.patches:
            pd = patch.patch_datas[key]
            g = pd.ghosts_nbr
            interior = pd.dataset[:][tuple(slice(n, -n) for n in g)]
            arrays["fp32"] = arrays["fp32"] or pd.dataset.dtype == np.float32
            lo = patch.box.lower
            global_array[
                lo[0] : lo[0] + interior.shape[0], lo[1] : lo[1] + interior.shape[1]
            ] = interior
        arrays[ilvl] = global_array
    return arrays


def fine_mean(fine):
    return 0.25 * (
        fine[0::2, 0::2] + fine[1::2, 0::2] + fine[0::2, 1::2] + fine[1::2, 1::2]
    )


class MHDSyncAverageTest(SimulatorTest):
    def tearDown(self):
        super().tearDown()
        ph.global_vars.sim = None

    def _run_and_check(self, with_body):
        diag_dir = f"phare_outputs/mhd_sync_average_{'body' if with_body else 'free'}"
        self.register_diag_dir_for_cleanup(diag_dir)
        Simulator(config(diag_dir, with_body)).run().reset()
        ph.global_vars.sim = None

        error = None
        if cpp.mpi_rank() == 0:
            try:
                self._check(diag_dir, with_body)
            except (AssertionError, KeyError, OSError, ValueError) as e:
                error = e

        from mpi4py import MPI

        if MPI.COMM_WORLD.bcast(error is not None, root=0):
            raise error if error is not None else RuntimeError("check failed on rank 0")

    def _check(self, diag_dir, with_body):
        node_x = np.arange(cells[0] + 1) * dl[0]
        node_y = np.arange(cells[1] + 1) * dl[1]
        NX, NY = np.meshgrid(node_x, node_y, indexing="ij")
        phi = np.sqrt((NX - center[0]) ** 2 + (NY - center[1]) ** 2) - radius
        corners = (phi[:-1, :-1], phi[1:, :-1], phi[:-1, 1:], phi[1:, 1:])
        phi_min = np.minimum.reduce(corners)
        phi_max = np.maximum.reduce(corners)
        evolved = phi_max > 0
        cut = evolved & (phi_min <= 0)

        compared = 0
        compared_cut = 0
        for time in timestamps[1:]:
            for qty, keys in (
                ("rho", ["mhdRho"]),
                ("rhoV", ["mhdRhoVx", "mhdRhoVy", "mhdRhoVz"]),
                ("Etot", ["mhdEtot"]),
            ):
                for key in keys:
                    levels = level_arrays(
                        os.path.join(diag_dir, f"mhd_{qty}.h5"), key, time
                    )
                    self.assertEqual(len(levels), 3)
                    rtol = 1e-6 if levels["fp32"] else 1e-10
                    coarse, mean = levels[0], fine_mean(levels[1])
                    covered = np.isfinite(mean)
                    if with_body:
                        covered &= evolved
                    self.assertTrue(np.all(np.isfinite(coarse)), f"{key} not finite")
                    np.testing.assert_allclose(
                        coarse[covered],
                        mean[covered],
                        rtol=rtol,
                        atol=rtol * np.abs(mean[covered]).max(),
                        err_msg=f"{key} t={time}",
                    )
                    compared += covered.sum()
                    if with_body:
                        compared_cut += (covered & cut).sum()

            P = level_arrays(os.path.join(diag_dir, "mhd_P.h5"), "mhdP", time)[0]
            self.assertTrue(np.all(np.isfinite(P)))
            if not with_body:
                self.assertGreaterEqual(
                    P.min(), pressure_floor * (1 - 1e-6), f"t={time}"
                )

        self.assertGreater(compared, 0)
        if with_body:
            self.assertGreater(compared_cut, 0)

    def test_covered_coarse_cells_are_fine_averages(self):
        self._run_and_check(with_body=False)

    def test_covered_coarse_cells_are_fine_averages_around_a_body(self):
        self._run_and_check(with_body=True)


if __name__ == "__main__":
    startMPI()
    unittest.main()
