import unittest
from unittest import mock

import numpy as np

from pyphare import pharein as ph
from pyphare.core import phare_utilities
from pyphare.pharein import global_vars, simulation, tagging
from pyphare.pharein.external_field import DipoleExternalField
from pyphare.pharein.initialize import general, hybrid, mhd
from pyphare.pharein.mhd_model import MHDModel


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.cells_array = [80, (80, 40), (80, 40, 12)]
        self.dl_array = [0.1, (0.1, 0.2), (0.1, 0.2, 0.3)]
        self.domain_size_array = [100.0, (100.0, 80.0), (100.0, 80.0, 20.0)]
        self.ndim = [1, 2]  # TODO https://github.com/PHAREHUB/PHARE/issues/232
        self.layout = "yee"
        self.time_step = 0.001
        self.time_step_nbr = 1000
        self.final_time = 1.0
        global_vars.sim = None

    def test_dl(self):
        for cells, domain_size, dim in zip(
            self.cells_array, self.domain_size_array, self.ndim
        ):
            j = simulation.Simulation(
                time_step_nbr=self.time_step_nbr,
                cells=cells,
                domain_size=domain_size,
                final_time=self.final_time,
            )

            if phare_utilities.none_iterable(domain_size, cells):
                domain_size = phare_utilities.listify(domain_size)
                cells = phare_utilities.listify(cells)

            for d in np.arange(dim):
                self.assertEqual(j.dl[d], domain_size[d] / float(cells[d]))

            global_vars.sim = None

    def test_boundaries_default_periodic(self):
        j = simulation.Simulation(
            time_step_nbr=1000,
            cells=80,
            domain_size=10,
            final_time=1.0,
        )

        for d in np.arange(j.ndim):
            self.assertTrue(j.periodicities[d])

    def test_boundary_types_kwarg_rejected(self):
        with self.assertRaises(ValueError):
            simulation.Simulation(
                time_step_nbr=1000,
                boundary_types="periodic",
                cells=80,
                domain_size=10,
                final_time=1000,
            )

    def test_time_step(self):
        s = simulation.Simulation(
            time_step_nbr=1000,
            cells=80,
            domain_size=10,
            final_time=10,
        )
        self.assertEqual(0.01, s.time_stepper.time_step)

    # ---- physical outer boundary conditions ----------------------------------

    def _mhd_kwargs(self, **overrides):
        kwargs = dict(
            time_step_nbr=1,
            cells=80,
            domain_size=10,
            final_time=1.0,
            model_options=["MHDModel"],
            mhd_timestepper="SSPRK4_5",
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
        )
        kwargs.update(overrides)
        return kwargs

    def test_boundaries_default_none(self):
        # without a domain_boundaries dict every location is periodic and defaults to 'none'
        global_vars.sim = None
        s = simulation.Simulation(**self._mhd_kwargs())
        self.assertEqual([True], s.periodicities)
        for loc in ("xlower", "xupper"):
            self.assertEqual("none", s.domain_boundaries[loc].type)

    def test_physical_direction_requires_both_locations(self):
        # giving only one side of a direction must be rejected
        global_vars.sim = None
        with self.assertRaises(ValueError):
            simulation.Simulation(
                **self._mhd_kwargs(domain_boundaries={"xlower": {"type": "open"}})
            )

    def test_inflow_velocity_scalar_normalized_signed(self):
        # a scalar inflow speed becomes the signed inward-normal component
        global_vars.sim = None
        s = simulation.Simulation(
            **self._mhd_kwargs(
                domain_boundaries={
                    "xlower": {
                        "type": "super-magnetofast-inflow",
                        "velocity": 2.0,
                        "density": 1.0,
                        "pressure": 1.0,
                        "B": [0.5, 1.0, 0.0],
                    },
                    "xupper": {"type": "open"},
                },
            )
        )
        self.assertEqual([False], s.periodicities)
        vx, vy, vz = s.domain_boundaries["xlower"].velocity
        self.assertEqual((2.0, 0.0, 0.0), (vx, vy, vz))  # +x inward at lower

    def test_inflow_scalar_B_rejected(self):
        # the magnetic field of an inflow must be a 3-vector, not a scalar
        global_vars.sim = None
        with self.assertRaises((TypeError, ValueError)):
            simulation.Simulation(
                **self._mhd_kwargs(
                    domain_boundaries={
                        "xlower": {
                            "type": "super-magnetofast-inflow",
                            "velocity": 2.0,
                            "density": 1.0,
                            "pressure": 1.0,
                            "B": 0.5,
                        },
                        "xupper": {"type": "open"},
                    },
                )
            )

    def _inflow_sim(self, velocity=2.0, B=(0.5, 1.0, 0.0), **overrides):
        global_vars.sim = None
        return simulation.Simulation(
            **self._mhd_kwargs(
                domain_boundaries={
                    "xlower": {
                        "type": "super-magnetofast-inflow",
                        "velocity": velocity,
                        "density": 1.0,
                        "pressure": 1.0,
                        "B": list(B),
                    },
                    "xupper": {"type": "open"},
                },
                **overrides,
            )
        )

    def test_inflow_consistent_normal_b_accepted(self):
        s = self._inflow_sim()
        MHDModel(bx=lambda x: 0.5 + 0.1 * np.sin(np.pi * x / 10.0), by=lambda x: 1.0)
        self.assertIs(s, global_vars.sim)
        self.assertIsNotNone(s.model)

    def test_inflow_inconsistent_normal_b_rejected(self):
        self._inflow_sim()
        with self.assertRaises(ValueError):
            MHDModel(bx=lambda x: 0.7 + x * 0, by=lambda x: 1.0)

    def test_inflow_oblique_inconsistent_normal_b_rejected(self):
        self._inflow_sim(velocity=[2.0, 0.5, 0.0])
        with self.assertRaises(ValueError):
            MHDModel(bx=lambda x: 0.7 + x * 0, by=lambda x: 1.0)

    def test_inflow_normal_b_checked_along_whole_face(self):
        self._inflow_sim(cells=(80, 40), domain_size=(10.0, 8.0))
        with self.assertRaises(ValueError):
            MHDModel(bx=lambda x, y: 0.5 + 0.01 * np.sin(2 * np.pi * y / 8.0))

    def test_inflow_normal_b_checked_at_upper_face(self):
        global_vars.sim = None
        simulation.Simulation(
            **self._mhd_kwargs(
                domain_boundaries={
                    "xlower": {"type": "open"},
                    "xupper": {
                        "type": "super-magnetofast-inflow",
                        "velocity": 2.0,
                        "density": 1.0,
                        "pressure": 1.0,
                        "B": [0.5, 1.0, 0.0],
                    },
                },
            )
        )
        with self.assertRaises(ValueError):
            MHDModel(bx=lambda x: np.where(x > 9.0, 0.6, 0.5), by=lambda x: 1.0)

    def tagging_simulation(self, **kwargs):
        return simulation.Simulation(
            time_step_nbr=self.time_step_nbr,
            cells=self.cells_array[0],
            domain_size=self.domain_size_array[0],
            final_time=self.final_time,
            max_nbr_levels=2,
            refinement="tagging",
            **kwargs,
        )

    def test_tagging_unknown_top_level_key_raises(self):
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(
                refinement="tagging",
                tagging={"method": "lohner-per-direction", "quantites": {"rho": 0.4}},
            )

    def test_tagging_without_refinement_raises(self):
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(
                max_nbr_levels=3,
                tagging={"method": "lohner-per-direction", "quantities": {"B": 0.1}},
            )

    def test_tagging_valid_spec_normalizes(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={
                "method": "lohner-per-direction",
                "quantities": {"B": 0.1},
                "params": {"reltol": 0.02},
            },
        )
        self.assertIsInstance(tag, tagging.LohnerPerDirectionTagging)
        self.assertEqual(tag.method, "lohner-per-direction")
        self.assertEqual(tag.quantities, [("B", 0.1)])
        self.assertEqual(tag.reltol, 0.02)
        self.assertEqual(tag.abstol, 1e-8)

    def test_tagging_lohner_default_params(self):
        tag = tagging.resolve_tagging(
            refinement="tagging", tagging={"method": "lohner-per-direction"}
        )
        self.assertEqual(tag.reltol, 0.01)
        self.assertEqual(tag.abstol, 1e-8)

    def test_tagging_lohner_cross_derivatives(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={"method": "lohner-cross-derivatives", "params": {"abstol": 1e-6}},
        )
        self.assertIsInstance(tag, tagging.LohnerCrossDerivativesTagging)
        self.assertEqual(tag.method, "lohner-cross-derivatives")
        self.assertEqual(tag.reltol, 0.01)
        self.assertEqual(tag.abstol, 1e-6)

    def test_tagging_old_lohner_name_rejected(self):
        with self.assertRaises(ValueError) as error:
            tagging.resolve_tagging(refinement="tagging", tagging={"method": "lohner"})
        self.assertIn("lohner-per-direction", str(error.exception))

    def test_tagging_none_when_not_tagging(self):
        self.assertIsNone(tagging.resolve_tagging())

    def test_tagging_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(refinement="tagging", tagging={"method": "wavelet"})

    def test_tagging_unknown_param_raises(self):
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(
                refinement="tagging",
                tagging={"method": "default", "params": {"reltol": 0.1}},
            )

    def test_tagging_not_available_rejected(self):
        for name in ("J", "Ey", "V", "pressure", "massDensity"):
            with self.assertRaises(ValueError) as error:
                tagging.resolve_tagging(
                    refinement="tagging", tagging={"quantities": {name: 0.1}}
                )
            self.assertIn("not available for tagging yet", str(error.exception))

    def test_tagging_duplicate_quantity_rejected(self):
        with self.assertRaises(ValueError) as error:
            tagging.resolve_tagging(
                refinement="tagging", tagging={"quantities": [("B", 0.1), ("B", 0.4)]}
            )
        self.assertIn("given more than once", str(error.exception))

    def test_tagging_fallback_uses_tagging_threshold(self):
        for spec in ({"method": "lohner-per-direction"}, {"quantities": {}}):
            tag = tagging.resolve_tagging(
                refinement="tagging", tagging_threshold=0.3, tagging=spec
            )
            self.assertEqual(tag.quantities, [("B", 0.3)])

    def test_re_numpify_does_not_corrupt_tagging_quantities(self):
        tag = tagging.LohnerPerDirectionTagging(quantities=[("B", 0.1), ("rho", 0.4)])
        simulation.re_numpify_recursive(tag, "quantities")
        self.assertEqual(tag.quantities, [("B", 0.1), ("rho", 0.4)])
        self.assertIsInstance(tag.quantities, list)
        self.assertIsInstance(tag.quantities[0], tuple)
        self.assertIsInstance(tag.quantities[0][1], float)

    def test_re_numpify_still_numpifies_homogeneous_numeric_lists(self):
        owner = simulation.Simulation.__new__(simulation.Simulation)
        object.__setattr__(owner, "cells", [80, 40, 12])
        simulation.re_numpify_recursive(owner, "cells")
        self.assertIsInstance(owner.cells, np.ndarray)
        np.testing.assert_array_equal(owner.cells, np.array([80, 40, 12]))

    def test_tagging_serialize_deserialize_roundtrip_preserves_quantities(self):
        sim = self.tagging_simulation(
            tagging={
                "method": "lohner-per-direction",
                "quantities": {"B": 0.1, "rho": 0.4},
                "params": {"reltol": 0.02},
            },
        )
        restored = simulation.deserialize(simulation.serialize(sim))
        self.assertEqual(restored.tagging.quantities, [("B", 0.1), ("rho", 0.4)])
        self.assertIsInstance(restored.tagging.quantities[0][1], float)
        self.assertIsInstance(restored.tagging.quantities[1][1], float)

    def test_default_path_populates_legacy_keys(self):
        sim = self.tagging_simulation()
        dp = FakeDictPopulator()
        general.populate_amr(dp, sim)

        base = "simulation/AMR/refinement/tagging/"
        tagging_calls = {
            path[len(base) :]: value
            for path, value in dp.calls.items()
            if path.startswith(base)
        }
        self.assertEqual(
            tagging_calls,
            {
                "method": "auto",
                "threshold": 0.1,
                "nbr_quantities": 1,
                "Q0/name": "B",
                "Q0/threshold": 0.1,
            },
        )
        self.assertEqual(sim.tagging.method, "default")

    def model_method(self, module, key):
        dp = FakeDictPopulator()
        with mock.patch.object(module, "DictPopulator", lambda: dp):
            module.populateDict(global_vars.sim)
        return dp.calls[f"simulation/AMR/refinement/tagging/{key}"]

    def test_method_reaches_model_method_keys(self):
        def one(x):
            return 1.0 + 0.0 * x

        for method in (
            "default",
            "lohner-per-direction",
            "lohner-cross-derivatives",
        ):
            global_vars.sim = None
            self.tagging_simulation(tagging={"method": method})
            ph.MaxwellianFluidModel(bx=one, by=one, bz=one, protons={"density": one})
            ph.ElectronModel(closure="isothermal", Te=0.0)
            self.assertEqual(self.model_method(hybrid, "hybrid_method"), method)

            global_vars.sim = None
            self.tagging_simulation(
                tagging={"method": method},
                model_options=["MHDModel"],
                reconstruction="Linear",
                limiter="VanLeer",
                riemann="Rusanov",
                mhd_timestepper="TVDRK2",
            )
            ph.MHDModel(
                density=one, vx=one, vy=one, vz=one, bx=one, by=one, bz=one, p=one
            )
            self.assertEqual(self.model_method(mhd, "mhd_method"), method)

    def _inflow_sim_2d(self, with_external_field):
        sim = self._inflow_sim(cells=(80, 40), domain_size=(10.0, 8.0))
        if with_external_field:
            DipoleExternalField(position=(5.0, 4.0), moment=(0.0, 1.0), radius=1.0)
        return sim

    def test_inflow_normal_b_skipped_for_b1_with_external_field(self):
        s = self._inflow_sim_2d(with_external_field=True)
        MHDModel(b1x=lambda x, y: 0.7 + 0 * x)
        self.assertIsNotNone(s.model)

    def test_inflow_normal_b_checked_for_b1_without_external_field(self):
        self._inflow_sim_2d(with_external_field=False)
        with self.assertRaises(ValueError):
            MHDModel(b1x=lambda x, y: 0.7 + 0 * x)

    def test_inflow_normal_b_checked_for_total_with_external_field(self):
        self._inflow_sim_2d(with_external_field=True)
        with self.assertRaises(ValueError):
            MHDModel(bx=lambda x, y: 0.7 + 0 * x)


class RecordingPopulator:
    def __init__(self):
        self.written = {}

    def add_double(self, path, value):
        self.written[path] = float(value)

    def add_string(self, path, value):
        self.written[path] = str(value)

    def add_array_as_vector(self, path, value):
        self.written[path] = [float(v) for v in value]


class TestInnerBoundaryAndFloors(unittest.TestCase):
    def setUp(self):
        global_vars.sim = None

    def tearDown(self):
        global_vars.sim = None

    @staticmethod
    def body(**overrides):
        ib = {
            "name": "earth",
            "shape": "sphere",
            "center": (1.0, 2.0),
            "radius": 0.5,
            "condition_type": "fixed-state",
            "density": 3.0,
            "pressure": 0.25,
        }
        ib.update(overrides)
        return ib

    def simulation(self, **kwargs):
        return simulation.Simulation(
            time_step=0.001, time_step_nbr=10, cells=(20, 20), dl=(0.1, 0.1), **kwargs
        )

    def test_inner_boundary_fixed_state_is_accepted(self):
        sim = self.simulation(inner_boundary=self.body())
        dp = RecordingPopulator()
        sim.inner_boundary.populate_dict(dp)
        base = "simulation/inner_boundary"
        self.assertEqual(dp.written[f"{base}/condition_type"], "fixed-state")
        self.assertEqual(dp.written[f"{base}/shape"], "sphere")
        self.assertEqual(dp.written[f"{base}/name"], "earth")
        self.assertEqual(dp.written[f"{base}/center"], [1.0, 2.0])
        self.assertEqual(dp.written[f"{base}/radius"], 0.5)
        self.assertEqual(dp.written[f"{base}/density"], 3.0)
        self.assertEqual(dp.written[f"{base}/pressure"], 0.25)

    def test_inner_boundary_fixed_density_is_accepted(self):
        ib = self.body(condition_type="fixed-density")
        del ib["pressure"]
        sim = self.simulation(inner_boundary=ib)
        dp = RecordingPopulator()
        sim.inner_boundary.populate_dict(dp)
        base = "simulation/inner_boundary"
        self.assertEqual(dp.written[f"{base}/condition_type"], "fixed-density")
        self.assertEqual(dp.written[f"{base}/density"], 3.0)
        self.assertNotIn(f"{base}/pressure", dp.written)

    def test_inner_boundary_fixed_density_rejects_pressure(self):
        with self.assertRaisesRegex(ValueError, "pressure"):
            self.simulation(inner_boundary=self.body(condition_type="fixed-density"))

    def test_inner_boundary_fixed_density_requires_density(self):
        ib = self.body(condition_type="fixed-density")
        del ib["pressure"]
        del ib["density"]
        with self.assertRaisesRegex(ValueError, "density"):
            self.simulation(inner_boundary=ib)

    def test_no_inner_boundary_by_default(self):
        self.assertIsNone(self.simulation().inner_boundary)

    def test_inner_boundary_rejects_plane(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.simulation(inner_boundary=self.body(shape="plane"))

    def test_inner_boundary_rejects_unknown_condition(self):
        with self.assertRaisesRegex(ValueError, "condition_type"):
            self.simulation(inner_boundary=self.body(condition_type="reflective"))

    def test_inner_boundary_requires_density_and_pressure(self):
        for key in ("density", "pressure"):
            ib = self.body()
            del ib[key]
            with self.assertRaisesRegex(ValueError, key):
                self.simulation(inner_boundary=ib)
            global_vars.sim = None

    def test_inner_boundary_rejects_non_positive_state(self):
        with self.assertRaisesRegex(ValueError, "density"):
            self.simulation(inner_boundary=self.body(density=0.0))

    def test_inner_boundary_center_must_match_dimension(self):
        with self.assertRaisesRegex(ValueError, "center"):
            self.simulation(inner_boundary=self.body(center=(1.0, 2.0, 3.0)))

    def test_inner_boundary_rejects_unknown_safe_state_key(self):
        with self.assertRaisesRegex(ValueError, "inactive_safe_state"):
            self.simulation(inner_boundary=self.body(inactive_safe_state={"B": 1.0}))

    def test_inner_boundary_safe_state_b1_broadcast(self):
        sim = self.simulation(
            inner_boundary=self.body(
                inactive_safe_state={
                    "B1": 2.0,
                    "density": 4.0,
                    "velocity": (1.0, 0.0, 0.5),
                }
            )
        )
        dp = RecordingPopulator()
        sim.inner_boundary.populate_dict(dp)
        base = "simulation/inner_boundary/inactive_safe_state"
        for c in "xyz":
            self.assertEqual(dp.written[f"{base}/B1/{c}"], 2.0)
        self.assertEqual(dp.written[f"{base}/density"], 4.0)
        self.assertEqual(dp.written[f"{base}/pressure"], 1.0)
        self.assertEqual(dp.written[f"{base}/velocity/x"], 1.0)
        self.assertEqual(dp.written[f"{base}/velocity/z"], 0.5)

    def test_floors_default_to_zero(self):
        sim = self.simulation()
        self.assertEqual(sim.pressure_floor, 0.0)
        self.assertEqual(sim.density_floor, 0.0)

    def test_floors_are_stored(self):
        sim = self.simulation(pressure_floor=1e-6, density_floor=1e-2)
        self.assertEqual(sim.pressure_floor, 1e-6)
        self.assertEqual(sim.density_floor, 1e-2)

    def test_negative_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pressure_floor must be"):
            self.simulation(pressure_floor=-1.0)


class FakeDictPopulator:
    def __init__(self):
        self.calls = {}

    def _record(self, path, value):
        assert path not in self.calls, f"duplicate path populated: {path}"
        self.calls[path] = value

    def __getattr__(self, name):
        if name in ("add_string", "add_int", "add_double", "add_bool"):
            return self._record
        if name.startswith("add_"):
            return lambda *args: None
        raise AttributeError(name)


class TestTaggingPopulateDict(unittest.TestCase):
    def test_lohner_multi_quantity_populate_dict(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={
                "method": "lohner-per-direction",
                "quantities": {"B": 0.1, "rho": 0.4},
                "params": {"reltol": 0.02, "abstol": 1e-20},
            },
        )
        dp = FakeDictPopulator()
        tag.populate_dict(dp)

        base = "simulation/AMR/refinement/tagging/"
        self.assertEqual(dp.calls[base + "nbr_quantities"], 2)
        self.assertEqual(dp.calls[base + "Q0/name"], "B")
        self.assertEqual(dp.calls[base + "Q0/threshold"], 0.1)
        self.assertEqual(dp.calls[base + "Q1/name"], "rho")
        self.assertEqual(dp.calls[base + "Q1/threshold"], 0.4)
        self.assertEqual(dp.calls[base + "params/reltol"], 0.02)
        self.assertEqual(dp.calls[base + "params/abstol"], 1e-20)
        self.assertNotIn(base + "method", dp.calls)


if __name__ == "__main__":
    unittest.main()
