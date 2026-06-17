import unittest
import numpy as np


import pyphare.pharein.global_vars as global_vars

from pyphare.core import phare_utilities
from pyphare.pharein import simulation
from pyphare.pharein import tagging


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.cells_array = [80, (80, 40), (80, 40, 12)]
        self.dl_array = [0.1, (0.1, 0.2), (0.1, 0.2, 0.3)]
        self.domain_size_array = [100.0, (100.0, 80.0), (100.0, 80.0, 20.0)]
        self.ndim = [1, 2]  # TODO https://github.com/PHAREHUB/PHARE/issues/232
        self.bcs = [
            "periodic",
            ("periodic", "periodic"),
            ("periodic", "periodic", "periodic"),
        ]
        self.layout = "yee"
        self.time_step = 0.001
        self.time_step_nbr = 1000
        self.final_time = 1.0
        global_vars.sim = None

    def test_dl(self):
        for cells, domain_size, dim, bc in zip(
            self.cells_array, self.domain_size_array, self.ndim, self.bcs
        ):
            j = simulation.Simulation(
                time_step_nbr=self.time_step_nbr,
                boundary_types=bc,
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

    def test_boundary_conditions(self):
        j = simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=1.0,
        )

        for d in np.arange(j.ndim):
            self.assertEqual("periodic", j.boundary_types[d])

    def test_assert_boundary_condition(self):
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
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
        )
        self.assertEqual(0.01, s.time_step)

    def test_tagging_unknown_top_level_key_raises(self):
        # a typo in a top-level tagging key must fail rather than silently fall back
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(
                refinement="tagging",
                tagging={"method": "lohner", "quantites": {"rho": 0.4}},
            )

    def test_tagging_without_refinement_raises(self):
        # a fully specified tagging= with refinement left at default must not be ignored
        with self.assertRaises(ValueError):
            tagging.resolve_tagging(
                max_nbr_levels=3,
                tagging={"method": "wavelet", "quantities": {"rho": 0.01}},
            )

    def test_tagging_valid_spec_normalizes(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={
                "method": "lohner",
                "quantities": {"B": 0.1},
                "params": {"reltol": 0.02},
            },
        )
        self.assertIsInstance(tag, tagging.LohnerTagging)
        self.assertEqual(tag.method, "lohner")
        self.assertEqual(tag.quantities, [("B", 0.1)])
        self.assertEqual(tag.reltol, 0.02)
        self.assertEqual(tag.abstol, 1e-30)  # unspecified -> field default

    def test_tagging_wavelet_level_scaling_is_bool(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={"method": "wavelet", "params": {"level_scaling": 0}},
        )
        self.assertIsInstance(tag, tagging.WaveletTagging)
        self.assertIs(tag.level_scaling, False)  # coerced 0 -> bool

    def test_tagging_none_when_not_tagging(self):
        self.assertIsNone(tagging.resolve_tagging())

    def test_re_numpify_does_not_corrupt_tagging_quantities(self):
        # regression test: Tagging.quantities is a list of (str, float) tuples; the general
        # re_numpify_recursive() walk used on deserialization must not hand this to np.array()
        # (which would silently upcast the whole array to strings, turning the float threshold
        # into the string "0.1").
        tag = tagging.LohnerTagging(quantities=[("B", 0.1), ("massDensity", 0.4)])
        simulation.re_numpify_recursive(tag, "quantities")
        self.assertEqual(tag.quantities, [("B", 0.1), ("massDensity", 0.4)])
        self.assertIsInstance(tag.quantities, list)
        self.assertIsInstance(tag.quantities[0], tuple)
        self.assertIsInstance(tag.quantities[0][1], float)

    def test_re_numpify_still_numpifies_homogeneous_numeric_lists(self):
        # the general fix must not break the original purpose of re_numpify_recursive: turning
        # plain numeric lists (as produced by de_numpify_recursive from ndarrays) back into
        # ndarrays, e.g. Simulation.cells / Simulation.dl or a Box's lower/upper bounds.
        owner = simulation.Simulation.__new__(simulation.Simulation)
        object.__setattr__(owner, "cells", [80, 40, 12])
        simulation.re_numpify_recursive(owner, "cells")
        self.assertIsInstance(owner.cells, np.ndarray)
        np.testing.assert_array_equal(owner.cells, np.array([80, 40, 12]))

    def test_tagging_serialize_deserialize_roundtrip_preserves_quantities(self):
        # end-to-end reproduction of the reported bug: a full Simulation
        # serialize()/deserialize() roundtrip (as used for restarts) must preserve the
        # tagging quantities' float thresholds, not coerce them to strings.
        j = simulation.Simulation(
            time_step_nbr=self.time_step_nbr,
            boundary_types=self.bcs[0],
            cells=self.cells_array[0],
            domain_size=self.domain_size_array[0],
            final_time=self.final_time,
            max_nbr_levels=2,
            refinement="tagging",
            tagging={
                "method": "lohner",
                "quantities": {"B": 0.1, "massDensity": 0.4},
                "params": {"reltol": 0.02},
            },
        )
        restored = simulation.deserialize(simulation.serialize(j))
        self.assertEqual(
            restored.tagging.quantities, [("B", 0.1), ("massDensity", 0.4)]
        )
        self.assertIsInstance(restored.tagging.quantities[0][1], float)
        self.assertIsInstance(restored.tagging.quantities[1][1], float)


class FakeDictPopulator:
    """Records add_*() calls made by Tagging.populate_dict(), mirroring the real
    pharein.initialize.general.dict_populator() interface (add_string/add_int/add_double/
    add_bool) without requiring the compiled C++ extension."""

    def __init__(self):
        self.calls = {}

    def _record(self, path, value):
        assert path not in self.calls, f"duplicate path populated: {path}"
        self.calls[path] = value

    def add_string(self, path, value):
        self._record(path, value)

    def add_int(self, path, value):
        self._record(path, value)

    def add_double(self, path, value):
        self._record(path, value)

    def add_bool(self, path, value):
        self._record(path, value)


class TestTaggingPopulateDict(unittest.TestCase):
    """End-to-end test of the tagging={...} -> resolve_tagging() -> Tagging.populate_dict(dp)
    path for the non-default methods (lohner/wavelet), which was previously only exercised
    (for the legacy default-B path) by functional tests, and only against a hand-built
    PHAREDict by the C++ unit tests (bypassing populate_dict() entirely)."""

    def test_lohner_multi_quantity_populate_dict(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={
                "method": "lohner",
                "quantities": {"B": 0.1, "massDensity": 0.4},
                "params": {"reltol": 0.02, "abstol": 1e-20},
            },
        )
        dp = FakeDictPopulator()
        tag.populate_dict(dp)

        base = "simulation/AMR/refinement/tagging/"
        self.assertEqual(dp.calls[base + "method"], "lohner")
        self.assertEqual(dp.calls[base + "nbr_quantities"], 2)
        self.assertEqual(dp.calls[base + "Q0/name"], "B")
        self.assertEqual(dp.calls[base + "Q0/threshold"], 0.1)
        self.assertEqual(dp.calls[base + "Q1/name"], "massDensity")
        self.assertEqual(dp.calls[base + "Q1/threshold"], 0.4)
        self.assertEqual(dp.calls[base + "params/reltol"], 0.02)
        self.assertEqual(dp.calls[base + "params/abstol"], 1e-20)

    def test_wavelet_multi_quantity_populate_dict(self):
        tag = tagging.resolve_tagging(
            refinement="tagging",
            tagging={
                "method": "wavelet",
                "quantities": {"B": 0.05, "chargeDensity": 0.2},
                "params": {"level_scaling": False},
            },
        )
        dp = FakeDictPopulator()
        tag.populate_dict(dp)

        base = "simulation/AMR/refinement/tagging/"
        self.assertEqual(dp.calls[base + "method"], "wavelet")
        self.assertEqual(dp.calls[base + "nbr_quantities"], 2)
        self.assertEqual(dp.calls[base + "Q0/name"], "B")
        self.assertEqual(dp.calls[base + "Q0/threshold"], 0.05)
        self.assertEqual(dp.calls[base + "Q1/name"], "chargeDensity")
        self.assertEqual(dp.calls[base + "Q1/threshold"], 0.2)
        self.assertIs(dp.calls[base + "params/level_scaling"], False)


if __name__ == "__main__":
    unittest.main()
