import unittest

from pyphare.pharein import boundary


class TestBoundaryStructural(unittest.TestCase):
    def test_default_all_none(self):
        resolved = boundary.resolve_boundary_conditions(
            2, boundary_types=("periodic", "periodic")
        )
        for loc in ("xlower", "xupper", "ylower", "yupper"):
            self.assertIsInstance(resolved[loc], boundary.NoneBC)
            self.assertEqual("none", resolved[loc].type)

    def test_physical_boundary_requires_type(self):
        with self.assertRaises(KeyError):
            boundary.resolve_boundary_conditions(
                2,
                boundary_types=("physical", "periodic"),
                model_options=["MHDModel"],
            )

    def test_physical_only_supported_by_mhd_model(self):
        with self.assertRaises(ValueError):
            boundary.resolve_boundary_conditions(
                2,
                boundary_types=("physical", "periodic"),
                boundary_conditions={"xlower": {"type": "open"}, "xupper": {"type": "open"}},
                model_options=["HybridModel"],
            )

    def test_periodic_location_rejects_non_none_type(self):
        with self.assertRaises(ValueError):
            boundary.resolve_boundary_conditions(
                2,
                boundary_types=("physical", "periodic"),
                boundary_conditions={
                    "xlower": {"type": "open"},
                    "xupper": {"type": "open"},
                    "ylower": {"type": "open"},
                },
                model_options=["MHDModel"],
            )

    def test_unknown_location_rejected(self):
        with self.assertRaises(ValueError):
            boundary.resolve_boundary_conditions(
                2,
                boundary_types=("physical", "periodic"),
                boundary_conditions={"not_a_location": {"type": "open"}},
                model_options=["MHDModel"],
            )

    def test_open_and_reflective_resolve(self):
        resolved = boundary.resolve_boundary_conditions(
            2,
            boundary_types=("physical", "periodic"),
            boundary_conditions={"xlower": {"type": "open"}, "xupper": {"type": "reflective"}},
            model_options=["MHDModel"],
        )
        self.assertIsInstance(resolved["xlower"], boundary.OpenBC)
        self.assertIsInstance(resolved["xupper"], boundary.ReflectiveBC)


if __name__ == "__main__":
    unittest.main()
