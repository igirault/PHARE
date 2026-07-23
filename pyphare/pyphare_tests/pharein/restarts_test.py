import unittest

from pyphare.pharein import boundary
from pyphare.pharein.restarts import _boundary_conditions_signature


class TestBoundaryConditionsSignature(unittest.TestCase):
    def test_identical_dataclass_instances_compare_equal(self):
        a = {"xlower": boundary.OpenBC(), "xupper": boundary.ReflectiveBC()}
        b = {"xlower": boundary.OpenBC(), "xupper": boundary.ReflectiveBC()}
        self.assertEqual(
            _boundary_conditions_signature(a), _boundary_conditions_signature(b)
        )

    def test_different_field_values_compare_unequal(self):
        a = {"xlower": boundary.FixedPressureOutflowBC(pressure=1.0)}
        b = {"xlower": boundary.FixedPressureOutflowBC(pressure=2.0)}
        self.assertNotEqual(
            _boundary_conditions_signature(a), _boundary_conditions_signature(b)
        )

    def test_unchanged_callable_source_compares_equal(self):
        def rho(x, t):
            return 1.0

        a = {
            "xlower": boundary.FreePressureInflowBC(
                density=rho, velocity=1.0, B=[0.0, 0.0, 0.0], location="xlower", ndim=1
            )
        }
        b = {
            "xlower": boundary.FreePressureInflowBC(
                density=rho, velocity=1.0, B=[0.0, 0.0, 0.0], location="xlower", ndim=1
            )
        }
        self.assertEqual(
            _boundary_conditions_signature(a), _boundary_conditions_signature(b)
        )

    def test_recreated_callable_with_same_source_compares_equal(self):
        def make_rho():
            def rho(x, t):
                return 1.0

            return rho

        a = {
            "xlower": boundary.FreePressureInflowBC(
                density=make_rho(),
                velocity=1.0,
                B=[0.0, 0.0, 0.0],
                location="xlower",
                ndim=1,
            )
        }
        b = {
            "xlower": boundary.FreePressureInflowBC(
                density=make_rho(),
                velocity=1.0,
                B=[0.0, 0.0, 0.0],
                location="xlower",
                ndim=1,
            )
        }
        self.assertEqual(
            _boundary_conditions_signature(a), _boundary_conditions_signature(b)
        )


if __name__ == "__main__":
    unittest.main()
