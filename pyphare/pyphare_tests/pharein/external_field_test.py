import unittest

import numpy as np
import pyphare.pharein.global_vars as global_vars
from pyphare.pharein import simulation
from pyphare.pharein.external_field import (
    DipoleExternalField,
    NoExternalField,
    UserDefinedExternalField,
    resolve_external_field,
)


def az_2d(x, y):
    return x + y


def az_2d_t(x, y, t):
    return (x + y) * t


def dazdt_2d(x, y, t):
    return x + y


def az_3d(x, y, z):
    return x + y + z


def ay_1d(x):
    return x


def resolve_user_defined(ndim, potential, **extra):
    """Resolve a user-defined declaration, narrowed to what the tests then read."""
    ef = resolve_external_field(
        ndim,
        external_field={"type": "user-defined", "potential": potential, **extra},
    )
    assert isinstance(ef, UserDefinedExternalField)
    return ef


class RecordingPopulator:
    """A dict_populator() stand-in recording what a description writes, and where."""

    def __init__(self):
        self.written = {}

    def add_double(self, path, value):
        self.written[path] = float(value)

    def add_enum_int(self, path, enum_name, member_name):
        self.written[path] = (enum_name, member_name)


class TestExternalFieldResolution(unittest.TestCase):
    def test_absent_gives_no_external_field(self):
        self.assertIsInstance(resolve_external_field(2), NoExternalField)
        self.assertIsInstance(
            resolve_external_field(2, external_field=None), NoExternalField
        )
        self.assertIsInstance(
            resolve_external_field(2, external_field={"type": "none"}), NoExternalField
        )

    def test_dipole_is_resolved(self):
        ef = resolve_external_field(
            2,
            external_field={
                "type": "dipole",
                "position": (0.5, 1.5),
                "moment": (0.0, 2.0),
            },
        )

        assert isinstance(ef, DipoleExternalField)
        self.assertEqual(ef.position, (0.5, 1.5))
        self.assertEqual(ef.moment, (0.0, 2.0))

    def test_user_defined_is_resolved(self):
        ef = resolve_user_defined(2, (None, None, az_2d))

        self.assertIsInstance(ef, UserDefinedExternalField)
        self.assertFalse(ef.is_time_dependent)
        self.assertIsNone(ef.potential_time_derivative)

        # the component the user gave is kept as is, the 'None' ones become callables
        self.assertIs(ef.potential[2], az_2d)
        self.assertTrue(all(callable(a) for a in ef.potential))

    def test_none_components_default_to_zero_shaped_like_the_coordinates(self):
        ef = resolve_user_defined(2, (None, None, az_2d))

        x = np.array([0.0, 1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])

        # an array, not a scalar: the pybind wrapper must not have to broadcast it
        for defaulted in (ef.potential[0], ef.potential[1]):
            np.testing.assert_array_equal(defaulted(x, y), np.zeros(x.size))

        np.testing.assert_array_equal(ef.potential[2](x, y), x + y)

    def test_user_defined_time_dependence_comes_from_the_signature(self):
        ef = resolve_user_defined(
            2,
            (None, None, az_2d_t),
            potential_time_derivative=(None, None, dazdt_2d),
        )

        dadt = ef.potential_time_derivative
        assert dadt is not None

        self.assertTrue(ef.is_time_dependent)
        self.assertIs(ef.potential[2], az_2d_t)
        self.assertIs(dadt[2], dazdt_2d)

        # the defaulted components take the time as a last argument too
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])
        for defaulted in (ef.potential[0], dadt[0]):
            np.testing.assert_array_equal(defaulted(x, y, 0.5), np.zeros(x.size))

    def test_user_defined_is_resolved_in_1d_and_3d(self):
        # in 1D a_x contributes to no curl term at all, and is expected to be None
        self.assertIs(resolve_user_defined(1, (None, ay_1d, None)).potential[1], ay_1d)
        self.assertIs(resolve_user_defined(3, (None, None, az_3d)).potential[2], az_3d)

    def test_invalid_declarations_are_rejected(self):
        dipole = {"type": "dipole", "position": (0.5, 1.5), "moment": (0.0, 2.0)}
        static = {"type": "user-defined", "potential": (None, None, az_2d)}
        timed = {
            "type": "user-defined",
            "potential": (None, None, az_2d_t),
            "potential_time_derivative": (None, None, dazdt_2d),
        }

        for ndim, ef in [
            (2, {"type": "quadrupole"}),  # unknown type
            (2, {"position": (0.5, 1.5)}),  # no type
            (2, "dipole"),  # not a dict
            (1, dipole),  # a dipole makes no sense in 1D
            (3, dipole),  # both vectors must have ndim components
            (2, {**dipole, "moment": (0.0, 1.0, 2.0)}),  # three components in 2D
            (2, {**dipole, "moment": (0.0, 0.0)}),  # a zero moment is no dipole
            (2, {k: v for k, v in dipole.items() if k != "moment"}),  # missing key
            (2, {**dipole, "value": 3}),  # unknown key
            #
            (2, {"type": "user-defined"}),  # no potential
            (2, {**static, "potential": (None, None, None)}),  # all components None
            (2, {**static, "potential": (None, az_2d)}),  # not three components
            (2, {**static, "potential": (None, None, 1.0)}),  # not a callable
            (2, {**static, "potential": (az_2d, None, az_2d_t)}),  # mixed signatures
            (3, static),  # signature does not match ndim
            (1, {**static, "potential": (ay_1d, None, None)}),  # a_x is unused in 1D
            (2, {**static, "moment": (0.0, 1.0)}),  # unknown key
            # the time dependence of the potential and of its derivative must agree
            (2, {k: v for k, v in timed.items() if k != "potential_time_derivative"}),
            (2, {**static, "potential_time_derivative": (None, None, dazdt_2d)}),
            (2, {**timed, "potential_time_derivative": (None, None, az_2d)}),
        ]:
            with self.assertRaises(ValueError):
                resolve_external_field(ndim, external_field=ef)


class TestExternalFieldPopulateDict(unittest.TestCase):
    def test_none_writes_only_its_type(self):
        dp = RecordingPopulator()
        NoExternalField().populate_dict(dp)

        self.assertEqual(
            dp.written,
            {
                "simulation/external_field/type": (
                    "ExternalFieldUpdaterType",
                    "none",
                )
            },
        )

    def test_dipole_writes_its_vectors_component_wise(self):
        dp = RecordingPopulator()
        DipoleExternalField((0.5, 1.5), (0.0, 1.0)).populate_dict(dp)

        self.assertEqual(
            dp.written,
            {
                "simulation/external_field/type": (
                    "ExternalFieldUpdaterType",
                    "dipole",
                ),
                # in 2D both vectors have x and y components only
                "simulation/external_field/position/x": 0.5,
                "simulation/external_field/position/y": 1.5,
                "simulation/external_field/moment/x": 0.0,
                "simulation/external_field/moment/y": 1.0,
            },
        )


class TestSimulationExternalField(unittest.TestCase):
    def setUp(self):
        global_vars.sim = None

    def tearDown(self):
        global_vars.sim = None

    def test_simulation_defaults_to_no_external_field(self):
        sim = simulation.Simulation(
            time_step=0.001, time_step_nbr=10, cells=(20, 20), dl=(0.1, 0.1)
        )
        self.assertIsInstance(sim.external_field, NoExternalField)

    def test_simulation_accepts_a_dipole(self):
        sim = simulation.Simulation(
            time_step=0.001,
            time_step_nbr=10,
            cells=(20, 20),
            dl=(0.1, 0.1),
            external_field={
                "type": "dipole",
                "position": (0.5, 1.5),
                "moment": (0.0, 2.0),
            },
        )
        self.assertIsInstance(sim.external_field, DipoleExternalField)

    def test_simulation_accepts_a_user_defined_field(self):
        sim = simulation.Simulation(
            time_step=0.001,
            time_step_nbr=10,
            cells=(20, 20),
            dl=(0.1, 0.1),
            external_field={"type": "user-defined", "potential": (None, None, az_2d)},
        )
        self.assertIsInstance(sim.external_field, UserDefinedExternalField)


if __name__ == "__main__":
    unittest.main()
