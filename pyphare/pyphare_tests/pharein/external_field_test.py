import unittest

import pyphare.pharein.global_vars as global_vars

from pyphare.pharein import simulation
from pyphare.pharein.external_field import (
    DipoleExternalField,
    NoExternalField,
    resolve_external_field,
)


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

        self.assertIsInstance(ef, DipoleExternalField)
        self.assertEqual(ef.position, (0.5, 1.5))
        self.assertEqual(ef.moment, (0.0, 2.0))

    def test_invalid_declarations_are_rejected(self):
        dipole = {"type": "dipole", "position": (0.5, 1.5), "moment": (0.0, 2.0)}

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


if __name__ == "__main__":
    unittest.main()
