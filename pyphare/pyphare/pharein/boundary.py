"""
Boundary-condition resolution and validation for pharein.Simulation.

'boundary_conditions' (the public Simulation constructor option) is a per-location dict:
  {"xlower": {"type": "open"}, "xupper": {"type": "super-magnetofast-inflow", "data": {...}}, ...}

resolve_boundary_conditions(ndim, **kwargs) is the single entry point used by Simulation's
checker() pipeline; it returns a dict[location] -> BoundaryCondition stored as
Simulation.boundary_conditions.
"""

from abc import ABC
from dataclasses import dataclass

from ..core import phare_utilities


@dataclass
class BoundaryCondition(ABC):
    """Base class holding the serialisation entry point common to every BC type."""

    type = None

    def populate_dict(self, bc_path, ndim):
        _add_string(f"{bc_path}/type", self.type)


@dataclass
class NoneBC(BoundaryCondition):
    type = "none"


@dataclass
class OpenBC(BoundaryCondition):
    type = "open"


@dataclass
class ReflectiveBC(BoundaryCondition):
    type = "reflective"


@dataclass
class SuperMagnetofastOutflowBC(BoundaryCondition):
    type = "super-magnetofast-outflow"


# ------------------------------------------------------------------------------

_TYPE_CTORS = {
    "none": NoneBC,
    "open": OpenBC,
    "reflective": ReflectiveBC,
    "super-magnetofast-outflow": SuperMagnetofastOutflowBC,
}
_DATA_CARRYING = set()  # populated in Task 2
_REQUIRED_DATA_KEYS = {}  # populated in Task 2


def resolve_boundary_conditions(ndim, **kwargs):
    all_directions = ["x", "y", "z"][:ndim]
    sides = ("lower", "upper")
    boundary_types = kwargs["boundary_types"]

    physical_directions = [
        direction
        for direction, boundary_type in zip(all_directions, boundary_types)
        if boundary_type == "physical"
    ]
    physical_boundary_locations = [
        f"{direction}{side}" for direction in physical_directions for side in sides
    ]

    model_options = phare_utilities.listify(kwargs.get("model_options", "HybridModel"))
    if physical_boundary_locations and "MHDModel" not in model_options:
        raise ValueError(
            "'physical' boundary_types are only supported by the MHDModel; "
            f"got model_options={model_options}"
        )

    all_boundary_locations = [
        f"{direction}{side}" for side in sides for direction in all_directions
    ]
    raw = kwargs.get("boundary_conditions", {})
    if not isinstance(raw, dict):
        raise TypeError("A dict should be passed to argument 'boundary_conditions'")

    for location in raw:
        if location not in all_boundary_locations:
            raise ValueError(
                f"Wrong boundary name {location}: should belong to {all_boundary_locations}"
            )

    resolved = {}
    for location in all_boundary_locations:
        bc = raw.get(location, {"type": "none"})
        if not isinstance(bc, dict):
            raise TypeError(
                f"A dict should be passed to the boundary {location} for specifying a "
                f"boundary condition"
            )
        if "type" not in bc:
            raise KeyError(
                f"No key 'type' found in the boundary_condition dict passed to {location}"
            )
        boundary_type = bc["type"]
        if boundary_type not in _TYPE_CTORS:
            raise ValueError(
                f"Boundary type {boundary_type} is not valid: it should belong to "
                f"{tuple(_TYPE_CTORS)}"
            )
        if location in physical_boundary_locations and boundary_type == "none":
            raise KeyError(
                f"{location} is a physical boundary and should be provided with a valid "
                f"type other than 'none'."
            )
        if location not in physical_boundary_locations and boundary_type != "none":
            raise ValueError(
                f"{location} is not a physical boundary (its direction is periodic) and must "
                f"have type 'none', got '{boundary_type}'"
            )

        data = bc.get("data", {})
        if boundary_type in _DATA_CARRYING:
            for key in _REQUIRED_DATA_KEYS[boundary_type]:
                if key not in data:
                    raise KeyError(
                        f"Boundary type '{boundary_type}' at '{location}' requires '{key}' "
                        f"inside 'data'"
                    )
            resolved[location] = _TYPE_CTORS[boundary_type](
                **data, location=location, ndim=ndim
            )
        else:
            if "data" in bc:
                raise ValueError(
                    f"Boundary type '{boundary_type}' at '{location}' takes no 'data' block, "
                    f"but one was provided. Supported data-carrying types: "
                    f"{sorted(_DATA_CARRYING)}"
                )
            resolved[location] = _TYPE_CTORS[boundary_type]()

    return resolved


def _add_string(path, val):
    import pybindlibs.dictator as pp

    pp.add_string(path, val)
