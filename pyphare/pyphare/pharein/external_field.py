"""
External magnetic field resolution and validation for pharein.Simulation.
"""

from abc import ABC
from dataclasses import dataclass

from ..core import phare_utilities

_AXES = ("x", "y", "z")


def _add_components(dp, path, values):
    """Write a vector as the x/y/z sub-dict shape the C++ parseDimXYZType expects."""
    for axis, value in zip(_AXES, values):
        dp.add_double(f"{path}/{axis}", value)


def _check_components(name, value, expected):
    values = phare_utilities.listify(value)

    if len(values) != expected:
        raise ValueError(
            f"Error: external_field '{name}' must have {expected} components, got {len(values)}"
        )
    if not all(phare_utilities.is_scalar(v) for v in values):
        raise ValueError(f"Error: external_field '{name}' components must be scalars")

    return tuple(float(v) for v in values)


@dataclass
class ExternalField(ABC):
    """
    Base class of the external field descriptions.

    'type' names the C++ ExternalFieldUpdaterType member this description maps onto.
    """

    type = None

    def populate_dict(self, dp):
        """Mirror the public `external_field` dict shape (type + per-type params) on the C++ side.

        `dp` is a dict_populator() (see pharein.initialize.general): an object exposing
        add_string/add_double/add_int/add_enum_int/... - passed in rather than imported, to
        avoid a circular import between this module and pharein.initialize.general.
        """
        dp.add_enum_int(
            "simulation/external_field/type", "ExternalFieldUpdaterType", self.type
        )


@dataclass
class NoExternalField(ExternalField):
    """No external field: B0 and its time derivative stay zero."""

    type = "none"


@dataclass
class DipoleExternalField(ExternalField):
    """
    A static magnetic dipole of moment 'moment' placed at 'position'.

    Both vectors have one component per dimension: in 2D the configuration is invariant along
    z and the moment lies in the (x, y) plane.
    """

    type = "dipole"

    position: tuple
    moment: tuple

    def populate_dict(self, dp):
        super().populate_dict(dp)
        _add_components(dp, "simulation/external_field/position", self.position)
        _add_components(dp, "simulation/external_field/moment", self.moment)


# ------------------------------------------------------------------------------


def _resolve_dict_external_field(external_field, *, ndim):
    valid_types = ("none", "dipole")
    type_ = external_field.get("type")
    if type_ not in valid_types:
        raise ValueError(
            f"Error: external_field dict requires 'type' in {valid_types}, got {type_!r}"
        )

    def _check_keys(allowed):
        extra = set(external_field) - allowed
        if extra:
            raise ValueError(
                f"Error: invalid external_field keys for type '{type_}': {sorted(extra)}, "
                f"allowed {sorted(allowed)}"
            )

    if type_ == "none":
        _check_keys({"type"})
        return NoExternalField()

    _check_keys({"type", "position", "moment"})
    if ndim == 1:
        raise ValueError("Error: a dipole external_field makes no sense in 1D")
    for key in ("position", "moment"):
        if key not in external_field:
            raise ValueError(f"Error: dipole external_field requires '{key}'")

    position = _check_components("position", external_field["position"], ndim)
    moment = _check_components("moment", external_field["moment"], ndim)

    if all(m == 0.0 for m in moment):
        raise ValueError("Error: dipole external_field 'moment' cannot be zero")

    return DipoleExternalField(position, moment)


def resolve_external_field(ndim, **kwargs):
    """
    Resolve the public 'external_field' Simulation option into a validated ExternalField.
    """
    external_field = kwargs.get("external_field")

    if external_field is None:
        return NoExternalField()

    if not isinstance(external_field, dict):
        raise ValueError(
            "Error: external_field must be a dict or an ExternalField, got "
            f"{type(external_field).__name__}"
        )

    return _resolve_dict_external_field(external_field, ndim=ndim)
