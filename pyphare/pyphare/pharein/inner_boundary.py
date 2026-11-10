import math
from dataclasses import dataclass

from pyphare.core import phare_utilities

_condition_types = ("fixed-state", "fixed-density")
_shapes = ("sphere",)
_keys = {"name", "shape", "center", "radius", "condition_type", "density", "pressure"}
_safe_state_keys = {"density", "pressure", "velocity", "B1"}


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Error: inner_boundary '{name}' must be finite and > 0, got {value}"
        )
    return value


def _vec3(name, value):
    values = [float(v) for v in phare_utilities.listify(value)]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(
            f"Error: inner_boundary inactive_safe_state '{name}' must be a scalar or a 3-list"
        )
    return tuple(values)


@dataclass(frozen=True)
class InactiveSafeState:
    density: float = 1.0
    pressure: float = 1.0
    velocity: tuple = (0.0, 0.0, 0.0)
    B1: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class InnerBoundary:
    name: str
    center: tuple
    radius: float
    density: float
    pressure: float | None
    safe_state: InactiveSafeState
    shape: str = "sphere"
    condition_type: str = "fixed-state"

    def populate_dict(self, dp):
        base = "simulation/inner_boundary"
        dp.add_string(f"{base}/name", self.name)
        dp.add_string(f"{base}/shape", self.shape)
        dp.add_string(f"{base}/condition_type", self.condition_type)
        dp.add_array_as_vector(f"{base}/center", list(self.center))
        dp.add_double(f"{base}/radius", self.radius)
        dp.add_double(f"{base}/density", self.density)
        if self.pressure is not None:
            dp.add_double(f"{base}/pressure", self.pressure)

        safe = f"{base}/inactive_safe_state"
        dp.add_double(f"{safe}/density", self.safe_state.density)
        dp.add_double(f"{safe}/pressure", self.safe_state.pressure)
        for c, v in zip("xyz", self.safe_state.velocity):
            dp.add_double(f"{safe}/velocity/{c}", v)
        for c, v in zip("xyz", self.safe_state.B1):
            dp.add_double(f"{safe}/B1/{c}", v)


def _resolve_safe_state(safe):
    if not isinstance(safe, dict):
        raise TypeError("Error: inner_boundary 'inactive_safe_state' must be a dict")
    unknown = set(safe) - _safe_state_keys
    if unknown:
        raise ValueError(
            f"Error: invalid inner_boundary inactive_safe_state keys {sorted(unknown)}, "
            f"valid keys are {sorted(_safe_state_keys)}"
        )
    return InactiveSafeState(
        density=_positive("inactive_safe_state density", safe.get("density", 1.0)),
        pressure=_positive("inactive_safe_state pressure", safe.get("pressure", 1.0)),
        velocity=_vec3("velocity", safe.get("velocity", 0.0)),
        B1=_vec3("B1", safe.get("B1", 0.0)),
    )


def resolve_inner_boundary(ndim, **kwargs):
    ib = kwargs.get("inner_boundary")
    if ib is None:
        return None
    if not isinstance(ib, dict):
        raise TypeError("Error: inner_boundary must be a dict")

    unknown = set(ib) - _keys - {"inactive_safe_state"}
    if unknown:
        raise ValueError(f"Error: invalid inner_boundary keys {sorted(unknown)}")

    condition_type = ib.get("condition_type", "fixed-state")
    required = ["name", "shape", "center", "radius", "density"]
    if condition_type != "fixed-density":
        required.append("pressure")
    for key in required:
        if key not in ib:
            raise ValueError(f"Error: inner_boundary requires '{key}'")
    if condition_type == "fixed-density" and "pressure" in ib:
        raise ValueError(
            "Error: inner_boundary condition_type 'fixed-density' does not take 'pressure'"
        )

    if ib["shape"] not in _shapes:
        raise ValueError(
            f"Error: inner_boundary shape '{ib['shape']}' is invalid, valid shapes are {_shapes}"
        )

    if condition_type not in _condition_types:
        raise ValueError(
            f"Error: inner_boundary condition_type '{condition_type}' is invalid, "
            f"valid types are {_condition_types}"
        )

    center = tuple(float(v) for v in phare_utilities.listify(ib["center"]))
    if len(center) != ndim:
        raise ValueError(
            f"Error: inner_boundary center must have {ndim} coordinates, got {len(center)}"
        )

    return InnerBoundary(
        name=str(ib["name"]),
        center=center,
        radius=_positive("radius", ib["radius"]),
        density=_positive("density", ib["density"]),
        pressure=_positive("pressure", ib["pressure"]) if "pressure" in ib else None,
        safe_state=_resolve_safe_state(ib.get("inactive_safe_state", {})),
        condition_type=condition_type,
    )
