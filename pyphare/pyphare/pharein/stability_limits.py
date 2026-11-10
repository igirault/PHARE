INTEGRATORS = ("Euler", "TVDRK2", "TVDRK3", "SSPRK4_5")

_HYPERBOLIC = {
    "Constant": (1.00, 1.00, 1.25, 2.19),
    "Linear": (0.5, 1.00, 1.17, 1.73),
    "WENO3": (None, 0.873, 1.62, 2.10),
    "WENOZ": (None, 1.00, 1.43, 1.98),
    "MP5": (None, 1.00, 1.43, 1.98),
}

_WHISTLER = (None, None, 1.36, 2.57)
_DIFFUSIVE = (1.00, 1.00, 1.25, 2.66)
_HYPER_RESISTIVE = (2.00, 2.00, 2.51, 5.33)


def _integrator_index(integrator):
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"Error: unknown time integrator {integrator!r}, expected one of {INTEGRATORS}"
        )
    return INTEGRATORS.index(integrator)


def hyperbolic_limit(reconstruction, integrator):
    if reconstruction not in _HYPERBOLIC:
        raise ValueError(
            f"Error: unknown reconstruction {reconstruction!r}, expected one of {tuple(_HYPERBOLIC)}"
        )
    return _HYPERBOLIC[reconstruction][_integrator_index(integrator)]


def whistler_limit(integrator):
    return _WHISTLER[_integrator_index(integrator)]


def diffusive_limit(integrator):
    return _DIFFUSIVE[_integrator_index(integrator)]


def hyper_resistive_limit(integrator):
    return _HYPER_RESISTIVE[_integrator_index(integrator)]
