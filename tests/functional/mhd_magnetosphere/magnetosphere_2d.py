import argparse
import sys

import numpy as np
from scipy.constants import e, m_p, mu_0

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--tagging-threshold", type=float, default=0.1)
    parser.add_argument("--flow-throughs", type=float, default=2.0)
    parser.add_argument("--dumps", type=int, default=40)
    parser.add_argument("--timestepper", default="TVDRK3")
    parser.add_argument("--diag-dir", default=None)
    parser.add_argument("--dump-window", type=float, nargs=3, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


ARGS = parse_args(sys.argv[1:])

small = 1e-12
gamma = 5.0 / 3.0

R_E = 6371.0e3
V_WIND = 400.0e3
N_WIND = 5.0e6
RHO_WIND = m_p * N_WIND
P_WIND = 2e-11
B_WIND = 5e-9
STANDOFF_DISTANCE = 10.0 * R_E
MAG_DIPOLE = (
    np.pi * np.sqrt(2.0) * STANDOFF_DISTANCE**2 * np.sqrt(RHO_WIND * V_WIND**2 / mu_0)
)
R_INNER = 5 * R_E


def dipole_2d_y(moment, mu, x, y):
    return (mu * moment / (2.0 * np.pi)) * (
        (y**2 - x**2) / (np.hypot(x, y) ** 4 + small)
    )


def dipole_2d_x(moment, mu, x, y):
    return (mu * moment / (2.0 * np.pi)) * (2.0 * x * y / (np.hypot(x, y) ** 4 + small))


B_MAX = np.abs(dipole_2d_y(MAG_DIPOLE, mu_0, R_INNER, 0.0))
V0 = B_MAX / np.sqrt(m_p * mu_0 * N_WIND)
D0 = V0 / (e * B_MAX / m_p)

re = R_E / D0
ib_radius = R_INNER / D0
u_wind = V_WIND / V0
p_wind = P_WIND / (B_MAX**2 / mu_0)
b_wind = B_WIND / B_MAX
rho_wind = RHO_WIND / (m_p * N_WIND)
m = MAG_DIPOLE / (B_MAX * D0**2 / mu_0)
rho_in = 100.0 * rho_wind
p_in = p_wind / 20.0

cells = np.array((120, 60))
domain_size = np.array([200.0 * re, 100.0 * re])
dl = domain_size / cells
center = (50.0 * re, 50.0 * re)
dipole_radius = ib_radius - 2 * dl.min()

flow_through = domain_size[0] / u_wind
final_time = ARGS.flow_throughs * flow_through
timestamps = np.linspace(0.0, final_time, ARGS.dumps + 1)
if ARGS.dump_window is not None:
    t0, t1, n = ARGS.dump_window
    timestamps = np.union1d(timestamps, np.linspace(t0, t1, int(n)))

diag_dir = ARGS.diag_dir or f"phare_outputs/mhd_magnetosphere_L{ARGS.levels}"

transition_width = 5 * re
transition_radius = 20 * re
cut_x = 20.0 * re
cut_w = 5.0 * re


def step(x, delta):
    return 0.5 * (1.0 + np.tanh(x / delta))


def dist(x, y):
    return np.hypot(x - center[0], y - center[1])


def density(x, y):
    return rho_wind + (rho_in - rho_wind) * step(
        transition_radius - dist(x, y), transition_width
    )


def vx(x, y):
    return u_wind * step(dist(x, y) - transition_radius, transition_width)


def zero(x, y):
    return 0.0 * x


def p(x, y):
    return p_wind + (p_in - p_wind) * step(
        transition_radius - dist(x, y), transition_width
    )


def _psi0(xr, yr):
    return (m / (2.0 * np.pi)) * xr / (np.hypot(xr, yr) ** 2 + small)


def _w(x):
    return step(cut_x - x, cut_w)


def a1z(x, y):
    xr, yr = x - center[0], y - center[1]
    draped = b_wind * xr * (1.0 - ib_radius**2 / (np.hypot(xr, yr) ** 2 + small))
    return draped - _w(x) * _psi0(xr, yr)


def b1x(x, y):
    return (a1z(x, y + 0.5 * dl[1]) - a1z(x, y - 0.5 * dl[1])) / dl[1]


def b1y(x, y):
    return -(a1z(x + 0.5 * dl[0], y) - a1z(x - 0.5 * dl[0], y)) / dl[0]


def config():

    sim = ph.Simulation(
        smallest_patch_size=20,
        time_step={"mode": "adaptive", "cfl_wave": 0.8},
        final_time=final_time,
        cells=tuple(cells),
        dl=tuple(dl),
        nesting_buffer=1,
        max_mhd_level=ARGS.levels,
        max_nbr_levels=ARGS.levels,
        hall=True,
        hyper_mode="spatial",
        hyper_resistivity=0.0,
        resistivity=0.0,
        eta=0.0,
        nu=0.01,
        gamma=gamma,
        pressure_floor=1e-12,
        density_floor=1e-2,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper=ARGS.timestepper,
        model_options=["MHDModel"],
        strict=True,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        external_field={
            "type": "dipole",
            "position": center,
            "moment": (0.0, -m),
            "radius": dipole_radius,
        },
        inner_boundary={
            "name": "earth",
            "shape": "sphere",
            "center": center,
            "radius": ib_radius,
            "condition_type": "fixed-state",
            "density": rho_in,
            "pressure": p_in,
            "inactive_safe_state": {"density": rho_in, "pressure": p_wind},
        },
        domain_boundaries={
            "xlower": {
                "type": "super-magnetofast-inflow",
                "density": rho_wind,
                "pressure": p_wind,
                "velocity": u_wind,
                "B": [0.0, -b_wind, 0.0],
            },
            "xupper": {"type": "open"},
            "ylower": {"type": "open"},
            "yupper": {"type": "open"},
        },
        refinement="tagging",
        tagging_threshold=ARGS.tagging_threshold,
    )

    ph.MHDModel(
        density=density, vx=vx, vy=zero, vz=zero, b1x=b1x, b1y=b1y, b1z=zero, p=p
    )

    for quantity in ["rho", "P", "V", "rhoV"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)
    for quantity in ["B", "B1"]:
        ph.ElectromagDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def main():
    startMPI()
    Simulator(config()).run().reset()
    ph.global_vars.sim = None


if __name__ == "__main__":
    main()
