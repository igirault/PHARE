# Simulation

## Setting up a simulation

A PHARE simulation is configured entirely from a Python script, by declaring
a small number of objects. Each is declared once (diagnostics may be declared
several times, once per output), and `Simulation` must always be declared
first - every other block below depends on it.

```python
# ------------ MANDATORY BLOCKS

ph.Simulation(
    # time evolution, domain, AMR, restart and diagnostics-output parameters
)

ph.MaxwellianFluidModel(
    # magnetic field profile and ion population(s), for a Hybrid simulation
)
# -- or, for an MHD simulation --
ph.MHDModel(
    # density, velocity, magnetic field and pressure profiles
)

ph.ElectronModel(
    # electron fluid closure - required for a Hybrid simulation
)

# ------------ END OF MANDATORY BLOCKS


# ------------ OPTIONAL BLOCKS

ph.LoadBalancer(
    # tune how work is redistributed across MPI ranks
)

ph.ElectromagDiagnostics(...)   # E and B field outputs
ph.FluidDiagnostics(...)        # ion moment outputs
ph.ParticleDiagnostics(...)     # particle outputs
ph.MHDDiagnostics(...)          # MHD fluid outputs
ph.InfoDiagnostics(...)         # run-monitoring outputs

# ------------ END OF OPTIONAL BLOCKS
```

See {doc}`models` for `MaxwellianFluidModel`/`MHDModel`/`ElectronModel`,
{doc}`diagnostics` for the diagnostics blocks, and {doc}`load_balancing` for
`LoadBalancer`. This page covers the `Simulation` block itself.

## The Simulation block

`Simulation` sets the overall numerical and physical parameters of the
run: how long it evolves and at what resolution, the physical domain and
mesh, adaptive mesh refinement, restart checkpoints, and where diagnostics
are written.

```{eval-rst}
.. autoclass:: pyphare.pharein.Simulation
```

## Domain and time, worked example

The domain can be given as any two of `cells` (number of mesh cells),
`dl` (mesh spacing) and `domain_size` (physical size) - the third is
derived. Likewise for time, any two of `time_step`, `time_step_nbr` and
`final_time`:

```python
import pyphare.pharein as ph

ph.Simulation(
    cells=(100, 100),
    dl=(0.2, 0.2),          # domain_size is derived: (20, 20)
    time_step=0.001,
    time_step_nbr=1000,     # final_time is derived: ~1.0
    refinement="tagging",
    max_nbr_levels=3,
)
```

```{note}
When `final_time` and `time_step` are both given, `time_step` is
kept exactly as given and `final_time` is treated as a target: the
actual number of steps is rounded to the nearest integer, so the run may
stop slightly before or after the requested `final_time`.
```

## Restarts and diagnostics output

See {doc}`restarts` for `restart_options` and {doc}`diagnostics` for
`diag_options` and the diagnostics blocks themselves.

(adaptive_time_step)=

## Adaptive time step

With `time_step={"mode": "adaptive", ...}` the MHD solver recomputes the time
step of each level from the current state. Three stability rates are added, each
divided by its own coefficient:

$$
\frac{1}{\Delta t} = \sum_d \frac{|v_d| + c_{f,d}}{C_\mathrm{hyp}\,\Delta x_d}
+ \sum_d \frac{v_{w,d} - v_A}{C_\mathrm{w}\,\Delta x_d}
+ \frac{2\eta}{C_\mathrm{diff}} \sum_d \frac{1}{\Delta x_d^2}
$$

where $C_\mathrm{hyp}$, $C_\mathrm{w}$ and $C_\mathrm{diff}$ are `cfl_hyperbolic`,
`cfl_whistler` and `cfl_diffusive`. $c_{f,d}$ is the fast magnetosonic speed in
direction $d$, and $v_{w,d}$ is the whistler phase speed at the grid scale
$k = \pi/\Delta x_d$. The whistler term is present only with `hall=True`. The
diffusive term is present only when `eta` is non-zero.

The rates add, rather than taking the smallest of three time steps, because the three
terms are evaluated in the same Runge-Kutta stages. If each coefficient stays below the
limit of its own term, the sum stays stable for any mix of the three.

### Stability limits

The tables give the largest stable value of each coefficient on its own. They come from
a linear (von Neumann) analysis of PHARE's discrete operators around a uniform state,
combined with the stability polynomial of each integrator:

- hyperbolic: the Rusanov flux, with each reconstruction at its linear weights, for
  every wave speed between $-S$ and $S$;
- whistler: the Hall electric field, with $J$ computed on the edges, averaged to the
  cell centres, reconstructed to the faces and averaged there;
- diffusive: $E = \eta J$ on the edges, which gives the compact Laplacian.

Use about 80% of the tabulated value: the analysis is linear, and limiters and WENO
weights are not.

Stability region of each integrator along the negative real axis (damping) and the
imaginary axis (dispersion):

| Integrator | real axis | imaginary axis |
|---|---|---|
| `Euler` | 2 | 0 |
| `TVDRK2` | 2 | 0 |
| `TVDRK3` | 2.51 | 1.73 |
| `SSPRK4_5` | 5.33 | 3.28 |

**cfl_hyperbolic**, Rusanov flux. The same values hold in 1D, 2D and 3D. `Linear` is
the smooth-region behaviour of the `VanLeer` and `MinMod` limiters, which fall back to
`Constant` at extrema. "unstable" means unstable for smooth solutions at any time step:

| Reconstruction | `Euler` | `TVDRK2` | `TVDRK3` | `SSPRK4_5` |
|---|---|---|---|---|
| `Constant` | 1.00 | 1.00 | 1.26 | 2.19 |
| `Linear` | unstable | 1.00 | 1.18 | 1.73 |
| `WENO3` | unstable | 0.87 | 1.63 | 2.10 |
| `WENOZ`, `MP5` | unstable | 0.10 | 1.43 | 1.99 |

**cfl_whistler**, Hall MHD without hyper-resistivity. The grid-scale whistler is
dispersive, and the integrator must reach along the imaginary axis. `Euler` and `TVDRK2`
do not reach it at all, and are unstable at any time step when $\Delta x \ll d_i$. The
values below are the limits for $\Delta x \ll d_i$. A linear analysis of the coupled
Alfvén-whistler system shows that the limit only grows at coarser resolution, so they
are safe at every $d_i/\Delta x$:

| Reconstruction | `Euler` | `TVDRK2` | `TVDRK3` | `SSPRK4_5` |
|---|---|---|---|---|
| `Constant` | unstable | unstable | 5.44 | 10.3 |
| `Linear` | unstable | unstable | 3.53 | 6.69 |
| `WENO3` | unstable | unstable | 4.02 | 7.61 |
| `WENOZ`, `MP5` | unstable | unstable | 3.56 | 6.74 |

These values are larger than 1. PHARE's Hall operator is weaker than the estimate
$(v_w - v_A)/\Delta x$ suggests, because $J$ is averaged and reconstructed before it
enters the flux.

Hyper-resistivity with `hyper_mode="spatial"` damps the grid-scale whistler. That makes
`Euler` and `TVDRK2` usable. The damping is measured by
$\tilde\nu = \nu\,(1 + \rho/|B|)$. `Euler` is stable up to about $2\pi\tilde\nu$ for every
reconstruction. `TVDRK2`:

| Reconstruction | $\tilde\nu = 0.01$ | 0.02 | 0.04 | 0.1 |
|---|---|---|---|---|
| `Constant` | 2.01 | 2.59 | 3.37 | 3.93 |
| `Linear` | 1.30 | 1.68 | 2.19 | 3.10 |
| `WENO3` | 1.50 | 1.93 | 2.52 | 3.57 |
| `WENOZ`, `MP5` | 1.35 | 1.75 | 2.28 | 3.22 |

Hyper-resistivity has no term of its own in the time step. With a large $\tilde\nu$ it
becomes the limit, and the `TVDRK3` and `SSPRK4_5` values decrease. At
$\tilde\nu = 0.1$ with `Linear` they are 4.83 and 6.00. With `hyper_mode="constant"` the
damping grows as $1/\Delta x^4$ and no coefficient bounds it.

**cfl_diffusive**, for any reconstruction and dimension: 1.00 (`Euler`),
1.00 (`TVDRK2`), 1.26 (`TVDRK3`), 2.67 (`SSPRK4_5`).

### Measured limits

1D whistler runs with `Linear`/`VanLeer`, $B = \rho = 1$, $\nu = 0.02$
(`hyper_mode="spatial"`). The table gives the largest stable coefficient and, in
brackets, the first unstable one. The model rows come from the coupled
Alfvén-whistler analysis, with one coefficient as in these runs.

| Case | `Euler` | `TVDRK2` | `TVDRK3` | `SSPRK4_5` |
|---|---|---|---|---|
| $\Delta x = 0.05\,d_i$, measured | 0.3 (0.4) | 2 (3) | 4 (5) | 6 (8) |
| $\Delta x = 0.05\,d_i$, model | | 2.30 | 4.57 | 6.73 |
| $\Delta x = 0.8\,d_i$, measured | | 2 | 3 | 4 |
| $\Delta x = 0.8\,d_i$, model | | 2.18 | 3.54 | 4.41 |

The model has no stable `Euler` step with `Linear`, because that pair is unstable for the
advective part (see `cfl_hyperbolic`). The measured runs held at 0.3, helped by the
limiter.

On a 2D Harris sheet with `SSPRK4_5` and `TVDRK3`, runs stayed stable up to
coefficients of 12 and 6, because the Hall term dominates only in part of the domain.
