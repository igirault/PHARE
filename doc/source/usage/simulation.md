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

With `time_step={"mode": "adaptive", "cfl": 0.8}` the MHD solver recomputes the time
step of each level from the current state. Each term of the scheme has a largest stable
coefficient $C^*$, a constant of the time integrator and, for the hyperbolic term, of the
reconstruction. Their rates are added, each divided by a fraction $f$ of that constant:

$$
\frac{1}{\Delta t} = \sum_d \frac{|v_d| + c_{f,d}}{f_h C_h^*\,\Delta x_d}
+ \sum_d \frac{k_d\,(v_{w,d} - v_A)}{f_w C_w^*}
+ \frac{4\eta}{f_d C_d^*} \sum_d \frac{1}{\Delta x_d^2}
+ \frac{\kappa}{f_d C_\nu^*} \Big(\sum_d \frac{4}{\Delta x_d^2}\Big)^2
$$

$c_{f,d}$ is the fast magnetosonic speed in direction $d$ and $v_{w,d}$ the whistler phase
speed at the discrete grid-scale wavenumber $k_d = 2/\Delta x_d$, where the compact
second difference has its largest eigenvalue $k_d^2 = 4/\Delta x_d^2$ (present with
`hall=True` only). Each rate is the largest eigenvalue of its discrete operator. $\kappa$ is the
hyper-resistive coefficient: $\nu\,\Delta x_\mathrm{min}^2 (|B|/\rho + 1)$ with
`hyper_mode="spatial"`, $\nu$ with `"constant"`.

- `cfl` (default 0.8) is the fraction used for every term. `cfl = 1` is the linear
  stability limit. Keep a margin: the Runge-Kutta stability regions of `TVDRK3` and
  `SSPRK4_5` are not convex, and limiters and WENO weights are nonlinear.
- `cfl_hyperbolic`, `cfl_whistler`, `cfl_diffusive` override the fraction of one term
  ($f_h$, $f_w$, $f_d$; the hyper-resistive term uses $f_d$). They default to `cfl`. A
  smaller `cfl_whistler` resolves whistlers in time: their phase error grows with the
  whistler Courant number.

The rates add, rather than taking the smallest of the time steps, because the terms are
evaluated in the same Runge-Kutta stages. If each term stays below its own limit, the
sum stays stable for any mix of them.

Adaptive runs reject the combinations that have no stable time step: `hall=True` with
`Euler` or `TVDRK2` (the grid-scale whistler is purely dispersive and these integrators do
not reach the imaginary axis), and `Euler` with `WENO3`, `WENOZ` or `MP5`. With a
constant time step these combinations are allowed; Hall runs then need hyper-resistivity
for `Euler` and `TVDRK2`.

### Stability limits

The constants come from a linear (von Neumann) analysis of PHARE's discrete operators
around a uniform state, combined with the stability polynomial of each integrator. They
are rounded down to 3 digits.

Stability region of each integrator along the negative real axis (damping, $\rho_r$) and
the imaginary axis (dispersion, $\rho_i$):

| Integrator | $\rho_r$ | $\rho_i$ |
|---|---|---|
| `Euler` | 2 | 0 |
| `TVDRK2` | 2 | 0 |
| `TVDRK3` | 2.51 | 1.73 |
| `SSPRK4_5` | 5.33 | 3.27 |

**Hyperbolic** ($C_h^*$), Rusanov flux, each reconstruction at its linear weights, for
every wave speed between $-S$ and $S$. The same values hold in 1D, 2D and 3D. `Linear`
is the smooth-region behaviour of the `VanLeer` and `MinMod` limiters:

| Reconstruction | `Euler` | `TVDRK2` | `TVDRK3` | `SSPRK4_5` |
|---|---|---|---|---|
| `Constant` | 1.00 | 1.00 | 1.25 | 2.19 |
| `Linear` | 0.5 | 1.00 | 1.17 | 1.73 |
| `WENO3` | rejected | 0.873 | 1.62 | 2.10 |
| `WENOZ`, `MP5` | rejected | 1.00 | 1.43 | 1.98 |

`Linear` with `Euler` is linearly unstable: 0.5 is the TVD bound of the limited
reconstruction. `WENOZ`/`MP5` with `TVDRK2` is weakly unstable at every time step for
smooth data (growth below $10^{-10}$ per step at 0.1); 1.00 is the `Linear` value.

**Whistler** ($C_w^*$), Hall MHD. The Hall electric field $(J\times B)/\rho$ is computed
on the Yee edges, which gives the compact operator $4\sin^2(\theta/2)/\Delta x^2$ for every
reconstruction. Its largest eigenvalue is $4|B|/(\rho\,\Delta x^2)$, which the estimate
$k(v_w - v_A)$ reaches for $\Delta x \ll d_i$, so $C_w^* = \rho_i$: 1.73 (`TVDRK3`), 3.27
(`SSPRK4_5`). A linear analysis of the coupled Alfvén-whistler system shows that the sum
of rates stays stable at every $d_i/\Delta x$, and the per-direction sum with $|B|$ in every
direction is stable for any field direction and cell aspect ratio in 2D and 3D.

**Resistive** ($C_d^* = \rho_r$, largest eigenvalue $4\eta\sum_d 1/\Delta x_d^2$): 2.00
(`Euler`), 2.00 (`TVDRK2`), 2.51 (`TVDRK3`), 5.33 (`SSPRK4_5`). **Hyper-resistive**
($C_\nu^* = \rho_r$): the same values.
Hyper-resistivity also damps the whistler, which the additive rule does not use: with
`TVDRK3` it is conservative.

### Measured limits

1D whistler runs with `Linear`/`VanLeer`, $B = \rho = 1$, $\nu = 0.02$
(`hyper_mode="spatial"`), run with a single absolute coefficient for the hyperbolic and
whistler terms. The table gives the ratio of the measured largest stable coefficient to
the one predicted by the linear analysis of the coupled Alfvén-whistler system (first
unstable value in brackets): the analysis predicts the measured limits within about 10%.

| Case | `TVDRK3` | `SSPRK4_5` |
|---|---|---|
| $\Delta x = 0.05\,d_i$ | 0.97 (1.10) | 0.97 (1.16) |
| $\Delta x = 0.8\,d_i$ | 0.95 (1.09) | 0.87 (1.09) |

On a 2D Harris sheet the runs stay stable further, because the Hall term dominates only in
part of the domain.
