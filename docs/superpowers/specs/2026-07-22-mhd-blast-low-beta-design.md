# Magnetized blast, low-β failure evidence — design

**Date:** 2026-07-22
**Branch:** `low-beta`
**Goal:** Demonstrate, with a reproducible test case, that PHARE's ideal-MHD solver
cannot handle low plasma-β plasma. Follow the magnetized blast problem of
Wu & Shu, *Provably Positive Discontinuous Galerkin Methods for Multidimensional
Ideal Magnetohydrodynamics*, SIAM J. Sci. Comput. (2018), DOI 10.1137/18M1168042,
Example 4.4 (originally Balsara & Spicer 1999).

## Motivation

PHARE MHD is a finite-volume Godunov scheme with no positivity-preserving limiter
and no pressure/density floor (verified: `to_primitive_converter.hpp` computes
`p = (γ-1)(etot - 0.5ρv² - 0.5b²)` with no clamp). At low β the magnetic energy
dominates internal energy, so discretization error in the total-energy update can
drive the reconstructed thermal pressure negative. Negative P → `sqrt(γP/ρ)` NaN
→ thrown negative-pressure exception + emergency dump (`useExceptionsInsteadOfMPIAbort`).
That crash, and the β at which it appears, is the evidence.

## Physical setup (paper, Example 4.4)

- Domain `[-0.5, 0.5]²`, plasma at rest, uniform density ρ = 1, adiabatic index γ = **1.4**.
- Explosion zone `r < 0.1`: pressure `pe`. Ambient `r > 0.1`: pressure `pa = 0.1`.
  `r = sqrt(x² + y²)`.
- Magnetic field uniform along x: `B = (Ba, 0, 0)`. `v = 0`.
- Plasma-beta `β = 2·pa / |B|²`.

## Unit convention (paper → PHARE)

PHARE MHD is Heaviside-Lorentz, μ₀ = 1: total energy
`etot = p/(γ-1) + 0.5ρv² + 0.5·B²` (`to_conservative_converter.hpp`), i.e. magnetic
pressure `B²/2` and `β = 2p/B²`. The paper uses the *same* β definition (`2p/|B|²`);
its `Ba = 100/√(4π)` is only the CGS→normalized rescaling of Balsara's original
`B = 100`. Plugging `100/√(4π) = 28.21` into PHARE reproduces β = 2.51×10⁻⁴ exactly.

**Decision:** express the field natively. β is the knob; derive
`Ba = sqrt(2·pa/β)`. No `√(4π)` in the code. A comment records the equivalence to
the paper's form.

## Configurations

`pa = 0.1`, γ = 1.4 throughout.

| Config    | role      | pe   | target β  | Ba = √(2·pa/β) | paper form   | tmax  | paper fate (no PP limiter) |
|-----------|-----------|------|-----------|----------------|--------------|-------|-----------------------------|
| reference | succeeds  | 1e3  | 2.5       | 0.2828         | 1/√(4π)      | 0.01  | runs clean                  |
| blast1    | fails     | 1e3  | 2.51e-4   | 28.21          | 100/√(4π)    | 0.01  | 3rd-order DG dies t≈2.85e-4 |
| blast2    | fails hard| 1e4  | 2.51e-6   | 282.1          | 1000/√(4π)   | 0.001 | dies t≈1.2e-5               |

## Numerical setup

- MPI: 10 ranks per run.
- Grid: **uniform single level** via `max_mhd_level = 1` alone (no `refinement`
  kwarg, no `max_nbr_levels`). Note: PHARE's DSL only accepts `max_nbr_levels`
  when `refinement` is set (`simulation.py:622`), so the two cannot be combined;
  `max_mhd_level = 1` with the default (`max_nbr_levels = 1`) already yields a
  single level with no AMR machinery. Isolates the
  solver — failure is the scheme, not AMR ghost/reflux artifacts.
- Resolution: 320×320 (paper). Domain modeled as `[0,1]²` with the circle centered at
  (0.5, 0.5); physics is translation-invariant, so identical to `[-0.5,0.5]²`.
- Solver: high-order `reconstruction="WENOZ"`, `limiter="None"`,
  `riemann="Rusanov"`, `mhd_timestepper="SSPRK4_5"`. No Hall, no resistivity, no
  hyper-resistivity. `gamma=1.4`. Requires adding the 2D permutation
  `2,1,4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false` to `res/sim/all.txt`
  (one `cpp_*` module rebuild).
- Boundaries: periodic (default; the shock does not reach the edge within tmax).
- `time_step`: fixed, chosen per config below the ideal CFL limit; blast2 needs a
  smaller step (stronger fast speed ∝ Ba).

## Architecture

Single file `tests/functional/mhd_blast/blast.py`, cloned from `mhd_rotor/rotor.py`.

- `config(pe, pa, beta, tmax, dt, label)` → builds `ph.Simulation` + `ph.MHDModel`
  (density/vx…/bx…/p closures with `Ba = np.sqrt(2*pa/beta)`), registers diagnostics,
  returns `sim`. One clear job: assemble one blast.
- `CONFIGS`: list of the three dicts above.
- Driver `run_all()`: for each config, reset globals (`ph.global_vars.sim = None`),
  build, run inside `try/except`, record outcome `{label, beta, survived, crash_time,
  min_pressure}`. Continue to next config on failure. After a caught MHD exception,
  `pkill` any leaked `mpirun` before the next run (MHD runs don't exit cleanly on
  caught exception).
- Outcomes written to `tests/functional/mhd_blast/README.md` (generated table) — the
  human-readable evidence artifact.

## Diagnostics / output

- Format: **`pharevtkhdf`** (`diag_options["format"]="pharevtkhdf"`). Viewed in ParaView.
- Quantities: `B` (ElectromagDiagnostics), `rho`, `V`, `P` (MHDDiagnostics), at ~5
  timestamps up to tmax.
- No matplotlib `Run().GetMHD*` plotting — that path is phareh5-only. Visual inspection
  is via ParaView on the `.vtkhdf` output.
- Per-config output dir `phare_outputs/blast/<label>`.

## Test registration

- `res/cmake/` via `phare_mpi_python3_exec`, guarded by `if(HighFive)`.
- Heavy execution level (11+) so it is excluded from default CI (`PHARE_EXEC_LEVEL_MAX`).
  This is an evidence/diagnostic case, not a pass/fail regression gate: the "failing"
  configs are *expected* to crash, so it must not run as an ordinary CI test.

## Success criteria

1. `reference` runs to `tmax` without exception.
2. `blast1` and `blast2` terminate early with a negative-pressure / NaN exception,
   at times comparable in order to the paper's failure times.
3. `README.md` table shows monotone: lower β → earlier crash (reference survives,
   blast1 dies, blast2 dies earliest), isolating β as the cause.

## Out of scope (YAGNI)

- No positivity-preserving limiter or pressure floor implementation (the point is to
  show the *absence* causes failure).
- No AMR, no Hall/resistive terms.
- No quantitative convergence study against a reference solution.
