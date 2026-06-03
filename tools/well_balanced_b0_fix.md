# Well-balanced B0 fix — scope

## Goal / invariant

Make the MHD scheme **exactly preserve** the background equilibrium

    V = 0,  B1 = uniform,  rho = uniform,  P = uniform,  B0 curl-free & div-free

i.e. zero spurious momentum/energy residual from rest, for any static B0. Regression
metric already exists: `tests/functional/mhd_2d_b0_well_balanced/` (uniform B0 gives
max|V| = 0 exactly today; dipole B0 must reach the same after the fix).

Confirmed earlier: single-valued B0 alone is NOT enough (Linear ≈ WENOZ ⇒ the
total-B Maxwell-stress *divergence* is the dominant spurious term, not the
reconstruction jump). So the flux must be reformulated, not just the reconstruction.

## Exact algebraic split (momentum)

Lorentz force `j×B`, `B = B0+B1`, `j = j0+j1`, `j0 = ∇×B0 = 0`:

    j×B = j1×B1   (kept inside B1 Maxwell-stress flux)
        + j1×B0   (cross  -> explicit SOURCE)
        + j0×B0 + j0×B1 = 0   (background, must never be discretized)

So momentum flux carries the **B1-only stress**

    T1 = rho u u + P I + ½ B1² I − B1 B1

and an explicit source `S_mom = j1 × B0`. At the equilibrium state above,
`j1 = ∇×B1 = 0` discretely (curl of a uniform field is exactly 0), so `∇·T1 = 0`
and `S_mom = 0` ⇒ well-balanced by construction.

Induction flux is unaffected (motional `E = −V×B` keeps total B; at rest it is 0).
Energy: doc form `(ρe_t+P)u + E×B1` already excludes B0 self-energy; at rest E=0,
so energy is automatically well-balanced. The split is an exact identity, so the
scheme stays analytically equivalent — only the discretization changes.

## The one real design decision: Riemann flux vs source split

HLLD star states (`hlld.hpp:170-357`) are built on the **total-B** MHD eigenstructure
(`etotL`, `p_tot`, transverse-B star states all use total B). Wave/fast speeds *must*
stay on total B (physical). The question is how to get a B1-stress conservative flux
out of a total-B Riemann solver. Two routes:

- **Route A — equilibrium-flux subtraction (low risk, local).**
  Leave HLLD/Rusanov untouched (total-B numerical flux `F_num`). At each face also
  evaluate the analytic background momentum flux `F0 = ½B0² I − B0 B0` using a single
  shared face B0 value, and subtract its divergence in the momentum update:
  `ρu_new = ρu − dt(∇·F_num − ∇·F0)`. Since `∇·F0 = −j0×B0 = 0` analytically,
  subtracting the *discrete* `∇·F0` cancels exactly the scheme's spurious background
  force. Fixes the magnetosphere case (B1 = 0 upstream → cross term absent). Does NOT
  fully cover `B1 ≠ 0` uniform (residual discrete `j0×B1` in the cross term).
  Smallest diff; no Riemann-solver changes.

- **Route B — source-term reformulation (robust, larger).**
  Change `MHD_equations::compute` momentum flux to B1-stress (`P+½B1²`, `B1B1`); keep
  total B only for wave speeds inside the Riemann solver; add `S_mom = j1×B0` as a
  cell-centered source after the flux divergence. Requires the Riemann solver to
  upwind a B1-stress flux with total-B speeds (decouple "flux function" from
  "wave-speed field" in `hlld.hpp`/`rusanov`), plus a source hook. Fully well-balanced
  for any uniform B1; matches BATSRUS/Tanaka background-field splitting. Higher risk:
  must re-verify HLLD star-state consistency and energy conservation.

## !! Mid-implementation finding (2026-06-03): culprit is the induction/CT, not momentum

The momentum B0-self removal below is implemented and correct, but it does NOT fix the
dipole well-balanced test (max|V| unchanged at 5.06e-5). A probe confirmed B0 IS
nonzero in `MHDEquations::compute`, so the fix is live but inert here.

Root cause is the **constrained transport**: in the rest run (V=0, B1=0 at t0), B1
grows to 8.5e-4 at the first step while V starts at 0 — i.e. the CT produces a spurious
electric field E ≠ 0 from the B0 gradient even though E = −V×B = 0 at rest. The
spurious E feeds the induction (∂B1/∂t = −∇×E) and bends the field lines. This is an
induction phenomenon, consistent with the observed symptom.

NEXT: make `src/core/numerics/constrained_transport/upwind_constrained_transport.hpp`
well-balanced — E must vanish for V=0 regardless of ∇B0 (the spurious term is almost
certainly reconstructed transverse-B0 jumps entering the edge-E averaging). Keep the
momentum-stress fix for the residual force. Re-test with the well-balanced suite.

## Implementation status (Stage 2, in progress)

The Rusanov/HLLD dissipation already uses the *reduced* conservatives (rhoV, B1,
Etot1) — no B0 — so the B0-self spurious force comes entirely from the physical flux
in `MHD_equations::compute`. Fix applied there: momentum flux = total Maxwell stress
MINUS B0 self-stress (`½B0²I − B0B0`). `B0 = 0` runs are byte-identical (no
regression). For `B1 = 0` this reduces the flux to pure pressure ⇒ **exactly**
well-balanced (machine zero). No Riemann-solver surgery, no source hook needed.

Residual for **uniform B1 ≠ 0**: the cross term leaves `j0×B1` where `j0` is the
*discrete* curl of B0 (truncation of an analytic curl-free field) — small and
*converging*, not machine zero. To make it exact too would require a discretely
curl-free B0 (B0 from a potential consistent with the Ampère stencil) — deferred;
the dominant, non-converging force (B0-self) is the one that bends the lines.

## Decision (locked)

- Target convention: **IMF lives in B1** (uniform, nonzero upstream). Route A's
  equilibrium-flux subtraction does not cancel the residual discrete `j0×B1`, so
  **Route B is the real solution** and is required.
- Staging: **Stage 1 = Route A** as an unblocking step + to build/validate the
  well-balancing regression machinery. Caveat — Route A is only fully well-balanced
  when `B1 = 0` upstream, so Stage 1 must temporarily keep the IMF folded into B0
  (as `case.py` does today: `b0y = dipole − b_in`). **Stage 2 = Route B**, which then
  permits moving the IMF into B1.

Note: a curl-source correction (`−j0×B` with discrete `j0=∇×B0`) was considered to
cover `j0×B1` cheaply, but it uses a different discrete operator than the FV
Maxwell-stress divergence, so it does NOT exactly cancel the spurious force. Exact
cancellation needs the same FV stencil → that is Route A (flux subtraction, B1=0) or
Route B (B1-stress flux). No cheap middle option.

### Stage 1 (Route A) task breakdown
1. Single-valued B0 at the interface — `reconstructor.hpp:31-37` (drop B0 L/R split;
   keep total B for wave speeds via the shared B0).
2. Background momentum flux operator `F0 = ½B0²I − B0B0` on faces + its divergence.
3. Subtract `dt ∇·F0` from `statenew.rhoV` after `fv_euler_`
   (`euler_using_computed_flux.hpp:34`).
4. Regression: assert dipole `max|V|` ≈ machine zero in
   `tests/functional/mhd_2d_b0_well_balanced/` (keep IMF in B0); wire as CTest.
5. No-regression: Orszag-Tang / `test_vtk_diagnostics_mhd` unchanged.

## Concrete changes

Ingredients already present:
- `J = ∇×B1` via `Ampere` (`solver_mhd_model_view.hpp:124`) = `j1`. For **ideal** runs
  Ampère is skipped (`compute_fluxes.hpp:54-57`) — enable it (or a B1-curl) when the
  source/correction is active.
- center/face→cell projections in `reconstructor.hpp` (`center_reconstruct`, `projection`)
  to co-locate `j1` (edge) and `B0` (face) at cell centres.
- `B0` is a `MHDState` member, face-centered, same Yee centering as `B1`
  (`mhd_state.hpp`).

### Route A files
- `src/core/numerics/reconstructions/reconstructor.hpp:31-37` — give B0 a single shared
  face value (drop L/R split for B0); B1 still reconstructed.
- New small operator (e.g. `src/core/numerics/mhd_background/background_flux.hpp`)
  computing face `F0 = ½B0²I − B0B0` and its divergence on rhoV.
- `src/core/numerics/finite_volume_euler/finite_volume_euler.hpp` /
  `time_integrator/euler_using_computed_flux.hpp:34` — subtract `dt ∇·F0` from
  `statenew.rhoV` right after `fv_euler_`.

### Route B additional files
- `src/core/numerics/MHD_equations/MHD_equations.hpp:31,42-44,56,70` — stress uses B1.
- `src/core/numerics/riemann_solvers/hlld.hpp`, `rusanov` — separate flux function (B1
  stress) from wave-speed field (total B).
- Source hook in the integrator: `S_mom = j1×B0` (cell-centered), added per RK substep
  in `euler_using_computed_flux.hpp`. Energy work term if needed for nonlinear
  consistency.

## Verification

1. **Well-balanced regression** (must pass): `tests/functional/mhd_2d_b0_well_balanced/`
   dipole B0 → `max|V|` at machine zero (today 3.3e-5). Add as a CTest assertion.
2. **No-regression correctness**: existing MHD tests (Orszag-Tang / flow-around-cylinder
   / `test_vtk_diagnostics_mhd`) unchanged; check energy conservation on a dynamic run.
3. **Magnetosphere**: rerun `mhd_2d_earth_like_magnetosphere`; upstream IMF lines
   straight until the bow shock.

## Doc

Update `PHARE-doc/doc/source/theory/mhd.md` "Splitting" section to document the
well-balanced momentum form (B1 stress + `j1×B0` source), noting `j0×B0 = j0×B1 = 0`
and why B0 self-stress is excluded from the discrete flux.

## Side cleanup (independent)

`hlld.hpp:570` (Hall path) operator-precedence: `Bn = SR*BcompR - SL*BcompL/(SR-SL)`
→ `(SR*BcompR - SL*BcompL)/(SR-SL)`. Ideal runs unaffected.
