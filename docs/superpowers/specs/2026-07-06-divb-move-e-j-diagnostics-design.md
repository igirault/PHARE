# divB relocation + E/J electromag diagnostics — design

Date: 2026-07-06
Branch: `derived-quantities` (worktree `PHARE-derived-quantities`)

## Goal

1. Move the MHD `divB` diagnostic from the `/mhd/` tree to the `/electromag/`-style tree (`ElectromagDiagnosticWriter`).
2. Add `E` and `J` to the electromag diagnostics:
   - `J`: new derived quantity for **both** Hybrid and MHD models.
   - `E`: new derived quantity for **MHD only** (Hybrid already stores/writes a real `E`).

Both `E` and `J` are freshly recomputed at diagnostic-write time from primary state, independent of the solver's internal transient fields (`state.E`/`state.J`), the same way `divB`/`V`/`P` are already recomputed independent of solver internals. The solver's own `state.E`/`state.J` are not reused — they are solver-internal intermediates (Riemann-flux-based for `E`; recomputed via `Ampere` mid-step for `J`), not guaranteed diagnostic-quality snapshots.

## Part 1 — divB relocation

- `src/core/data/derived_quantity/mhd_derived_quantities.hpp`: `MhdDivB` class unchanged (still rank-0 `DerivedQuantity`, name `"divB"`).
- `src/diagnostic/detail/types/electromag.hpp` (`ElectromagDiagnosticWriter`):
  - Add `Model_t`/`ModelView_t`/`physical_quantity_type` typedefs (same pattern as `mhd.hpp`).
  - Add an `"EM_divB"` code path in `createFiles`, `getDataSetInfo`, `initDataSets`, `write`, pulling from `h5Writer.modelView().derivedQuantities().quantities<0>()`, using `core::derived_scalar_view` exactly as `mhd.hpp` does today.
  - Guard the scalar (`quantities<0>()`) handling with `if constexpr (solver::is_mhd_model_v<Model_t>)` since Hybrid's `ModelView` has no scalar derived-quantity scratch buffer and doesn't need one.
- `src/diagnostic/detail/types/mhd.hpp` (`MHDDiagnosticWriter`):
  - Remove `"divB"` from the `createFiles` tree list and from the header doc comment.
  - No other change — the existing generic `quantities<0>()`/`quantities<1>()` loops keep serving `P` and `V` under `/mhd/`.
- Python `pyphare/pyphare/pharein/diagnostics.py`:
  - Remove `"divB"` from `MHDDiagnostics.mhd_quantities`.
  - Add `"divB"` to `ElectromagDiagnostics.em_quantities`. Resulting quantity path: `/EM_divB` (was `/mhd/divB`).
- Downstream references to update:
  - `pyphare/pyphare/pharesee/run/run.py::GetMHDdivB` — path/tree lookup.
  - `pyphare/pyphare/pharesee/hierarchy/hierarchy_utils.py` — the `"divB": "mhdDivB"` name mapping (verify it still keys correctly off the new path suffix).
  - `tests/simulator/test_mhd_derived_diagnostics.py` — switch the diagnostic construction for `divB` from `MHDDiagnostics` to `ElectromagDiagnostics`, update `mhd_quantities` list used in the test.
  - `tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp` — unaffected (tests the `MhdDivB` class directly, not the writer wiring).

## Part 2 — J (Hybrid + MHD)

New rank-1 `DerivedQuantity`, name `"J"`, `VectorCentering::Elike`, computed via the same curl-of-B formulas already used by `core::Ampere` (inlined directly in the derived-quantity class, consistent with the existing convention that derived quantities are self-contained — see `MhdDivB`).

- `MhdCurrentDensity` in `mhd_derived_quantities.hpp`, added to `makeMhdDerivedQuantities` via `registry.add<1>(...)`.
- `HybridCurrentDensity` in new `src/core/data/derived_quantity/hybrid_derived_quantities.hpp`, plus a new `makeHybridDerivedQuantities<State, GridLayout>()` factory function.
- Hybrid `ModelView` (`src/diagnostic/diagnostic_model_view.hpp`):
  - Currently constructs `derived_` via its default constructor (empty registry, unused). Change to initialize with `core::makeHybridDerivedQuantities<State_t, GridLayout>()`.
  - Add `derivedVecScratch_` member tagged `HybridQuantity::Vector::VecElike` and a `derivedVecScratch()` accessor, mirroring MHD's `derivedVecScratch_`/`derivedVecScratch()`. (No scalar scratch needed — no rank-0 Hybrid derived quantities are being added.)
- `ElectromagDiagnosticWriter`: add a generic `"EM_J"` path (no `if constexpr` needed — both models now have the registry + vector scratch) parallel to `EM_B`/`EM_E`, following the same `quantities<1>()` dispatch already used for divB/rank-0.
- Python `diagnostics.py`: add `"J"` to `ElectromagDiagnostics.em_quantities` → path `/EM_J`. Valid for both Hybrid and MHD sims (the writer dispatch table already registers `ElectromagDiagnosticWriter` for both).

## Part 3 — E (MHD only)

Hybrid's `getE()`/electromag `E` output is untouched (already a real stored field, already written).

New `MhdElectricField` in `mhd_derived_quantities.hpp`, rank-1, name `"E"`, `VectorCentering::Elike`. Point-wise, self-contained (recomputes its own local J via the same inline curl formulas as `MhdCurrentDensity` — no cross-dependency between derived quantities):

```
E = -V×B                      (ideal, using bulk V, not an electron Ve)
  + (J×B)/rho                 (Hall term, standard generalized Ohm's law form)
  + eta·J                     (resistive)
  + hyper-resistive(J, B, rho) (constant or spatial mode, matching existing nu/hyper_mode semantics)
```

This mirrors the algebra in `upwind_constrained_transport.hpp`'s `resistive_contribution_`/`hyperresistive_contribution_` (plus the added Hall term), but without the Riemann-reconstruction machinery — this is an independent diagnostic quantity, not a replication of the CT scheme's discrete internals. Component projections (`GridLayout::project<BxToEx>` etc.) follow the same pattern already used in `ohm.hpp`.

### New state plumbing required

`eta`, `nu`, `hyper_mode` are not currently reachable from `MhdState`/`ModelView` — they live in `ComputeFluxes`'s `GodunovInfo` (solver-level config). Mirroring the existing `gamma_` precedent on `MhdState`:

- `src/core/models/mhd_state.hpp`: add `eta_`, `nu_`, `hyperMode_` members, parsed from dict in the constructor alongside `gamma_`.
- `pyphare/pyphare/pharein/initialize/mhd.py`: add new dict entries mirroring the existing gamma line:
  ```python
  add_double("simulation/mhd_state/eta", sim.eta)
  add_double("simulation/mhd_state/nu", sim.nu)
  add_string("simulation/mhd_state/hyper_mode", sim.hyper_mode)
  ```
- `src/diagnostic/diagnostic_model_view.hpp`: MHD `ModelView`'s `makeMhdDerivedQuantities` call gains `eta`/`nu`/`hyper_mode` args alongside the existing `gamma`.

## Testing

- Existing `test_mhd_derived_diagnostics.py` extended (or a sibling test added) to cover `EM_divB`, `EM_J`, `EM_E` paths for MHD; a Hybrid-side functional test for `EM_J`.
- `tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp` gets new unit tests for `MhdCurrentDensity` and `MhdElectricField` (e.g. J=0 for uniform B; E ideal-only check with resistivity/hyper-resistivity/Hall all zeroed).
- New unit test file for `HybridCurrentDensity` alongside existing hybrid derived-quantity tests (if any exist — otherwise co-located with the new header).

## Out of scope

- VTK writer path (`detail/vtk_types/`) — divB was never exposed there; not adding it now.
- Replicating the CT scheme's exact discrete E (Riemann/reconstruction-based) — the diagnostic E is an independent, physically-consistent-but-simpler formula.
