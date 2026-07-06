# divB relocation + E/J electromag diagnostics — design

Date: 2026-07-06
Branch: `derived-quantities` (worktree `PHARE-derived-quantities`)

## Goal

1. Move the MHD `divB` diagnostic from the `/mhd/` tree to the `/electromag/`-style tree (`ElectromagDiagnosticWriter`).
2. Add `E` and `J` to the electromag diagnostics:
   - `J`: new derived quantity for **both** Hybrid and MHD models.
   - `E`: new derived quantity for **MHD only** (Hybrid already stores/writes a real `E`).

Both `E` and `J` are freshly recomputed at diagnostic-write time from primary state, independent of the solver's internal transient fields (`state.E`/`state.J`), the same way `divB`/`V`/`P` are already recomputed independent of solver internals. The solver's own `state.E`/`state.J` are not reused — they are solver-internal intermediates (Riemann-flux-based for `E`; recomputed via `Ampere` mid-step for `J`), not guaranteed diagnostic-quality snapshots.

Both the legacy HDF5 diagnostics (`src/diagnostic/detail/types/`, `PHARE::diagnostic::h5`) and the vtkhdf diagnostics (`src/diagnostic/detail/vtk_types/`, `PHARE::diagnostic::vtkh5`) must be covered — the two writer families are separate implementations that happen to mirror each other's tree/quantity naming.

## Part 1 — divB relocation

- `src/core/data/derived_quantity/mhd_derived_quantities.hpp`: `MhdDivB` class unchanged (still rank-0 `DerivedQuantity`, name `"divB"`).
- `src/diagnostic/detail/types/electromag.hpp` (`ElectromagDiagnosticWriter`):
  - Add `Model_t`/`ModelView_t`/`physical_quantity_type` typedefs (same pattern as `mhd.hpp`).
  - Add an `"EM_divB"` code path in `createFiles`, `getDataSetInfo`, `initDataSets`, `write`, pulling from `h5Writer.modelView().derivedQuantities().quantities<0>()`, using `core::derived_scalar_view` exactly as `mhd.hpp` does today.
  - Guard the scalar (`quantities<0>()`) handling with `if constexpr (solver::is_mhd_model_v<Model_t>)` since Hybrid's `ModelView` has no scalar derived-quantity scratch buffer and doesn't need one.
- `src/diagnostic/detail/types/mhd.hpp` (`MHDDiagnosticWriter`):
  - Remove `"divB"` from the `createFiles` tree list and from the header doc comment.
  - No other change — the existing generic `quantities<0>()`/`quantities<1>()` loops keep serving `P` and `V` under `/mhd/`.
- `src/diagnostic/detail/vtk_types/fluid.hpp` (`MhdFluidInitializer`/`MhdFluidWriter`): **no code change**. Both already loop generically over `derivedQuantities().quantities<0>()/<1>()` under the `/mhd/` tree (no hardcoded quantity-name list), so `divB` is already reachable there today. Removing `"divB"` from the Python `mhd_quantities` list (below) is sufficient to stop it being requested under `/mhd/`.
- `src/diagnostic/detail/vtk_types/electromag.hpp` (`ElectromagDiagnosticWriter`, vtkhdf): unlike its H5 sibling, this one **hardcodes** `EM_B`/`EM_E` in both `setup()` and `write()` — no generic derived-quantity loop exists yet. Add:
  - `Model_t` typedef (`H5Writer::ModelView::Model_t`, same style as `vtk_types/fluid.hpp`'s `FluidDiagnosticWriter`).
  - An `"EM_divB"` branch in `setup()`'s `init` lambda (`initializer.initFieldFileLevel(level)`) and in `write()`'s `write_quantity` lambda, using `derivedQuantities().quantities<0>()` + `core::derived_scalar_view` + `dq->compute(...)` + `writer.writeField(...)`, guarded by `if constexpr (solver::is_mhd_model_v<Model_t>)` — mirroring `MhdFluidWriter`'s derived-quantity branch in `fluid.hpp`.
- Python `pyphare/pyphare/pharein/diagnostics.py`:
  - Remove `"divB"` from `MHDDiagnostics.mhd_quantities`.
  - Add `"divB"` to `ElectromagDiagnostics.em_quantities`. Resulting quantity path: `/EM_divB` (was `/mhd/divB`). Same path/name used by both writer families since `type="electromag"` dispatches to the matching `ElectromagDiagnosticWriter` in whichever writer (`h5writer.hpp` or `vtkh5_writer.hpp`) is active.
- Downstream references to update:
  - `pyphare/pyphare/pharesee/run/run.py::GetMHDdivB` — path/tree lookup.
  - `pyphare/pyphare/pharesee/hierarchy/hierarchy_utils.py` — the `"divB": "mhdDivB"` name mapping (verify it still keys correctly off the new path suffix).
  - `tests/simulator/test_mhd_derived_diagnostics.py` — switch the diagnostic construction for `divB` from `MHDDiagnostics` to `ElectromagDiagnostics`, update `mhd_quantities` list used in the test.
  - `tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp` — unaffected (tests the `MhdDivB` class directly, not the writer wiring).

## Part 2 — J (Hybrid + MHD)

New rank-1 `DerivedQuantity`, name `"J"`, `VectorCentering::Elike`, computed by constructing `core::Ampere<GridLayout>{layout}` and calling it on `(B, out)` — the existing, already-shared-between-Hybrid-and-MHD curl-of-B operator (`src/core/numerics/ampere/ampere.hpp`), reused as-is rather than reimplemented.

- `MhdCurrentDensity` in `mhd_derived_quantities.hpp`, added to `makeMhdDerivedQuantities` via `registry.add<1>(...)`.
- `HybridCurrentDensity` in new `src/core/data/derived_quantity/hybrid_derived_quantities.hpp`, plus a new `makeHybridDerivedQuantities<State, GridLayout>()` factory function.
- Hybrid `ModelView` (`src/diagnostic/diagnostic_model_view.hpp`):
  - Currently constructs `derived_` via its default constructor (empty registry, unused). Change to initialize with `core::makeHybridDerivedQuantities<State_t, GridLayout>()`.
  - Add `derivedVecScratch_` member tagged `HybridQuantity::Vector::VecElike` and a `derivedVecScratch()` accessor, mirroring MHD's `derivedVecScratch_`/`derivedVecScratch()`. (No scalar scratch needed — no rank-0 Hybrid derived quantities are being added.)
- `ElectromagDiagnosticWriter` (H5, `types/electromag.hpp`): add a generic `"EM_J"` path (no `if constexpr` needed — both models now have the registry + vector scratch) parallel to `EM_B`/`EM_E`, following the same `quantities<1>()` dispatch already used for `EM_divB`.
- `ElectromagDiagnosticWriter` (vtkhdf, `vtk_types/electromag.hpp`): same `"EM_J"` branch added to `setup()`/`write()`, using `initTensorFieldFileLevel<1>` + `core::derived_vector_view` + `writer.writeTensorField<1>(...)`, no `if constexpr` guard (both models have the registry + vector scratch after the `ModelView` change above).
- Python `diagnostics.py`: add `"J"` to `ElectromagDiagnostics.em_quantities` → path `/EM_J`. Valid for both Hybrid and MHD sims (both writer dispatch tables already register `ElectromagDiagnosticWriter` for both model types, in both `h5writer.hpp` and `vtkh5_writer.hpp`).

## Part 3 — E (MHD only)

Hybrid's `getE()`/electromag `E` output is untouched (already a real stored field, already written).

New `MhdElectricField` in `mhd_derived_quantities.hpp`, rank-1, name `"E"`, `VectorCentering::Elike`. Point-wise; recomputes its own local J via `core::Ampere` internally (same operator `MhdCurrentDensity` uses, called independently — no cross-dependency on the `MhdCurrentDensity` derived-quantity object itself, just on the same shared `Ampere` numerics class). Since `Field`/`VecFieldT` are non-owning views, the local J is backed by three `std::vector<double>` sized/shaped like `out`'s own components (same `Elike` centering, so no extra ModelView-level scratch buffer is needed — this is a self-contained, per-call heap allocation, acceptable since diagnostics are not a hot path). Formula, evaluated per-point after J is filled:

```
E = -V×B                      (ideal, using bulk V, not an electron Ve)
  + (J×B)/rho                 (Hall term, standard generalized Ohm's law form)
  + eta·J                     (resistive)
  + hyper-resistive(J, B, rho) (constant or spatial mode, matching existing nu/hyper_mode semantics)
```

This mirrors the algebra in `upwind_constrained_transport.hpp`'s `resistive_contribution_`/`hyperresistive_contribution_` (plus the added Hall term), but without the Riemann-reconstruction machinery — this is an independent diagnostic quantity, not a replication of the CT scheme's discrete internals. Component projections (`GridLayout::project<BxToEx>` etc.) follow the same pattern already used in `ohm.hpp`.

Since `MhdElectricField` is registered through the same `makeMhdDerivedQuantities` registry as `MhdCurrentDensity`/`MhdDivB`, it needs no separate wiring, but the existing `"EM_E"` branch in both `ElectromagDiagnosticWriter`s currently calls `modelView().getE()` **unconditionally** for any model — for MHD that throws today (`"E not currently available in MHD diagnostics"`). That single branch must be split in two:
  - `if constexpr (solver::is_hybrid_model_v<Model_t>)`: unchanged, calls `getE()` (real stored field).
  - `if constexpr (solver::is_mhd_model_v<Model_t>)`: new, uses the derived-quantity registry exactly like `EM_J`/`EM_divB`.
- MHD `ModelView::getE()` (both `const`/non-`const` overloads in `diagnostic_model_view.hpp`) can then be deleted — it existed only to throw, and nothing else calls it once this split lands.

### New state plumbing required

`eta`, `nu`, `hyper_mode` are not currently reachable from `MhdState`/`ModelView` — they live in `ComputeFluxes`'s `GodunovInfo` (solver-level config, itself built via `core::OhmInfo::FROM(dict)` parsing `"resistivity"`/`"hyper_resistivity"`/`"hyper_mode"` keys). Mirroring the existing `gamma_` precedent on `MhdState`, and reusing `core::OhmInfo::FROM` rather than re-parsing:

- `src/core/models/mhd_state.hpp`: add a `core::OhmInfo const ohmInfo_` member, built via `core::OhmInfo::FROM(dict)` in the dict-based constructor (default-initialized to `{0.0, 0.0, HyperMode::constant}` in the name-only constructor), plus `eta()`/`nu()`/`hyperMode()` accessors mirroring `gamma()`. Requires including `core/numerics/ohm/ohm.hpp` (no circular dependency — that header only depends on generic grid/vecfield/initializer headers).
- `pyphare/pyphare/pharein/initialize/mhd.py`: add new dict entries under the `mhd_state` subtree, reusing the same key names (`"resistivity"`, `"hyper_resistivity"`, `"hyper_mode"`) that `OhmInfo::FROM` already expects elsewhere (`fv_method`, `constrained_transport`):
  ```python
  add_double("simulation/mhd_state/resistivity", sim.eta)
  add_double("simulation/mhd_state/hyper_resistivity", sim.nu)
  add_string("simulation/mhd_state/hyper_mode", sim.hyper_mode)
  ```
- `src/diagnostic/diagnostic_model_view.hpp`: MHD `ModelView`'s `makeMhdDerivedQuantities` call gains `eta()`/`nu()`/`hyperMode()` args alongside the existing `gamma()`.

## Testing

- Existing `test_mhd_derived_diagnostics.py` extended (or a sibling test added) to cover `EM_divB`, `EM_J`, `EM_E` paths for MHD, via **both** the phareh5 (legacy H5) and vtkhdf outputs (the existing test already checks divB via both, per its docstring — extend the same way); a Hybrid-side functional test for `EM_J` similarly covering both writer paths.
- `tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp` gets new unit tests for `MhdCurrentDensity` and `MhdElectricField` (e.g. J=0 for uniform B; E ideal-only check with resistivity/hyper-resistivity/Hall all zeroed).
- New unit test file for `HybridCurrentDensity` alongside existing hybrid derived-quantity tests (if any exist — otherwise co-located with the new header).

## Out of scope

- Replicating the CT scheme's exact discrete E (Riemann/reconstruction-based) — the diagnostic E is an independent, physically-consistent-but-simpler formula.
