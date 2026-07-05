# Derived-quantity interface for diagnostics

Date: 2026-07-05
Status: approved

## Problem

Post-processed dump quantities (velocity, pressure, divB, ...) are hand-wired today.
Each derived quantity requires touching: the python whitelist
(`pyphare/pharein/diagnostics.py`), the writer's
`createFiles`/`getDataSetInfo`/`initDataSets`/`compute`/`write` in **each** writer
stack (phareh5 `src/diagnostic/detail/types/`, vtkhdf
`src/diagnostic/detail/vtk_types/`), a dedicated model-owned buffer (`V_diag_`,
`P_diag_`), the `ModelView` accessor for it, and the model's
`registerResources`/`allocate`. Only MHD `V` and `P` exist in C++; `divB` is
python-side only. The computation logic is duplicated per writer stack
(`MHDDiagnosticWriter::compute` and `MhdFluidComputer`).

## Goal

One interface for "compute a scalar/vector/tensor field from the primary variables
of a state (hybrid or MHD), into a provided buffer, possibly time-dependent". At
init, a list of instances is registered corresponding to the requested
post-processed dump quantities. Writers become generic consumers: the computation
is defined once, shared by all diagnostic formats.

## Decisions (from brainstorming)

1. **Model scope**: model-agnostic core interface; MHD-only concrete
   implementations first (`V`, `P` ported; `divB` added).
2. **Rank handling**: single abstract class template on rank (0 = scalar,
   1 = vector, 2 = tensor later), via a per-rank traits struct. No fat interface
   (three overloads with runtime throws was considered and rejected). v1
   implements ranks 0 and 1 only.
3. **State access**: interface templated on `State` type (`MHDState`,
   `HybridState`); no runtime state adapter. Constants such as gamma are captured
   in the implementation's constructor.
4. **Buffers**: shared scratch, not per-quantity SAMRAI resources. Writer-owned
   raw memory block (`std::vector<double>`) grown lazily to the most demanding
   (rank, centering) case for the current patch; `Field`/`VecField` views are
   constructed per patch over that block. Compute and write are fused per patch.
   No ResourcesManager registration: scratch is transient and never crosses
   patches, so `setOnPatch` machinery is unnecessary. `Field(name, qty, ptr,
   dims)` supports this directly (same pattern as the test-side
   `UsableTensorField` fixtures).
5. **Centerings covered (v1)**: scalars: cell-centered, node-centered; vectors:
   cell-centered, E-like, B-like. No tensors in v1.
6. **Writer scope**: both stacks (phareh5 and vtkhdf) from the start, removing the
   duplicated per-stack compute logic.
7. **Python surface**: keep existing diagnostic types and file paths. User still
   writes `MHDDiagnostics(quantity="P")`; whitelist extended with `"divB"`.
   `/mhd/V`, `/mhd/P` paths unchanged; pharesee untouched.

## Architecture

### Core interface (`src/core/data/derived_quantity/`)

```cpp
template<typename State, typename GridLayout, std::size_t rank>
struct derived_traits;
// per-rank: out_t   (Field / VecField / SymTensorField from State's types)
//           centering_t (rank 0: cell|node; rank 1: cell|Elike|Blike)

template<typename State, typename GridLayout, std::size_t rank>
class DerivedQuantity
{
    using traits = derived_traits<State, GridLayout, rank>;

public:
    virtual ~DerivedQuantity() = default;
    virtual std::string name() const = 0;                 // "P", "divB", ...
    virtual typename traits::centering_t centering() const = 0;
    virtual void compute(State const& state, GridLayout const& layout,
                         typename traits::out_t& out, double time) const = 0;
};
```

Implementations fill the ghost box (as `eosEtotToPOnGhostBox` does today) so that
vtkhdf primal re-centering has valid ghost values.

v1 concrete implementations (MHD):
- `MhdVelocity` : rank 1, cell-centered — `V = rhoV / rho`.
- `MhdPressure` : rank 0, cell-centered, gamma captured at construction —
  `P = (gamma-1)(Etot - 1/2 rho v^2 - 1/2 B^2)`.
- `MhdDivB` : rank 0, cell-centered — divergence of the face-centered B.

These reuse/absorb the pointwise formulas of `ToPrimitiveConverter`.

### Registry and factory

```cpp
template<typename State, typename GridLayout>
struct DerivedQuantityRegistry
{
    // std::tuple of vectors of std::unique_ptr<DerivedQuantity<State, GridLayout, rank>>
    template<std::size_t rank> auto& quantities();
    // lookup: name -> (rank, index); miss means "raw state field, write directly"
};
```

A per-model factory `makeDerivedQuantities<Model>(requested_names, dict)`
constructs instances at init for the requested quantity names only. A requested
name that is neither a raw state field nor a known derived quantity throws at
init (fail fast, not at dump time). The registry lives in `ModelView`
(`src/diagnostic/diagnostic_model_view.hpp`), which is already the model-to-writer
bridge; writers query it.

### Scratch buffers

```cpp
class DerivedScratch   // one per writer instance; no SAMRAI involvement
{
    std::vector<double> mem_;   // grown lazily to worst case for current patch
public:
    template<std::size_t rank>
    auto view(/* centering_t<rank> */, GridLayout const&);  // Field/VecField over mem_
};
```

Sizing uses `layout.allocSize(qty)` on the ghost box. Requires generic
centering-only entries in the physical-quantity enum (cell/node scalar;
cell/E-like/B-like vector), wired into the `GridLayout` centering maps, because
`evalOnGhostBox`, `allocSize`, and writer re-centering all query centering via
`field.physicalQuantity()`.

### Writer integration

For each requested quantity, in both writer stacks:
- raw state field -> write directly (unchanged);
- otherwise registry lookup -> per patch: build scratch view, call
  `compute(state, layout, view, time)`, write the view.

The `compute(diag)` phase of `DiagnosticsManager::dump` becomes a no-op for these
writers; work moves into the per-patch write visit.

Deleted once `V`/`P` are ported: `V_diag_`, `P_diag_`, `tmpField_`, `tmpVec_` in
`MHDModel` and `ModelView` (including their `registerResources`/`allocate`
plumbing), `MHDDiagnosticWriter::compute` (phareh5), and the `MhdFluidComputer`
compute logic (vtkhdf).

### Python

`mhd_quantities` gains `"divB"`. `V` and `P` requests behave identically to
today, same file paths (`/mhd/V`, `/mhd/P`), so pharesee getters are untouched.
`heat_capacity_ratio` stays available in the diagnostic attributes; gamma for
`MhdPressure` is taken from the simulation dict at registry construction.

## Error handling

- Unknown quantity name: rejected by the python whitelist first; if it reaches
  C++, the registry factory throws at init.
- Scratch view requested for a (rank, centering) not in the v1 set: compile-time
  error where possible, otherwise throw at registry construction.

## Testing

- C++ unit tests (mirror the `to_primitive` converter tests): build an MHD state
  with analytic fields on a layout, run each computer, compare against the
  formula. For `MhdDivB`: construct B as a discrete curl of a vector potential,
  assert divB ~ 0 at machine precision.
- Functional: small MHD run dumping `V`, `P`, `divB` in both formats (phareh5 and
  vtkhdf); compare `divB` against python `_compute_divB`, and `V`/`P` against the
  existing `GetMHDV`/`GetMHDP` results from master.
