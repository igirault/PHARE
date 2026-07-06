# Generalize inflow BC prescription; remove adaptive-outflow and dt infra

Branch: `feature/outer-physical-bc`. Three independent changes to the outer physical
MHD boundary conditions.

## Task 1 — space-time functions for all inflow quantities

### Goal

Today only the magnetic field **B** at an inflow boundary may be prescribed as a
space-time callable (used for IMF turning); density, velocity, and pressure are plain
constants. Generalize so that **every** inflow quantity independently accepts either a
constant or a callable `f(x[,y[,z]], t)`, for both inflow BC types:
`super-magnetofast-inflow` and `free-pressure-inflow`.

`free-pressure-inflow` prescribes ρ, v, B (not pressure — pressure stays Neumann), so
"all quantities" there means ρ, v, B.

### Non-goals

- No change to outflow / reflective / open boundaries.
- Pressure remains Neumann for free-pressure-inflow (that is the defining feature of the
  type).
- No new BC classes: composition is done with wrapper `SpaceTimeFunction`s fed to the
  existing `FieldDirichletBoundaryCondition` (its scalar and per-component
  `SpaceTimeFunction` ctors already exist:
  `field_dirichlet_boundary_condition.hpp:54,61`).

### Batch-evaluation invariant (hard requirement)

`FieldDirichletBoundaryCondition::apply` already evaluates a prescribed function **once
per component per BC application**: it collects every ghost-box node coordinate into
per-dimension vectors and makes a single call `(*fn_[i])(coords..., ctx.time)` returning
a `Span<double>` (`field_dirichlet_boundary_condition.hpp:119-138`). All composition
helpers introduced below MUST preserve this: a composed function, when invoked with the
batch coordinate spans + time, calls each of its input functions exactly once (batch) and
combines the returned spans element-wise. This mirrors the existing `linComb2_`
(`boundary_factory.hpp:176-187`). No helper may evaluate its inputs point-by-point.

### C++ design (`src/core/boundary/boundary_factory.hpp`)

Prescribed scalars that are Dirichlet directly (ρ as `rho_bc`, pressure as `P_bc`):
- constant → constant `FieldDirichletBoundaryCondition` (today's fast path, no function
  eval),
- callable → `SpaceTimeFunction` Dirichlet.

Composed conservative quantities — momentum ρv and the motional field E = −v×B — cannot
be prescribed directly (the user gives primitives ρ, v, B). Rule:
- if **all** their inputs are constant → precompute the double/array (today's fast path,
  `vToRhoV` / `inflow_motional_E_`),
- if **any** input is a function → build a wrapper `SpaceTimeFunction` composing the
  inputs, wrapping the constant inputs locally as constant-functions.

New static helpers, next to `linComb2_` / `motionalEFunction_`:
- `constFunction_(double c)` → `SpaceTimeFunction` returning a constant-valued span sized
  to the coordinate batch (used to lift a constant input into the composition).
- `mulFunction_(f1, f2)` → element-wise product of two space-time functions (batch call
  each, multiply spans). Used for momentum component ρv_i = ρ·v_i.
- `crossFunction_(Vfns, Bfns)` → the three components of −V×B from two function-vectors,
  generalizing `motionalEFunction_` (which currently assumes v constant). Built from
  `mulFunction_` + span subtraction, batch-safe.
- a per-quantity parse helper that, given the dict key and its `<key>_is_function` flag
  (existing `isFunctionXYZ_` convention, `boundary_factory.hpp:168-172`), returns either a
  constant (`double` / `std::array<double,3>`) or a `SpaceTimeFunction` /
  `std::array<SpaceTimeFunction,3>`.

Both `register_super_magnetofast_inflow_conditions_` (`boundary_factory.hpp:334`) and
`register_free_pressure_inflow_conditions_` (`:462`) branch each prescribable quantity on
its function flag:
- ρ: `rho_bc` and the direct `rho` scalar condition become function-or-constant Dirichlet.
- v: feeds momentum (`rhoV` / `rhoV_bc`) and motional E; composed per the rule above.
- P (super-magnetofast only): `P_bc` and the direct pressure path become
  function-or-constant Dirichlet. (free-pressure keeps `P_bc` Neumann.)
- B: already function-capable; keep as-is, but its motional-E composition now goes through
  the generalized `crossFunction_` so a function v combines with a function/constant B.
- The `DivergenceFreeTransverseDirichlet` regrid fallback for B already accepts both
  constant and function forms; unchanged.

The energy term (`FieldTotalEnergyFromPressureBoundaryCondition`, "TEFP") already
reconstructs Etot from its ρ / ρv / B / P sub-BCs, so feeding it function sub-BCs needs no
change to TEFP itself.

### Python design

`pyphare/pyphare/pharein/simulation.py`:
- Remove the B-only `allow_time_function` gate (`simulation.py:315-328`) that rejects a
  callable for any quantity other than B.
- `_check_super_magnetofast_inflow_data` / `_check_free_pressure_inflow_data`: validate
  each quantity as float-or-callable (callable signature = existing B convention).

`pyphare/pyphare/pharein/initialize/general.py`:
- Generalize the B serializer (`_add_inflow_magnetic_field`, `:120-140`) into a
  per-quantity helper writing `<name>_is_function` (bool) plus either a plain double or a
  `SpaceTimeFunction` via the existing `_add_bc_value` → `addSpaceTimeFunction`
  (`:104-116`). Apply to density, velocity (per component), pressure, B.

### Tests

- C++ unit: extend the inflow BC tests to cover a function-prescribed ρ / v / P (assert
  ghost values match the analytic function at `ctx.time`, and that a constant given as a
  function equals the constant path).
- Python: extend `simulation_test.py` BC-validation tests — a callable is now accepted for
  ρ / v / P (was rejected).
- Functional smoke: a super-magnetofast-inflow case with a time-varying ρ (or v) held via
  the function path; assert the inflow value tracks the prescribed function.

## Task 2 — remove the adaptive-outflow boundary type

Adaptive-outflow (per-column choice between zero-gradient and Dirichlet pressure via a
fast-speed criterion) did not prove useful. Delete every reference:

- `src/core/boundary/boundary_defs.hpp:18,209` — `BoundaryType::AdaptiveOutflow` enum
  value + `"adaptive-outflow"` string.
- `src/core/boundary/boundary_factory.hpp` — dispatch case (`:126-134`) and
  `register_adaptive_outflow_conditions_` (`:644-708`).
- `src/core/numerics/boundary_condition/field_adaptive_outflow_pressure_boundary_condition.hpp`
  — delete the file.
- `FieldBoundaryConditionType::AdaptiveOutflowPressure` enum value
  (`field_boundary_condition.hpp`) + its `create` case in
  `field_boundary_condition_factory.hpp`.
- `pyphare/pyphare/pharein/simulation.py` — `_check_adaptive_outflow_data` (`:403`) and
  `"adaptive-outflow"` in `valid_bc_types` (`:456,524`).
- `tests/core/numerics/boundary_condition/test_field_boundary_conditions_adaptive_outflow.cpp`
  — delete file + its CMake registration.

## Task 3 — remove the dead dt-passing infrastructure

`dt` was threaded solver → messenger → patch-strategy → BC context for the
characteristic (NSCBC/LODI) outflow BCs, which were dropped from this PR. No live BC reads
`ctx.dt` (grep confirms zero uses in `boundary_condition/*.hpp`). Absolute time
(`ctx.time` / `newTime` / `fill_time`) stays — it drives the SpaceTimeFunction eval and
must not be touched.

Delete:
- `src/core/numerics/boundary_condition/boundary_condition_context.hpp:30` — `double dt;`
  member (+ its NSCBC/LODI comment lines 12-18).
- `src/amr/data/field/refine/field_refine_patch_strategy.hpp` — `setDt(double)` +
  member `dt_` (`:165-168,354`) and the ctx `dt` assignment (`:228-229`).
- `src/amr/messengers/mhd_messenger.hpp` — the `dt` parameter of `fillMomentsGhosts` and
  the `strat->setDt(dt)` call (`:478-489`).
- `src/amr/solvers/time_integrator/euler_using_computed_flux.hpp:33` — drop the `dt`
  argument at the `fillMomentsGhosts` call site.

`accessor_old` (previous-substage state in the context) is likewise only consumed by
adaptive-outflow; once Task 2 removes adaptive-outflow, check whether `accessor_old`
becomes dead too and remove it if so (verify no other BC reads it before deleting).

## Sequencing

Tasks are independent. Suggested order: **2** (removes adaptive-outflow, which also frees
`accessor_old`), then **3** (dt removal), then **1** (the feature). Each task builds and
its tests pass before moving on.

## Verification

- `tools/cmake.sh` build (dev config), then the outer-BC C++ unit tests + the pharein
  simulation Python tests green.
- A functional inflow smoke run: no NaN, prescribed inflow quantity tracks its function.
