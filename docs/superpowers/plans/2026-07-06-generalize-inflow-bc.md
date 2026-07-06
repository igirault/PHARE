# Generalize Inflow BC + Remove adaptive-outflow & dt-infra — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every MHD inflow quantity (density, velocity, pressure, B) be prescribed as a space-time function, and delete the unused adaptive-outflow boundary type and the dead dt-passing scaffolding.

**Architecture:** Inflow BC values reach the solver through `FieldDirichletBoundaryCondition`, which already batch-evaluates a `SpaceTimeFunction<dim>` once per component per apply. We generalize the boundary factory to build, per prescribable quantity, either a constant Dirichlet (fast path) or a function Dirichlet; conservative composites (momentum ρv, motional E = −v×B) are composed from primitive functions via new batch-safe helper functions in a small dedicated header. Python serializes each quantity as constant-or-function with an explicit `<name>_is_function` flag. Adaptive-outflow and the `dt` context field are excised.

**Tech Stack:** C++20 (header-only `core/`), SAMRAI, pybind11, pyphare (Python), GoogleTest, CTest, `uv`.

## Global Constraints

- This worktree has NO `tools/cmake.sh` (main-repo only) and its `build/` uses Makefiles with `test:BOOL=ON`. Build with **focused targets, threads ≤ 10** (12 OOMs): `cd build && uv run cmake --build . --target <name>... -j 10`. Do NOT run a full `cmake --build .` (all template permutations → OOM). Redirect build output to a log file, never pipe through `tail` (a prior run got signaled/truncated by the pipe).
- Run tests via `uv run ctest` from `build/` (never bare test binaries — embedded Python needs PYTHONPATH). Boundary C++ test targets: `test-boundary_condition`, `test-field-bc-<suffix>`. To validate a header that only heavy targets include (e.g. `boundary_factory.hpp` → AMR), build `test-messenger` / `test-models`.
- Callable convention (decided): every prescribable inflow value is a constant OR a space-time callable `f(x[, y[, z]], t)` returning a per-node scalar (scalar-in-space is broadcast). Vector quantities (velocity, B) are 3-sequences whose components are each independently float-or-callable. This is a BREAKING change to the old single time-only vector B callable `f(t) -> [Bx,By,Bz]`.
- `SpaceTimeFunction<dim>` = `std::function<std::shared_ptr<Span<double>>(<dim spatial spans>, double time)>` (`src/initializer/data_provider.hpp:84`). It is evaluated at absolute `ctx.time`.
- Batch-evaluation invariant (HARD): a composed function, when invoked, calls each input function exactly once over the whole coordinate batch and combines element-wise. Never evaluate inputs point-by-point. Mirror the existing `linComb2_` (`boundary_factory.hpp:176-187`).
- Keep `time`/`newTime`/`fill_time` everywhere; only `dt` is removed. Do NOT touch the `rhoOld_`/`rhoVold_`/`EtotOld_` messenger buffers — they are the old-time operand of SAMRAI `addTimeRefiner` coarse-fine temporal interpolation, not just characteristic-BC scratch.

---

### Task 1: Remove the adaptive-outflow boundary type

Delete every reference to adaptive-outflow. It compiles/links today, so removal is verified by a clean build plus the boundary test suite still passing (minus the deleted test).

**Files:**
- Modify: `src/core/boundary/boundary_defs.hpp:18` (enum), `:209` (string map)
- Modify: `src/core/boundary/boundary_factory.hpp:130-138` (dispatch case), delete `register_adaptive_outflow_conditions_` (`:632-712`)
- Delete: `src/core/numerics/boundary_condition/field_adaptive_outflow_pressure_boundary_condition.hpp`
- Modify: `src/core/numerics/boundary_condition/field_boundary_condition.hpp:28` (enum value `AdaptiveOutflowPressure`)
- Modify: `src/core/numerics/boundary_condition/field_boundary_condition_factory.hpp:119-130` (the `AdaptiveOutflowPressure` `create` branch + its `#include`)
- Modify: `pyphare/pyphare/pharein/simulation.py` — delete `_check_adaptive_outflow_data` (`:403-421`), remove `"adaptive-outflow"` from `valid_bc_types` (`:456`) and its dispatch `elif` (`:526-527`)
- Delete: `tests/core/numerics/boundary_condition/test_field_boundary_conditions_adaptive_outflow.cpp`
- Modify: `tests/core/numerics/boundary_condition/CMakeLists.txt:25` (drop `adaptive_outflow` from the foreach suffix list)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: removes `BoundaryType::AdaptiveOutflow`, `FieldBoundaryConditionType::AdaptiveOutflowPressure`, and the `"adaptive-outflow"` string. No later task references these.

- [ ] **Step 1: Delete the C++ BC class file and its test**

```bash
cd /var/home/girault/Documents/postdoc_lpp/dev/PHARE-outer-bc
git rm src/core/numerics/boundary_condition/field_adaptive_outflow_pressure_boundary_condition.hpp
git rm tests/core/numerics/boundary_condition/test_field_boundary_conditions_adaptive_outflow.cpp
```

- [ ] **Step 2: Remove the enum value + string map entry in `boundary_defs.hpp`**

Delete the `AdaptiveOutflow` enumerator (last in the `BoundaryType` enum — also remove the trailing comma now on `FixedPressureOutflow`) and the `{"adaptive-outflow", BoundaryType::AdaptiveOutflow},` line in `typeMap_`.

- [ ] **Step 3: Remove the factory dispatch case + register function in `boundary_factory.hpp`**

Delete the entire `case BoundaryType::AdaptiveOutflow:` block (`:130-138`) and the whole `register_adaptive_outflow_conditions_(...)` static method (`:632-712`).

- [ ] **Step 4: Remove the field-BC enum value + factory branch**

In `field_boundary_condition.hpp`, delete the `AdaptiveOutflowPressure` enumerator (`:28`) and fix the preceding comma. In `field_boundary_condition_factory.hpp`, delete the `else if constexpr (type == FieldBoundaryConditionType::AdaptiveOutflowPressure)` branch (`:119-130`) and the `#include ".../field_adaptive_outflow_pressure_boundary_condition.hpp"` at the top.

- [ ] **Step 5: Remove the Python validation + registration**

In `pyphare/pyphare/pharein/simulation.py`: delete the `_check_adaptive_outflow_data` function (`:403-421`), remove `"adaptive-outflow",` from the `valid_bc_types` tuple (`:456`), and delete the `elif bc_type == "adaptive-outflow":` dispatch pair (`:526-527`).

- [ ] **Step 6: Drop the test from CMake**

Edit `tests/core/numerics/boundary_condition/CMakeLists.txt:21-25`: remove `adaptive_outflow` from the `foreach(suffix ...)` list.

- [ ] **Step 7: Build**

Run: `bash tools/cmake.sh`
Expected: build completes, no reference-to-`AdaptiveOutflow` errors.

- [ ] **Step 8: Run the boundary test suite**

Run: `cd build && uv run ctest -R 'test-field-bc-|^test-boundary_condition$' --output-on-failure`
Expected: all remaining `test-field-bc-*` + `test-boundary_condition` PASS; no `adaptive_outflow` target exists.

- [ ] **Step 9: Run the pharein simulation Python test**

Run: `cd build && uv run ctest -R 'py3_test-pharein-simulation' --output-on-failure`
Expected: PASS (adaptive-outflow no longer referenced).

- [ ] **Step 10: Commit**

```bash
cd /var/home/girault/Documents/postdoc_lpp/dev/PHARE-outer-bc
git add -A
git commit -m "refactor: remove unused adaptive-outflow boundary type

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Remove the dead dt-passing infrastructure

`dt` was threaded solver → messenger → patch strategy → BC context for characteristic (NSCBC/LODI) outflow, which is not in this PR. No live BC reads `ctx.dt`. Remove the thread; keep `time`. Leave `accessor_old`/old-id-maps in place (separable follow-up; see note at end).

**Files:**
- Modify: `src/core/numerics/boundary_condition/boundary_condition_context.hpp:30` (drop `double dt;`)
- Modify: `src/amr/data/field/refine/field_refine_patch_strategy.hpp` — drop ctor init `dt_{0.0}` (`:132`), `setDt` (`:168`), member `dt_` (`:354`), and the `dt_` argument in the ctx aggregate (`:229`)
- Modify: `src/amr/messengers/mhd_messenger.hpp:478-487` — drop the `dt` parameter of `fillMomentsGhosts` and the three `setDt(dt)` loops
- Modify: `src/amr/solvers/time_integrator/euler_using_computed_flux.hpp:33` — drop the `dt` argument at the `fillMomentsGhosts` call
- Modify: `tests/core/numerics/boundary_condition/hybrid_bc_test_fixtures.hpp:76-78` and `mhd_bc_test_fixtures.hpp:76-78` — drop the `dt` param from `makeCtx` and the aggregate

**Interfaces:**
- Consumes: nothing.
- Produces: `BoundaryConditionContext{accessor_new, accessor_old, time}` (3 fields); `fillMomentsGhosts(state, level, fillTime)` (no dt); `makeCtx(acc, time=0.0)`.

- [ ] **Step 1: Update the two test fixtures first (they define the expected ctx shape)**

In BOTH `hybrid_bc_test_fixtures.hpp` and `mhd_bc_test_fixtures.hpp`, change `makeCtx`:

```cpp
// substage state pass the same accessor for both new and old. time=0 by default.
auto makeCtx(NullFieldAccessorT<FieldT> const& acc, double time = 0.0)
{
    return PHARE::core::BoundaryConditionContext<FieldT, HybridQuantity>{acc, acc, time};
}
```

(mhd fixture analogously: `BoundaryConditionContext<FieldMHD<dim>, MHDQuantity>{acc, acc, time}`.)

- [ ] **Step 2: Drop `dt` from the context struct**

In `boundary_condition_context.hpp`, delete the `double dt;` member (`:30`) and trim the doc comment sentence referring to "the substage time step" / "integrate over `dt`" (lines 12-18) to mention only `accessor_new`/`accessor_old`/`time`.

- [ ] **Step 3: Drop `dt_`/`setDt` from the patch strategy**

In `field_refine_patch_strategy.hpp`: remove `, dt_{0.0}` from the ctor init list (`:132`), the `void setDt(double const dt) { dt_ = dt; }` method (`:168`) and its doc comment (`:164-167`), the `double dt_;` member (`:354`), and change the ctx aggregate (`:228-229`) to:

```cpp
core::BoundaryConditionContext<field_type, physical_quantity_type> const ctx{
    fieldAccessor, fieldAccessorOld, fill_time};
```

- [ ] **Step 4: Drop `dt` from `fillMomentsGhosts`**

In `mhd_messenger.hpp`, replace the method head + the three setDt loops (`:478-488`) with:

```cpp
void fillMomentsGhosts(MHDStateT& state, level_t const& level, double const fillTime)
{
    setNaNsOnFieldGhosts(state.rho, level);
    setNaNsOnVecfieldGhosts(state.rhoV, level);
    setNaNsOnFieldGhosts(state.Etot, level);
    rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
    momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
    totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
}
```

- [ ] **Step 5: Drop the `dt` argument at the call site**

In `euler_using_computed_flux.hpp:33`, change `bc.fillMomentsGhosts(statenew, level, newTime, dt);` to `bc.fillMomentsGhosts(statenew, level, newTime);`. If `dt` is now unused in that scope, leave the surrounding signature untouched (other consumers may still need it) — only drop the argument to this one call. Verify with a grep that no other `fillMomentsGhosts(` call passes 4 args.

Run: `grep -rn "fillMomentsGhosts(" src/`
Expected: every call now passes exactly 3 args.

- [ ] **Step 6: Build**

Run: `bash tools/cmake.sh`
Expected: clean build; no `dt`/`setDt` errors.

- [ ] **Step 7: Run the boundary + messenger tests**

Run: `cd build && uv run ctest -R 'test-field-bc-|^test-boundary_condition$|test-messenger' --output-on-failure`
Expected: all PASS (fixtures updated, ctx shape consistent). If `test-messenger` needs PYTHONPATH, it is already run via ctest so this is handled.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove dead dt-passing BC infrastructure (was for characteristic BCs)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

**Note (out of scope):** `accessor_old`, the patch-strategy `old_scalar_ids_`/`old_vector_ids_`, the messenger `oldScalarIdMap_`/`oldVectorIdMap_`, and the `Pold_`/`Vold_`/`Jold_` buffers are now-dead characteristic-BC scaffolding (adaptive-outflow was their only consumer). They can be removed in a follow-up, but `rhoOld_`/`rhoVold_`/`EtotOld_` must stay (SAMRAI `addTimeRefiner` operands). Left untouched here to keep this task safe and bounded.

---

### Task 3: Batch-safe inflow-composition helpers (new header + unit test)

Add a small, directly unit-testable header of free functions that compose `SpaceTimeFunction`s: a constant lifted to a function, a product of two functions, and a linear combination of two products (the building block for −v×B). These supersede the private `linComb2_`/`motionalEFunction_` in the factory.

**Files:**
- Create: `src/core/boundary/boundary_inflow_compose.hpp`
- Create test: `tests/core/numerics/boundary_condition/test_field_boundary_conditions_inflow_compose.cpp`
- Modify: `tests/core/numerics/boundary_condition/CMakeLists.txt` (add `inflow_compose` suffix)

**Interfaces:**
- Consumes: `initializer::SpaceTimeFunction<dim>`, `PHARE::core::Span`/`VectorSpan` (from `data_provider.hpp` / span headers).
- Produces (namespace `PHARE::core::inflow_compose`):
  - `template<std::size_t dim> initializer::SpaceTimeFunction<dim> constFunction(double c);`
  - `template<std::size_t dim> initializer::SpaceTimeFunction<dim> mulFunction(initializer::SpaceTimeFunction<dim> f, initializer::SpaceTimeFunction<dim> g);`
  - `template<std::size_t dim> initializer::SpaceTimeFunction<dim> prodComb2(double a, initializer::SpaceTimeFunction<dim> f1, initializer::SpaceTimeFunction<dim> g1, double b, initializer::SpaceTimeFunction<dim> f2, initializer::SpaceTimeFunction<dim> g2);`
  - `template<std::size_t dim> std::array<initializer::SpaceTimeFunction<dim>, 3> negCrossFunction(std::array<initializer::SpaceTimeFunction<dim>, 3> const& V, std::array<initializer::SpaceTimeFunction<dim>, 3> const& B);` — returns the three components of E = −V×B.

- [ ] **Step 1: Write the failing unit test**

Create `tests/core/numerics/boundary_condition/test_field_boundary_conditions_inflow_compose.cpp`:

```cpp
#include "gtest/gtest.h"

#include "core/boundary/boundary_inflow_compose.hpp"
#include "initializer/data_provider.hpp"

#include <memory>
#include <vector>

using namespace PHARE::core::inflow_compose;
using PHARE::core::Span;
using PHARE::core::VectorSpan;
using PHARE::initializer::SpaceTimeFunction;

namespace
{
// A 1D space-time function f(x, t) built from a plain lambda over the batch.
SpaceTimeFunction<1> make1D(std::function<double(double, double)> f)
{
    return [f](std::vector<double> const& x, double t) -> std::shared_ptr<Span<double>> {
        std::vector<double> out(x.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            out[k] = f(x[k], t);
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}
} // namespace

TEST(InflowCompose, ConstFunctionBroadcastsToNodeCount)
{
    auto c = constFunction<1>(2.5);
    std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    auto s = c(x, 7.0); // time is ignored
    ASSERT_EQ(s->size(), x.size());
    for (std::size_t k = 0; k < x.size(); ++k)
        EXPECT_DOUBLE_EQ((*s)[k], 2.5);
}

TEST(InflowCompose, MulFunctionMultipliesElementwise)
{
    auto f = make1D([](double x, double) { return x; });
    auto g = make1D([](double, double t) { return t; });
    auto p = mulFunction<1>(f, g); // x * t
    std::vector<double> x{1.0, 2.0, 4.0};
    auto s = p(x, 3.0);
    ASSERT_EQ(s->size(), x.size());
    EXPECT_DOUBLE_EQ((*s)[0], 3.0);
    EXPECT_DOUBLE_EQ((*s)[1], 6.0);
    EXPECT_DOUBLE_EQ((*s)[2], 12.0);
}

TEST(InflowCompose, ProdComb2LinearlyCombinesTwoProducts)
{
    auto one = constFunction<1>(1.0);
    auto x   = make1D([](double x, double) { return x; });
    auto y2  = make1D([](double, double) { return 2.0; });
    // a*f1*g1 + b*f2*g2 = (-1)*x*1 + (3)*2*1 = -x + 6
    auto r = prodComb2<1>(-1.0, x, one, 3.0, y2, one);
    std::vector<double> xs{0.0, 1.0, 5.0};
    auto s = r(xs, 0.0);
    EXPECT_DOUBLE_EQ((*s)[0], 6.0);
    EXPECT_DOUBLE_EQ((*s)[1], 5.0);
    EXPECT_DOUBLE_EQ((*s)[2], 1.0);
}

TEST(InflowCompose, NegCrossMatchesMinusVCrossB)
{
    // Uniform V = (1,2,3), B = (4,5,6); E = -V x B = -(2*6-3*5, 3*4-1*6, 1*5-2*4)
    //                                             = -(-3, 6, -3) = (3, -6, 3)
    std::array<SpaceTimeFunction<1>, 3> V{constFunction<1>(1.0), constFunction<1>(2.0),
                                          constFunction<1>(3.0)};
    std::array<SpaceTimeFunction<1>, 3> B{constFunction<1>(4.0), constFunction<1>(5.0),
                                          constFunction<1>(6.0)};
    auto E = negCrossFunction<1>(V, B);
    std::vector<double> x{0.0, 1.0};
    auto ex = E[0](x, 0.0);
    auto ey = E[1](x, 0.0);
    auto ez = E[2](x, 0.0);
    EXPECT_DOUBLE_EQ((*ex)[0], 3.0);
    EXPECT_DOUBLE_EQ((*ey)[0], -6.0);
    EXPECT_DOUBLE_EQ((*ez)[0], 3.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

- [ ] **Step 2: Register the test target and confirm it fails to build (header missing)**

Edit `tests/core/numerics/boundary_condition/CMakeLists.txt:21-25`, append `inflow_compose` to the foreach suffix list.

Run: `bash tools/cmake.sh`
Expected: FAIL — `boundary_inflow_compose.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

Create `src/core/boundary/boundary_inflow_compose.hpp`:

```cpp
#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP
#define PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP

#include "initializer/data_provider.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace PHARE::core::inflow_compose
{
// All helpers preserve the batch-evaluation invariant: each input function is invoked
// exactly once over the whole coordinate batch, then combined element-wise. They mirror
// the shape of BoundaryFactory::linComb2_.

/** @brief A constant lifted into a space-time function: returns c at every node. Sized to
 * the first spatial coordinate span (the node count). Time is ignored. */
template<std::size_t dim>
initializer::SpaceTimeFunction<dim> constFunction(double const c)
{
    return [c](auto const&... args) -> std::shared_ptr<Span<double>> {
        auto const& first = std::get<0>(std::forward_as_tuple(args...));
        std::vector<double> out(first.size(), c);
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}

/** @brief Element-wise product f*g of two space-time functions. */
template<std::size_t dim>
initializer::SpaceTimeFunction<dim> mulFunction(initializer::SpaceTimeFunction<dim> f,
                                                initializer::SpaceTimeFunction<dim> g)
{
    return [f = std::move(f), g = std::move(g)](
               auto const&... args) -> std::shared_ptr<Span<double>> {
        auto sf = f(args...);
        auto sg = g(args...);
        std::vector<double> out(sf->size());
        for (std::size_t k = 0; k < out.size(); ++k)
            out[k] = (*sf)[k] * (*sg)[k];
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}

/** @brief a*f1*g1 + b*f2*g2, element-wise. Building block for -v x B components. */
template<std::size_t dim>
initializer::SpaceTimeFunction<dim>
prodComb2(double const a, initializer::SpaceTimeFunction<dim> f1,
          initializer::SpaceTimeFunction<dim> g1, double const b,
          initializer::SpaceTimeFunction<dim> f2, initializer::SpaceTimeFunction<dim> g2)
{
    return [a, b, f1 = std::move(f1), g1 = std::move(g1), f2 = std::move(f2),
            g2 = std::move(g2)](auto const&... args) -> std::shared_ptr<Span<double>> {
        auto s1 = f1(args...);
        auto h1 = g1(args...);
        auto s2 = f2(args...);
        auto h2 = g2(args...);
        std::vector<double> out(s1->size());
        for (std::size_t k = 0; k < out.size(); ++k)
            out[k] = a * (*s1)[k] * (*h1)[k] + b * (*s2)[k] * (*h2)[k];
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}

/** @brief The three components of the ideal motional field E = -V x B from two
 * function-vectors. Generalizes BoundaryFactory::motionalEFunction_ to a time-varying V.
 *   E_x = -(V_y B_z - V_z B_y) = -V_y B_z + V_z B_y
 *   E_y = -(V_z B_x - V_x B_z) = -V_z B_x + V_x B_z
 *   E_z = -(V_x B_y - V_y B_x) = -V_x B_y + V_y B_x */
template<std::size_t dim>
std::array<initializer::SpaceTimeFunction<dim>, 3>
negCrossFunction(std::array<initializer::SpaceTimeFunction<dim>, 3> const& V,
                 std::array<initializer::SpaceTimeFunction<dim>, 3> const& B)
{
    return {prodComb2<dim>(-1.0, V[1], B[2], 1.0, V[2], B[1]),
            prodComb2<dim>(-1.0, V[2], B[0], 1.0, V[0], B[2]),
            prodComb2<dim>(-1.0, V[0], B[1], 1.0, V[1], B[0])};
}

} // namespace PHARE::core::inflow_compose

#endif // PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP
```

Note: if `Span`/`VectorSpan` are not visible via `data_provider.hpp`, add the include that defines them (search: `grep -rn "class VectorSpan" src/`). Match whatever `boundary_factory.hpp` already includes for `VectorSpan`.

- [ ] **Step 4: Build and run the unit test**

Run: `bash tools/cmake.sh && cd build && uv run ctest -R 'test-field-bc-inflow_compose' --output-on-failure`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: batch-safe SpaceTimeFunction composition helpers for inflow BCs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Generalize the inflow factory registrations (C++)

Make `register_super_magnetofast_inflow_conditions_` and `register_free_pressure_inflow_conditions_` branch each prescribable quantity on its `<key>_is_function` flag: constant → today's fast Dirichlet; function → `SpaceTimeFunction` Dirichlet; composites (ρv, E) built with Task 3 helpers, lifting constant inputs via `constFunction`.

**Files:**
- Modify: `src/core/boundary/boundary_factory.hpp` — `#include "core/boundary/boundary_inflow_compose.hpp"`; add small per-quantity parse helpers; rewrite the two register functions (`:334-450`, `:462-551`); the constant-B `motionalEFunction_`/`linComb2_`/`inflow_motional_E_` may be dropped if fully superseded (keep `inflow_motional_E_` only if still used by the all-constant fast path — it is; keep it).

**Interfaces:**
- Consumes Task 3: `inflow_compose::{constFunction, mulFunction, negCrossFunction}`.
- Produces: no new external symbols; behavior change only. Verified via a functional run in Task 6 and the existing C++ Dirichlet function test (mechanism) already green.

- [ ] **Step 1: Add per-quantity parse helpers next to `isFunctionXYZ_`**

In `boundary_factory.hpp`, after `isFunctionXYZ_` (`:172`), add helpers that read a scalar or a 3-vector as EITHER a constant OR a `SpaceTimeFunction`, plus a lifter to a function array. Use the existing `parseDimXYZType<T,3>` for both `double` and `_space_time_function` element types (already used for B).

```cpp
// Lift a constant 3-vector to three constant-valued space-time functions.
static std::array<_space_time_function, 3> liftConst3_(std::array<double, 3> const& c)
{
    return {inflow_compose::constFunction<dimension>(c[0]),
            inflow_compose::constFunction<dimension>(c[1]),
            inflow_compose::constFunction<dimension>(c[2])};
}

// A prescribable 3-vector as functions, whether given as constants or functions.
static std::array<_space_time_function, 3>
vecAsFunctions_(initializer::PHAREDict const& data, std::string const& key)
{
    if (isFunctionXYZ_(data, key))
        return initializer::parseDimXYZType<_space_time_function, 3>(data, key);
    return liftConst3_(initializer::parseDimXYZType<double, 3>(data, key));
}
```

For a scalar (`density`, `pressure`) the register functions branch inline on `isFunctionXYZ_(data, key)` and either read `data[key].to<double>()` or `data[key].to<_space_time_function>()` (confirm the scalar accessor: the serializer writes a single `SpaceTimeFunction` at `data/<key>` when callable — see Task 5; a scalar function is stored directly, not under `/x/y/z`).

- [ ] **Step 2: Rewrite `register_super_magnetofast_inflow_conditions_`**

Replace the body's constant-only reads and the `BisFn`-only E/B branch with per-quantity branching. Key structure (rho shown for scalars; momentum & E composed):

```cpp
using STF = _space_time_function;
bool const rhoIsFn = isFunctionXYZ_(data, "density");
bool const pIsFn   = isFunctionXYZ_(data, "pressure");
bool const vIsFn   = isFunctionXYZ_(data, "velocity");
bool const bIsFn   = isFunctionXYZ_(data, "B");

// --- scalar Dirichlet sub/main BCs: rho ---
auto rho_bc = rhoIsFn
    ? std::shared_ptr<ScalarBcType>{FieldBoundaryConditionFactory::create<
          FieldBoundaryConditionType::Dirichlet, FieldT, GridLayoutT>(
          data["density"].template to<STF>())}
    : std::shared_ptr<ScalarBcType>{FieldBoundaryConditionFactory::create<
          FieldBoundaryConditionType::Dirichlet, FieldT, GridLayoutT>(
          data["density"].template to<double>())};

// --- pressure P_bc: constant or function ---
auto P_bc = pIsFn
    ? std::shared_ptr<ScalarBcType>{FieldBoundaryConditionFactory::create<
          FieldBoundaryConditionType::Dirichlet, FieldT, GridLayoutT>(
          data["pressure"].template to<STF>())}
    : std::shared_ptr<ScalarBcType>{FieldBoundaryConditionFactory::create<
          FieldBoundaryConditionType::Dirichlet, FieldT, GridLayoutT>(
          data["pressure"].template to<double>())};

// --- momentum rhoV: constant fast-path if rho and v both constant, else composed ---
std::shared_ptr<VectorBcType> rhoV_bc;
if (!rhoIsFn && !vIsFn)
{
    auto const rho  = data["density"].template to<double>();
    auto const v    = initializer::parseDimXYZType<double, 3>(data, "velocity");
    auto const rhoV = vToRhoV(rho, v);
    rhoV_bc = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
        FieldBoundaryConditionType::Dirichlet, VecFieldT, GridLayoutT>(rhoV)};
}
else
{
    auto const rhoFn = rhoIsFn ? data["density"].template to<STF>()
                               : inflow_compose::constFunction<dimension>(
                                     data["density"].template to<double>());
    auto const vFns  = vecAsFunctions_(data, "velocity");
    std::array<STF, 3> rhoVfns{inflow_compose::mulFunction<dimension>(rhoFn, vFns[0]),
                               inflow_compose::mulFunction<dimension>(rhoFn, vFns[1]),
                               inflow_compose::mulFunction<dimension>(rhoFn, vFns[2])};
    rhoV_bc = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
        FieldBoundaryConditionType::Dirichlet, VecFieldT, GridLayoutT>(rhoVfns)};
}

// --- B sub-BC + motional E + regrid fallback ---
std::shared_ptr<VectorBcType> B_bc;
bool const eNeedsFn = vIsFn || bIsFn;
std::array<double, 3> E{};
std::array<STF, 3> Efns{};
if (!eNeedsFn)
{
    auto const v = initializer::parseDimXYZType<double, 3>(data, "velocity");
    auto const B = initializer::parseDimXYZType<double, 3>(data, "B");
    E    = inflow_motional_E_(v, B);
    B_bc = make_inflow_B_bc_<VecFieldT, VectorBcType>(B);
    boundary->template registerRegridFallbackCondition<
        FieldBoundaryConditionType::DivergenceFreeTransverseDirichlet>(
        PhysicalQuantityT::Vector::B, B);
}
else
{
    auto const vFns = vecAsFunctions_(data, "velocity");
    auto const Bfns = vecAsFunctions_(data, "B");
    Efns = inflow_compose::negCrossFunction<dimension>(vFns, Bfns);
    B_bc = make_inflow_B_bc_<VecFieldT, VectorBcType>(Bfns);
    boundary->template registerRegridFallbackCondition<
        FieldBoundaryConditionType::DivergenceFreeTransverseDirichlet>(
        PhysicalQuantityT::Vector::B, Bfns);
}
```

Then the scalar loop registers rho as `Dirichlet(rho_bc-equivalent)` — i.e. reuse the same const/function choice as `rho_bc` (register the direct rho condition with the constant or the function), Etot as `TotalEnergyFromPressure(rho_bc, rhoV_bc, B_bc, P_bc, thermo)`, others `None`. The vector loop registers `rhoV` via the const-or-function value used for `rhoV_bc`, `E` as `Dirichlet(E)` or `Dirichlet(Efns)` per `eNeedsFn`, `B` as `None`.

To avoid duplicating the const/function ternary for the direct rho and rhoV registrations, register those direct conditions by passing the already-built sub-BC choice: register rho with `data["density"].to<double>()`/`to<STF>()` mirroring `rhoIsFn`, and rhoV with the same `rhoV`/`rhoVfns` value computed above (hoist `rhoV`/`rhoVfns` out of the `if` so both the sub-BC and the direct condition use them).

- [ ] **Step 3: Rewrite `register_free_pressure_inflow_conditions_` the same way**

Identical to Step 2 except: there is no prescribed pressure — `P_bc` stays `Neumann` (unchanged, `:502-504`), and there is no direct pressure branch. rho, velocity→momentum, B→E, and the energy compound BC use the same const/function branching. The `E` vector condition is `Dirichlet(E)`/`Dirichlet(Efns)`.

- [ ] **Step 4: Include the compose header + drop the now-unused constant helpers**

Add `#include "core/boundary/boundary_inflow_compose.hpp"` near the top of `boundary_factory.hpp`. Remove `linComb2_` (`:176-187`) and `motionalEFunction_` (`:192-198`) if no longer referenced (the function path now uses `negCrossFunction`); keep `inflow_motional_E_` (still used by the all-constant fast path). Confirm with grep before deleting.

Run: `grep -n "linComb2_\|motionalEFunction_" src/core/boundary/boundary_factory.hpp`
Expected: no remaining references before deletion.

- [ ] **Step 5: Build**

Run: `bash tools/cmake.sh`
Expected: clean build.

- [ ] **Step 6: Run the boundary + model + messenger tests**

Run: `cd build && uv run ctest -R 'test-field-bc-|^test-boundary_condition$|test-models|test-messenger' --output-on-failure`
Expected: all PASS (behavior unchanged for the all-constant case; the constant IMF/inflow cases still route through the fast path).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: per-quantity space-time inflow prescription in MHD boundary factory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Generalize the Python inflow serialization + validation

Every inflow quantity accepts a constant or a space-time callable `f(x[,y[,z]],t)`, serialized with a `<name>_is_function` flag mirroring the existing B path; the B-only time-callable form is removed.

**Files:**
- Modify: `pyphare/pyphare/pharein/simulation.py` — replace `_normalize_B`/`_normalize_inflow_magnetic_field` callable handling; update `_check_inflow_data` and `_check_free_pressure_inflow_data` to accept per-quantity callables; `_normalize_inflow_velocity` to allow per-component callables
- Modify: `pyphare/pyphare/pharein/initialize/general.py` — generalize `_add_inflow_magnetic_field` into a per-quantity serializer writing `<name>_is_function` + double-or-SpaceTimeFunction; call it for density, velocity, pressure, B

**Interfaces:**
- Consumes: `_add_bc_value` (`general.py:114`), `addSpaceTimeFunction` (`:104`), `add_bool`.
- Produces: dict layout consumed by Task 4's factory: for a scalar quantity `<key>`, `data/<key>_is_function` (bool) + either `data/<key>` (double) or `data/<key>` (SpaceTimeFunction). For a vector quantity, `data/<key>_is_function` (bool) + `data/<key>/{x,y,z}` each double-or-SpaceTimeFunction.

- [ ] **Step 1: Simplify validation to per-quantity float-or-callable**

In `simulation.py`, replace `_normalize_B` (`:311-346`) and drop its `allow_time_function` machinery. New helpers validate a value as float-or-callable (a callable is accepted as-is; a constant coerced to float). Vectors validate each of 3 components independently.

```python
def _is_inflow_callable(v):
    return callable(v)


def _normalize_inflow_scalar(location, key, val, positive=False):
    """A prescribable inflow scalar: a float or a space-time callable f(x[,y[,z]],t)."""
    if callable(val):
        return val
    if not isinstance(val, (int, float)) or (positive and val <= 0):
        raise ValueError(
            f"'{key}' at inflow boundary '{location}' must be a "
            f"{'positive ' if positive else ''}scalar or a callable f(x,y,z,t), got {val!r}"
        )
    return float(val)


def _normalize_inflow_vector(location, key, vec):
    """A prescribable inflow 3-vector: each component a float or a callable f(x[,y[,z]],t)."""
    try:
        comps = list(vec)
    except TypeError:
        raise TypeError(
            f"'{key}' at inflow boundary '{location}' must be a 3-vector (each component a "
            f"float or a callable f(x,y,z,t)), got {vec!r}"
        )
    if len(comps) != 3:
        raise ValueError(
            f"'{key}' at inflow boundary '{location}' must be a 3-vector, "
            f"got a {len(comps)}-element sequence"
        )
    return [
        c if callable(c) else _normalize_inflow_scalar(location, f"{key}[{i}]", c)
        for i, c in enumerate(comps)
    ]
```

Delete `_normalize_B` and the old `_normalize_inflow_magnetic_field`/`_normalize_inflow_velocity`; replace their uses below.

- [ ] **Step 2: Update `_check_inflow_data` (super-magnetofast)**

```python
def _check_inflow_data(location, bc):
    """Validate the 'data' sub-dict for a super-magnetofast-inflow BC.

    density, pressure, velocity, and B may each be a constant or a space-time callable
    f(x[, y[, z]], t) (per component for velocity/B)."""
    data = bc.get("data", {})
    for key in ("density", "pressure", "velocity", "B"):
        if key not in data:
            raise KeyError(f"Inflow BC at '{location}' requires '{key}' inside 'data'")
    data["density"] = _normalize_inflow_scalar(location, "density", data["density"], positive=True)
    data["pressure"] = _normalize_inflow_scalar(location, "pressure", data["pressure"], positive=True)
    data["velocity"] = _normalize_inflow_vector(location, "velocity", data["velocity"])
    data["B"] = _normalize_inflow_vector(location, "B", data["B"])
    bc["data"] = data
```

(Note: `positive=True` only constrains constants; a callable is not range-checked.)

- [ ] **Step 3: Update `_check_free_pressure_inflow_data`**

```python
def _check_free_pressure_inflow_data(location, bc):
    """Free-pressure inflow: density, velocity, B prescribable (constant or callable);
    pressure is not prescribed (Neumann)."""
    data = bc.get("data", {})
    for key in ("density", "velocity", "B"):
        if key not in data:
            raise KeyError(
                f"Free-pressure inflow BC at '{location}' requires '{key}' inside 'data'"
            )
    data["density"] = _normalize_inflow_scalar(location, "density", data["density"], positive=True)
    data["velocity"] = _normalize_inflow_vector(location, "velocity", data["velocity"])
    data["B"] = _normalize_inflow_vector(location, "B", data["B"])
    bc["data"] = data
```

- [ ] **Step 4: Generalize the serializer in `general.py`**

Replace `_add_inflow_magnetic_field` (`:121-143`) with per-quantity serializers and call them for all inflow quantities:

```python
def _add_inflow_scalar(bc_path, data, key, ndim):
    """Serialise a prescribable inflow scalar: a constant double or a space-time function,
    tagged by '<key>_is_function'."""
    val = data[key]
    add_bool(f"{bc_path}/data/{key}_is_function", callable(val))
    _add_bc_value(f"{bc_path}/data/{key}", val, ndim)


def _add_inflow_vector(bc_path, data, key, ndim):
    """Serialise a prescribable inflow 3-vector: each component a constant double or a
    space-time function, tagged by '<key>_is_function' (true if any component is callable)."""
    comps = data[key]
    add_bool(f"{bc_path}/data/{key}_is_function", any(callable(c) for c in comps))
    for axis, c in zip("xyz", comps):
        _add_bc_value(f"{bc_path}/data/{key}/{axis}", c, ndim)
```

IMPORTANT: `parseDimXYZType<_space_time_function,3>` on the C++ side expects EVERY component to be a function when `<key>_is_function` is true. So when a vector is tagged as a function (any component callable), lift the constant components to constant callables before serialization so all three slots hold functions:

```python
def _add_inflow_vector(bc_path, data, key, ndim):
    comps = list(data[key])
    is_fn = any(callable(c) for c in comps)
    add_bool(f"{bc_path}/data/{key}_is_function", is_fn)
    if is_fn:
        comps = [c if callable(c) else (lambda *a, _c=float(c): _c) for c in comps]
    for axis, c in zip("xyz", comps):
        _add_bc_value(f"{bc_path}/data/{key}/{axis}", c, ndim)
```

The lifted constant `lambda *a, _c: _c` returns a scalar; `space_time_fn_wrapper` broadcasts it to the node count (`general.py:72-73`). (C++ still lifts constant scalars for composition too, but this keeps the vector-function dict slot uniform.)

Then, in `populateDict`'s inflow-BC serialization block (search where `_add_inflow_magnetic_field` was called), call:
- super-magnetofast: `_add_inflow_scalar(bc_path, data, "density", ndim)`, `_add_inflow_scalar(..., "pressure", ...)`, `_add_inflow_vector(..., "velocity", ...)`, `_add_inflow_vector(..., "B", ...)`.
- free-pressure: same minus `"pressure"`.

Locate that block: `grep -n "_add_inflow_magnetic_field\|super-magnetofast\|free-pressure" pyphare/pyphare/pharein/initialize/general.py` and replace the per-quantity writes accordingly. Density/velocity/pressure were previously written as plain doubles nearby — replace those writes with the new helpers so the `<key>_is_function` flag is always present.

- [ ] **Step 5: Verify the scalar-function C++ read path**

The C++ scalar branch (Task 4) reads `data["density"].to<_space_time_function>()`. Confirm `_add_bc_value` on a callable stores a single `SpaceTimeFunction` retrievable as `_space_time_function` at `data/density`. If the dict stores it under a nested key, align the C++ read. (It calls `addSpaceTimeFunction(path, fn, ndim)` at exactly `path=data/density`, so `data["density"].to<_space_time_function>()` is correct.)

- [ ] **Step 6: Run the pharein simulation test**

Run: `cd build && uv run ctest -R 'py3_test-pharein-simulation' --output-on-failure`
Expected: PASS (existing constant-inflow cases still validate/serialize).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: per-quantity constant-or-callable inflow prescription in pyphare

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end tests — Python validation + functional smoke

Add validation tests proving a callable is now accepted for ρ/v/P, and a functional MHD smoke run proving a time-varying inflow quantity actually drives the boundary.

**Files:**
- Modify: `pyphare/.../tests/.../simulation_test.py` (locate the existing BC-validation tests: `grep -rn "super-magnetofast-inflow\|free-pressure-inflow" pyphare` under the test dir)
- Create or extend a functional test under `tests/functional/` mirroring an existing MHD inflow smoke (e.g. reuse the `mhd_harris_with_boundaries` or `smoke_shock` pattern from the branch); assert the inflow density tracks a prescribed `f(t)`

**Interfaces:**
- Consumes: the full stack from Tasks 4-5.
- Produces: regression coverage. Terminal deliverable.

- [ ] **Step 1: Add Python validation tests (callable now accepted)**

In `simulation_test.py`, add tests asserting that a super-magnetofast-inflow with a callable density (and a callable velocity component, and a callable pressure) validates without raising, and that a free-pressure-inflow with a callable density validates. Model them on the existing BC-validation tests in that file. Example shape:

```python
def test_inflow_accepts_callable_density(self):
    ph.global_vars.sim = None
    inflow = {
        "type": "super-magnetofast-inflow",
        "data": {
            "density": lambda x, t: 1.0 + 0.0 * x + 0.0 * t,
            "pressure": 0.5,
            "velocity": [2.0, 0.0, 0.0],
            "B": [1.0, 0.0, 0.0],
        },
    }
    sim = ph.Simulation(
        # ... minimal valid MHD sim kwargs used by the sibling BC tests ...
        boundary_types=["physical", "periodic"][: /*ndim*/],
        boundary_conditions={"xlower": inflow, "xupper": {"type": "fixed-pressure-outflow", "data": {"pressure": 0.5}}},
    )
    self.assertTrue("xlower" in sim.boundary_conditions)
    ph.global_vars.sim = None
```

Fill in the minimal Simulation kwargs by copying the setup of the nearest existing inflow BC-validation test in the same file (do not invent kwargs).

- [ ] **Step 2: Run the Python validation tests**

Run: `cd build && uv run ctest -R 'py3_test-pharein-simulation' --output-on-failure`
Expected: PASS including the new tests.

- [ ] **Step 3: Write a functional smoke with a time-varying inflow density**

Copy the branch's existing MHD inflow smoke script (find it: `grep -rln "super-magnetofast-inflow" tests/functional`; the branch also has `scratchpad/smoke_shock.py`). Change the inflow `density` to a callable, e.g. `density=lambda x, t: 1.0 + 0.1 * math.sin(t)` (uniform in space), run ~30 steps, and read back the inflow-edge density from the raw HDF5 (per memory: never count via pharesee for face fields; for cell-centered ρ pharesee is fine) asserting it tracks `1.0 + 0.1*sin(t)` at the boundary column within tolerance and is NaN-free.

Register it via the appropriate CMake macro (`phare_mpi_python3_exec` / `add_no_mpi_python3_test`) mirroring the sibling functional MHD tests, at the same exec level as `mhd_harris_with_boundaries` (level 11) if it needs MPI, else a no-mpi level.

- [ ] **Step 4: Run the functional smoke**

Follow the harness run conventions (sandbox-off, venv python, `pkill -f mpirun || true` before running, one run at a time — see the running-PHARE-sims notes). Run the registered test:

Run: `cd build && uv run ctest -R '<new_functional_test_name>' --output-on-failure`
Expected: PASS — no NaN, inflow ρ tracks the prescribed function.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: space-time inflow prescription — validation + functional smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1↔spec §Task2 (adaptive removal); Task 2↔spec §Task3 (dt removal, scoped to dt only per the buffer-disentangling risk found during planning); Tasks 3-5↔spec §Task1 (helpers, factory, Python); Task 6↔spec §Tests. The `accessor_old` follow-up is explicitly deferred with rationale.
- **Batch invariant:** enforced in Task 3 helpers (one call per input per apply) and tested (`InflowCompose.*`).
- **Type consistency:** `SpaceTimeFunction<dim>`, `_space_time_function` alias, `constFunction/mulFunction/prodComb2/negCrossFunction` names match across Tasks 3-4. Dict layout (`<key>_is_function` + double/SpaceTimeFunction) matches between Task 5 (writer) and Task 4 (reader).
- **Open verification points flagged inline:** the scalar `SpaceTimeFunction` dict read (Task 5 Step 5), the `VectorSpan` include (Task 3 Step 3), and the exact `populateDict` inflow block (Task 5 Step 4) are all marked to confirm against the tree during execution rather than assumed.
