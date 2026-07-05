# Derived-Quantity Diagnostics Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generic `DerivedQuantity<State, GridLayout, rank>` interface computing post-processed fields (V, P, divB) from primary variables into transient scratch buffers, consumed by both diagnostic writer stacks (phareh5 + vtkhdf), replacing the hand-wired `V_diag_`/`P_diag_` machinery.

**Architecture:** Abstract class template on rank (0 scalar, 1 vector) with per-rank traits; a per-model registry living in `ModelView`; writer-owned non-SAMRAI scratch memory viewed through `Field`/`VecField` per patch; compute+write fused per patch. Centering-only physical quantities are enum-value aliases (`ScalarCellCentered = P` style, following the `ScalarAllPrimal` precedent) so no GridLayout changes are needed.

**Tech Stack:** C++20 header-only templates in `src/core/data/derived_quantity/`, gtest unit tests, python functional test via ctest.

**Spec:** `docs/superpowers/specs/2026-07-05-derived-quantities-design.md`

## Global Constraints

- Build with `./tools/cmake.sh` from repo root (NOT bare `uv run cmake --build build`). Builds are long; expect minutes.
- devMode is ON: `-Werror -Wall -Wextra`. Name-omit unused parameters (`double /*time*/`).
- Run tests from `build/`: `uv run ctest -R '<regex>' --output-on-failure`. Python tests appear as `py3_<name>`; embedded-python tests MUST run via ctest (never the bare binary) for PYTHONPATH.
- Register tests via macros from `res/cmake/def.cmake` (`add_no_mpi_phare_test`, `phare_python3_exec`), never raw `add_test`.
- clang-format each created/modified file: `clang-format -i <file>` (4-space indent, 100 col, SortIncludes: false).
- Commit after each task with a `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- If a run leaks `mpirun` (MHD runs don't always exit on caught exceptions): `pkill -f mpirun || true` before rerunning.

---

### Task 1: Centering enums + generic quantity aliases

**Files:**
- Create: `src/core/data/derived_quantity/centering.hpp`
- Modify: `src/core/mhd/mhd_quantities.hpp` (Scalar enum ~line 56, Vector enum line 58)
- Modify: `src/core/hybrid/hybrid_quantities.hpp` (Scalar enum ~line 37, Vector enum line 39)
- Create: `tests/core/data/derived_quantity/test_centering.cpp`
- Create: `tests/core/data/derived_quantity/CMakeLists.txt`
- Modify: `tests/core/data/CMakeLists.txt` (add_subdirectory; check the file — if subdirs are globbed automatically, skip)

**Interfaces:**
- Produces: `PHARE::core::ScalarCentering{cell,node}`, `PHARE::core::VectorCentering{cell,Elike,Blike}`, `scalar_qty<PhysicalQuantity>(ScalarCentering) -> PhysicalQuantity::Scalar`, `vector_qty<PhysicalQuantity>(VectorCentering) -> PhysicalQuantity::Vector`. Enum aliases `MHDQuantity::Scalar::{ScalarCellCentered,ScalarNodeCentered}`, `MHDQuantity::Vector::{VecCellCentered,VecElike,VecBlike}`, `HybridQuantity::Scalar::ScalarNodeCentered`, `HybridQuantity::Vector::{VecElike,VecBlike}`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/core/data/derived_quantity/test_centering.cpp
#include "core/data/derived_quantity/centering.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/hybrid/hybrid_quantities.hpp"

#include "gtest/gtest.h"

using namespace PHARE::core;

TEST(DerivedCentering, mhdAliasesResolveToExistingQuantities)
{
    EXPECT_EQ(scalar_qty<MHDQuantity>(ScalarCentering::cell), MHDQuantity::Scalar::P);
    EXPECT_EQ(scalar_qty<MHDQuantity>(ScalarCentering::node), MHDQuantity::Scalar::ScalarAllPrimal);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::cell), MHDQuantity::Vector::V);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::Elike), MHDQuantity::Vector::E);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::Blike), MHDQuantity::Vector::B);
}

TEST(DerivedCentering, hybridAliasesResolveToExistingQuantities)
{
    EXPECT_EQ(scalar_qty<HybridQuantity>(ScalarCentering::node), HybridQuantity::Scalar::rho);
    EXPECT_EQ(vector_qty<HybridQuantity>(VectorCentering::Elike), HybridQuantity::Vector::E);
    EXPECT_EQ(vector_qty<HybridQuantity>(VectorCentering::Blike), HybridQuantity::Vector::B);
}

TEST(DerivedCentering, hybridCellCenteredThrows)
{
    EXPECT_THROW(scalar_qty<HybridQuantity>(ScalarCentering::cell), std::runtime_error);
    EXPECT_THROW(vector_qty<HybridQuantity>(VectorCentering::cell), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

```cmake
# tests/core/data/derived_quantity/CMakeLists.txt
cmake_minimum_required (VERSION 3.20.1)

project(test-derived-quantity)

set(SOURCES test_centering.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  phare_core
  phare_initializer
  ${GTEST_LIBS})

add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})
```

Check `tests/core/data/CMakeLists.txt` (or wherever `mhd_state` gets added — `grep -rn mhd_state tests/CMakeLists.txt tests/core/CMakeLists.txt tests/core/data/CMakeLists.txt`) and register `derived_quantity` the same way.

- [ ] **Step 2: Run test to verify it fails**

Run: `./tools/cmake.sh` (or configure-only first build), expect compile FAILURE: `centering.hpp: No such file`.

- [ ] **Step 3: Write implementation**

Enum aliases — in `src/core/mhd/mhd_quantities.hpp`, inside `enum class Scalar`, after `count`:

```cpp
        count,

        // centering-only aliases for derived-quantity scratch fields (share values
        // with existing quantities of the same centering; see ScalarAllPrimal)
        ScalarCellCentered = P,
        ScalarNodeCentered = ScalarAllPrimal
```

Inside `enum class Vector` (line 58):

```cpp
    enum class Vector {
        V, B, rhoV, E, J, VecFlux_x, VecFlux_y, VecFlux_z, VecAllPrimal,
        // centering-only aliases for derived-quantity scratch fields
        VecCellCentered = V,
        VecElike = E,
        VecBlike = B
    };
```

In `src/core/hybrid/hybrid_quantities.hpp`, inside `enum class Scalar` after `count`:

```cpp
        count,

        // centering-only alias for derived-quantity scratch fields (moments are all-primal)
        ScalarNodeCentered = rho
```

Inside `enum class Vector`:

```cpp
    enum class Vector { B, E, J, V, VecElike = E, VecBlike = B };
```

New header:

```cpp
// src/core/data/derived_quantity/centering.hpp
#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_CENTERING_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_CENTERING_HPP

#include "core/mhd/mhd_quantities.hpp"
#include "core/hybrid/hybrid_quantities.hpp"

#include <stdexcept>
#include <type_traits>

namespace PHARE::core
{
enum class ScalarCentering { cell, node };
enum class VectorCentering { cell, Elike, Blike };


template<typename PhysicalQuantity>
auto scalar_qty(ScalarCentering const centering)
{
    if constexpr (std::is_same_v<PhysicalQuantity, MHDQuantity>)
        return centering == ScalarCentering::cell ? MHDQuantity::Scalar::ScalarCellCentered
                                                  : MHDQuantity::Scalar::ScalarNodeCentered;
    else
    {
        if (centering == ScalarCentering::cell)
            throw std::runtime_error("no cell-centered scalar quantity for hybrid");
        return HybridQuantity::Scalar::ScalarNodeCentered;
    }
}


template<typename PhysicalQuantity>
auto vector_qty(VectorCentering const centering)
{
    using Vector = typename PhysicalQuantity::Vector;

    if (centering == VectorCentering::Elike)
        return Vector::VecElike;
    if (centering == VectorCentering::Blike)
        return Vector::VecBlike;

    if constexpr (std::is_same_v<PhysicalQuantity, MHDQuantity>)
        return MHDQuantity::Vector::VecCellCentered;
    else
        throw std::runtime_error("no cell-centered vector quantity for hybrid");
}

} // namespace PHARE::core

#endif
```

- [ ] **Step 4: Build and run test**

Run: `./tools/cmake.sh` then `cd build && uv run ctest -R '^test-derived-quantity$' --output-on-failure`
Expected: PASS. Also confirm nothing else broke: aliases share enum values, so `-Wswitch` in existing switches stays silent.

- [ ] **Step 5: Commit**

```bash
git add src/core/data/derived_quantity/centering.hpp src/core/mhd/mhd_quantities.hpp \
        src/core/hybrid/hybrid_quantities.hpp tests/core/data/derived_quantity/
git add -u
git commit -m "feat: centering-only quantity aliases + derived-quantity centering enums"
```

---

### Task 2: DerivedQuantity interface + registry

**Files:**
- Create: `src/core/data/derived_quantity/derived_quantity.hpp`
- Modify: `src/core/models/hybrid_state.hpp` (~line 30, add typedefs)
- Test: `tests/core/data/derived_quantity/test_derived_quantity.cpp` (+ add to `SOURCES`… note the CMake project has one executable per source-set; simplest: one executable per test file — add a second `add_executable` block, or extend SOURCES if the file uses gtest_discover; follow the pattern used in `tests/core/data/mhd_state/CMakeLists.txt` and create separate project `test-derived-quantity-interface` in the same CMakeLists)

**Interfaces:**
- Consumes: `ScalarCentering`, `VectorCentering` from Task 1.
- Produces:
  - `template<typename State, typename GridLayout, std::size_t rank> struct derived_traits` with `out_t` (`State::field_type` rank 0, `State::vecfield_type` rank 1) and `centering_t` (`ScalarCentering` rank 0, `VectorCentering` rank 1).
  - `template<typename State, typename GridLayout, std::size_t rank> class DerivedQuantity` with pure virtuals `std::string name() const`, `centering_t centering() const`, `void compute(State const&, GridLayout const&, out_t&, double time) const`.
  - `template<typename State, typename GridLayout> class DerivedQuantityRegistry` with `template<std::size_t rank> void add(std::unique_ptr<DerivedQuantity<State,GridLayout,rank>>)`, `template<std::size_t rank> DerivedQuantity<State,GridLayout,rank> const* find(std::string const&) const` (nullptr on miss), `template<std::size_t rank> auto const& quantities() const`.
  - `HybridState` gains `using vecfield_type = typename Electromag::vecfield_type; using field_type = typename vecfield_type::field_type;`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/core/data/derived_quantity/test_derived_quantity.cpp
#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/models/mhd_state.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t  = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t = TestGridLayout<YeeLayout_t>;
using State_t      = MHDState<VecFieldMHD<dim>>;

struct Ones : DerivedQuantity<State_t, YeeLayout_t, 0>
{
    std::string name() const override { return "ones"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }
    void compute(State_t const& /*state*/, YeeLayout_t const& layout, out_t& out,
                 double /*time*/) const override
    {
        layout.evalOnGhostBox(out, [&](auto const&... args) { out(args...) = 1.0; });
    }
};

TEST(DerivedQuantityRegistry, findsRegisteredQuantityByNameAndRank)
{
    DerivedQuantityRegistry<State_t, YeeLayout_t> registry;
    registry.add<0>(std::make_unique<Ones>());

    auto const* dq = registry.find<0>("ones");
    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->name(), "ones");
    EXPECT_EQ(dq->centering(), ScalarCentering::cell);

    EXPECT_EQ(registry.find<0>("nope"), nullptr);
    EXPECT_EQ(registry.find<1>("ones"), nullptr);
    EXPECT_EQ(registry.quantities<0>().size(), 1u);
}

TEST(DerivedQuantityRegistry, computeFillsGhostBox)
{
    GridLayout_t layout{10};
    UsableMHDState<dim> state{layout, "state"};
    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};

    Ones{}.compute(*state, layout, out, 0.0);

    for (auto const& v : out)
        EXPECT_DOUBLE_EQ(v, 1.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

Notes for implementer: check `UsableFieldMHD` constructor signature in `tests/core/data/field/test_usable_field_fixtures_mhd.hpp` — it is `(name, layout, qty)` and derives from the Field view; if iteration (`begin/end`) is on the inner Grid, adapt the final loop (e.g. `for (std::size_t i = 0; i < out.size(); ++i) EXPECT_DOUBLE_EQ(out.data()[i], 1.0);`). `evalOnGhostBox` covers the whole ghost box which for a Field equals its full allocation, so all values are written.

Add to `tests/core/data/derived_quantity/CMakeLists.txt` a second executable:

```cmake
project(test-derived-quantity-interface)
add_executable(${PROJECT_NAME} test_derived_quantity.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE phare_core phare_initializer ${GTEST_LIBS})
add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})
```

- [ ] **Step 2: Run test to verify it fails**

Compile failure: `derived_quantity.hpp: No such file`.

- [ ] **Step 3: Write implementation**

```cpp
// src/core/data/derived_quantity/derived_quantity.hpp
#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_QUANTITY_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_QUANTITY_HPP

#include "core/data/derived_quantity/centering.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace PHARE::core
{
template<typename State, typename GridLayout, std::size_t rank>
struct derived_traits;

template<typename State, typename GridLayout>
struct derived_traits<State, GridLayout, 0>
{
    using out_t       = typename State::field_type;
    using centering_t = ScalarCentering;
};

template<typename State, typename GridLayout>
struct derived_traits<State, GridLayout, 1>
{
    using out_t       = typename State::vecfield_type;
    using centering_t = VectorCentering;
};


/** Interface for computing a derived (post-processed) quantity from the primary
 *  variables of a State into a caller-provided buffer, over the ghost box. */
template<typename State, typename GridLayout, std::size_t rank>
class DerivedQuantity
{
    using traits = derived_traits<State, GridLayout, rank>;

public:
    using out_t       = typename traits::out_t;
    using centering_t = typename traits::centering_t;

    virtual ~DerivedQuantity() = default;

    virtual std::string name() const                    = 0;
    virtual centering_t centering() const               = 0;
    virtual void compute(State const& state, GridLayout const& layout, out_t& out,
                         double time) const             = 0;
};


template<typename State, typename GridLayout>
class DerivedQuantityRegistry
{
    template<std::size_t rank>
    using DQ = DerivedQuantity<State, GridLayout, rank>;

public:
    template<std::size_t rank>
    void add(std::unique_ptr<DQ<rank>> dq)
    {
        std::get<rank>(quantities_).push_back(std::move(dq));
    }

    template<std::size_t rank>
    DQ<rank> const* find(std::string const& name) const
    {
        for (auto const& dq : std::get<rank>(quantities_))
            if (dq->name() == name)
                return dq.get();
        return nullptr;
    }

    template<std::size_t rank>
    auto const& quantities() const
    {
        return std::get<rank>(quantities_);
    }

private:
    std::tuple<std::vector<std::unique_ptr<DQ<0>>>, std::vector<std::unique_ptr<DQ<1>>>>
        quantities_;
};

} // namespace PHARE::core

#endif
```

In `src/core/models/hybrid_state.hpp`, next to the existing `using VecField = typename Electromag::vecfield_type;` (~line 30) add:

```cpp
        using vecfield_type = typename Electromag::vecfield_type;
        using field_type    = typename vecfield_type::field_type;
```

- [ ] **Step 4: Build and run**

`./tools/cmake.sh && cd build && uv run ctest -R '^test-derived-quantity' --output-on-failure` → both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/data/derived_quantity/derived_quantity.hpp src/core/models/hybrid_state.hpp \
        tests/core/data/derived_quantity/
git commit -m "feat: DerivedQuantity interface + per-rank registry"
```

---

### Task 3: DerivedScratch (shared non-SAMRAI scratch views)

**Files:**
- Create: `src/core/data/derived_quantity/derived_scratch.hpp`
- Test: `tests/core/data/derived_quantity/test_derived_scratch.cpp` (third executable `test-derived-scratch` in same CMakeLists, same block as Task 2's)

**Interfaces:**
- Consumes: `scalar_qty`/`vector_qty` (Task 1).
- Produces: `template<typename VecField_t, typename PhysicalQuantity> class DerivedScratch` with:
  - `template<typename GridLayout> Field_t scalar(ScalarCentering, GridLayout const&)` — Field view over internal memory, shape `layout.allocSize(qty)`.
  - `template<typename GridLayout> VecField_t vector(VectorCentering, GridLayout const&)` — VecField whose 3 components view disjoint segments.
  - `template<std::size_t rank, typename GridLayout> auto view(centering, layout)` dispatching to the two above.
  - Memory grows lazily, never shrinks; successive calls reuse the same block.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/core/data/derived_quantity/test_derived_scratch.cpp
#include "core/data/derived_quantity/derived_scratch.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t  = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t = TestGridLayout<YeeLayout_t>;
using VecField_t   = VecFieldMHD<dim>;
using Scratch_t    = DerivedScratch<VecField_t, MHDQuantity>;

TEST(DerivedScratch, scalarViewHasAllocSizeShape)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto f = scratch.scalar(ScalarCentering::cell, layout);
    EXPECT_TRUE(f.isUsable());
    EXPECT_EQ(f.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarCellCentered));

    auto n = scratch.scalar(ScalarCentering::node, layout);
    EXPECT_EQ(n.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarNodeCentered));
}

TEST(DerivedScratch, vectorComponentsViewDisjointSegments)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto vf         = scratch.vector(VectorCentering::Blike, layout);
    auto const qtys = MHDQuantity::componentsQuantities(MHDQuantity::Vector::VecBlike);

    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(vf[i].isUsable());
        EXPECT_EQ(vf[i].shape(), layout.allocSize(qtys[i]));
    }
    // disjoint: end of comp i == start of comp i+1
    EXPECT_EQ(vf[0].data() + vf[0].size(), vf[1].data());
    EXPECT_EQ(vf[1].data() + vf[1].size(), vf[2].data());
}

TEST(DerivedScratch, memoryIsReusedAcrossCalls)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto a        = scratch.scalar(ScalarCentering::cell, layout);
    a.data()[0]   = 42.0;
    auto b        = scratch.scalar(ScalarCentering::cell, layout);
    EXPECT_EQ(a.data(), b.data()); // same block: shared scratch
    EXPECT_DOUBLE_EQ(b.data()[0], 42.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

- [ ] **Step 2: Run to verify failure** (missing header).

- [ ] **Step 3: Write implementation**

```cpp
// src/core/data/derived_quantity/derived_scratch.hpp
#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP

#include "core/data/derived_quantity/centering.hpp"
#include "core/utilities/types.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

namespace PHARE::core
{
/** Transient scratch memory for derived-quantity outputs. One raw block, grown
 *  lazily to the most demanding request; Field/VecField views are built per
 *  patch over it. Deliberately NOT a SAMRAI resource: values never survive the
 *  patch visit, so ResourcesManager/setOnPatch machinery is unnecessary. */
template<typename VecField_t, typename PhysicalQuantity>
class DerivedScratch
{
public:
    using Field_t                   = typename VecField_t::field_type;
    static constexpr auto dimension = Field_t::dimension;

    template<typename GridLayout>
    Field_t scalar(ScalarCentering const centering, GridLayout const& layout)
    {
        auto const qty   = scalar_qty<PhysicalQuantity>(centering);
        auto const shape = layout.allocSize(qty);
        ensure_(product_(shape));
        return Field_t{"PHARE_derived_scratch", qty, mem_.data(), shape};
    }

    template<typename GridLayout>
    VecField_t vector(VectorCentering const centering, GridLayout const& layout)
    {
        auto const qty  = vector_qty<PhysicalQuantity>(centering);
        auto const qtys = PhysicalQuantity::componentsQuantities(qty);

        VecField_t vf{"PHARE_derived_scratch_vec", qty};

        std::array<std::array<std::uint32_t, dimension>, 3> shapes;
        std::size_t total = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            shapes[i] = layout.allocSize(qtys[i]);
            total += product_(shapes[i]);
        }
        ensure_(total);

        std::size_t offset = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            Field_t component{vf[i].name(), qtys[i], mem_.data() + offset, shapes[i]};
            vf[i].setBuffer(&component);
            offset += product_(shapes[i]);
        }
        return vf;
    }

    template<std::size_t rank, typename GridLayout, typename Centering>
    auto view(Centering const centering, GridLayout const& layout)
    {
        static_assert(rank <= 1, "tensor scratch not implemented");
        if constexpr (rank == 0)
            return scalar(centering, layout);
        else
            return vector(centering, layout);
    }

private:
    static std::size_t product_(std::array<std::uint32_t, dimension> const& shape)
    {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                               std::multiplies<std::size_t>{});
    }

    void ensure_(std::size_t const n)
    {
        if (mem_.size() < n)
            mem_.resize(n);
    }

    std::vector<double> mem_;
};

} // namespace PHARE::core

#endif
```

Implementer notes: `Field(name, qty, data, dims)` ctor is `src/core/data/field/field.hpp:34`; `setBuffer` asserts matching names (`field.hpp:64`) — that's why the temp component is constructed with `vf[i].name()`. `TensorField::operator[]` is `tensorfield.hpp:181`. If `mem_.resize` reallocates on the vector() call after scalar(), earlier views dangle — that's fine by contract (views are per-patch transient, one active quantity at a time); the reuse test above sizes equal so pointers stay stable.

- [ ] **Step 4: Build + run** `uv run ctest -R '^test-derived-scratch$' --output-on-failure` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/data/derived_quantity/derived_scratch.hpp tests/core/data/derived_quantity/
git commit -m "feat: DerivedScratch non-SAMRAI shared scratch views"
```

---

### Task 4: MHD concrete implementations + factory

**Files:**
- Create: `src/core/data/derived_quantity/mhd_derived_quantities.hpp`
- Modify: `src/core/models/mhd_state.hpp` (add `gamma()` accessor, ~line 128)
- Test: `tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp` (fourth executable `test-mhd-derived-quantities`)

**Interfaces:**
- Consumes: `DerivedQuantity`, `DerivedQuantityRegistry` (Task 2); pointwise `rhoVToV`, `eosEtotToP` from `core/numerics/primite_conservative_converter/to_primitive_converter.hpp`; `GridLayout::deriv<Direction>`, `GridLayout::project<faceXToCellCenter>`.
- Produces:
  - `MhdVelocity<State, GridLayout>` (rank 1, cell): `V = rhoV/rho`, name `"V"`.
  - `MhdPressure<State, GridLayout>` (rank 0, cell, ctor `(double gamma)`): `P = (γ-1)(Etot - ½ρv² - ½B²)`, name `"P"`, B projected face→cell.
  - `MhdDivB<State, GridLayout>` (rank 0, cell): `divB = Σ ∂B_i/∂x_i`, name `"divB"`.
  - `template<typename State, typename GridLayout> DerivedQuantityRegistry<State, GridLayout> makeMhdDerivedQuantities(double gamma)`.
  - `MHDState::gamma() const -> double`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/core/data/derived_quantity/test_mhd_derived_quantities.cpp
#include "core/data/derived_quantity/mhd_derived_quantities.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <cmath>

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t  = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t = TestGridLayout<YeeLayout_t>;
using State_t      = MHDState<VecFieldMHD<dim>>;

struct MhdDerived : public ::testing::Test
{
    GridLayout_t layout{10};
    UsableMHDState<dim> state{layout, "state"};

    void fill(auto& field, double const v)
    {
        for (std::size_t i = 0; i < field.size(); ++i)
            field.data()[i] = v;
    }
};

TEST_F(MhdDerived, velocityIsMomentumOverDensity)
{
    fill(state.rho, 2.0);
    fill(state.rhoV[0], 2.0);
    fill(state.rhoV[1], 4.0);
    fill(state.rhoV[2], 6.0);

    UsableVecFieldMHD<dim> out{"out", layout, MHDQuantity::Vector::VecCellCentered};
    MhdVelocity<State_t, YeeLayout_t>{}.compute(*state, layout, out, 0.0);

    auto& V = static_cast<VecFieldMHD<dim>&>(out);
    layout.evalOnGhostBox(V[0], [&](auto const&... args) {
        EXPECT_DOUBLE_EQ(V[0](args...), 1.0);
        EXPECT_DOUBLE_EQ(V[1](args...), 2.0);
        EXPECT_DOUBLE_EQ(V[2](args...), 3.0);
    });
}

TEST_F(MhdDerived, pressureRecoversEosValue)
{
    double const gamma = 5. / 3.;
    double const p_ref = 0.7;

    fill(state.rho, 1.0);
    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(state.rhoV[c], 0.0);
        fill(state.B[c], 0.0);
    }
    fill(state.Etot, p_ref / (gamma - 1.0));

    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};
    MhdPressure<State_t, YeeLayout_t>{gamma}.compute(*state, layout, out, 0.0);

    layout.evalOnGhostBox(out, [&](auto const&... args) {
        EXPECT_NEAR(out(args...), p_ref, 1e-12);
    });
}

TEST_F(MhdDerived, divBOfLinearFieldIsConstant)
{
    // Bx(i,j) = i (in local index space) => dBx/dx = 1/dx; By, Bz uniform.
    auto& Bx = state.B[0];
    for (std::uint32_t i = 0; i < Bx.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bx.shape()[1]; ++j)
            Bx(i, j) = static_cast<double>(i);
    fill(state.B[1], 1.0);
    fill(state.B[2], 1.0);

    double const inv_dx = 1.0 / layout.meshSize()[0];

    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};
    MhdDivB<State_t, YeeLayout_t>{}.compute(*state, layout, out, 0.0);

    layout.evalOnGhostBox(out, [&](auto const&... args) {
        EXPECT_NEAR(out(args...), inv_dx, 1e-10 * inv_dx);
    });
}

TEST_F(MhdDerived, divBOfDiscreteCurlIsZero)
{
    // Az on z-edges (primal,primal); Bx = dAz/dy on x-faces, By = -dAz/dx on
    // y-faces. Discrete divB then cancels to machine precision.
    auto& Bx = state.B[0];
    auto& By = state.B[1];

    auto const nx = Bx.shape()[0]; // primal x
    auto const ny = By.shape()[1]; // primal y

    auto Az = [&](std::uint32_t const i, std::uint32_t const j) {
        return std::sin(0.7 * i) * std::cos(1.3 * j);
    };

    double const inv_dx = 1.0 / layout.meshSize()[0];
    double const inv_dy = 1.0 / layout.meshSize()[1];

    for (std::uint32_t i = 0; i < Bx.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bx.shape()[1]; ++j)
            Bx(i, j) = (Az(i, j + 1) - Az(i, j)) * inv_dy;

    for (std::uint32_t i = 0; i < By.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < By.shape()[1]; ++j)
            By(i, j) = -(Az(i + 1, j) - Az(i, j)) * inv_dx;

    fill(state.B[2], 0.3);
    (void)nx; (void)ny;

    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};
    MhdDivB<State_t, YeeLayout_t>{}.compute(*state, layout, out, 0.0);

    layout.evalOnGhostBox(out, [&](auto const&... args) {
        EXPECT_NEAR(out(args...), 0.0, 1e-12 * inv_dx);
    });
}

TEST_F(MhdDerived, factoryRegistersVPandDivB)
{
    auto registry = makeMhdDerivedQuantities<State_t, YeeLayout_t>(5. / 3.);
    EXPECT_NE(registry.find<1>("V"), nullptr);
    EXPECT_NE(registry.find<0>("P"), nullptr);
    EXPECT_NE(registry.find<0>("divB"), nullptr);
    EXPECT_EQ(registry.find<0>("V"), nullptr);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

Implementer notes: `state.B[0]` etc. on `UsableMHDState` are `UsableVecFieldMHD` members — component access may be `state.B[0]` if operator[] forwards, otherwise use the Grid members / `getComponent`; check `tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp` and adapt accessors (the deref `*state` yields the bound `MHDState`, whose `B[i]` are Field views — prefer writing through `(*state).B[i]`). The divB curl test relies on Bx being (primal-x, dual-y) and By (dual-x, primal-y) in the MHD Yee layout — verify with `layout.centering(...)` if the shape arithmetic looks off by one, and check `deriv<Direction>` next/prev index tables (`gridlayout.hpp:535`) to confirm the exact discrete stencil before debugging test values.

- [ ] **Step 2: Run to verify failure** (missing header).

- [ ] **Step 3: Write implementation**

In `src/core/models/mhd_state.hpp` add a public accessor (after `initialize`, before the field members):

```cpp
        NO_DISCARD double gamma() const { return gamma_; }
```

```cpp
// src/core/data/derived_quantity/mhd_derived_quantities.hpp
#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP

#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/utilities/index/index.hpp"

#include <memory>
#include <string>

namespace PHARE::core
{
template<typename State, typename GridLayout>
class MhdVelocity : public DerivedQuantity<State, GridLayout, 1>
{
    using Super = DerivedQuantity<State, GridLayout, 1>;

public:
    using typename Super::out_t;

    std::string name() const override { return "V"; }
    VectorCentering centering() const override { return VectorCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        auto& Vx = out(Component::X);
        auto& Vy = out(Component::Y);
        auto& Vz = out(Component::Z);

        layout.evalOnGhostBox(Vx, [&](auto const&... args) {
            point_(state, Vx, Vy, Vz, {args...});
        });
    }

private:
    static void point_(State const& state, auto& Vx, auto& Vy, auto& Vz,
                       MeshIndex<GridLayout::dimension> const index)
    {
        auto&& [vx, vy, vz]
            = rhoVToV(state.rho(index), state.rhoV(Component::X)(index),
                      state.rhoV(Component::Y)(index), state.rhoV(Component::Z)(index));
        Vx(index) = vx;
        Vy(index) = vy;
        Vz(index) = vz;
    }
};


template<typename State, typename GridLayout>
class MhdPressure : public DerivedQuantity<State, GridLayout, 0>
{
    using Super = DerivedQuantity<State, GridLayout, 0>;

public:
    using typename Super::out_t;

    explicit MhdPressure(double const gamma)
        : gamma_{gamma}
    {
    }

    std::string name() const override { return "P"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        layout.evalOnGhostBox(out, [&](auto const&... args) {
            point_(gamma_, state, out, {args...});
        });
    }

private:
    static void point_(double const gamma, State const& state, out_t& P,
                       MeshIndex<GridLayout::dimension> const index)
    {
        auto const& rho = state.rho;
        auto const vx   = state.rhoV(Component::X)(index) / rho(index);
        auto const vy   = state.rhoV(Component::Y)(index) / rho(index);
        auto const vz   = state.rhoV(Component::Z)(index) / rho(index);

        auto const bx = GridLayout::template project<GridLayout::faceXToCellCenter>(
            state.B(Component::X), index);
        auto const by = GridLayout::template project<GridLayout::faceYToCellCenter>(
            state.B(Component::Y), index);
        auto const bz = GridLayout::template project<GridLayout::faceZToCellCenter>(
            state.B(Component::Z), index);

        auto const etot = state.Etot(index);
        P(index)        = eosEtotToP(gamma, rho(index), vx, vy, vz, bx, by, bz, etot);
    }

    double gamma_;
};


template<typename State, typename GridLayout>
class MhdDivB : public DerivedQuantity<State, GridLayout, 0>
{
    using Super = DerivedQuantity<State, GridLayout, 0>;

public:
    using typename Super::out_t;

    std::string name() const override { return "divB"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        auto const& Bx = state.B(Component::X);
        auto const& By = state.B(Component::Y);
        auto const& Bz = state.B(Component::Z);

        layout.evalOnGhostBox(out, [&](auto const&... args) {
            point_(layout, Bx, By, Bz, out, {args...});
        });
    }

private:
    static void point_(GridLayout const& layout, auto const& Bx, auto const& By, auto const& Bz,
                       out_t& out, MeshIndex<GridLayout::dimension> const index)
    {
        out(index) = layout.template deriv<Direction::X>(Bx, index);
        if constexpr (GridLayout::dimension >= 2)
            out(index) += layout.template deriv<Direction::Y>(By, index);
        if constexpr (GridLayout::dimension == 3)
            out(index) += layout.template deriv<Direction::Z>(Bz, index);
    }
};


template<typename State, typename GridLayout>
DerivedQuantityRegistry<State, GridLayout> makeMhdDerivedQuantities(double const gamma)
{
    DerivedQuantityRegistry<State, GridLayout> registry;
    registry.template add<1>(std::make_unique<MhdVelocity<State, GridLayout>>());
    registry.template add<0>(std::make_unique<MhdPressure<State, GridLayout>>(gamma));
    registry.template add<0>(std::make_unique<MhdDivB<State, GridLayout>>());
    return registry;
}

} // namespace PHARE::core

#endif
```

Implementer notes: `eosEtotToP` takes `auto& etot` (non-const) — bind through a named `auto const etot` local as shown works because `auto&` deduces `double const&`. `Direction` comes from `core/data/grid/gridlayoutdefs.hpp` (transitively included). If `project<faceZToCellCenter>` is ill-formed in 2D builds, guard bz with `if constexpr (GridLayout::dimension == 3)` and use `state.B(Component::Z)(index)` otherwise — check how `to_primitive_converter.hpp:108-110` compiles in 2D first (it does, so the direct call is fine).

- [ ] **Step 4: Build + run** `uv run ctest -R '^test-mhd-derived-quantities$' --output-on-failure` → PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/data/derived_quantity/mhd_derived_quantities.hpp src/core/models/mhd_state.hpp \
        tests/core/data/derived_quantity/
git commit -m "feat: MHD derived quantities V, P, divB + factory"
```

---

### Task 5: ModelView wiring (registry + state accessor)

**Files:**
- Modify: `src/diagnostic/diagnostic_model_view.hpp` (MHD ModelView ~line 267-345; hybrid ModelView ~line 148-264)

**Interfaces:**
- Consumes: `makeMhdDerivedQuantities` (Task 4), `DerivedQuantityRegistry` (Task 2).
- Produces (on both ModelView specializations):
  - `using State_t = ...` (MHD: `Model::state_type`; hybrid: `std::decay_t<decltype(std::declval<Model&>().state)>`).
  - `auto& state()` / `auto const& state() const` → `model_.state`.
  - `auto const& derivedQuantities() const` → `core::DerivedQuantityRegistry<State_t, GridLayout>`.
  - MHD registry populated from `makeMhdDerivedQuantities<State_t, GridLayout>(model.state.gamma())`; hybrid registry empty.

- [ ] **Step 1: Modify MHD ModelView**

Add include at top of `diagnostic_model_view.hpp`:

```cpp
#include "core/data/derived_quantity/mhd_derived_quantities.hpp"
```

In the MHD specialization, replace the inherited constructor (`using BaseModelView<...>::BaseModelView;`) with:

```cpp
    using Super   = BaseModelView<ModelView<Hierarchy, Model>, Hierarchy, Model>;
    using State_t = typename Model::state_type;
    using GridLayout = typename Super::GridLayout;

    ModelView(Hierarchy& hierarchy, Model& model)
        : Super{hierarchy, model}
        , derived_{core::makeMhdDerivedQuantities<State_t, GridLayout>(model.state.gamma())}
    {
    }

    NO_DISCARD auto& state() { return this->model_.state; }
    NO_DISCARD auto const& state() const { return this->model_.state; }

    NO_DISCARD auto const& derivedQuantities() const { return derived_; }
```

and add the member next to the existing protected fields:

```cpp
    core::DerivedQuantityRegistry<State_t, GridLayout> derived_;
```

(Keep `getV()`/`getP()`/`V_diag_`/`P_diag_` for now — deleted in Task 8 after the writers stop using them.)

- [ ] **Step 2: Modify hybrid ModelView**

```cpp
    using State_t    = std::decay_t<decltype(std::declval<Model&>().state)>;
    using GridLayout = typename Super::GridLayout;

    NO_DISCARD auto& state() { return this->model_.state; }
    NO_DISCARD auto const& state() const { return this->model_.state; }

    NO_DISCARD auto const& derivedQuantities() const { return derived_; }
```

member: `core::DerivedQuantityRegistry<State_t, GridLayout> derived_;` (never populated — hybrid impls come later). The hybrid specialization already defines `using Super = BaseModelView<...>` at line 152 — reuse it.

- [ ] **Step 3: Build everything**

`./tools/cmake.sh` → clean build (this instantiates the registry for both models; catches missing hybrid typedefs).

- [ ] **Step 4: Run existing diagnostic + simulator tests**

`cd build && uv run ctest -R 'diag|test-derived' --output-on-failure` → PASS (no behavior change yet).

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/diagnostic_model_view.hpp
git commit -m "feat: derived-quantity registry + state accessor on ModelView"
```

---

### Task 6: phareh5 MHD writer migration (+ divB)

**Files:**
- Modify: `src/diagnostic/detail/h5writer.hpp` (patch-layout plumbing, ~lines 199, 240, 305, 342)
- Modify: `src/diagnostic/detail/types/mhd.hpp` (full rewrite of compute/getDataSetInfo/initDataSets/write/createFiles)

**Interfaces:**
- Consumes: `modelView.derivedQuantities()`, `modelView.state()` (Task 5), `DerivedScratch` (Task 3), `h5Writer.timestamp()`.
- Produces: `H5Writer::patchLayout()` returning `GridLayout const&` valid during visit callbacks; MHD phareh5 writer handling any registry quantity generically; `"divB"` in the candidate file list.

- [ ] **Step 1: Add patch-layout plumbing to H5Writer**

In `h5writer.hpp`, next to `std::string patchPath_;` (line 199):

```cpp
    std::optional<GridLayout> patchLayout_; // set per-patch during visits, like patchPath_
```

accessor next to `patchPath()` (line 240):

```cpp
    NO_DISCARD auto& patchLayout() const { return *patchLayout_; }
```

In `collectPatchAttributes` (line 305) and `writePatch` (line 342), first statement of the lambda body (the GridLayout is already the first parameter; name it in `collectPatchAttributes` where it is currently anonymous):

```cpp
        patchLayout_ = gridLayout;
```

Add `#include <optional>` if absent. Verify `GridLayout` is copy-constructible (it is — `ToPrimitiveConverter` stores one by value).

- [ ] **Step 2: Rewrite `types/mhd.hpp`**

Changes, keeping the file's existing style:

1. Includes: drop `to_primitive_converter.hpp`; add:

```cpp
#include "core/data/derived_quantity/derived_scratch.hpp"
```

2. Class members/aliases (inside `MHDDiagnosticWriter`):

```cpp
    using ModelView_t = std::decay_t<decltype(std::declval<H5Writer&>().modelView())>;
    using Model_t     = typename ModelView_t::Model_t;
    using Scratch_t   = core::DerivedScratch<typename Model_t::vecfield_type,
                                             typename Model_t::physical_quantity_type>;

    Scratch_t scratch_;
```

3. `compute()` becomes a no-op:

```cpp
template<typename H5Writer>
void MHDDiagnosticWriter<H5Writer>::compute(DiagnosticProperties&)
{
    // derived quantities are computed per patch during write(), into scratch views
}
```

4. `createFiles` — extend candidates:

```cpp
    checkCreateFileFor_(diagnostic, fileData_, tree, "rho", "V", "P", "rhoV", "Etot", "divB");
```

5. `getDataSetInfo` — keep raw branches for `rho`, `rhoV`, `Etot`; DELETE the `V`/`P` branches; append generic derived loops (after the raw branches):

```cpp
    auto const& derived = h5Writer.modelView().derivedQuantities();
    auto const& layout  = h5Writer.patchLayout();

    for (auto const& dq : derived.template quantities<0>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
        {
            auto field = scratch_.scalar(dq->centering(), layout);
            infoDS(field, dq->name(), patchAttributes[lvlPatchID]["mhd"]);
        }

    for (auto const& dq : derived.template quantities<1>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
        {
            auto vecfield = scratch_.vector(dq->centering(), layout);
            infoVF(vecfield, dq->name(), patchAttributes[lvlPatchID]["mhd"]);
        }
```

Also delete the now-unused `auto& V = ...getV(); auto& P = ...getP();` locals.

6. `initDataSets` — inside `initPatch`, keep raw branches (`rho`, `rhoV`, `Etot`), DELETE `V`/`P` branches, append:

```cpp
        auto const& derived = h5Writer.modelView().derivedQuantities();
        for (auto const& dq : derived.template quantities<0>())
            if (isActiveDiag(diagnostic, tree, dq->name()))
                initDS(path, attr["mhd"], dq->name(), null);
        for (auto const& dq : derived.template quantities<1>())
            if (isActiveDiag(diagnostic, tree, dq->name()))
                initVF(path, attr["mhd"], dq->name(), null);
```

7. `write` — keep raw branches, DELETE `V`/`P` branches and the `getV()/getP()` locals, append:

```cpp
    auto& modelView     = h5Writer.modelView();
    auto const& derived = modelView.derivedQuantities();
    auto const& layout  = h5Writer.patchLayout();
    auto const time     = h5Writer.timestamp();

    for (auto const& dq : derived.template quantities<0>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
        {
            auto field = scratch_.scalar(dq->centering(), layout);
            dq->compute(modelView.state(), layout, field, time);
            writeDS(path + dq->name(), field);
        }

    for (auto const& dq : derived.template quantities<1>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
        {
            auto vecfield = scratch_.vector(dq->centering(), layout);
            dq->compute(modelView.state(), layout, vecfield, time);
            writeTF(path + dq->name(), vecfield);
        }
```

Note the registry's `GridLayout` (from ModelView) and the writer's `GridLayout` (from H5Writer) must be the same type — they both come from the Model; if the compiler disagrees, use `typename ModelView_t::GridLayout` consistently.

- [ ] **Step 3: Build** `./tools/cmake.sh` → clean.

- [ ] **Step 4: Run existing test suite**

`cd build && uv run ctest -R 'diag' --output-on-failure` and any MHD python tests currently registered (`uv run ctest -N | grep -i mhd`) → PASS. V/P outputs must be identical to before (same formulas, same ghost-box fill, same dataset paths).

- [ ] **Step 5: Commit**

```bash
git add src/diagnostic/detail/h5writer.hpp src/diagnostic/detail/types/mhd.hpp
git commit -m "feat: phareh5 MHD writer consumes derived-quantity registry (adds divB)"
```

---

### Task 7: vtkhdf fluid writer migration (+ divB)

**Files:**
- Modify: `src/diagnostic/detail/vtk_types/fluid.hpp`

**Interfaces:**
- Consumes: same as Task 6; `VTKFileWriter::writeField(field, layout)` / `writeTensorField<1>(tf, layout)`; `VTKFileInitializer::initFieldFileLevel(ilvl)` / `initTensorFieldFileLevel<1>(ilvl)`; `h5Writer_.timestamp()`.
- Produces: vtkhdf MHD path handling any registry quantity; `MhdFluidComputer` deleted.

- [ ] **Step 1: Rewrite the MHD structs**

1. Includes: drop `to_primitive_converter.hpp`; add `#include "core/data/derived_quantity/derived_scratch.hpp"`.

2. Add to `FluidDiagnosticWriter` class:

```cpp
    using Scratch_t = core::DerivedScratch<typename Model_t::vecfield_type,
                                           typename Model_t::physical_quantity_type>;
    Scratch_t scratch_;
```

(`Model_t` alias exists at line 26.)

3. `MhdFluidInitializer::operator()` — keep raw branches (`rho`, `rhoV`, `Etot`), DELETE explicit `P`/`V` branches, append before `return std::nullopt;`:

```cpp
    auto const& derived = modelView.derivedQuantities();
    for (auto const& dq : derived.template quantities<0>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
            return file_initializer.initFieldFileLevel(ilvl);
    for (auto const& dq : derived.template quantities<1>())
        if (isActiveDiag(diagnostic, tree, dq->name()))
            return file_initializer.template initTensorFieldFileLevel<1>(ilvl);
```

4. `MhdFluidWriter::operator()(auto const& layout)` — keep raw branches, DELETE `P`/`V` branches and `getP()/getV()` locals, append as the final `else` block:

```cpp
    else
    {
        auto const& derived = modelView.derivedQuantities();
        auto const time     = writer->h5Writer_.timestamp();

        for (auto const& dq : derived.template quantities<0>())
            if (isActiveDiag(diagnostic, tree, dq->name()))
            {
                auto field = writer->scratch_.scalar(dq->centering(), layout);
                dq->compute(modelView.state(), layout, field, time);
                file_writer.writeField(field, layout);
            }
        for (auto const& dq : derived.template quantities<1>())
            if (isActiveDiag(diagnostic, tree, dq->name()))
            {
                auto vecfield = writer->scratch_.vector(dq->centering(), layout);
                dq->compute(modelView.state(), layout, vecfield, time);
                file_writer.template writeTensorField<1>(vecfield, layout);
            }
    }
```

5. Delete `MhdFluidComputer` struct + its `operator()` definition; `compute(DiagnosticProperties&)` becomes:

```cpp
template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::compute(DiagnosticProperties&)
{
    // derived quantities are computed per patch during write(), into scratch views
}
```

Note: `writeField` internally re-centers to all-primal using `modelView.tmpField()` and reads ghost values — this is why `compute` fills the ghost box.

- [ ] **Step 2: Build** `./tools/cmake.sh` → clean.

- [ ] **Step 3: Run tests** `cd build && uv run ctest -R 'diag|vtk' --output-on-failure` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/diagnostic/detail/vtk_types/fluid.hpp
git commit -m "feat: vtkhdf MHD fluid writer consumes derived-quantity registry (adds divB)"
```

---

### Task 8: Delete V_diag_/P_diag_ machinery + python whitelist

**Files:**
- Modify: `src/amr/physical_models/mhd_model.hpp` (lines 45-47, 60-61, 80-81)
- Modify: `src/diagnostic/diagnostic_model_view.hpp` (MHD ModelView: `getV/getP`, `V_diag_/P_diag_`, resource tuple)
- Modify: `pyphare/pyphare/pharein/diagnostics.py` (line 219)

**Interfaces:**
- Consumes: nothing new. Produces: no `V_diag_`/`P_diag_` anywhere; `mhd_quantities` includes `"divB"`.

- [ ] **Step 1: Verify no remaining users**

```bash
grep -rn "getV()\|getP()\|V_diag_\|P_diag_\|diagnostics_V_\|diagnostics_P_" src/ tests/
```

Expected: hits only in `mhd_model.hpp` and `diagnostic_model_view.hpp`. If a test references them, fix that test in this task.

- [ ] **Step 2: Delete**

- `mhd_model.hpp`: remove the two member declarations (lines 45-47), the two `allocate` lines (60-61), the two `registerResources` lines (80-81).
- `diagnostic_model_view.hpp` (MHD specialization): remove `getV()`/`getP()` (all four overloads), remove `V_diag_`/`P_diag_` members, change both `getCompileTimeResourcesViewList()` to `return std::forward_as_tuple(tmpField_, tmpVec_);` (tmp fields STAY — vtkhdf primal conversion uses them).

- [ ] **Step 3: Python whitelist**

```python
    mhd_quantities = ["rho", "V", "P", "rhoV", "Etot", "divB"]
```

- [ ] **Step 4: Build + full relevant suite**

`./tools/cmake.sh && cd build && uv run ctest -R 'diag|mhd|derived' --output-on-failure` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/amr/physical_models/mhd_model.hpp src/diagnostic/diagnostic_model_view.hpp \
        pyphare/pyphare/pharein/diagnostics.py
git commit -m "refactor: drop V_diag_/P_diag_ SAMRAI buffers; whitelist divB"
```

---

### Task 9: Functional test (both formats)

**Files:**
- Create: `tests/simulator/test_mhd_derived_diagnostics.py`
- Modify: `tests/simulator/CMakeLists.txt` (register, guarded by `if(HighFive)` — mirror how neighbouring MHD/diagnostic python tests are registered; check `tests/simulator/initialize/test_init_mhd.py`'s registration for the harness pattern)

**Interfaces:**
- Consumes: everything above, end to end.

- [ ] **Step 1: Write the test**

Model the simulation setup on `tests/simulator/initialize/test_init_mhd.py` (read it first; reuse its MHD simulation config style). Test skeleton — adapt config details to what that file provides:

```python
# tests/simulator/test_mhd_derived_diagnostics.py
"""
Functional check of derived-quantity diagnostics (V, P, divB) in phareh5 format:
dumped V and P must match manual reconstruction from conservative dumps, and
divB of the (divergence-free) initial condition must vanish to machine precision.
"""
import os
import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator
from tests.simulator import SimulatorTest


out_dir = "phare_outputs/mhd_derived_diagnostics"
time_step = 0.001
final_time = 0.001  # a couple of steps is enough; we check t=0 dumps
gamma = 5.0 / 3.0


def config():
    sim = ph.Simulation(
        # copy the MHD simulation kwargs from test_init_mhd.py (model="MHD",
        # cells, dl, time_step, final_time, gamma=gamma, diag_options with
        # format "phareh5" and directory out_dir, ...)
    )
    # copy the ph.MHDModel / initial condition setup from test_init_mhd.py,
    # choosing a B initial condition defined via vector potential (divergence
    # free discretely), e.g. the orszag-tang setup if present.

    timestamps = [0.0]
    for q in ["rho", "V", "P", "rhoV", "Etot", "divB"]:
        ph.MHDDiagnostics(quantity=q, write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    return sim


class MHDDerivedDiagnosticsTest(SimulatorTest):
    def test_derived_quantities(self):
        ph.global_vars.sim = None
        sim = config()
        Simulator(sim).run().reset()

        from pyphare.pharesee.run import Run

        run = Run(out_dir)
        t = 0.0

        rho = run.GetMHDrho(t)
        rhoV = run.GetMHDrhoV(t)
        V = run.GetMHDV(t)
        P = run.GetMHDP(t)
        Etot = run.GetMHDEtot(t)

        # V == rhoV / rho, patch by patch
        for ilvl in rho.levels():
            for p_rho, p_rhoV, p_V in zip(
                rho.level(ilvl).patches,
                rhoV.level(ilvl).patches,
                V.level(ilvl).patches,
            ):
                for c in ["x", "y", "z"]:
                    np.testing.assert_allclose(
                        p_V.patch_datas[f"V_{c}"].dataset[:],
                        p_rhoV.patch_datas[f"rhoV_{c}"].dataset[:]
                        / p_rho.patch_datas["rho"].dataset[:],
                        rtol=1e-12,
                    )

        # divB ~ 0 (initial condition is divergence-free by construction)
        divB = run.GetMHDdivB(t)
        for ilvl in divB.levels():
            for patch in divB.level(ilvl).patches:
                np.testing.assert_allclose(
                    patch.patch_datas["divB"].dataset[:], 0.0, atol=1e-11
                )

        # P == (gamma-1) * (Etot - 0.5*rho*V^2 - 0.5*B^2), sampled sanity check
        # on interior values of the first patch (B needs face->cell averaging;
        # keep tolerance loose or reconstruct with the same projection).
        self.assertTrue(P is not None and Etot is not None)


if __name__ == "__main__":
    unittest.main()
```

Implementer notes (read before coding):
1. `run.GetMHDdivB` does not exist yet — add it in `pyphare/pyphare/pharesee/run/run.py` mirroring `GetMHDP` (same pattern, file `mhd_divB.h5`, quantity key `divB`). Check the existing `GetMHDrho`/`GetMHDP` implementations (~run.py:184-239) for the exact hierarchy-loading call and patch-data key naming; adapt the test's key names (`"rho"` vs `"mhd_rho"` etc.) to what actually comes back — inspect an existing MHD test that reads these getters.
2. Patch-data keys for vector components may differ (`V_x` vs `Vx`) — verify against `GetMHDV` usage in existing tests before hard-coding.
3. If no existing MHD config in `tests/simulator/initialize/test_init_mhd.py` is divergence-free by vector potential, use the orszag-tang config from `tests/functional/mhd_orszagtang/orszag_tang.py` (Bx = -B0 sin(y), By = B0 sin(2x) is discretely divergence-free since each component is constant along its own direction).
4. The P cross-check requires face→cell projection of B; simplest faithful reconstruction: `bx_cc = 0.5*(Bx[1:,:] + Bx[:-1,:])` etc. on primal-stripped arrays. If the index bookkeeping gets fragile, assert instead that P is finite, positive, and equals the configured initial pressure profile at t=0 within 1e-10 (the initial condition is known analytically).
5. vtkhdf smoke check: append a second run with `diag_options` format `"pharevtkhdf"` into a second directory, then assert the expected `.h5`/`.vtkhdf` files exist and `h5py`-open them, checking datasets are finite (`np.isfinite(...).all()`). Look at how existing vtkhdf tests (if any: `grep -rn pharevtkhdf tests/`) validate output; if none, file-exists + finite-data is enough.
6. Registration in `tests/simulator/CMakeLists.txt`: `phare_python3_exec(9 mhd_derived_diagnostics test_mhd_derived_diagnostics.py ${CMAKE_CURRENT_BINARY_DIR})` style — copy the exact macro/arity used by neighbouring entries, inside the `if(HighFive)` guard.

- [ ] **Step 2: Run to verify it fails usefully**

Before adding `GetMHDdivB`: the test errors with `AttributeError: GetMHDdivB` — confirms the test is wired into ctest:
`cd build && uv run ctest -R 'py3.*derived' --output-on-failure`

- [ ] **Step 3: Add `GetMHDdivB` to pharesee** (mirror `GetMHDP` in `pyphare/pyphare/pharesee/run/run.py`).

- [ ] **Step 4: Run test**

`cd build && uv run ctest -R 'py3.*derived' --output-on-failure` → PASS. (`pkill -f mpirun || true` first if a previous run leaked.)

- [ ] **Step 5: Full suite sanity**

`cd build && uv run ctest -j 12 --output-on-failure` → no regressions vs. the failures already present on master (if any — note them).

- [ ] **Step 6: Commit**

```bash
git add tests/simulator/test_mhd_derived_diagnostics.py tests/simulator/CMakeLists.txt \
        pyphare/pyphare/pharesee/run/run.py
git commit -m "test: functional MHD derived-diagnostics check (V, P, divB; phareh5 + vtkhdf)"
```
