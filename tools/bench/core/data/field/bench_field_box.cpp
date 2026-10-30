#include "core/data/field/field_box.hpp"
#include "core/utilities/types.hpp"

#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"

#include "benchmark/benchmark.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#if __has_include("core/utilities/box/box_spans.hpp")
#include "core/utilities/box/box_spans.hpp"
#define PHARE_BENCH_HAS_FOR_EACH_POINT 1
#elif __has_include("core/utilities/box/box_slabs.hpp")
#include "core/utilities/box/box_slabs.hpp"
#define PHARE_BENCH_HAS_FOR_EACH_POINT 1
#endif


namespace PHARE::core::bench
{

enum class Shape { full, thin_slow, thin_fast };
enum class Op { copy, add };

std::uint32_t constexpr cells_for(std::size_t const dim)
{
    return dim == 1 ? 1024 : dim == 2 ? 128 : 40;
}

template<std::size_t dim>
struct Setup
{
    static constexpr PHARE::SimOpts opts{dim, 1};
    using PHARE_Types  = core::PHARE_Types<opts>;
    using GridLayout_t = TestGridLayout<typename PHARE_Types::Hybrid::GridLayout_t>;
    using Grid_t       = PHARE_Types::Hybrid::Grid_t;

    Setup()
    {
        for (std::size_t i = 0; i < a.size(); ++i)
            a.data()[i] = static_cast<double>(i % 97);
    }

    GridLayout_t layout{cells_for(dim)};
    Grid_t a_grid{"a", layout, HybridQuantity::Scalar::rho, 1.};
    Grid_t b_grid{"b", layout, HybridQuantity::Scalar::rho, 0.};
    Grid_t::field_type& a = *&a_grid;
    Grid_t::field_type& b = *&b_grid;
};

template<std::size_t dim>
Box<std::uint32_t, dim> local_box(auto const& shape, Shape const kind, std::uint32_t const offset)
{
    Box<std::uint32_t, dim> box;
    for (std::size_t d = 0; d < dim; ++d)
    {
        box.lower[d] = offset;
        box.upper[d] = shape[d] - 3 + offset;
    }
    if (kind == Shape::thin_slow)
        box.upper[0] = box.lower[0] + 2;
    if (kind == Shape::thin_fast)
        box.upper[dim - 1] = box.lower[dim - 1] + 2;
    return box;
}


template<std::size_t dim>
void BM_evalOnBox_pointwise(benchmark::State& state)
{
    Setup<dim> s;
    auto& a = s.a;
    auto& b = s.b;
    for (auto _ : state)
    {
        s.layout.evalOnBox(b, [&](auto const... ijk) { b(ijk...) = 2. * a(ijk...) + 1.; });
        benchmark::DoNotOptimize(b.data());
        benchmark::ClobberMemory();
    }
}

template<std::size_t dim>
void BM_evalOnBox_stencil(benchmark::State& state)
{
    Setup<dim> s;
    auto& a = s.a;
    auto& b = s.b;
    for (auto _ : state)
    {
        s.layout.evalOnBox(b, [&](auto const... ijk) {
            std::array<std::uint32_t, dim> const p{ijk...};
            auto fast = p;
            auto slow = p;
            ++fast[dim - 1];
            ++slow[0];
            b(p) = a(fast) + a(slow) - 2. * a(p);
        });
        benchmark::DoNotOptimize(b.data());
        benchmark::ClobberMemory();
    }
}

template<std::size_t dim, Shape kind, Op op>
void BM_operate_on_fields(benchmark::State& state)
{
    using Operator = std::conditional_t<op == Op::copy, SetEqual<double>, PlusEquals<double>>;
    Setup<dim> s;
    auto const src_box = local_box<dim>(s.a.shape(), kind, 0);
    auto const dst_box = local_box<dim>(s.b.shape(), kind, 2);
    FieldBox src{s.a, s.layout, src_box};
    FieldBox dst{s.b, s.layout, dst_box};
    for (auto _ : state)
    {
        operate_on_fields<Operator>(dst, src);
        benchmark::DoNotOptimize(s.b.data());
        benchmark::ClobberMemory();
    }
    state.counters["cells"] = static_cast<double>(src_box.size());
}

template<std::size_t dim, Shape kind>
void BM_append_to(benchmark::State& state)
{
    Setup<dim> s;
    auto const box = local_box<dim>(s.a.shape(), kind, 0);
    FieldBox src{s.a, s.layout, box};
    std::vector<double> buffer;
    buffer.reserve(box.size());
    for (auto _ : state)
    {
        buffer.clear();
        src.append_to(buffer);
        benchmark::DoNotOptimize(buffer.data());
        benchmark::ClobberMemory();
    }
    state.counters["cells"] = static_cast<double>(box.size());
}

template<std::size_t dim, Shape kind>
void BM_set_from(benchmark::State& state)
{
    Setup<dim> s;
    auto const box = local_box<dim>(s.b.shape(), kind, 2);
    FieldBox dst{s.b, s.layout, box};
    std::vector<double> const buffer(box.size(), 3.);
    for (auto _ : state)
    {
        dst.template set_from<SetEqual<double>>(buffer);
        benchmark::DoNotOptimize(s.b.data());
        benchmark::ClobberMemory();
    }
    state.counters["cells"] = static_cast<double>(box.size());
}

#if defined(PHARE_BENCH_HAS_FOR_EACH_POINT)
template<std::size_t dim, Shape kind>
void BM_for_each_point(benchmark::State& state)
{
    Setup<dim> s;
    auto const box = local_box<dim>(s.a.shape(), kind, 1);
    auto& a        = s.a;
    auto& b        = s.b;
    for (auto _ : state)
    {
        for_each_point(box, [&](auto const& point) { b(point) = 2. * a(point) + 1.; });
        benchmark::DoNotOptimize(b.data());
        benchmark::ClobberMemory();
    }
    state.counters["cells"] = static_cast<double>(box.size());
}
#endif


BENCHMARK_TEMPLATE(BM_evalOnBox_pointwise, 1);
BENCHMARK_TEMPLATE(BM_evalOnBox_pointwise, 2);
BENCHMARK_TEMPLATE(BM_evalOnBox_pointwise, 3);
BENCHMARK_TEMPLATE(BM_evalOnBox_stencil, 1);
BENCHMARK_TEMPLATE(BM_evalOnBox_stencil, 2);
BENCHMARK_TEMPLATE(BM_evalOnBox_stencil, 3);

BENCHMARK_TEMPLATE(BM_operate_on_fields, 1, Shape::full, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 1, Shape::full, Op::add);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 2, Shape::full, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 2, Shape::full, Op::add);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 2, Shape::thin_slow, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 2, Shape::thin_fast, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 2, Shape::thin_fast, Op::add);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 3, Shape::full, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 3, Shape::full, Op::add);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 3, Shape::thin_slow, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 3, Shape::thin_fast, Op::copy);
BENCHMARK_TEMPLATE(BM_operate_on_fields, 3, Shape::thin_fast, Op::add);

BENCHMARK_TEMPLATE(BM_append_to, 1, Shape::full);
BENCHMARK_TEMPLATE(BM_append_to, 2, Shape::full);
BENCHMARK_TEMPLATE(BM_append_to, 2, Shape::thin_fast);
BENCHMARK_TEMPLATE(BM_append_to, 3, Shape::full);
BENCHMARK_TEMPLATE(BM_append_to, 3, Shape::thin_slow);
BENCHMARK_TEMPLATE(BM_append_to, 3, Shape::thin_fast);

BENCHMARK_TEMPLATE(BM_set_from, 1, Shape::full);
BENCHMARK_TEMPLATE(BM_set_from, 2, Shape::full);
BENCHMARK_TEMPLATE(BM_set_from, 2, Shape::thin_fast);
BENCHMARK_TEMPLATE(BM_set_from, 3, Shape::full);
BENCHMARK_TEMPLATE(BM_set_from, 3, Shape::thin_slow);
BENCHMARK_TEMPLATE(BM_set_from, 3, Shape::thin_fast);

#if defined(PHARE_BENCH_HAS_FOR_EACH_POINT)
BENCHMARK_TEMPLATE(BM_for_each_point, 1, Shape::full);
BENCHMARK_TEMPLATE(BM_for_each_point, 2, Shape::full);
BENCHMARK_TEMPLATE(BM_for_each_point, 2, Shape::thin_fast);
BENCHMARK_TEMPLATE(BM_for_each_point, 3, Shape::full);
BENCHMARK_TEMPLATE(BM_for_each_point, 3, Shape::thin_slow);
BENCHMARK_TEMPLATE(BM_for_each_point, 3, Shape::thin_fast);
#endif

} // namespace PHARE::core::bench


BENCHMARK_MAIN();
