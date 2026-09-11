/**
 * Benchmark of the cost of filling a field on a grid from a user-specified function.
 *
 * Three families of variants fill the exact same field (the ghost box of one Yee-centered
 * quantity) with the exact same values, and are checksum-verified against each other:
 *
 *   cpp_*    plain C++ evaluation, no Python involved                       (the reference)
 *   py_now   the current PHARE path: initializer::InitFunction<dim>, i.e.
 *            std::function<shared_ptr<Span<double>>(std::vector<double> const&...)> bound to a
 *            Python callable, with pyphare's py_fn_wrapper/fn_wrapper on the Python side
 *   py_np_*  the proposed path: coordinates handed over as zero-copy numpy views, the returned
 *            ndarray read in place, and (optionally) the index/coordinate vectors cached
 *
 * The ladder between py_now and py_np_cached_flat isolates each proposed change:
 *
 *   py_now                 vector -> py::list -> np.asarray  (two full copies per coordinate)
 *   py_np                  zero-copy numpy views in and out       [suggestion 2]
 *   py_np_cached           + index/coord vectors built once and reused
 *                            (models a field re-stamped every timestep, e.g. B0(x,t))
 *                                                                 [suggestion 1]
 *   py_np_cached_flat      + contiguous scatter instead of per-node AMRToLocal(Point{...})
 *   py_np_out              + the field's own buffer handed over as a writable `out` array, so the
 *                            user function fills it in place: no result allocation, no copy back
 *                            (costs an API change: user functions write instead of returning)
 *
 * Two user functions are benchmarked: `trivial` (returns a scalar, so the measurement is pure
 * binding-layer overhead) and `sincos` (a representative numpy expression).
 */

#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "benchmark/benchmark.h"

#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/models/options/hybrid_options.hpp"
#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/utilities/types.hpp"

#include "phare_simulator_options.hpp"
#include "initializer/data_provider.hpp"
#include "python3/pybind_def.hpp"

#include <map>
#include <cmath>
#include <tuple>
#include <string>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

namespace PHARE::bench
{
using PHARE::core::Point;
using PHARE::core::Span;

// -----------------------------------------------------------------------------------------------
// types
// -----------------------------------------------------------------------------------------------

// the coordinate functor used by field_user_initializer.hpp, as a named type
struct NodeCoords
{
    template<typename GridLayout, typename Field, typename... Args>
    auto operator()(GridLayout& layout, Field& field, Args const&... args) const
    {
        return layout.fieldNodeCoordinates(field, args...);
    }
};

template<std::size_t dim>
auto constexpr sim_opts = PHARE::SimOpts{dim, 1};

template<std::size_t dim>
auto constexpr field_opts = PHARE::HybridFieldOptions<sim_opts<dim>>{};

template<std::size_t dim>
auto constexpr hybrid_opts = PHARE::HybridOptions<field_opts<dim>>{};

template<std::size_t dim>
using GridLayout_t = PHARE::core::GridLayout<hybrid_opts<dim>>;

template<std::size_t dim>
using Grid_t = PHARE::core::Grid<PHARE::core::NdArrayVector<dim>, PHARE::core::HybridQuantity::Scalar>;

// the quantity being filled: Bx is dual in x, primal elsewhere (a non-trivial centering)
auto constexpr qty = PHARE::core::HybridQuantity::Scalar::Bx;


// the "improved" function signature: numpy arrays in, numpy array out
template<typename ReturnType, std::size_t dim>
struct NpFunctionHelper
{
};
template<>
struct NpFunctionHelper<double, 1>
{
    using type = std::function<PHARE::pydata::py_array_t<double>(py::object const&)>;
};
template<>
struct NpFunctionHelper<double, 2>
{
    using type = std::function<PHARE::pydata::py_array_t<double>(py::object const&, py::object const&)>;
};
template<>
struct NpFunctionHelper<double, 3>
{
    using type = std::function<PHARE::pydata::py_array_t<double>(py::object const&, py::object const&,
                                                                py::object const&)>;
};
template<std::size_t dim>
using NpFunction = typename NpFunctionHelper<double, dim>::type;


// the in-place signature: coordinates plus the destination array, nothing returned
template<typename ReturnType, std::size_t dim>
struct NpOutFunctionHelper
{
};
template<>
struct NpOutFunctionHelper<double, 1>
{
    using type = std::function<void(py::object const&, py::object const&)>;
};
template<>
struct NpOutFunctionHelper<double, 2>
{
    using type = std::function<void(py::object const&, py::object const&, py::object const&)>;
};
template<>
struct NpOutFunctionHelper<double, 3>
{
    using type = std::function<void(py::object const&, py::object const&, py::object const&,
                                    py::object const&)>;
};
template<std::size_t dim>
using NpOutFunction = typename NpOutFunctionHelper<double, dim>::type;


// -----------------------------------------------------------------------------------------------
// the C++ user functions (must mirror the Python ones exactly, checksums are compared)
// -----------------------------------------------------------------------------------------------

enum class UserFn { trivial, sincos };

template<std::size_t dim, UserFn fn, typename Point_t>
double evaluate(Point_t const& p)
{
    if constexpr (fn == UserFn::trivial)
    {
        (void)p;
        return 1.5;
    }
    else
    {
        double constexpr two_pi = 2 * 3.14159265358979323846;
        double out              = std::sin(two_pi * p[0]);
        for (std::size_t i = 1; i < dim; ++i)
            out *= std::cos(two_pi * p[i]);
        return out;
    }
}


// -----------------------------------------------------------------------------------------------
// embedded module mirroring what cpp_etc exposes to pyphare
// -----------------------------------------------------------------------------------------------

PYBIND11_EMBEDDED_MODULE(bench_cpp_etc, m)
{
    py::class_<core::Span<double>, py::smart_holder>(m, "Span");
    py::class_<PHARE::pydata::PyArrayWrapper<double>, py::smart_holder, core::Span<double>>(
        m, "PyWrapper");
    m.def("makePyArrayWrapper", PHARE::pydata::makePyArrayWrapper<double>);
}


// pyphare/pyphare/pharein/initialize/general.py, verbatim where it matters, plus the proposed
// numpy binding-layer wrapper and the user functions.
static char const* python_source = R"PY(
import numpy as np
from bench_cpp_etc import makePyArrayWrapper


def is_scalar(arg):  # pyphare.core.phare_utilities.is_scalar
    return not isinstance(arg, (list, tuple)) and not isinstance(arg, np.ndarray)


class py_fn_wrapper:  # pyphare.pharein.initialize.general.py_fn_wrapper
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *xyz):
        args = [np.asarray(arg) for arg in xyz]
        ret = self.fn(*args)
        if isinstance(ret, list):
            ret = np.asarray(ret)
        if is_scalar(ret):
            ret = np.full(len(args[-1]), ret)
        return ret


class fn_wrapper(py_fn_wrapper):  # pyphare.pharein.initialize.general.fn_wrapper
    def __call__(self, *xyz):
        return makePyArrayWrapper(super().__call__(*xyz))


class np_fn_wrapper:  # proposed: coordinates already are numpy views, return the ndarray as is
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *xyz):
        ret = self.fn(*xyz)
        if is_scalar(ret):
            ret = np.full(xyz[-1].shape, ret, dtype=np.float64)
        return ret


class np_out_fn_wrapper:  # proposed further step: the user fills the buffer it is handed
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        *xyz, out = args
        self.fn(*xyz, out=out)


def user_trivial(*xyz):
    return 1.5


def user_sincos(*xyz):
    out = np.sin(2.0 * np.pi * xyz[0])
    for c in xyz[1:]:
        out = out * np.cos(2.0 * np.pi * c)
    return out


def user_trivial_out(*xyz, out):
    out[...] = 1.5


def user_sincos_out(*xyz, out):
    np.sin(2.0 * np.pi * xyz[0], out=out)
    for c in xyz[1:]:
        out *= np.cos(2.0 * np.pi * c)
)PY";


// -----------------------------------------------------------------------------------------------
// fixture: one layout, one grid, the coordinate vectors, and the bound python functions
// -----------------------------------------------------------------------------------------------

template<std::size_t dim>
struct Fixture
{
    using Layout_t  = GridLayout_t<dim>;
    using Indices_t = std::vector<PHARE::core::tuple_fixed_type<int, dim>>;
    using Coords_t  = PHARE::core::tuple_fixed_type<std::vector<double>, dim>;

    Fixture(std::uint32_t const cells, UserFn const user_fn)
        : layout{PHARE::core::ConstArray<double, dim>(1. / cells),
                 PHARE::core::ConstArray<std::uint32_t, dim>(cells),
                 Point<double, dim>{PHARE::core::ConstArray<double, dim>(0)}}
        , grid{"bench_field", layout, qty}
        , field{*(&grid)}
        , indices{layout.indices(layout.AMRGhostBoxFor(field))}
        , coords{layout.template indexesToCoordVectors</*WithField=*/true>(indices, field,
                                                                          NodeCoords{})}
    {
        std::string const name = user_fn == UserFn::trivial ? "user_trivial" : "user_sincos";
        auto mod               = py::module_::import("bench_field_init_py");
        // py::object, not auto: attr() yields a lazy accessor that keeps the char* it was given
        py::object const raw     = mod.attr(name.c_str());
        py::object const raw_out = mod.attr((name + "_out").c_str());

        current      = mod.attr("fn_wrapper")(raw).template cast<initializer::InitFunction<dim>>();
        improved     = mod.attr("np_fn_wrapper")(raw).template cast<NpFunction<dim>>();
        improved_out = mod.attr("np_out_fn_wrapper")(raw_out).template cast<NpOutFunction<dim>>();
    }

    Layout_t layout;
    Grid_t<dim> grid;
    typename Grid_t<dim>::field_type& field;

    Indices_t indices;
    Coords_t coords;

    initializer::InitFunction<dim> current;
    NpFunction<dim> improved;
    NpOutFunction<dim> improved_out;

    double checksum() const
    {
        double sum = 0;
        for (std::size_t i = 0; i < grid.size(); ++i)
            sum += grid.data()[i];
        return sum;
    }

    void reset() { grid.zero(); }
};


// -----------------------------------------------------------------------------------------------
// the variants
// -----------------------------------------------------------------------------------------------

// plain C++: idiomatic evaluation, one pass, no intermediate vectors
template<std::size_t dim, UserFn fn>
void fill_cpp_direct(Fixture<dim>& f)
{
    auto const& layout = f.layout;
    auto& field        = f.field;

    for (auto const& indiceTuple : f.indices)
        std::apply(
            [&](auto const&... args) {
                field(layout.AMRToLocal(Point{args...}))
                    = evaluate<dim, fn>(layout.fieldNodeCoordinates(field, args...));
            },
            indiceTuple);
}

// plain C++ but through the same coordinate-vector machinery the Python path uses, to separate
// the cost of that machinery from the cost of the binding layer
template<std::size_t dim, UserFn fn>
void fill_cpp_via_vectors(Fixture<dim>& f)
{
    auto const& layout = f.layout;
    auto& field        = f.field;

    auto const indices = layout.indices(layout.AMRGhostBoxFor(field));
    auto const coords = layout.template indexesToCoordVectors</*WithField=*/true>(indices, field,
                                                                                 NodeCoords{});

    auto const ptrs = std::apply(
        [](auto const&... vecs) { return std::array<double const*, dim>{vecs.data()...}; }, coords);

    std::vector<double> out(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        Point<double, dim> p;
        for (std::size_t d = 0; d < dim; ++d)
            p[d] = ptrs[d][i];
        out[i] = evaluate<dim, fn>(p);
    }

    for (std::size_t cell_idx = 0; cell_idx < indices.size(); cell_idx++)
        std::apply([&](auto&... args) { field(layout.AMRToLocal(Point{args...})) = out[cell_idx]; },
                   indices[cell_idx]);
}

// the current PHARE path, unmodified
template<std::size_t dim, UserFn fn>
void fill_py_now(Fixture<dim>& f)
{
    core::FieldUserFunctionInitializer::initialize(f.field, f.layout, f.current);
}


template<typename Coords, std::size_t... I>
auto zero_copy_views(Coords const& coords, std::index_sequence<I...>)
{
    // py::array_t over an existing buffer: the capsule base tells pybind not to own nor copy
    return std::make_tuple(py::array_t<double>{
        static_cast<py::ssize_t>(std::get<I>(coords).size()), std::get<I>(coords).data(),
        py::capsule{std::get<I>(coords).data(), [](void*) {}}}...);
}

// proposed binding layer, coordinate vectors still rebuilt on every call
template<std::size_t dim, UserFn fn>
void fill_py_np(Fixture<dim>& f)
{
    auto const& layout = f.layout;
    auto& field        = f.field;

    auto const indices = layout.indices(layout.AMRGhostBoxFor(field));
    auto const coords = layout.template indexesToCoordVectors</*WithField=*/true>(indices, field,
                                                                                 NodeCoords{});

    auto views      = zero_copy_views(coords, std::make_index_sequence<dim>{});
    auto const grid = std::apply([&](auto&... args) { return f.improved(args...); }, views);
    auto const* src = static_cast<double const*>(grid.request().ptr);

    for (std::size_t cell_idx = 0; cell_idx < indices.size(); cell_idx++)
        std::apply([&](auto&... args) { field(layout.AMRToLocal(Point{args...})) = src[cell_idx]; },
                   indices[cell_idx]);
}

// proposed binding layer + cached index/coordinate vectors
template<std::size_t dim, UserFn fn>
void fill_py_np_cached(Fixture<dim>& f)
{
    auto const& layout = f.layout;
    auto& field        = f.field;

    auto views      = zero_copy_views(f.coords, std::make_index_sequence<dim>{});
    auto const grid = std::apply([&](auto&... args) { return f.improved(args...); }, views);
    auto const* src = static_cast<double const*>(grid.request().ptr);

    for (std::size_t cell_idx = 0; cell_idx < f.indices.size(); cell_idx++)
        std::apply([&](auto&... args) { field(layout.AMRToLocal(Point{args...})) = src[cell_idx]; },
                   f.indices[cell_idx]);
}

// proposed binding layer + cached vectors + contiguous scatter
template<std::size_t dim, UserFn fn>
void fill_py_np_cached_flat(Fixture<dim>& f)
{
    auto views      = zero_copy_views(f.coords, std::make_index_sequence<dim>{});
    auto const grid = std::apply([&](auto&... args) { return f.improved(args...); }, views);
    auto const* src = static_cast<double const*>(grid.request().ptr);

    std::copy(src, src + f.grid.size(), f.grid.data());
}

// proposed binding layer + cached vectors + the field buffer itself as the destination array: the
// user function writes into it, so nothing is allocated for the result and nothing is copied back
template<std::size_t dim, UserFn fn>
void fill_py_np_out(Fixture<dim>& f)
{
    auto views = zero_copy_views(f.coords, std::make_index_sequence<dim>{});
    auto out   = py::array_t<double>{static_cast<py::ssize_t>(f.grid.size()), f.grid.data(),
                                     py::capsule{f.grid.data(), [](void*) {}}};

    std::apply([&](auto&... args) { f.improved_out(args..., out); }, views);
}


// -----------------------------------------------------------------------------------------------
// registration
// -----------------------------------------------------------------------------------------------

template<std::size_t dim, UserFn fn, auto filler>
void bench(::benchmark::State& state)
{
    auto const cells = static_cast<std::uint32_t>(state.range(0));
    Fixture<dim> f{cells, fn};

    filler(f); // warm up, and leave a filled field for the checksum

    state.counters["nodes"] = static_cast<double>(f.grid.size());
    state.counters["sum"]   = f.checksum();

    for (auto _ : state)
    {
        filler(f);
        ::benchmark::DoNotOptimize(f.grid.data());
        ::benchmark::ClobberMemory();
    }
}

#define PHARE_BENCH(dim, fn, name)                                                                 \
    BENCHMARK_TEMPLATE(bench, dim, UserFn::fn, fill_##name<dim, UserFn::fn>)                       \
        ->Name(std::string{#name} + "/" #dim "d/" #fn)                                             \
        ->RangeMultiplier(2)                                                                       \
        ->Range(64, 512)                                                                           \
        ->Unit(::benchmark::kMicrosecond);

#define PHARE_BENCH_3D(dim, fn, name)                                                              \
    BENCHMARK_TEMPLATE(bench, dim, UserFn::fn, fill_##name<dim, UserFn::fn>)                       \
        ->Name(std::string{#name} + "/" #dim "d/" #fn)                                             \
        ->RangeMultiplier(2)                                                                       \
        ->Range(16, 64)                                                                            \
        ->Unit(::benchmark::kMicrosecond);

#define PHARE_BENCH_ALL(macro, dim, fn)                                                            \
    macro(dim, fn, cpp_direct) macro(dim, fn, cpp_via_vectors) macro(dim, fn, py_now)              \
        macro(dim, fn, py_np) macro(dim, fn, py_np_cached) macro(dim, fn, py_np_cached_flat)       \
            macro(dim, fn, py_np_out)

PHARE_BENCH_ALL(PHARE_BENCH, 2, trivial)
PHARE_BENCH_ALL(PHARE_BENCH, 2, sincos)
PHARE_BENCH_ALL(PHARE_BENCH_3D, 3, sincos)


// -----------------------------------------------------------------------------------------------
// correctness: every variant must produce the same field
// -----------------------------------------------------------------------------------------------

template<std::size_t dim, UserFn fn>
void verify()
{
    std::uint32_t constexpr cells = dim == 3 ? 16 : 64;

    auto run = [&](auto&& filler) {
        Fixture<dim> f{cells, fn};
        f.reset();
        filler(f);
        return f.checksum();
    };

    auto const ref = run(fill_cpp_direct<dim, fn>);
    auto const all = std::vector<double>{run(fill_cpp_via_vectors<dim, fn>),  //
                                         run(fill_py_now<dim, fn>),          //
                                         run(fill_py_np<dim, fn>),           //
                                         run(fill_py_np_cached<dim, fn>),    //
                                         run(fill_py_np_cached_flat<dim, fn>),
                                         run(fill_py_np_out<dim, fn>)};

    for (std::size_t i = 0; i < all.size(); ++i)
        if (std::abs(all[i] - ref) > 1e-9 * (1 + std::abs(ref)))
            throw std::runtime_error{"bench_field_init: variant " + std::to_string(i)
                                     + " checksum " + std::to_string(all[i]) + " != reference "
                                     + std::to_string(ref)};

    std::cout << "verified dim=" << dim << " fn=" << (fn == UserFn::trivial ? "trivial" : "sincos")
              << " checksum=" << ref << std::endl;
}


// -----------------------------------------------------------------------------------------------
// summary: googlebench prints one line per case, which is unreadable when what you want is to
// compare variants. Collect the runs and print them pivoted: one row per variant, one column per
// grid size, each cell the time and its ratio to the plain C++ reference.
// -----------------------------------------------------------------------------------------------

// registration order, which is also the order of the summary rows
static std::vector<std::string> const variant_order{
    "cpp_direct", "cpp_via_vectors",   "py_now",     "py_np",
    "py_np_cached", "py_np_cached_flat", "py_np_out"};
static std::string const reference_variant = "cpp_direct";

struct Collector : ::benchmark::ConsoleReporter
{
    struct Key
    {
        std::string group;   // "2d / trivial"
        std::string variant; //
        int cells;

        bool operator<(Key const& o) const
        {
            return std::tie(group, variant, cells) < std::tie(o.group, o.variant, o.cells);
        }
    };

    explicit Collector(bool const quiet)
        : quiet_{quiet}
    {
    }

    bool ReportContext(Context const& context) override
    {
        return quiet_ ? true : ConsoleReporter::ReportContext(context);
    }

    void ReportRuns(std::vector<Run> const& runs) override
    {
        for (auto const& run : runs)
        {
            // with --benchmark_repetitions the mean aggregate is the one to keep
            if (run.run_type == Run::RT_Aggregate && run.aggregate_name != "mean")
                continue;

            auto const parts = split(run.benchmark_name(), '/'); // variant/2d/fn/cells
            if (parts.size() != 4)
                continue;

            auto const nodes = run.counters.find("nodes");
            Key const key{parts[1] + " / " + parts[2], parts[0], std::stoi(parts[3])};

            times_[key] = run.GetAdjustedRealTime();
            if (nodes != run.counters.end())
                nodes_[{key.group, "", key.cells}] = nodes->second.value;
            if (std::find(groups_.begin(), groups_.end(), key.group) == groups_.end())
                groups_.push_back(key.group);
        }

        if (!quiet_)
            ConsoleReporter::ReportRuns(runs);
    }

    void print_summary(std::ostream& os) const
    {
        if (times_.empty())
            return;

        os << "\n" << std::string(96, '=') << "\n"
           << "summary - microseconds per field fill, (relative to " << reference_variant << ")\n"
           << std::string(96, '=') << "\n";

        for (auto const& group : groups_)
        {
            auto const cells = cells_of(group);

            os << "\n  " << group << "\n";
            os << "  " << std::left << std::setw(20) << "cells" << std::right;
            for (auto const c : cells)
                os << std::setw(10) << c << std::setw(8) << "";
            os << "\n  " << std::left << std::setw(20) << "nodes" << std::right;
            for (auto const c : cells)
                os << std::setw(10) << human(nodes_.at({group, "", c})) << std::setw(8) << "";
            os << "\n  " << std::string(20 + 18 * cells.size(), '-') << "\n";

            for (auto const& variant : variant_order)
            {
                if (!std::any_of(cells.begin(), cells.end(), [&](auto const c) {
                        return times_.count({group, variant, c});
                    }))
                    continue;

                os << "  " << std::left << std::setw(20) << variant << std::right;
                for (auto const c : cells)
                {
                    auto const it = times_.find({group, variant, c});
                    if (it == times_.end())
                    {
                        os << std::setw(10) << "-" << std::setw(8) << "";
                        continue;
                    }
                    auto const ref = times_.find({group, reference_variant, c});
                    os << std::setw(10) << time_str(it->second) << std::setw(8)
                       << (ref == times_.end() || variant == reference_variant
                               ? std::string{}
                               : delta_str(it->second, ref->second));
                }
                os << "\n";
            }
        }
        os << std::endl;
    }

private:
    static std::vector<std::string> split(std::string const& s, char const sep)
    {
        std::vector<std::string> out;
        std::size_t start = 0, at = 0;
        while ((at = s.find(sep, start)) != std::string::npos)
            out.push_back(s.substr(start, at - start)), start = at + 1;
        out.push_back(s.substr(start));
        return out;
    }

    static std::string fmt(char const* f, double const v)
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), f, v);
        return buf;
    }

    static std::string time_str(double const us)
    {
        return fmt(us < 10 ? "%.2f" : us < 1000 ? "%.1f" : "%.0f", us);
    }

    // signed relative difference to the reference, in percent
    static std::string delta_str(double const t, double const ref)
    {
        auto const pct = (t / ref - 1) * 100;
        return fmt(std::abs(pct) < 10 ? "%+.1f%%" : "%+.0f%%", pct);
    }

    static std::string human(double const n)
    {
        return n < 1e3 ? fmt("%.0f", n) : n < 1e6 ? fmt("%.1fk", n / 1e3) : fmt("%.1fM", n / 1e6);
    }

    std::vector<int> cells_of(std::string const& group) const
    {
        std::vector<int> out;
        for (auto const& [key, _] : times_)
            if (key.group == group && std::find(out.begin(), out.end(), key.cells) == out.end())
                out.push_back(key.cells);
        std::sort(out.begin(), out.end());
        return out;
    }

    bool quiet_;
    std::vector<std::string> groups_;
    std::map<Key, double> times_;
    std::map<Key, double> nodes_;
};

} // namespace PHARE::bench


int main(int argc, char** argv)
{
    py::scoped_interpreter guard{};

    auto mod = py::module_::create_extension_module("bench_field_init_py", nullptr,
                                                    new py::module_::module_def);
    py::exec(PHARE::bench::python_source, mod.attr("__dict__"));
    py::module_::import("sys").attr("modules")["bench_field_init_py"] = mod;

    try
    {
        PHARE::bench::verify<2, PHARE::bench::UserFn::trivial>();
        PHARE::bench::verify<2, PHARE::bench::UserFn::sincos>();
        PHARE::bench::verify<3, PHARE::bench::UserFn::sincos>();
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // --summary-only is ours, not googlebench's: drop it before Initialize sees it
    bool quiet = false;
    for (int i = 1; i < argc; ++i)
        if (std::string{argv[i]} == "--summary-only")
        {
            quiet = true;
            for (int j = i--; j < argc - 1; ++j)
                argv[j] = argv[j + 1];
            --argc;
        }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    PHARE::bench::Collector collector{quiet};
    ::benchmark::RunSpecifiedBenchmarks(&collector);
    collector.print_summary(std::cout);
    ::benchmark::Shutdown();
    return 0;
}
