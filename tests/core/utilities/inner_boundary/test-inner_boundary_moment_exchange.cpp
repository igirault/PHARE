#include "gtest/gtest.h"

#include "core/data/field/field.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_manager.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"
#include "core/models/options/mhd_options.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "simulator/simulator_def.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace PHARE::core;

constexpr double adiabaticIndex = 5.0 / 3.0;

inline constexpr PHARE::SimOpts wenozOptions{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::WENOZ,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::None,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<wenozOptions> wenozFieldOptions{};

using Layout      = GridLayout<PHARE::MHDOptions<wenozFieldOptions>{}>;
using ScalarField = Field<2, MHDQuantity::Scalar, double>;
using MeshData    = InnerBoundaryMeshData<2, MHDQuantity>;
using Cell        = std::pair<int, int>;

struct StoredScalar
{
    StoredScalar(std::string const& name, MHDQuantity::Scalar qty, Layout const& layout)
        : storage{layout.allocSize(qty)}
        , field{name, qty, storage.data(), storage.shape()}
    {
    }
    NdArrayVector<2, double> storage;
    ScalarField field;
};

struct State
{
    explicit State(Layout const& layout)
        : rho_s{"rho", MHDQuantity::Scalar::rho, layout}
        , P_s{"P", MHDQuantity::Scalar::P, layout}
        , Etot1_s{"Etot1", MHDQuantity::Scalar::Etot1, layout}
        , rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV}
        , B1{"B1", layout, MHDQuantity::Vector::B1}
        , E{"E", layout, MHDQuantity::Vector::E}
    {
    }
    StoredScalar rho_s;
    StoredScalar P_s;
    StoredScalar Etot1_s;
    ScalarField& rho   = rho_s.field;
    ScalarField& P     = P_s.field;
    ScalarField& Etot1 = Etot1_s.field;
    UsableVecFieldMHD<2> rhoV;
    UsableVecFieldMHD<2> B1;
    UsableVecFieldMHD<2> E;
};

using Manager = InnerBoundaryManager<MHDQuantity, ScalarField, Layout, State>;

double pressureAt(double x, double y)
{
    return 1.0 + 0.3 * std::sin(x / 5.0) + 0.2 * std::cos(y / 7.0);
}

struct Patch
{
    static constexpr int size = 8;

    Patch(int lx, int ly)
        : layout{{1.0, 1.0},
                 {static_cast<std::uint32_t>(size), static_cast<std::uint32_t>(size)},
                 {static_cast<double>(lx), static_cast<double>(ly)},
                 Box<int, 2>{{lx, ly}, {lx + size - 1, ly + size - 1}}}
        , phi_storage{layout.allocSize(MHDQuantity::Scalar::NodeCentered)}
        , manager{std::make_unique<SphereInnerBoundary<2>>("body", Point<double, 2>{32.0, 32.0},
                                                           6.4),
                  InnerBoundaryConditionType::FixedDensity,
                  {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                  {MHDQuantity::Vector::rhoV, MHDQuantity::Vector::B1},
                  adiabaticIndex,
                  FixedStateValues{1.0, 0.0}}
        , state{layout}
    {
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
            elem_storages.emplace_back(
                layout.allocSize(MeshData::scalarFromCentering(MeshData::idxToCentering(i))));
        auto& md = manager.getMeshData();
        ScalarField phi{md.signedDistanceAtNodes.name(), MHDQuantity::Scalar::NodeCentered,
                        phi_storage.data(), phi_storage.shape()};
        md.signedDistanceAtNodes.setBuffer(&phi);
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
        {
            auto const qty = MeshData::scalarFromCentering(MeshData::idxToCentering(i));
            ScalarField tmp{md.elemStatus[i].name(), qty, elem_storages[i].data(),
                            elem_storages[i].shape()};
            md.elemStatus[i].setBuffer(&tmp);
        }
        md.ghostElemsData._data    = &ghost_array;
        md.degradedElemsData._data = &degraded_array;
        manager.classify(layout);

        forEachGhostBoxCell([&](std::uint32_t i, std::uint32_t j, Cell c) {
            double const x = c.first + 0.5, y = c.second + 0.5;
            state.rho(i, j)   = 1.0;
            state.P(i, j)     = pressureAt(x, y);
            state.Etot1(i, j) = state.P(i, j) / (adiabaticIndex - 1.0);
        });
        for (auto comp : {Component::X, Component::Y, Component::Z})
        {
            auto& r = state.rhoV(comp);
            auto& b = state.B1(comp);
            for (auto i = 0u; i < r.shape()[0]; ++i)
                for (auto j = 0u; j < r.shape()[1]; ++j)
                    r(i, j) = 0.0;
            for (auto i = 0u; i < b.shape()[0]; ++i)
                for (auto j = 0u; j < b.shape()[1]; ++j)
                    b(i, j) = 0.0;
        }
    }

    template<typename Fn>
    void forEachGhostBoxCell(Fn&& fn) const
    {
        auto const s0
            = static_cast<int>(layout.physicalStartIndex(QtyCentering::dual, Direction::X));
        auto const lo     = layout.AMRBox().lower;
        auto const& shape = state.Etot1.shape();
        for (auto i = 0u; i < shape[0]; ++i)
            for (auto j = 0u; j < shape[1]; ++j)
                fn(i, j, Cell{static_cast<int>(i) - s0 + lo[0], static_cast<int>(j) - s0 + lo[1]});
    }

    Cell global(Point<std::uint32_t, 2> const& local) const
    {
        auto const s0
            = static_cast<int>(layout.physicalStartIndex(QtyCentering::dual, Direction::X));
        auto const lo = layout.AMRBox().lower;
        return {static_cast<int>(local[0]) - s0 + lo[0], static_cast<int>(local[1]) - s0 + lo[1]};
    }

    auto& cellGhosts()
    {
        return manager.getMeshData().getGhostDataFromCentering(
            layout.centering(MHDQuantity::Scalar::Etot1));
    }

    Layout layout;
    NdArrayVector<2, double> phi_storage;
    std::vector<NdArrayVector<2, double>> elem_storages;
    GhostElemPack<2>::ghost_elem_array_type ghost_array{};
    GhostElemPack<2>::ghost_elem_array_type degraded_array{};
    Manager manager;
    State state;
    std::vector<double> saved;
};

std::vector<std::unique_ptr<Patch>> tilePatches()
{
    std::vector<std::unique_ptr<Patch>> patches;
    for (int lx = 16; lx < 48; lx += Patch::size)
        for (int ly = 16; ly < 48; ly += Patch::size)
            patches.push_back(std::make_unique<Patch>(lx, ly));
    return patches;
}

void emulateFillIfNaN(std::vector<std::unique_ptr<Patch>>& patches, bool reverse)
{
    std::vector<std::map<Cell, double>> sources(patches.size());
    for (std::size_t p = 0; p < patches.size(); ++p)
        patches[p]->forEachGhostBoxCell([&](std::uint32_t i, std::uint32_t j, Cell c) {
            sources[p][c] = patches[p]->state.Etot1(i, j);
        });

    for (std::size_t d = 0; d < patches.size(); ++d)
        patches[d]->forEachGhostBoxCell([&](std::uint32_t i, std::uint32_t j, Cell c) {
            for (std::size_t k = 0; k < patches.size(); ++k)
            {
                auto const s = reverse ? patches.size() - 1 - k : k;
                if (s == d)
                    continue;
                auto it = sources[s].find(c);
                if (it == sources[s].end())
                    continue;
                auto& value = patches[d]->state.Etot1(i, j);
                if (std::isnan(value))
                    value = it->second;
            }
        });
}

} // namespace

TEST(InnerBoundaryMomentExchange, everyComputedElementEndsWithItsComputedValue)
{
    for (bool reverse : {false, true})
    {
        auto patches = tilePatches();

        std::map<Cell, double> computed;
        for (auto& p : patches)
        {
            p->manager.applyToMoments(p->layout,
                                      InnerBCContext<State>{p->state, p->state, 0.0, 0.0});
            for (auto const& g : p->cellGhosts())
                if (g.mirrorIsInterpolable)
                {
                    auto const c     = p->global(g.index);
                    auto const value = p->state.Etot1(g.index);
                    auto [it, fresh] = computed.emplace(c, value);
                    if (!fresh)
                    {
                        EXPECT_NEAR(it->second, value, 1e-12) << "patches disagree";
                    }
                }
        }
        ASSERT_FALSE(computed.empty());

        for (auto& p : patches)
            p->manager.markUncomputedMoments(p->layout, p->state, p->saved);

        emulateFillIfNaN(patches, reverse);

        for (auto& p : patches)
            p->manager.restoreUncomputedMoments(p->layout, p->state, p->saved);

        std::size_t checked = 0, wrong = 0;
        for (auto& p : patches)
            for (auto const& g : p->cellGhosts())
            {
                auto const c  = p->global(g.index);
                auto const it = computed.find(c);
                if (it == computed.end())
                {
                    EXPECT_TRUE(g.stale);
                    continue;
                }
                ++checked;
                if (std::abs(p->state.Etot1(g.index) - it->second) > 1e-12)
                    ++wrong;
                EXPECT_FALSE(g.stale);
            }
        EXPECT_GT(checked, 0u);
        EXPECT_EQ(wrong, 0u) << "reverse=" << reverse;
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
