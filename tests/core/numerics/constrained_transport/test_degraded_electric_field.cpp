#include "gtest/gtest.h"

#include "core/data/grid/gridlayout.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/models/options/mhd_options.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/field/zero_field.hpp"
#include "core/numerics/constrained_transport/non_ideal_electric_field.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "core/numerics/reconstructions/constant.hpp"
#include "core/numerics/reconstructions/linear.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

namespace
{
using namespace PHARE::core;

constexpr double adiabaticIndex = 5.0 / 3.0;

inline constexpr PHARE::SimOpts linearOptions{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::Linear,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::VanLeer,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<linearOptions> linearFieldOptions{};

using Layout    = GridLayout<PHARE::MHDOptions<linearFieldOptions>{}>;
using FieldT    = FieldMHD<2>;
using VecFieldT = VecFieldMHD<2>;
using MeshData  = InnerBoundaryMeshData<2, MHDQuantity>;

template<typename L>
using LinearVL = LinearReconstruction<L, VanLeerLimiter>;
template<typename L>
using WenoZ = WENOZReconstruction<L>;
template<typename L>
using Constant = ConstantReconstruction<L>;

using GodunovT     = Godunov<Layout, WenoZ, Rusanov, MHDEquations<false>>;
using WenozUct     = UpwindConstrainedTransport<Layout, WenoZ>;
using LinearUct    = UpwindConstrainedTransport<Layout, LinearVL>;
using ConstantUct  = UpwindConstrainedTransport<Layout, Constant>;
using NonIdealT    = NonIdealElectricFieldState<VecFieldT>;
using ElemIndex    = Point<std::uint32_t, 2>;
using ProbeIndices = std::array<MeshIndex<2>, 3>;

ZeroVecField<VecFieldT> const zeroB0{MHDQuantity::Vector::B};
OhmInfo const ohmInfo{0.0, 0.0, HyperMode::constant};

struct Storage
{
    explicit Storage(Layout const& layout)
        : layout_{layout}
    {
    }

    void bind(auto& resourceUser)
    {
        std::apply([&](auto&... views) { (bindOne_(views), ...); },
                   resourceUser.getCompileTimeResourcesViewList());
    }

    void bindVec(VecFieldT& v) { bindOne_(v); }

private:
    void bindOne_(FieldT& f)
    {
        fields_.emplace_back(
            std::make_unique<UsableFieldMHD<2>>(f.name(), layout_, f.physicalQuantity()));
        fields_.back()->set_on(f);
    }

    void bindOne_(VecFieldT& v)
    {
        vecs_.emplace_back(
            std::make_unique<UsableVecFieldMHD<2>>(v.name(), layout_, v.physicalQuantity()));
        vecs_.back()->set_on(v);
    }

    Layout const& layout_;
    std::vector<std::unique_ptr<UsableFieldMHD<2>>> fields_;
    std::vector<std::unique_ptr<UsableVecFieldMHD<2>>> vecs_;
};

struct DegradedElectricFieldFixture
{
    Layout layout{{1.0, 1.0}, {16u, 16u}, {0.0, 0.0}};
    UsableMHDState<2> state{layout, "state"};
    UpwindConstrainedTransportState<VecFieldT> ct{};
    NonIdealT inactive{false};
    NonIdealT active{true};
    AllFluxes<FieldT, VecFieldT> fluxes{{"rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                                        {"rhoV_fx", MHDQuantity::Vector::VecFlux_x},
                                        {"B_fx", MHDQuantity::Vector::VecFlux_x},
                                        {"Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},
                                        {"rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                                        {"rhoV_fy", MHDQuantity::Vector::VecFlux_y},
                                        {"B_fy", MHDQuantity::Vector::VecFlux_y},
                                        {"Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},
                                        {"rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                                        {"rhoV_fz", MHDQuantity::Vector::VecFlux_z},
                                        {"B_fz", MHDQuantity::Vector::VecFlux_z},
                                        {"Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}};
    Storage storage{layout};

    MeshData meshData{"test"};
    GhostElemPack<2>::ghost_elem_array_type ghostArray{};
    GhostElemPack<2>::ghost_elem_array_type degradedArray{};

    GodunovT godunov{GodunovInfo{ohmInfo, adiabaticIndex}, layout};

    DegradedElectricFieldFixture()
    {
        storage.bind(ct);
        storage.bind(fluxes);
        storage.bindVec(active.E());
        meshData.ghostElemsData._data    = &ghostArray;
        meshData.degradedElemsData._data = &degradedArray;

        layout.evalOnGhostBox(state.rho, [&](auto i, auto j) {
            auto const x                = static_cast<double>(i);
            auto const y                = static_cast<double>(j);
            state.rho(i, j)             = 1.0 + 0.1 * x + 0.02 * x * x + 0.01 * y * y;
            state.P(i, j)               = 1.0 + 0.05 * x + 0.01 * x * x;
            state.V(Component::X)(i, j) = 0.2 + 0.01 * y * y;
            state.V(Component::Y)(i, j) = 0.1 + 0.02 * x * x;
            state.V(Component::Z)(i, j) = 0.05;
        });
        for (auto c : {Component::X, Component::Y, Component::Z})
            layout.evalOnGhostBox(state.B1(c), [&](auto i, auto j) {
                auto const x      = static_cast<double>(i);
                auto const y      = static_cast<double>(j);
                state.B1(c)(i, j) = c == Component::X   ? 0.3 + 0.01 * y * y
                                    : c == Component::Y ? 0.1 + 0.01 * x * x
                                                        : 0.2 + 0.005 * x * y + 0.003 * x * x;
            });

        std::array<double, 3> const Euniform{0.01, 0.02, 0.03};
        for (auto c : {Component::X, Component::Y, Component::Z})
            layout.evalOnGhostBox(active.E()(c), [&](auto i, auto j) {
                active.E()(c)(i, j) = Euniform[static_cast<std::size_t>(c)];
            });

        godunov(ct, inactive, state, zeroB0, fluxes);
    }

    MeshIndex<2> at(Component c, std::size_t ix, std::size_t iy) const
    {
        auto const& Ec = state.E(c);
        return {layout.physicalStartIndex(Ec, Direction::X) + ix,
                layout.physicalStartIndex(Ec, Direction::Y) + iy};
    }

    void list(Component c, MeshIndex<2> const& idx)
    {
        meshData.getDegradedElemsFromCentering(layout.centering(state.E(c)))
            .push_back(
                {ElemIndex{static_cast<std::uint32_t>(idx[0]), static_cast<std::uint32_t>(idx[1])},
                 {},
                 {},
                 false});
    }

    template<typename Uct>
    std::array<double, 3> run(NonIdealT const& nonIdeal, ProbeIndices const& probes)
    {
        Uct{ohmInfo, layout}(ct, nonIdeal, state, zeroB0);
        return read(probes);
    }

    std::array<double, 3> read(ProbeIndices const& probes) const
    {
        return {state.E(Component::Z)(probes[0]), state.E(Component::Y)(probes[1]),
                state.E(Component::Z)(probes[2])};
    }

    ProbeIndices probes() const
    {
        return {at(Component::Z, 6, 5), at(Component::Y, 7, 4), at(Component::Z, 10, 9)};
    }

    void listProbes(ProbeIndices const& p)
    {
        list(Component::Z, p[0]);
        list(Component::Y, p[1]);
    }
};

} // namespace

TEST(DegradedElectricField, secondOrderListedEdgeMatchesLinearUct)
{
    for (bool withNonIdeal : {false, true})
    {
        DegradedElectricFieldFixture fix;
        auto const& nonIdeal = withNonIdeal ? fix.active : fix.inactive;
        auto const probes    = fix.probes();

        auto const expected = fix.run<LinearUct>(nonIdeal, probes);

        auto const wenoz = fix.run<WenozUct>(nonIdeal, probes);
        EXPECT_GT(std::abs(wenoz[0] - expected[0]), 1e-10);
        EXPECT_GT(std::abs(wenoz[1] - expected[1]), 1e-10);
        fix.listProbes(probes);
        WenozUct{ohmInfo, fix.layout}.degrade_E_near_inner_boundary(
            fix.ct, nonIdeal, fix.state, zeroB0, fix.meshData, NearBodyScheme::SecondOrder);
        auto const got = fix.read(probes);

        EXPECT_NEAR(got[0], expected[0], 1e-12) << "nonIdeal " << withNonIdeal;
        EXPECT_NEAR(got[1], expected[1], 1e-12) << "nonIdeal " << withNonIdeal;
    }
}

TEST(DegradedElectricField, unlistedEdgeKeepsWenozValue)
{
    DegradedElectricFieldFixture fix;
    auto const probes = fix.probes();

    auto const wenoz = fix.run<WenozUct>(fix.active, probes);
    fix.listProbes(probes);
    WenozUct{ohmInfo, fix.layout}.degrade_E_near_inner_boundary(
        fix.ct, fix.active, fix.state, zeroB0, fix.meshData, NearBodyScheme::SecondOrder);

    EXPECT_EQ(fix.read(probes)[2], wenoz[2]);
}

TEST(DegradedElectricField, firstOrderMatchesConstantIdealUct)
{
    DegradedElectricFieldFixture fix;
    auto const probes = fix.probes();

    auto const expected = fix.run<ConstantUct>(fix.inactive, probes);

    fix.run<WenozUct>(fix.active, probes);
    fix.listProbes(probes);
    WenozUct{ohmInfo, fix.layout}.degrade_E_near_inner_boundary(
        fix.ct, fix.active, fix.state, zeroB0, fix.meshData, NearBodyScheme::FirstOrder);
    auto const got = fix.read(probes);

    EXPECT_NEAR(got[0], expected[0], 1e-12);
    EXPECT_NEAR(got[1], expected[1], 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
