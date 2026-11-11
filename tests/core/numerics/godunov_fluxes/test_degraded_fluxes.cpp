#include "gtest/gtest.h"

#include "core/data/grid/gridlayout.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/models/options/mhd_options.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/field/zero_field.hpp"
#include "core/numerics/constrained_transport/non_ideal_electric_field.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "core/numerics/reconstructions/constant.hpp"
#include "core/numerics/reconstructions/linear.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <limits>
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

using GodunovT              = Godunov<Layout, LinearVL, Rusanov, MHDEquations<false>>;
using WenozGodunovT         = Godunov<Layout, WenoZ, Rusanov, MHDEquations<false>>;
using ConstantReconstructor = Reconstructor<ConstantReconstruction<Layout>>;
using LinearVLReconstructor = Reconstructor<LinearReconstruction<Layout, VanLeerLimiter>>;

ZeroVecField<VecFieldT> const zeroB0{MHDQuantity::Vector::B};

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

template<typename GodunovType>
struct DegradedFluxFixtureT
{
    Layout layout{{1.0, 1.0}, {16u, 16u}, {0.0, 0.0}};
    UsableMHDState<2> state{layout, "state"};
    UpwindConstrainedTransportState<VecFieldT> ct{};
    NonIdealElectricFieldState<VecFieldT> dissipative{false};
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

    GodunovType godunov{GodunovInfo{{0.0, 0.0, HyperMode::constant}, adiabaticIndex}, layout};

    explicit DegradedFluxFixtureT(bool const quadratic = false)
    {
        storage.bind(ct);
        storage.bind(fluxes);
        meshData.ghostElemsData._data    = &ghostArray;
        meshData.degradedElemsData._data = &degradedArray;

        layout.evalOnGhostBox(state.rho, [&](auto i, auto j) {
            auto const x                = static_cast<double>(i);
            auto const x2               = quadratic ? x * x : 0.0;
            state.rho(i, j)             = 1.0 + 0.1 * x + 0.02 * x2;
            state.P(i, j)               = 1.0 + 0.05 * x + 0.01 * x2;
            state.V(Component::X)(i, j) = 0.2;
            state.V(Component::Y)(i, j) = 0.0;
            state.V(Component::Z)(i, j) = 0.0;
        });
        for (auto c : {Component::X, Component::Y, Component::Z})
            layout.evalOnGhostBox(state.B1(c), [&](auto i, auto j) {
                state.B1(c)(i, j) = c == Component::X ? 0.3 : (c == Component::Y ? 0.1 : 0.0);
            });
    }

    auto faceXCentering() const { return layout.centering(MHDQuantity::Scalar::ScalarFlux_x); }

    MeshIndex<2> faceIndex(std::size_t ix, std::size_t iy) const
    {
        auto const start = layout.physicalStartIndex(QtyCentering::primal, Direction::X);
        return {start + ix, start + iy};
    }

    void list(MeshIndex<2> const& idx)
    {
        meshData.getDegradedElemsFromCentering(faceXCentering())
            .push_back({{static_cast<std::uint32_t>(idx[0]), static_cast<std::uint32_t>(idx[1])},
                        {},
                        {},
                        false});
    }
};

using DegradedFluxFixture      = DegradedFluxFixtureT<GodunovT>;
using WenozDegradedFluxFixture = DegradedFluxFixtureT<WenozGodunovT>;

auto linearVanLeerFlux(auto& state, MeshIndex<2> const& idx)
{
    auto [uL, uR] = LinearVLReconstructor::reconstruct<Direction::X>(state, zeroB0, idx);
    MHDEquations<false> equations{adiabaticIndex};
    Rusanov riemann{adiabaticIndex};
    auto const fL = equations.compute<Direction::X>(uL);
    auto const fR = equations.compute<Direction::X>(uR);
    return riemann.solve<Direction::X>(uL, uR, fL, fR);
}

} // namespace

TEST(DegradedFluxes, listedFaceGetsFirstOrderIdealFlux)
{
    DegradedFluxFixture fix;
    auto& layout = fix.layout;

    auto const start = layout.physicalStartIndex(QtyCentering::primal, Direction::X);
    MeshIndex<2> const listed{start + 6, start + 5};
    MeshIndex<2> const unlisted{start + 9, start + 5};

    fix.godunov(fix.ct, fix.state, zeroB0, fix.fluxes);

    double const highOrderListed   = fix.fluxes.rho_fx(listed);
    double const highOrderUnlisted = fix.fluxes.rho_fx(unlisted);

    fix.meshData.getDegradedElemsFromCentering(fix.faceXCentering())
        .push_back({{static_cast<std::uint32_t>(listed[0]), static_cast<std::uint32_t>(listed[1])},
                    {},
                    {},
                    false});

    fix.godunov.degrade_fluxes_near_inner_boundary(fix.ct, fix.meshData, fix.state, zeroB0,
                                                   fix.fluxes, NearBodyScheme::FirstOrder);

    auto [uL, uR] = ConstantReconstructor::reconstruct<Direction::X>(fix.state, zeroB0, listed);
    MHDEquations<false> equations{adiabaticIndex};
    Rusanov riemann{adiabaticIndex};
    auto const fL       = equations.compute<Direction::X>(uL);
    auto const fR       = equations.compute<Direction::X>(uR);
    auto const expected = riemann.solve<Direction::X>(uL, uR, fL, fR);

    EXPECT_NEAR(fix.fluxes.rho_fx(listed), expected.rho, 1e-12);
    EXPECT_NEAR(fix.fluxes.Etot_fx(listed), expected.Etot(), 1e-12);
    EXPECT_GT(std::abs(fix.fluxes.rho_fx(listed) - highOrderListed), 1e-6);
    EXPECT_EQ(fix.fluxes.rho_fx(unlisted), highOrderUnlisted);
}

TEST(DegradedFluxes, secondOrderListedFaceMatchesLinearVanLeerFlux)
{
    WenozDegradedFluxFixture fix{true};
    auto const listed   = fix.faceIndex(6, 5);
    auto const unlisted = fix.faceIndex(9, 5);

    fix.godunov(fix.ct, fix.state, zeroB0, fix.fluxes);
    double const wenozListed   = fix.fluxes.rho_fx(listed);
    double const wenozUnlisted = fix.fluxes.rho_fx(unlisted);

    fix.list(listed);
    fix.godunov.degrade_fluxes_near_inner_boundary(fix.ct, fix.meshData, fix.state, zeroB0,
                                                   fix.fluxes, NearBodyScheme::SecondOrder);

    auto const expected = linearVanLeerFlux(fix.state, listed);
    EXPECT_NEAR(fix.fluxes.rho_fx(listed), expected.rho, 1e-12);
    EXPECT_NEAR(fix.fluxes.Etot_fx(listed), expected.Etot(), 1e-12);
    EXPECT_GT(std::abs(fix.fluxes.rho_fx(listed) - wenozListed), 1e-8);
    EXPECT_EQ(fix.fluxes.rho_fx(unlisted), wenozUnlisted);
}

TEST(DegradedFluxes, noneSchemeLeavesFluxesUntouched)
{
    WenozDegradedFluxFixture fix{true};
    auto const listed = fix.faceIndex(6, 5);

    fix.godunov(fix.ct, fix.state, zeroB0, fix.fluxes);
    double const rho  = fix.fluxes.rho_fx(listed);
    double const etot = fix.fluxes.Etot_fx(listed);

    fix.list(listed);
    fix.godunov.degrade_fluxes_near_inner_boundary(fix.ct, fix.meshData, fix.state, zeroB0,
                                                   fix.fluxes, NearBodyScheme::None);

    EXPECT_EQ(fix.fluxes.rho_fx(listed), rho);
    EXPECT_EQ(fix.fluxes.Etot_fx(listed), etot);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
