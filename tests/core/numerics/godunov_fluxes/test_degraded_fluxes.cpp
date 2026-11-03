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
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

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

using GodunovT              = Godunov<Layout, LinearVL, Rusanov, MHDEquations<false>>;
using ConstantReconstructor = Reconstructor<ConstantReconstruction<Layout>>;

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

struct DegradedFluxFixture
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

    GodunovT godunov{GodunovInfo{{0.0, 0.0, HyperMode::constant}, adiabaticIndex}, layout};

    DegradedFluxFixture()
    {
        storage.bind(ct);
        storage.bind(fluxes);
        meshData.ghostElemsData._data    = &ghostArray;
        meshData.degradedElemsData._data = &degradedArray;

        layout.evalOnGhostBox(state.rho, [&](auto i, auto j) {
            state.rho(i, j)             = 1.0 + 0.1 * static_cast<double>(i);
            state.P(i, j)               = 1.0 + 0.05 * static_cast<double>(i);
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
};

} // namespace

TEST(DegradedFluxes, listedFaceGetsFirstOrderIdealFlux)
{
    DegradedFluxFixture fix;
    auto& layout = fix.layout;

    auto const start = layout.physicalStartIndex(QtyCentering::primal, Direction::X);
    MeshIndex<2> const listed{start + 6, start + 5};
    MeshIndex<2> const unlisted{start + 9, start + 5};

    fix.godunov(fix.ct, fix.dissipative, fix.state, zeroB0, fix.fluxes);

    double const highOrderListed   = fix.fluxes.rho_fx(listed);
    double const highOrderUnlisted = fix.fluxes.rho_fx(unlisted);

    fix.meshData.getDegradedElemsFromCentering(fix.faceXCentering())
        .push_back({{static_cast<std::uint32_t>(listed[0]), static_cast<std::uint32_t>(listed[1])},
                    {},
                    {},
                    false});

    fix.godunov.degrade_fluxes_near_inner_boundary(fix.ct, fix.meshData, fix.state, zeroB0,
                                                   fix.fluxes);

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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
