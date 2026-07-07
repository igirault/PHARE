#include "core/data/derived_quantity/mhd_derived_quantities.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/numerics/ampere/ampere.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <cmath>

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t                = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t               = TestGridLayout<YeeLayout_t>;
using State_t                    = MHDState<VecFieldMHD<dim>>;

struct MhdDerived : public ::testing::Test
{
    GridLayout_t layout{10};
    UsableMHDState<dim> state{layout, std::string{"state"}};

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

    layout.evalOnGhostBox(out,
                          [&](auto const&... args) { EXPECT_NEAR(out(args...), p_ref, 1e-12); });
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

    layout.evalOnGhostBox(
        out, [&](auto const&... args) { EXPECT_NEAR(out(args...), inv_dx, 1e-10 * inv_dx); });
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
    (void)nx;
    (void)ny;

    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};
    MhdDivB<State_t, YeeLayout_t>{}.compute(*state, layout, out, 0.0);

    layout.evalOnGhostBox(
        out, [&](auto const&... args) { EXPECT_NEAR(out(args...), 0.0, 1e-12 * inv_dx); });
}

TEST_F(MhdDerived, factoryRegistersVPandDivB)
{
    auto registry
        = makeMhdDerivedQuantities<State_t, YeeLayout_t>(5. / 3., 0.0, 0.0, HyperMode::constant,
                                                         /*hall=*/false);
    EXPECT_NE(registry.find<1>("V"), nullptr);
    EXPECT_NE(registry.find<0>("P"), nullptr);
    EXPECT_NE(registry.find<0>("divB"), nullptr);
    EXPECT_EQ(registry.find<0>("V"), nullptr);
}

TEST_F(MhdDerived, currentDensityMatchesAmpereDirectly)
{
    // Bx(i,j) = i, By(i,j) = j, Bz = 0 => J = curl(B) is a simple, known
    // combination of the mesh spacings; check MhdCurrentDensity agrees
    // exactly with calling core::Ampere directly on the same B.
    auto& Bx = state.B[0];
    auto& By = state.B[1];
    for (std::uint32_t i = 0; i < Bx.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bx.shape()[1]; ++j)
            Bx(i, j) = static_cast<double>(i);
    for (std::uint32_t i = 0; i < By.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < By.shape()[1]; ++j)
            By(i, j) = static_cast<double>(j);
    fill(state.B[2], 0.0);

    UsableVecFieldMHD<dim> out{"out", layout, MHDQuantity::Vector::VecElike};
    MhdCurrentDensity<State_t, YeeLayout_t>{}.compute(*state, layout, out, 0.0);

    UsableVecFieldMHD<dim> expected{"expected", layout, MHDQuantity::Vector::VecElike};
    Ampere<YeeLayout_t>{layout}(static_cast<VecFieldMHD<dim> const&>(state.B),
                                static_cast<VecFieldMHD<dim>&>(expected));

    auto& J         = static_cast<VecFieldMHD<dim>&>(out);
    auto& expectedJ = static_cast<VecFieldMHD<dim>&>(expected);
    layout.evalOnGhostBox(
        J[0], [&](auto const&... args) { EXPECT_DOUBLE_EQ(J[0](args...), expectedJ[0](args...)); });
    layout.evalOnGhostBox(
        J[1], [&](auto const&... args) { EXPECT_DOUBLE_EQ(J[1](args...), expectedJ[1](args...)); });
    layout.evalOnGhostBox(
        J[2], [&](auto const&... args) { EXPECT_DOUBLE_EQ(J[2](args...), expectedJ[2](args...)); });
}

TEST_F(MhdDerived, factoryRegistersJ)
{
    auto registry
        = makeMhdDerivedQuantities<State_t, YeeLayout_t>(5. / 3., 0.0, 0.0, HyperMode::constant,
                                                         /*hall=*/false);
    EXPECT_NE(registry.find<1>("J"), nullptr);
    EXPECT_NE(registry.find<1>("E"), nullptr);
}

TEST_F(MhdDerived, electricFieldIdealOnlyMatchesMinusVCrossB)
{
    // eta = nu = 0 => E should reduce to -V x B + Hall term. Zero out J (B
    // uniform => curl(B) = 0) so only the ideal term (-V x B) survives, and
    // check it against a hand-computed cross product at a single point.
    fill(state.rho, 2.0);
    fill(state.rhoV[0], 2.0); // Vx = 1
    fill(state.rhoV[1], 4.0); // Vy = 2
    fill(state.rhoV[2], 0.0); // Vz = 0
    fill(state.B[0], 0.0);
    fill(state.B[1], 0.0);
    fill(state.B[2], 3.0); // uniform B => J = curl(B) = 0, Hall term vanishes too

    UsableVecFieldMHD<dim> out{"out", layout, MHDQuantity::Vector::VecElike};
    MhdElectricField<State_t, YeeLayout_t>{0.0, 0.0, HyperMode::constant, /*hall=*/true}.compute(
        *state, layout, out, 0.0);

    // -V x B with V=(1,2,0), B=(0,0,3): (-V x B) = (-(2*3-0*0), -(0*0-1*3), -(1*0-2*0))
    //                                            = (-6, 3, 0)
    // MhdElectricField only fills 2 cells in from the edge of the ghost box (Ampere
    // itself shrinks J by 1, and the hyper-resistive laplacian needs one more J
    // neighbor beyond that), so check over that same shrunk region rather than the
    // full ghost box. Each component has a different (edge) centering, so each
    // must be checked over its own shrunk ghost box rather than reusing E[0]'s.
    auto& E = static_cast<VecFieldMHD<dim>&>(out);
    Point<std::uint32_t, dim> shrink;
    for (std::size_t i = 0; i < dim; ++i)
        shrink[i] = 2;
    layout.evalOnShrinkedGhostBox(
        E[0], shrink, [&](auto const&... args) { EXPECT_NEAR(E[0](args...), -6.0, 1e-10); });
    layout.evalOnShrinkedGhostBox(
        E[1], shrink, [&](auto const&... args) { EXPECT_NEAR(E[1](args...), 3.0, 1e-10); });
    layout.evalOnShrinkedGhostBox(
        E[2], shrink, [&](auto const&... args) { EXPECT_NEAR(E[2](args...), 0.0, 1e-10); });
}

TEST_F(MhdDerived, electricFieldHallTermUsesPerComponentRhoProjection)
{
    // Regression test for a bug where the Hall term's density projection
    // (rhoE) in hall_() was always computed with cellCenterToEdgeX,
    // regardless of the requested E component. Here we exercise the Y
    // component: with V=0 the ideal term vanishes, so E_y reduces exactly to
    // the Hall term (Jz*bx - Jx*bz)/rhoE, where rhoE must be projected with
    // cellCenterToEdgeY (shift in X), not cellCenterToEdgeX (shift in Y).
    //
    // rho varies quadratically with the local y-index and is uniform in x,
    // so cellCenterToEdgeY (which only averages in x) reproduces rho's exact
    // pointwise value in y, while cellCenterToEdgeX (which averages in y)
    // does not -- making the two projections numerically distinguishable.
    auto& rho = state.rho;
    for (std::uint32_t i = 0; i < rho.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < rho.shape()[1]; ++j)
            rho(i, j) = static_cast<double>(j * j) + 1.0; // +1 to keep it away from 0

    for (std::size_t c = 0; c < 3; ++c)
        fill(state.rhoV[c], 0.0); // V = 0 => ideal term vanishes, only Hall term remains

    auto& Bx = state.B[0];
    auto& By = state.B[1];
    auto& Bz = state.B[2];
    for (std::uint32_t i = 0; i < Bx.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bx.shape()[1]; ++j)
            Bx(i, j) = 2.0 * static_cast<double>(j);
    for (std::uint32_t i = 0; i < By.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < By.shape()[1]; ++j)
            By(i, j) = static_cast<double>(i);
    for (std::uint32_t i = 0; i < Bz.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bz.shape()[1]; ++j)
            Bz(i, j) = static_cast<double>(j);

    UsableVecFieldMHD<dim> out{"out", layout, MHDQuantity::Vector::VecElike};
    MhdElectricField<State_t, YeeLayout_t>{0.0, 0.0, HyperMode::constant, /*hall=*/true}.compute(
        *state, layout, out, 0.0);

    UsableVecFieldMHD<dim> J{"J", layout, MHDQuantity::Vector::VecElike};
    Ampere<YeeLayout_t>{layout}(static_cast<VecFieldMHD<dim> const&>(state.B),
                                static_cast<VecFieldMHD<dim>&>(J));

    auto& E        = static_cast<VecFieldMHD<dim>&>(out);
    auto& Jvec     = static_cast<VecFieldMHD<dim>&>(J);
    auto const& Bx_ = static_cast<VecFieldMHD<dim> const&>(state.B)[0];
    auto const& Bz_ = static_cast<VecFieldMHD<dim> const&>(state.B)[2];

    Point<std::uint32_t, dim> shrink;
    for (std::size_t i = 0; i < dim; ++i)
        shrink[i] = 2;

    layout.evalOnShrinkedGhostBox(E[1], shrink, [&](auto const&... args) {
        MeshIndex<dim> const index{args...};

        auto const bx = YeeLayout_t::template project<YeeLayout_t::BxToEy>(Bx_, index);
        auto const bz = YeeLayout_t::template project<YeeLayout_t::BzToEy>(Bz_, index);
        auto const rhoE_correct
            = YeeLayout_t::template project<YeeLayout_t::cellCenterToEdgeY>(rho, index);
        auto const rhoE_wrong
            = YeeLayout_t::template project<YeeLayout_t::cellCenterToEdgeX>(rho, index);

        // Sanity check: the two projections must actually differ here, or this
        // test wouldn't be able to distinguish the fix from the bug.
        ASSERT_NE(rhoE_correct, rhoE_wrong);

        auto const expected
            = (Jvec[2](args...) * bx - Jvec[0](args...) * bz) / rhoE_correct;
        EXPECT_NEAR(E[1](args...), expected, 1e-10 * std::abs(expected) + 1e-12);
    });
}

TEST_F(MhdDerived, electricFieldHyperresistiveSpatialUsesPerComponentProjection)
{
    // Regression test for a bug where the spatial hyper-resistive term's density and
    // magnetic field projections were always computed with the same component (X),
    // regardless of the requested E component. Here we exercise the Y component:
    // with eta=0 the ohmic term vanishes, and with V=0 the ideal term vanishes,
    // so E_y reduces exactly to the Hall term + hyper-resistive term. The hyper-resistive
    // term is -nu * (b/rho + 1) * laplacian(J_y), where b and rho must be projected
    // with component Y (cellCenterToEdgeY and *ToEy projections), not X.
    //
    // rho varies quadratically with the local y-index and is uniform in x,
    // so cellCenterToEdgeY reproduces rho's exact pointwise value in y,
    // while cellCenterToEdgeX (which averages in y) does not -- making the two
    // projections numerically distinguishable.
    //
    // For nonzero laplacian(J), we set B so that curl(B) = J is non-constant AND
    // non-linear in space. In particular, 2D Ampere gives J_y = -dBz/dx, so Bz
    // must be cubic in i: a merely quadratic Bz(i) would make J_y linear in i,
    // whose second derivative (and hence its laplacian) is identically zero and
    // the guard below would never fire.

    auto& rho = state.rho;
    for (std::uint32_t i = 0; i < rho.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < rho.shape()[1]; ++j)
            rho(i, j) = static_cast<double>(j * j) + 1.0;

    for (std::size_t c = 0; c < 3; ++c)
        fill(state.rhoV[c], 0.0); // V = 0 => ideal term vanishes

    auto& Bx = state.B[0];
    auto& By = state.B[1];
    auto& Bz = state.B[2];
    // Set B so curl(B) is non-constant (produces non-zero laplacian(J))
    for (std::uint32_t i = 0; i < Bx.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bx.shape()[1]; ++j)
            Bx(i, j) = static_cast<double>(i * j);
    for (std::uint32_t i = 0; i < By.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < By.shape()[1]; ++j)
            By(i, j) = static_cast<double>(i) * static_cast<double>(i + 1);
    for (std::uint32_t i = 0; i < Bz.shape()[0]; ++i)
        for (std::uint32_t j = 0; j < Bz.shape()[1]; ++j)
            Bz(i, j) = static_cast<double>(i) * static_cast<double>(i + 1)
                           * static_cast<double>(i + 2)
                       + static_cast<double>(j) * static_cast<double>(j + 1);

    UsableVecFieldMHD<dim> out{"out", layout, MHDQuantity::Vector::VecElike};
    MhdElectricField<State_t, YeeLayout_t>{0.0, 1.0, HyperMode::spatial, /*hall=*/true}.compute(
        *state, layout, out, 0.0);

    UsableVecFieldMHD<dim> J{"J", layout, MHDQuantity::Vector::VecElike};
    Ampere<YeeLayout_t>{layout}(static_cast<VecFieldMHD<dim> const&>(state.B),
                                static_cast<VecFieldMHD<dim>&>(J));

    auto& E        = static_cast<VecFieldMHD<dim>&>(out);
    auto& Jvec     = static_cast<VecFieldMHD<dim>&>(J);
    auto const& Bx_ = static_cast<VecFieldMHD<dim> const&>(state.B)[0];
    auto const& By_ = static_cast<VecFieldMHD<dim> const&>(state.B)[1];
    auto const& Bz_ = static_cast<VecFieldMHD<dim> const&>(state.B)[2];

    Point<std::uint32_t, dim> shrink;
    for (std::size_t i = 0; i < dim; ++i)
        shrink[i] = 2;

    layout.evalOnShrinkedGhostBox(E[1], shrink, [&](auto const&... args) {
        MeshIndex<dim> const index{args...};

        auto const bx = YeeLayout_t::template project<YeeLayout_t::BxToEy>(Bx_, index);
        auto const by = YeeLayout_t::template project<YeeLayout_t::ByToEy>(By_, index);
        auto const bz = YeeLayout_t::template project<YeeLayout_t::BzToEy>(Bz_, index);
        auto const rhoE_correct
            = YeeLayout_t::template project<YeeLayout_t::cellCenterToEdgeY>(rho, index);
        auto const rhoE_wrong
            = YeeLayout_t::template project<YeeLayout_t::cellCenterToEdgeX>(rho, index);

        // Sanity check: the two projections must actually differ here.
        ASSERT_NE(rhoE_correct, rhoE_wrong);

        auto const b = std::sqrt(bx * bx + by * by + bz * bz);
        auto const Jy_laplacian = layout.laplacian(Jvec[1], index);

        // Only check at points where the hyper-resistive term is substantial
        if (std::abs(Jy_laplacian) > 1e-14)
        {
            // E_y = Hall term + hyper-resistive term
            // Hall: (J_z * b_x - J_x * b_z) / rho_E
            // Hyper: -nu * (b/rho + 1) * minMeshSize^2 * laplacian(J_y)
            auto const m   = layout.meshSize();
            auto const dl  = std::min(m[0], m[1]);
            auto const dl2 = dl * dl;
            auto const hall_term = (Jvec[2](args...) * bx - Jvec[0](args...) * bz) / rhoE_correct;
            auto const hyper_term = -1.0 * (b / rhoE_correct + 1.0) * dl2 * Jy_laplacian;
            auto const expected = hall_term + hyper_term;

            EXPECT_NEAR(E[1](args...), expected, 1e-10 * (std::abs(expected) + 1.0) + 1e-12);
        }
    });
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
