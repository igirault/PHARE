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
    auto registry = makeMhdDerivedQuantities<State_t, YeeLayout_t>(5. / 3.);
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

    auto& J        = static_cast<VecFieldMHD<dim>&>(out);
    auto& expectedJ = static_cast<VecFieldMHD<dim>&>(expected);
    layout.evalOnGhostBox(J[0], [&](auto const&... args) {
        EXPECT_DOUBLE_EQ(J[0](args...), expectedJ[0](args...));
        EXPECT_DOUBLE_EQ(J[1](args...), expectedJ[1](args...));
        EXPECT_DOUBLE_EQ(J[2](args...), expectedJ[2](args...));
    });
}

TEST_F(MhdDerived, factoryRegistersJ)
{
    auto registry = makeMhdDerivedQuantities<State_t, YeeLayout_t>(5. / 3.);
    EXPECT_NE(registry.find<1>("J"), nullptr);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
