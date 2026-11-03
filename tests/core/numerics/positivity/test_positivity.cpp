#include "gtest/gtest.h"

#include "core/data/grid/gridlayout.hpp"
#include "core/models/options/mhd_options.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/numerics/primite_conservative_converter/positivity_floors.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/data/field/zero_field.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{
using namespace PHARE::core;

constexpr double adiabaticIndex = 5.0 / 3.0;
constexpr double eps            = 1e-12;

inline constexpr PHARE::SimOpts constantOptions{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::Constant,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::None,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<constantOptions> constantFieldOptions{};

using TestLayout = GridLayout<PHARE::MHDOptions<constantFieldOptions>{}>;

struct StateFixture
{
    TestLayout layout{{1.0, 1.0}, {8u, 8u}, {0.0, 0.0}};
    UsableMHDState<2> state{layout, "state"};

    StateFixture()
    {
        layout.evalOnGhostBox(state.rho, [&](auto i, auto j) {
            state.rho(i, j)   = 1.0 + 0.1 * static_cast<double>(i);
            state.Etot1(i, j) = 2.0;
            for (auto c : {Component::X, Component::Y, Component::Z})
                state.rhoV(c)(i, j) = c == Component::X ? 0.2 : 0.0;
        });
        for (auto c : {Component::X, Component::Y, Component::Z})
            layout.evalOnGhostBox(state.B1(c), [&](auto i, auto j) {
                state.B1(c)(i, j) = c == Component::Y ? 0.5 : 0.0;
            });
    }

    MeshIndex<2> cell(std::uint32_t i, std::uint32_t j) const
    {
        auto const s = layout.physicalStartIndex(QtyCentering::dual, Direction::X);
        return {s + i, s + j};
    }

    auto convert(PositivityFloors const& floors = {})
    {
        return ToPrimitiveConverter<TestLayout>{layout}(
            adiabaticIndex, state.rho, state.rhoV, state.B1, state.Etot1, state.V, state.P, floors);
    }

    double b1yAtCell(MeshIndex<2> idx) const
    {
        return TestLayout::project<MHDQuantity::Scalar::By, MHDQuantity::Scalar::rho>(
            state.B1(Component::Y), idx);
    }
};

template<typename GridLayout, int left, int right>
struct ScaledReconstruction
{
    using GridLayout_t            = GridLayout;
    static constexpr auto nghosts = 1;

    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        return std::make_pair(left * F(index), right * F(index));
    }

    template<auto direction, auto src, auto dst, typename Field>
    static auto center_reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        auto const u = GridLayout::template project<src, dst>(F, index);
        return std::make_pair(left * u, right * u);
    }
};

template<typename L>
using NegativeLeft = ScaledReconstruction<L, -1, 3>;

template<typename L>
using PositiveBoth = ScaledReconstruction<L, 2, 3>;

using NegativeReconstructor = Reconstructor<NegativeLeft<TestLayout>>;
using PositiveReconstructor = Reconstructor<PositiveBoth<TestLayout>>;

using VecFieldT = VecFieldMHD<2>;

ZeroVecField<VecFieldT> const zeroB0{MHDQuantity::Vector::B};

} // namespace

TEST(PositivityFloors, pressureFloorRewritesEtot1)
{
    StateFixture fix;
    auto const idx   = fix.cell(3, 3);
    double const rho = fix.state.rho(idx), vx = fix.state.rhoV(Component::X)(idx) / rho;
    fix.state.Etot1(idx)
        = eosPToEtot1(adiabaticIndex, rho, vx, 0., 0., 0., fix.b1yAtCell(idx), 0., -1e-3);

    auto const counts = fix.convert({0.0, 1e-2});

    EXPECT_NEAR(fix.state.P(idx), 1e-2, eps);
    EXPECT_NEAR(eosEtot1ToP(adiabaticIndex, rho, vx, 0., 0., 0., fix.b1yAtCell(idx), 0.,
                            fix.state.Etot1(idx)),
                1e-2, eps);
    EXPECT_EQ(counts.pressure, 1u);
    EXPECT_EQ(counts.density, 0u);
}

TEST(PositivityFloors, pressureFloorCatchesNaN)
{
    StateFixture fix;
    auto const idx       = fix.cell(2, 5);
    fix.state.Etot1(idx) = std::numeric_limits<double>::quiet_NaN();

    fix.convert({0.0, 1e-2});

    EXPECT_NEAR(fix.state.P(idx), 1e-2, eps);
    EXPECT_TRUE(std::isfinite(fix.state.Etot1(idx)));
}

TEST(PositivityFloors, densityFloorKeepsMomentum)
{
    StateFixture fix;
    auto const idx     = fix.cell(4, 1);
    fix.state.rho(idx) = 1e-5;

    auto const counts = fix.convert({1e-2, 0.0});

    EXPECT_NEAR(fix.state.rho(idx), 1e-2, eps);
    EXPECT_NEAR(fix.state.rhoV(Component::X)(idx), 0.2, eps);
    EXPECT_NEAR(fix.state.V(Component::X)(idx), 0.2 / 1e-2, 1e-9);
    EXPECT_EQ(counts.density, 1u);
}

TEST(PositivityFloors, floorsOffChangeNothing)
{
    StateFixture fix;
    auto const idx       = fix.cell(3, 3);
    fix.state.rho(idx)   = 1e-5;
    fix.state.Etot1(idx) = -1.0;

    StateFixture ref;
    ref.state.rho(idx)   = 1e-5;
    ref.state.Etot1(idx) = -1.0;

    auto const counts = fix.convert();
    ToPrimitiveConverter<TestLayout>{ref.layout}(adiabaticIndex, ref.state.rho, ref.state.rhoV,
                                                 ref.state.B1, ref.state.Etot1, ref.state.V,
                                                 ref.state.P);

    EXPECT_EQ(counts.density + counts.pressure, 0u);
    fix.layout.evalOnGhostBox(fix.state.rho, [&](auto i, auto j) {
        EXPECT_EQ(fix.state.rho(i, j), ref.state.rho(i, j));
        EXPECT_EQ(fix.state.Etot1(i, j), ref.state.Etot1(i, j));
        EXPECT_TRUE(fix.state.P(i, j) == ref.state.P(i, j)
                    || (std::isnan(fix.state.P(i, j)) && std::isnan(ref.state.P(i, j))));
    });
}

TEST(ReconstructionPositivity, reconstructionFallsBackOnNonPositive)
{
    StateFixture fix;
    fix.convert();
    auto const idx  = fix.cell(4, 4);
    auto const prev = TestLayout::previous<Direction::X>(idx);

    auto const [uL, uR] = NegativeReconstructor::reconstruct<Direction::X>(fix.state, zeroB0, idx);

    EXPECT_EQ(uL.rho, fix.state.rho(prev));
    EXPECT_EQ(uR.rho, fix.state.rho(idx));
    EXPECT_EQ(uL.P, fix.state.P(prev));
    EXPECT_EQ(uR.P, fix.state.P(idx));
    EXPECT_EQ(uL.V.x, -fix.state.V(Component::X)(idx));
}

TEST(ReconstructionPositivity, reconstruction_unchanged_for_positive_state)
{
    StateFixture fix;
    fix.convert();
    auto const idx = fix.cell(4, 4);

    auto const [uL, uR] = PositiveReconstructor::reconstruct<Direction::X>(fix.state, zeroB0, idx);

    EXPECT_EQ(uL.rho, 2 * fix.state.rho(idx));
    EXPECT_EQ(uR.rho, 3 * fix.state.rho(idx));
    EXPECT_EQ(uL.P, 2 * fix.state.P(idx));
    EXPECT_EQ(uR.P, 3 * fix.state.P(idx));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
