#include "core/data/user/user_field_updater.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/types.hpp"

#include "phare_core.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include "tests/core/models/test_external_field_fixtures.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PHARE;
using namespace PHARE::core;


namespace
{
//! an MHD-enabled option set, the axes other than the dimension are irrelevant here
template<std::size_t dim>
constexpr SimOpts mhd_opts{.dimension           = dim,
                           .interp_order        = 1,
                           .reconstruction_type = MHDOpts::ReconstructionType::Constant,
                           .slope_limiter_type  = MHDOpts::SlopeLimiterType::None,
                           .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};

template<std::size_t dim>
using GridLayout_t = typename PHARE_Types<mhd_opts<dim>>::MHD::GridLayout_t;

double constexpr nan = std::numeric_limits<double>::quiet_NaN();


/**
 * @brief dim E-centered vecfields, coordinates[d] meant to hold coordinate d, as the external
 * field coordinates cache does. Buffers start as NaN so that an unwritten node shows up.
 *
 * Like on a patch, the vecfields are views whose buffers are set from grids owned elsewhere.
 */
template<std::size_t dim_>
struct CoordinatesSetup
{
    auto static constexpr dim   = dim_;
    auto static constexpr cells = 10;

    using VecField_t = VecFieldMHD<dim>;

    TestGridLayout<GridLayout_t<dim>> layout{cells};

    std::array<UsableVecFieldMHD<dim>, dim> grids
        = for_N<dim, for_N_R_mode::make_array>([&](auto d) {
              return UsableVecFieldMHD<dim>{"coords_" + std::to_string(d), layout,
                                            MHDQuantity::Vector::E};
          });

    std::vector<VecField_t> coordinates;

    CoordinatesSetup()
    {
        coordinates.reserve(dim);
        for (auto& grid : grids)
        {
            coordinates.emplace_back(grid.super().name(), MHDQuantity::Vector::E);
            grid.set_on(coordinates.back());
            for (std::size_t c = 0; c < 3; ++c)
                for (auto& v : coordinates.back()[c])
                    v = nan;
        }
    }

    void fill() { UserFieldUpdater::fillCoordinates(layout, std::span{coordinates}); }
};


template<typename SetupT>
struct FillCoordinatesTest : public ::testing::Test
{
    SetupT setup;
};

using Setups = ::testing::Types<CoordinatesSetup<1>, CoordinatesSetup<2>, CoordinatesSetup<3>>;
TYPED_TEST_SUITE(FillCoordinatesTest, Setups);

} // namespace


TYPED_TEST(FillCoordinatesTest, writesTheNodeCoordinatesOfEachComponent)
{
    auto& setup = this->setup;
    setup.fill();

    std::size_t mismatches = 0;
    for (std::size_t d = 0; d < TypeParam::dim; ++d)
        for (std::size_t c = 0; c < 3; ++c)
        {
            auto& field = setup.coordinates[d][c];
            setup.layout.evalOnGhostBox(field, [&](auto... ijk) {
                auto const x = setup.layout.fieldNodeCoordinates(
                    field, setup.layout.localToAMR(Point{ijk...}));
                if (field(ijk...) != x[d])
                    ++mismatches;
            });
        }
    EXPECT_EQ(mismatches, 0u);
}


//! the ghost box iteration covers the whole allocated buffer: the contiguous copy of a user
//! function result into a field relies on it
TYPED_TEST(FillCoordinatesTest, writesEveryAllocatedNode)
{
    auto& setup = this->setup;
    setup.fill();

    std::size_t unwritten = 0;
    for (auto& vecfield : setup.coordinates)
        for (std::size_t c = 0; c < 3; ++c)
            for (auto const& v : vecfield[c])
                if (std::isnan(v))
                    ++unwritten;
    EXPECT_EQ(unwritten, 0u);
}


TYPED_TEST(FillCoordinatesTest, throwsWithoutOneVecfieldPerDimension)
{
    auto& setup       = this->setup;
    auto const tooFew = std::span{setup.coordinates}.first(TypeParam::dim - 1);
    EXPECT_THROW(UserFieldUpdater::fillCoordinates(setup.layout, tooFew), std::runtime_error);
}


namespace
{
//! a formula whose value tells the component, the time and every coordinate apart
template<std::size_t dim>
double formula(std::size_t component, Point<double, dim> const& x, double t)
{
    double value = 10. * component + t;
    for (std::size_t d = 0; d < dim; ++d)
        value += (d + 1) * x[d];
    return value;
}

template<std::size_t dim>
std::array<SpaceTimeFunction<dim>, 3> formulaFunctions()
{
    auto component = [](std::size_t c) {
        return spaceTimeFunction<dim>(
            [c](Point<double, dim> const& x, double t) { return formula<dim>(c, x, t); });
    };
    return {component(0), component(1), component(2)};
}

//! a user function returning one value too many
template<std::size_t dim>
std::array<SpaceTimeFunction<dim>, 3> wrongSizeFunctions()
{
    auto f = [](auto const& x, auto const&... rest) {
        auto const t = std::get<sizeof...(rest) - 1>(std::forward_as_tuple(rest...));
        return std::static_pointer_cast<Span<double>>(
            std::make_shared<VectorSpan<double>>(std::vector<double>(x.size + 1, t)));
    };
    return {f, f, f};
}


template<std::size_t dim_>
struct UpdateSetup : public CoordinatesSetup<dim_>
{
    using Super = CoordinatesSetup<dim_>;
    using Super::dim;
    using Super::layout;

    double static constexpr time = 1.5;

    UsableVecFieldMHD<dim> onTheFly{"onTheFly", layout, MHDQuantity::Vector::E};
    UsableVecFieldMHD<dim> cached{"cached", layout, MHDQuantity::Vector::E};

    void updateOnTheFly(std::array<SpaceTimeFunction<dim>, 3> const& funcs)
    {
        UserFieldUpdater::update(onTheFly.super(), layout, funcs, time);
    }

    void updateCached(std::array<SpaceTimeFunction<dim>, 3> const& funcs)
    {
        this->fill();
        UserFieldUpdater::update(cached.super(), funcs, time,
                                 std::span<VecFieldMHD<dim> const>{this->coordinates});
    }
};

template<typename SetupT>
struct UpdateTest : public ::testing::Test
{
    SetupT setup;
};

using UpdateSetups = ::testing::Types<UpdateSetup<1>, UpdateSetup<2>, UpdateSetup<3>>;
TYPED_TEST_SUITE(UpdateTest, UpdateSetups);

} // namespace


//! also checks the order: a value copied to the wrong node would not match its coordinates
TYPED_TEST(UpdateTest, onTheFlyEvaluatesTheUserFunctionAtEachNode)
{
    auto& setup        = this->setup;
    auto constexpr dim = TypeParam::dim;
    setup.updateOnTheFly(formulaFunctions<dim>());

    std::size_t mismatches = 0;
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto& field = setup.onTheFly[c];
        setup.layout.evalOnGhostBox(field, [&](auto... ijk) {
            auto const x
                = setup.layout.fieldNodeCoordinates(field, setup.layout.localToAMR(Point{ijk...}));
            if (field(ijk...) != formula<dim>(c, x, setup.time))
                ++mismatches;
        });
    }
    EXPECT_EQ(mismatches, 0u);
}

TYPED_TEST(UpdateTest, cachedMatchesOnTheFly)
{
    auto& setup      = this->setup;
    auto const funcs = formulaFunctions<TypeParam::dim>();
    setup.updateOnTheFly(funcs);
    setup.updateCached(funcs);

    std::size_t mismatches = 0;
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const& a = setup.onTheFly[c];
        auto const& b = setup.cached[c];
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a.data()[i] != b.data()[i])
                ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u);
}

TYPED_TEST(UpdateTest, throwsWhenTheUserFunctionReturnsTheWrongSize)
{
    auto& setup      = this->setup;
    auto const funcs = wrongSizeFunctions<TypeParam::dim>();
    EXPECT_THROW(setup.updateOnTheFly(funcs), std::runtime_error);
    EXPECT_THROW(setup.updateCached(funcs), std::runtime_error);
}

TYPED_TEST(UpdateTest, cachedThrowsWithoutOneCoordinatesVecfieldPerDimension)
{
    auto& setup = this->setup;
    setup.fill();
    auto const tooFew
        = std::span<VecFieldMHD<TypeParam::dim> const>{setup.coordinates}.first(TypeParam::dim - 1);
    EXPECT_THROW(UserFieldUpdater::update(setup.cached.super(), formulaFunctions<TypeParam::dim>(),
                                          setup.time, tooFew),
                 std::runtime_error);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
