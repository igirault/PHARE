#include "core/models/external_field_updater_factory.hpp"
#include "core/models/external_field_updater_defs.hpp"
#include "core/models/external_field_updater_dipole.hpp"

#include "initializer/data_provider.hpp"

#include "phare_core.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include "tests/core/models/test_external_field_fixtures.hpp"

#include "gtest/gtest.h"

#include <string>

using namespace PHARE;
using namespace PHARE::core;


namespace
{
//! an MHD-enabled option set, the axes other than the dimension are irrelevant here
constexpr SimOpts mhd_opts{.dimension            = 2,
                           .interp_order         = 1,
                           .time_integrator_type = MHDOpts::TimeIntegratorType::Euler,
                           .reconstruction_type  = MHDOpts::ReconstructionType::Constant,
                           .slope_limiter_type   = MHDOpts::SlopeLimiterType::None,
                           .riemann_solver_type  = MHDOpts::RiemannSolverType::Rusanov};

using MHDTypes     = PHARE_Types<mhd_opts>::MHD;
using GridLayout_t = MHDTypes::GridLayout_t;
using VecField_t   = MHDTypes::VecField_t;
using Factory_t    = ExternalFieldUpdaterFactory<VecField_t, GridLayout_t>;
using Dipole_t     = ExternalFieldUpdaterDipole<VecField_t, GridLayout_t>;
using None_t       = ExternalFieldUpdaterNone<VecField_t, GridLayout_t>;

auto constexpr cells = 8u;

//! kept outside the domain so that the dipole field stays smooth on the whole mesh
auto const position = Dipole_t::point_type{-0.5, 0.5};
auto const moment   = Dipole_t::vector_type{0.3, -0.2};


void putType(initializer::PHAREDict& dict, ExternalFieldUpdaterType type)
{
    dict["type"] = static_cast<int>(type);
}

template<std::size_t size>
void putVector(initializer::PHAREDict& dict, std::string const& key,
               Point<double, size> const& value)
{
    std::array constexpr axes{"x", "y", "z"};
    for (std::size_t i = 0; i < size; ++i)
        dict[key][axes[i]] = value[i];
}

//! a complete, valid dipole dict
initializer::PHAREDict dipoleDict()
{
    initializer::PHAREDict dict;
    putType(dict, ExternalFieldUpdaterType::Dipole);
    putVector(dict, "position", position);
    putVector(dict, "moment", moment);
    return dict;
}


/**
 * @brief runs an updater on a fresh set of fields and returns the resulting B0
 *
 * The layout and the fields are owned by the holder so that the vecfield stays valid for the
 * comparison.
 *
 * NOTE: the name cannot be Run: ::testing::Test declares a member function of that name, which
 * hides the type at test scope.
 */
struct UpdaterRun
{
    TestGridLayout<GridLayout_t> layout{cells};

    UsableExternalField<2> externalField{"external", layout};
    UsableVecFieldMHD<2> a0{"a0", layout, MHDQuantity::Vector::E};

    template<typename Updater>
    explicit UpdaterRun(Updater& updater)
    {
        updater(externalField, a0, layout, 0.);
    }
};

//! largest absolute difference between two B0, all components, over the whole ghost box
double maxDifference(UpdaterRun& lhs, UpdaterRun& rhs)
{
    double maxDiff = 0.;
    for_N<3>([&](auto i) {
        constexpr auto component = static_cast<Component>(decltype(i)::value);
        auto& lhsField           = lhs.externalField.B0(component);
        auto& rhsField           = rhs.externalField.B0(component);
        for (std::size_t k = 0; k < lhsField.size(); ++k)
            maxDiff = std::max(maxDiff, std::abs(lhsField.data()[k] - rhsField.data()[k]));
    });
    return maxDiff;
}

} // namespace


TEST(ExternalFieldUpdaterFactory, defaultsToNoExternalFieldWhenTypeIsAbsent)
{
    initializer::PHAREDict dict;
    auto updater = Factory_t::create(dict);

    ASSERT_NE(updater, nullptr);
    EXPECT_NE(dynamic_cast<None_t*>(updater.get()), nullptr);
}


TEST(ExternalFieldUpdaterFactory, createsTheNoneUpdater)
{
    initializer::PHAREDict dict;
    putType(dict, ExternalFieldUpdaterType::None);
    auto updater = Factory_t::create(dict);

    ASSERT_NE(updater, nullptr);
    EXPECT_NE(dynamic_cast<None_t*>(updater.get()), nullptr);
    EXPECT_FALSE(updater->isTimeDependent());
}


TEST(ExternalFieldUpdaterFactory, createsTheDipoleUpdater)
{
    auto const dict = dipoleDict();
    auto updater    = Factory_t::create(dict);

    ASSERT_NE(updater, nullptr);
    EXPECT_NE(dynamic_cast<Dipole_t*>(updater.get()), nullptr);
    EXPECT_FALSE(updater->isTimeDependent());
}


/**
 * @brief the dict parameters must reach the dipole constructor, in the right order
 *
 * Comparing the field produced by the factory-built updater with that of a directly
 * constructed one is what makes a swapped position/moment, or a truncated component, visible.
 */
TEST(ExternalFieldUpdaterFactory, forwardsThePositionAndMomentToTheDipole)
{
    auto const dict = dipoleDict();
    auto fromDict   = Factory_t::create(dict);
    Dipole_t direct{position, moment};

    UpdaterRun fromDictRun{*fromDict};
    UpdaterRun directRun{direct};

    EXPECT_DOUBLE_EQ(maxDifference(fromDictRun, directRun), 0.);
}


TEST(ExternalFieldUpdaterFactory, throwsOnAMissingDipoleParameter)
{
    initializer::PHAREDict dict;
    putType(dict, ExternalFieldUpdaterType::Dipole);
    putVector(dict, "position", position); // no "moment"

    EXPECT_THROW(Factory_t::create(dict), std::runtime_error);
}


/**
 * @brief a missing component must name itself, not surface later as a type error
 */
TEST(ExternalFieldUpdaterFactory, throwsNamingAMissingVectorComponent)
{
    initializer::PHAREDict dict;
    putType(dict, ExternalFieldUpdaterType::Dipole);
    putVector(dict, "position", position);
    dict["moment"]["x"] = moment[0]; // no "y"

    try
    {
        Factory_t::create(dict);
        FAIL() << "expected a missing component to throw";
    }
    catch (std::runtime_error const& e)
    {
        EXPECT_NE(std::string{e.what()}.find("invalid key: y"), std::string::npos)
            << "got: " << e.what();
    }
}


TEST(ExternalFieldUpdaterFactory, throwsOnAnIncompleteVectorParameter)
{
    auto dict           = dipoleDict();
    dict["moment"]["y"] = std::string{"not a double"};

    EXPECT_THROW(Factory_t::create(dict), std::runtime_error);
}


TEST(ExternalFieldUpdaterFactory, throwsOnAnUnimplementedType)
{
    initializer::PHAREDict dict;
    putType(dict, ExternalFieldUpdaterType::UserDefined);

    EXPECT_THROW(Factory_t::create(dict), std::runtime_error);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
