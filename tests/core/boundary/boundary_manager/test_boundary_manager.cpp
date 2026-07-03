#include "core/boundary/boundary_factory.hpp"
#include "core/boundary/boundary_manager.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/numerics/thermo/ideal_gas_thermo.hpp"

#include "initializer/data_provider.hpp"

#include "simulator/phare_types.hpp"

#include "gtest/gtest.h"

#include <string>

using namespace PHARE::core;

constexpr size_t dimension = 3;
constexpr PHARE::SimOpts opts{dimension};
constexpr std::size_t rank   = 1;
using types                  = PHARE::amr::PHARE_Types<opts>::core_types;
using grid_type              = types::Grid_MHD;
using field_type             = grid_type::field_type;
using grid_layout_type       = types::GridLayout_MHD;
using physical_quantity_type = MHDQuantity;
using boundary_type          = Boundary<physical_quantity_type, field_type, grid_layout_type>;
using boundary_manager_type = BoundaryManager<physical_quantity_type, field_type, grid_layout_type>;

boundary_manager_type createBoundaryManager()
{
    PHARE::initializer::PHAREDict dict;
    dict["xlower"]["type"] = std::string{"none"};
    dict["xupper"]["type"] = std::string{"none"};
    dict["ylower"]["type"] = std::string{"reflective"};
    dict["yupper"]["type"] = std::string{"reflective"};
    dict["zlower"]["type"] = std::string{"open"};
    dict["zupper"]["type"] = std::string{"open"};


    boundary_manager_type bm{dict, {}, {}};

    return bm;
}

TEST(BoundaryManager, hasPriorityPolicyByDirection)
{
    auto bm = createBoundaryManager();
    bm.setPriorityPolicy(boundary_manager_type::PriorityPolicy::ByDirection);

    for (size_t i = 0; i < NUM_3D_EDGES; ++i)
    {
        auto codim2loc            = static_cast<Codim2BoundaryLocation>(i);
        BoundaryLocation actual   = bm.getMasterBoundaryLocation(codim2loc);
        BoundaryLocation expected = getAdjacentBoundaryLocations(codim2loc)[1];
        EXPECT_EQ(actual, expected);
    }

    for (size_t i = 0; i < NUM_3D_NODES; ++i)
    {
        auto codim3loc            = static_cast<Codim2BoundaryLocation>(i);
        BoundaryLocation actual   = bm.getMasterBoundaryLocation(codim3loc);
        BoundaryLocation expected = getAdjacentBoundaryLocations(codim3loc)[1];
        EXPECT_EQ(actual, expected);
    }
}

TEST(BoundaryManager, hasPriorityPolicyByBoundaryTypes)
{
    auto bm = createBoundaryManager();
    bm.setPriorityPolicy(boundary_manager_type::PriorityPolicy::ByBoundaryType);

    for (size_t i = 0; i < NUM_3D_EDGES; ++i)
    {
        auto codim2loc                = static_cast<Codim2BoundaryLocation>(i);
        BoundaryLocation masterLoc    = bm.getMasterBoundaryLocation(codim2loc);
        boundary_type& masterBoundary = *(bm.getBoundary(masterLoc));
        std::array adjacentLocations  = getAdjacentBoundaryLocations(codim2loc);
        for (auto loc : adjacentLocations)
        {
            boundary_type& adjacentBoundary = *(bm.getBoundary(loc));
            EXPECT_TRUE(masterBoundary.getType() >= adjacentBoundary.getType());
        }
    }

    for (size_t i = 0; i < NUM_3D_NODES; ++i)
    {
        auto codim3loc                = static_cast<Codim2BoundaryLocation>(i);
        BoundaryLocation masterLoc    = bm.getMasterBoundaryLocation(codim3loc);
        boundary_type& masterBoundary = *(bm.getBoundary(masterLoc));
        std::array adjacentLocations  = getAdjacentBoundaryLocations(codim3loc);
        for (auto loc : adjacentLocations)
        {
            boundary_type& adjacentBoundary = *(bm.getBoundary(loc));
            EXPECT_TRUE(masterBoundary >= adjacentBoundary);
        }
    }
}

// ─── inflow regrid fallback keying ───────────────────────────────────────────
//
// The magnetic init/regrid schedules fill the B1 field, so the fallback lookup key
// is Vector::B1; a fallback registered under Vector::B can never fire (the lookup
// misses and falls through to the normal B1 condition, None at inflow).

using boundary_factory_type = BoundaryFactory<physical_quantity_type, field_type, grid_layout_type>;

namespace
{
PHARE::initializer::PHAREDict inflowBoundaryDict(std::string const& type, bool BisFunction)
{
    PHARE::initializer::PHAREDict dict;
    dict["type"]                = type;
    dict["data"]["density"]     = 1.0;
    dict["data"]["velocity"]["x"] = 2.0;
    dict["data"]["velocity"]["y"] = 0.0;
    dict["data"]["velocity"]["z"] = 0.0;
    if (type == "super-magnetofast-inflow")
        dict["data"]["pressure"] = 1.0;

    if (BisFunction)
    {
        auto uniformFn = [](double base) {
            return PHARE::initializer::TimeFunction<dimension>{
                [base](std::vector<double> const& x, std::vector<double> const&,
                       std::vector<double> const&, double t) {
                    std::vector<double> out(x.size(), base + t);
                    return std::shared_ptr<Span<double>>{
                        std::make_shared<VectorSpan<double>>(std::move(out))};
                }};
        };
        dict["data"]["B_is_function"] = true;
        dict["data"]["B"]["x"]        = uniformFn(0.0);
        dict["data"]["B"]["y"]        = uniformFn(-1.0);
        dict["data"]["B"]["z"]        = uniformFn(0.0);
    }
    else
    {
        dict["data"]["B"]["x"] = 0.0;
        dict["data"]["B"]["y"] = -1.0;
        dict["data"]["B"]["z"] = 0.0;
    }
    return dict;
}

void expectB1KeyedFallback(std::string const& type, bool BisFunction)
{
    auto thermo = std::make_shared<IdealGasThermo>(5.0 / 3.0);
    auto boundary
        = boundary_factory_type::create(BoundaryLocation::XLower,
                                        inflowBoundaryDict(type, BisFunction),
                                        {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                                        {MHDQuantity::Vector::rhoV, MHDQuantity::Vector::E,
                                         MHDQuantity::Vector::B1},
                                        thermo);

    auto fallback = boundary->getRegridFallbackCondition(MHDQuantity::Vector::B1);
    ASSERT_NE(fallback, nullptr) << type << " BisFunction=" << BisFunction;
    EXPECT_EQ(fallback->getType(), FieldBoundaryConditionType::B1FromBtot)
        << type << " BisFunction=" << BisFunction;
    EXPECT_EQ(boundary->getRegridFallbackCondition(MHDQuantity::Vector::B), nullptr)
        << type << " BisFunction=" << BisFunction;
}
} // namespace

TEST(BoundaryFactory, superMagnetofastInflowRegridFallbackIsB1FromBtotKeyedOnB1)
{
    expectB1KeyedFallback("super-magnetofast-inflow", /*BisFunction=*/false);
}

TEST(BoundaryFactory, superMagnetofastInflowTimeVaryingRegridFallbackIsB1FromBtotKeyedOnB1)
{
    expectB1KeyedFallback("super-magnetofast-inflow", /*BisFunction=*/true);
}

TEST(BoundaryFactory, freePressureInflowRegridFallbackIsB1FromBtotKeyedOnB1)
{
    expectB1KeyedFallback("free-pressure-inflow", /*BisFunction=*/false);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
