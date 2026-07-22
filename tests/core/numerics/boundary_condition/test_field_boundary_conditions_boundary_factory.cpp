#include "gtest/gtest.h"

#include "core/boundary/boundary_factory.hpp"
#include "core/numerics/thermo/ideal_gas_thermo.hpp"
#include "initializer/data_provider.hpp"

#include "tests/core/numerics/boundary_condition/mhd_bc_test_fixtures.hpp"

#include <map>
#include <string>
#include <vector>

using namespace PHARE::core;
using PHARE::initializer::PHAREDict;

// Table-driven check that the boundary factory maps each supported boundary-type string to the
// expected FieldBoundaryConditionType for every registered MHD quantity. This is the layer where
// two real fall-through bugs were already found (None->Reflective on a missing break, and the
// dead-motional-E path), so pinning the exact per-quantity dispatch here guards against their
// regression. No patch data or SAMRAI needed: only the factory's construction wiring is exercised.

namespace
{
using Factory = BoundaryFactory<MHDQuantity, FieldMHD<1>, GridLayoutMHD1D>;
using Scalar  = MHDQuantity::Scalar;
using Vector  = MHDQuantity::Vector;
using FBC     = FieldBoundaryConditionType;

double constexpr gamma = 5.0 / 3.0;

// the quantities the MHD model registers boundary conditions for (see mhd_model.hpp)
std::vector<Scalar> const mhdScalars{Scalar::rho, Scalar::Etot};
std::vector<Vector> const mhdVectors{Vector::B, Vector::E, Vector::rhoV};

// A fully-populated inflow 'data' sub-dict: constant density/pressure and constant
// velocity/B 3-vectors, with the "<key>_is_function" flags the C++ readers expect.
void fillInflowData(PHAREDict& dict)
{
    dict["data"]["density"]              = 1.0;
    dict["data"]["density_is_function"]  = false;
    dict["data"]["pressure"]             = 2.0;
    dict["data"]["pressure_is_function"] = false;

    dict["data"]["velocity_is_function"] = false;
    dict["data"]["velocity"]["x"]        = 3.0;
    dict["data"]["velocity"]["y"]        = 0.0;
    dict["data"]["velocity"]["z"]        = 0.0;

    dict["data"]["B_is_function"] = false;
    dict["data"]["B"]["x"]        = 0.75;
    dict["data"]["B"]["y"]        = 1.0;
    dict["data"]["B"]["z"]        = 0.0;
}

PHAREDict dictFor(std::string const& type)
{
    PHAREDict dict;
    dict["type"] = type;
    if (type == "super-magnetofast-inflow" || type == "free-pressure-inflow")
        fillInflowData(dict);
    else if (type == "fixed-pressure-outflow")
        dict["data"]["pressure"] = 2.0;
    return dict;
}

struct ExpectedDispatch
{
    std::map<Scalar, FBC> scalars;
    std::map<Vector, FBC> vectors;
};

void checkDispatch(std::string const& type, ExpectedDispatch const& expected)
{
    auto thermo   = std::make_shared<IdealGasThermo>(gamma);
    auto boundary = Factory::create(BoundaryLocation::XLower, dictFor(type), mhdScalars, mhdVectors,
                                    thermo);
    ASSERT_NE(boundary, nullptr) << "factory returned null for type '" << type << "'";

    for (auto const& [qty, expectedType] : expected.scalars)
    {
        auto bc = boundary->getFieldCondition(qty);
        ASSERT_NE(bc, nullptr) << "type '" << type << "' left scalar quantity "
                               << static_cast<int>(qty) << " with no condition";
        EXPECT_EQ(bc->getType(), expectedType)
            << "type '" << type << "', scalar quantity " << static_cast<int>(qty)
            << ": got " << static_cast<int>(bc->getType()) << ", expected "
            << static_cast<int>(expectedType);
    }
    for (auto const& [qty, expectedType] : expected.vectors)
    {
        auto bc = boundary->getFieldCondition(qty);
        ASSERT_NE(bc, nullptr) << "type '" << type << "' left vector quantity "
                               << static_cast<int>(qty) << " with no condition";
        EXPECT_EQ(bc->getType(), expectedType)
            << "type '" << type << "', vector quantity " << static_cast<int>(qty)
            << ": got " << static_cast<int>(bc->getType()) << ", expected "
            << static_cast<int>(expectedType);
    }
}
} // namespace

TEST(BoundaryFactory, NoneLeavesEveryQuantityUntouched)
{
    checkDispatch("none", {{{Scalar::rho, FBC::None}, {Scalar::Etot, FBC::None}},
                           {{Vector::B, FBC::None},
                            {Vector::E, FBC::None},
                            {Vector::rhoV, FBC::None}}});
}

TEST(BoundaryFactory, Reflective)
{
    checkDispatch("reflective",
                  {{{Scalar::rho, FBC::Neumann}, {Scalar::Etot, FBC::Neumann}},
                   {{Vector::B, FBC::DivergenceFreeTransverseNeumann},
                    {Vector::E, FBC::AntiSymmetric},
                    {Vector::rhoV, FBC::Symmetric}}});
}

TEST(BoundaryFactory, OpenAndSuperMagnetofastOutflowShareDispatch)
{
    ExpectedDispatch const open{{{Scalar::rho, FBC::Neumann},
                                 {Scalar::Etot, FBC::TotalEnergyFromPressure}},
                                {{Vector::B, FBC::DivergenceFreeTransverseNeumann},
                                 {Vector::E, FBC::None},
                                 {Vector::rhoV, FBC::Neumann}}};
    checkDispatch("open", open);
    checkDispatch("super-magnetofast-outflow", open);
}

TEST(BoundaryFactory, SuperMagnetofastInflowAndFreePressureInflowShareDispatch)
{
    ExpectedDispatch const inflow{{{Scalar::rho, FBC::Dirichlet},
                                   {Scalar::Etot, FBC::TotalEnergyFromPressure}},
                                  {{Vector::B, FBC::DivergenceFreeTransverseDirichlet},
                                   {Vector::E, FBC::None},
                                   {Vector::rhoV, FBC::Dirichlet}}};
    checkDispatch("super-magnetofast-inflow", inflow);
    checkDispatch("free-pressure-inflow", inflow);
}

TEST(BoundaryFactory, FixedPressureOutflow)
{
    checkDispatch("fixed-pressure-outflow",
                  {{{Scalar::rho, FBC::Neumann},
                    {Scalar::Etot, FBC::TotalEnergyFromPressure}},
                   {{Vector::B, FBC::DivergenceFreeTransverseNeumann},
                    {Vector::E, FBC::None},
                    {Vector::rhoV, FBC::Neumann}}});
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
