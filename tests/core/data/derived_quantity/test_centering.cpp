#include "core/data/derived_quantity/centering.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/hybrid/hybrid_quantities.hpp"

#include "gtest/gtest.h"

using namespace PHARE::core;

TEST(DerivedCentering, mhdAliasesResolveToExistingQuantities)
{
    EXPECT_EQ(scalar_qty<MHDQuantity>(ScalarCentering::cell), MHDQuantity::Scalar::P);
    EXPECT_EQ(scalar_qty<MHDQuantity>(ScalarCentering::node), MHDQuantity::Scalar::ScalarAllPrimal);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::cell), MHDQuantity::Vector::V);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::Elike), MHDQuantity::Vector::E);
    EXPECT_EQ(vector_qty<MHDQuantity>(VectorCentering::Blike), MHDQuantity::Vector::B);
}

TEST(DerivedCentering, hybridAliasesResolveToExistingQuantities)
{
    EXPECT_EQ(scalar_qty<HybridQuantity>(ScalarCentering::node), HybridQuantity::Scalar::rho);
    EXPECT_EQ(vector_qty<HybridQuantity>(VectorCentering::Elike), HybridQuantity::Vector::E);
    EXPECT_EQ(vector_qty<HybridQuantity>(VectorCentering::Blike), HybridQuantity::Vector::B);
}

TEST(DerivedCentering, hybridCellCenteredThrows)
{
    EXPECT_THROW(scalar_qty<HybridQuantity>(ScalarCentering::cell), std::runtime_error);
    EXPECT_THROW(vector_qty<HybridQuantity>(VectorCentering::cell), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
