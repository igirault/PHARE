#include "core/data/derived_quantity/derived_scratch.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t                = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t               = TestGridLayout<YeeLayout_t>;
using VecField_t                 = VecFieldMHD<dim>;
using Scratch_t                  = DerivedScratch<VecField_t, MHDQuantity>;

TEST(DerivedScratch, scalarViewHasAllocSizeShape)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto f = scratch.scalar(ScalarCentering::cell, layout);
    EXPECT_TRUE(f.isUsable());
    EXPECT_EQ(f.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarCellCentered));

    auto n = scratch.scalar(ScalarCentering::node, layout);
    EXPECT_EQ(n.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarNodeCentered));
}

TEST(DerivedScratch, vectorComponentsViewDisjointSegments)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto vf         = scratch.vector(VectorCentering::Blike, layout);
    auto const qtys = MHDQuantity::componentsQuantities(MHDQuantity::Vector::VecBlike);

    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(vf[i].isUsable());
        EXPECT_EQ(vf[i].shape(), layout.allocSize(qtys[i]));
    }
    // disjoint: end of comp i == start of comp i+1
    EXPECT_EQ(vf[0].data() + vf[0].size(), vf[1].data());
    EXPECT_EQ(vf[1].data() + vf[1].size(), vf[2].data());
}

TEST(DerivedScratch, memoryIsReusedAcrossCalls)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto a      = scratch.scalar(ScalarCentering::cell, layout);
    a.data()[0] = 42.0;
    auto b      = scratch.scalar(ScalarCentering::cell, layout);
    EXPECT_EQ(a.data(), b.data()); // same block: shared scratch
    EXPECT_DOUBLE_EQ(b.data()[0], 42.0);
}

TEST(DerivedScratch, viewDispatchesToScalarAndVector)
{
    GridLayout_t layout{10};
    Scratch_t scratch;

    auto f = scratch.template view<0>(ScalarCentering::cell, layout);
    EXPECT_TRUE(f.isUsable());
    EXPECT_EQ(f.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarCellCentered));

    auto vf         = scratch.template view<1>(VectorCentering::Blike, layout);
    auto const qtys = MHDQuantity::componentsQuantities(MHDQuantity::Vector::VecBlike);
    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(vf[i].isUsable());
        EXPECT_EQ(vf[i].shape(), layout.allocSize(qtys[i]));
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
