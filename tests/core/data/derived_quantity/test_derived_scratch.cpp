#include "core/data/derived_quantity/derived_scratch.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t                = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t               = TestGridLayout<YeeLayout_t>;
using VecField_t                 = VecFieldMHD<dim>;

TEST(DerivedScratch, scalarViewHasAllocSizeShapeAndAliasesBacking)
{
    GridLayout_t layout{10};
    UsableFieldMHD<dim> backing{"scratch", layout, MHDQuantity::Scalar::ScalarAllPrimal};

    auto view = derived_scalar_view<MHDQuantity>(backing.super(), ScalarCentering::cell, layout);
    EXPECT_TRUE(view.isUsable());
    EXPECT_EQ(view.shape(), layout.allocSize(MHDQuantity::Scalar::ScalarCellCentered));
    EXPECT_EQ(view.data(), backing.super().data());

    view.data()[0] = 42.0;
    EXPECT_DOUBLE_EQ(backing.super().data()[0], 42.0);
}

TEST(DerivedScratch, vectorViewComponentsAliasBackingComponents)
{
    GridLayout_t layout{10};
    UsableVecFieldMHD<dim> vbacking{"vscratch", layout, MHDQuantity::Vector::VecAllPrimal};

    auto view = derived_vector_view<MHDQuantity>(vbacking.super(), VectorCentering::Blike, layout);
    auto const qtys = MHDQuantity::componentsQuantities(MHDQuantity::Vector::VecBlike);

    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(view[i].isUsable());
        EXPECT_EQ(view[i].shape(), layout.allocSize(qtys[i]));
        EXPECT_EQ(view[i].data(), vbacking.super()[i].data());
        EXPECT_LE(view[i].size(), vbacking.super()[i].size());
    }
}

TEST(DerivedScratch, nodeAndVectorCenteringsFitInsideAllPrimal)
{
    GridLayout_t layout{10};
    UsableFieldMHD<dim> backing{"scratch", layout, MHDQuantity::Scalar::ScalarAllPrimal};
    UsableVecFieldMHD<dim> vbacking{"vscratch", layout, MHDQuantity::Vector::VecAllPrimal};

    for (auto const centering : {ScalarCentering::cell, ScalarCentering::node})
    {
        auto view = derived_scalar_view<MHDQuantity>(backing.super(), centering, layout);
        EXPECT_LE(detail::product<dim>(view.shape()), backing.super().size());
    }

    for (auto const centering :
         {VectorCentering::cell, VectorCentering::Elike, VectorCentering::Blike})
    {
        auto view = derived_vector_view<MHDQuantity>(vbacking.super(), centering, layout);
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_LE(detail::product<dim>(view[i].shape()), vbacking.super()[i].size());
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
