#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/models/mhd_state.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"

#include "gtest/gtest.h"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"

using namespace PHARE::core;

static constexpr std::size_t dim = 2;
using YeeLayout_t                = GridLayout<GridLayoutImplYeeMHD<dim, 1>>;
using GridLayout_t               = TestGridLayout<YeeLayout_t>;
using State_t                    = MHDState<VecFieldMHD<dim>>;

struct Ones : DerivedQuantity<State_t, YeeLayout_t, 0>
{
    std::string name() const override { return "ones"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }
    void compute(State_t const& /*state*/, YeeLayout_t const& layout, out_t& out,
                 double /*time*/) const override
    {
        layout.evalOnGhostBox(out, [&](auto const&... args) { out(args...) = 1.0; });
    }
};

TEST(DerivedQuantityRegistry, findsRegisteredQuantityByNameAndRank)
{
    DerivedQuantityRegistry<State_t, YeeLayout_t> registry;
    registry.add<0>(std::make_unique<Ones>());

    auto const* dq = registry.find<0>("ones");
    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->name(), "ones");
    EXPECT_EQ(dq->centering(), ScalarCentering::cell);

    EXPECT_EQ(registry.find<0>("nope"), nullptr);
    EXPECT_EQ(registry.find<1>("ones"), nullptr);
    EXPECT_EQ(registry.quantities<0>().size(), 1u);
}

TEST(DerivedQuantityRegistry, computeFillsGhostBox)
{
    GridLayout_t layout{10};
    UsableMHDState<dim> state{layout, "state"};
    UsableFieldMHD<dim> out{"out", layout, MHDQuantity::Scalar::ScalarCellCentered};

    Ones{}.compute(*state, layout, out, 0.0);

    for (auto const& v : out)
        EXPECT_DOUBLE_EQ(v, 1.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
