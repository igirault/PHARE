#include "core/data/field/field_box_spans.hpp"

#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"

#include "gtest/gtest.h"

#include <cstddef>


namespace PHARE::core
{

template<std::size_t _dim>
struct TestParam
{
    auto constexpr static dim = _dim;
};


template<typename Param>
class FieldBoxSpansTest : public ::testing::Test
{
    static constexpr std::size_t dim    = Param::dim;
    static constexpr std::size_t interp = 1;

    static constexpr PHARE::SimOpts opts{dim, interp};
    using PHARE_Types  = PHARE::core::PHARE_Types<opts>;
    using GridLayout_t = TestGridLayout<typename PHARE_Types::Hybrid::GridLayout_t>;
    using Grid_t       = PHARE_Types::Hybrid::Grid_t;

public:
    GridLayout_t layout{9};
    Grid_t dst{"rho", layout, PHARE::core::HybridQuantity::Scalar::rho, 0};
    Grid_t src{"rho", layout, PHARE::core::HybridQuantity::Scalar::rho, 0};
};

using Permutations_t = testing::Types<TestParam<1>, TestParam<2>, TestParam<3>>;

TYPED_TEST_SUITE(FieldBoxSpansTest, Permutations_t, );

TYPED_TEST(FieldBoxSpansTest, test_field_box_spans_nd)
{
    auto& layout          = this->layout;
    auto& dst             = this->dst;
    auto& src             = this->src;
    auto const domain_box = layout.domainBoxFor(dst);

    for (auto const& bix : domain_box)
        src(bix) = product(bix);

    auto const d_spans = make_field_box_spans(domain_box, dst);
    auto const s_spans = make_field_box_spans(domain_box, src);

    auto d_it = d_spans.begin();
    for (auto const& s_span : s_spans)
    {
        auto const d_span = *d_it++;

        for (std::size_t j = 0; j < s_span.size(); ++j)
            d_span[j] += s_span[j];
    }


    for (auto const& bix : domain_box)
    {
        EXPECT_EQ(dst(bix), product(bix));
    }

    EXPECT_EQ(sum(dst), sum(src));
}



TYPED_TEST(FieldBoxSpansTest, test_field_box_point_spans_copies_do_not_alias)
{
    if constexpr (TypeParam::dim > 1)
    {
        auto& dst             = this->dst;
        auto const domain_box = this->layout.domainBoxFor(dst);

        auto const spans = make_field_box_point_spans(domain_box, dst);
        auto const base  = spans.begin();

        auto a = base;
        auto b = base;
        auto c = b;

        auto const [a_span, a_point] = *a;
        auto const* const a_data     = a_span.data();
        auto const a_start           = a_point;

        ++b;
        auto const [b_span, b_point] = *b;

        EXPECT_NE(&a_span, &b_span);
        EXPECT_NE(&a_point, &b_point);
        EXPECT_EQ(a_span.data(), a_data);
        EXPECT_EQ(a_point, a_start);
        EXPECT_NE(b_span.data(), a_data);
        EXPECT_EQ(a_span.data(), &dst(a_point));
        EXPECT_EQ(b_span.data(), &dst(b_point));

        auto const [c_span, c_point] = *c;
        EXPECT_EQ(c_point, a_start);
        EXPECT_EQ(c_span.data(), a_data);
        EXPECT_EQ(a_span.data(), a_data);
    }
}

} // namespace PHARE::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
