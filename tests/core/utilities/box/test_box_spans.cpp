#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"

#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"

#include "gtest/gtest.h"

#include <cstdint>
#include <ranges>
#include <vector>


namespace PHARE::core
{



TEST(BoxSpansTest, test_reverse_iterator)
{
    static constexpr std::size_t dim = 3;
    static constexpr std::size_t IDX = dim - 1;
    static constexpr PHARE::SimOpts opts{dim, 1};
    using PHARE_Types  = PHARE::core::PHARE_Types<opts>;
    using GridLayout_t = TestGridLayout<typename PHARE_Types::Hybrid::GridLayout_t>;

    GridLayout_t layout{3};

    std::vector<Point<int, dim>> points;

    for (auto const& [start, size] : make_box_spans(layout.AMRBox()) | std::views::reverse)
    {
        auto point = start;
        point[IDX] += size;
        for (; point[IDX]-- > start[IDX];)
            points.emplace_back(point);
    }

    std::vector<Point<int, dim>> const expected{
        {2, 2, 2}, {2, 2, 1}, {2, 2, 0}, //
        {2, 1, 2}, {2, 1, 1}, {2, 1, 0}, //
        {2, 0, 2}, {2, 0, 1}, {2, 0, 0}, //
        {1, 2, 2}, {1, 2, 1}, {1, 2, 0}, //
        {1, 1, 2}, {1, 1, 1}, {1, 1, 0}, //
        {1, 0, 2}, {1, 0, 1}, {1, 0, 0}, //
        {0, 2, 2}, {0, 2, 1}, {0, 2, 0}, //
        {0, 1, 2}, {0, 1, 1}, {0, 1, 0}, //
        {0, 0, 2}, {0, 0, 1}, {0, 0, 0},
    };

    EXPECT_EQ(expected, points);
}


TEST(BoxSpansTest, test_reverse_iterator_unsigned_box)
{
    static constexpr std::size_t dim = 3;
    static constexpr std::size_t IDX = dim - 1;

    Box<std::uint32_t, dim> const box{{0, 0, 0}, {2, 2, 2}};

    std::vector<Point<std::uint32_t, dim>> points;

    auto const spans = make_box_spans(box);

    for (auto it = spans.end(); it-- != spans.begin();)
    {
        auto const [start, size] = *it;
        auto point               = start;
        point[IDX] += size;
        for (; point[IDX]-- > start[IDX];)
            points.emplace_back(point);
    }

    std::vector<Point<std::uint32_t, dim>> const expected{
        {2u, 2u, 2u}, {2u, 2u, 1u}, {2u, 2u, 0u}, //
        {2u, 1u, 2u}, {2u, 1u, 1u}, {2u, 1u, 0u}, //
        {2u, 0u, 2u}, {2u, 0u, 1u}, {2u, 0u, 0u}, //
        {1u, 2u, 2u}, {1u, 2u, 1u}, {1u, 2u, 0u}, //
        {1u, 1u, 2u}, {1u, 1u, 1u}, {1u, 1u, 0u}, //
        {1u, 0u, 2u}, {1u, 0u, 1u}, {1u, 0u, 0u}, //
        {0u, 2u, 2u}, {0u, 2u, 1u}, {0u, 2u, 0u}, //
        {0u, 1u, 2u}, {0u, 1u, 1u}, {0u, 1u, 0u}, //
        {0u, 0u, 2u}, {0u, 0u, 1u}, {0u, 0u, 0u},
    };

    EXPECT_EQ(expected, points);
}


TEST(BoxSpansTest, test_range_loop)
{
    std::size_t static constexpr dim = 3;
    Box<std::uint32_t, dim> box{{0, 0, 0}, {9, 9, 9}};
    std::size_t elements = 0;

    for (auto const& [start, size] : make_box_spans(box))
        elements += size;

    EXPECT_EQ(elements, 10 * 10 * 10);
}


template<std::size_t dim>
std::size_t count_with_iterators(Box<std::uint32_t, dim> const& box)
{
    std::size_t elements = 0;
    auto const spans     = make_box_spans(box);

    for (auto it = spans.begin(); it != spans.end(); ++it)
    {
        auto const [start, size] = *it;
        elements += size;
    }
    return elements;
}

TEST(BoxSpansTest, test_iter_loop_dim1)
{
    EXPECT_EQ(count_with_iterators(Box<std::uint32_t, 1>{{0}, {9}}), 10);
}

TEST(BoxSpansTest, test_iter_loop_dim2)
{
    EXPECT_EQ(count_with_iterators(Box<std::uint32_t, 2>{{0, 0}, {9, 9}}), 10 * 10);
}

TEST(BoxSpansTest, test_iter_loop_dim3)
{
    EXPECT_EQ(count_with_iterators(Box<std::uint32_t, 3>{{0, 0, 0}, {9, 9, 9}}), 10 * 10 * 10);
}


TEST(BoxSpansTest, test_ill_formed_box_is_empty)
{
    auto const count = [](auto const& box) {
        std::size_t elements = 0;
        for (auto const& [start, size] : make_box_spans(box))
            elements += size;
        return elements;
    };

    EXPECT_EQ(count(Box<int, 1>{{5}, {3}}), 0);
    EXPECT_EQ(count(Box<int, 2>{{0, 5}, {9, 3}}), 0);
    EXPECT_EQ(count(Box<int, 2>{{5, 0}, {3, 9}}), 0);
    EXPECT_EQ(count(Box<int, 3>{{5, 0, 0}, {3, 9, 9}}), 0);
    EXPECT_EQ(count(Box<int, 3>{{0, 5, 0}, {9, 3, 9}}), 0);
    EXPECT_EQ(count(Box<int, 3>{{0, 0, 5}, {9, 9, 3}}), 0);
    EXPECT_EQ(count(Box<std::uint32_t, 3>{{0, 0, 5}, {9, 9, 3}}), 0);
}


TEST(BoxSpansTest, test_copies_do_not_alias)
{
    std::size_t static constexpr dim = 3;
    Box<std::uint32_t, dim> const box{{0, 0, 0}, {2, 2, 2}};

    auto const spans = make_box_spans(box);
    auto const base  = spans.begin();

    auto a = base;
    auto b = base;
    auto c = b;

    auto const [a_point, a_size] = *a;
    Point<std::uint32_t, dim> const a_start{a_point};

    ++b;
    auto const [b_point, b_size] = *b;

    EXPECT_NE(&a_point, &b_point);
    EXPECT_EQ(a_point, a_start);
    EXPECT_EQ(a_start, (Point<std::uint32_t, dim>{0u, 0u, 0u}));
    EXPECT_EQ(b_point, (Point<std::uint32_t, dim>{0u, 1u, 0u}));
    EXPECT_EQ(a_size, b_size);

    auto const [c_point, c_size] = *c;
    EXPECT_EQ(c_point, a_start);
    EXPECT_EQ(a_point, a_start);
}


TEST(BoxSpansTest, test_size_matches_iteration)
{
    Box<int, 3> const box{{1, -2, 3}, {4, 2, 7}};
    auto const spans = make_box_spans(box);

    EXPECT_EQ(spans.size(), 4u * 5u);
    EXPECT_EQ(std::ranges::distance(spans.begin(), spans.end()), 4 * 5);
}


TEST(BoxSpansTest, test_for_each_point_matches_box_iterator)
{
    auto const check = [](auto const& box) {
        using Point_t = std::decay_t<decltype(box.lower)>;
        std::vector<Point_t> expected, visited;
        for (auto const& point : box)
            expected.emplace_back(point);
        for_each_point(box, [&](auto const& point) { visited.emplace_back(point); });
        EXPECT_EQ(expected, visited);
    };

    check(Box<int, 1>{{-3}, {4}});
    check(Box<int, 2>{{-3, 2}, {4, 5}});
    check(Box<int, 3>{{-3, 2, -1}, {4, 5, 1}});
    check(Box<std::uint32_t, 3>{{0, 2, 1}, {4, 5, 1}});
}

} // namespace PHARE::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
