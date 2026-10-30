#ifndef PHARE_CORE_UTILITIES_BOX_BOX_SPANS_HPP
#define PHARE_CORE_UTILITIES_BOX_BOX_SPANS_HPP

#include "core/utilities/box/box.hpp"
#include "core/utilities/point/point.hpp"

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>


namespace PHARE::core
{

template<typename T, std::size_t dim>
class BoxSpanIterator
{
public:
    using value_type      = std::pair<Point<T, dim>, std::uint32_t>;
    using difference_type = std::ptrdiff_t;

    BoxSpanIterator() = default;

    BoxSpanIterator(Box<T, dim> const& box, std::uint32_t const idx_)
        : start{box.lower}
        , size{extent(box, dim - 1)}
        , idx{idx_}
    {
        if constexpr (dim == 2)
            start[0] += idx;
        if constexpr (dim == 3)
        {
            wrap_lower = box.lower[1];
            wrap_upper = box.upper[1];
            if (auto const n1 = extent(box, 1); n1 > 0)
            {
                start[0] += idx / n1;
                start[1] += idx % n1;
            }
        }
    }

    static std::uint32_t extent(Box<T, dim> const& box, std::size_t const d)
    {
        return box.upper[d] >= box.lower[d] ? box.upper[d] - box.lower[d] + 1 : 0;
    }

    value_type operator*() const { return {start, size}; }

    BoxSpanIterator& operator++()
    {
        ++idx;
        if constexpr (dim == 2)
            ++start[0];
        if constexpr (dim == 3)
            if (++start[1] > wrap_upper)
            {
                start[1] = wrap_lower;
                ++start[0];
            }
        return *this;
    }

    BoxSpanIterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    BoxSpanIterator& operator--()
    {
        --idx;
        if constexpr (dim == 2)
            --start[0];
        if constexpr (dim == 3)
        {
            if (start[1] == wrap_lower)
            {
                start[1] = wrap_upper;
                --start[0];
            }
            else
                --start[1];
        }
        return *this;
    }

    BoxSpanIterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    bool operator==(BoxSpanIterator const& that) const { return idx == that.idx; }

    difference_type operator-(BoxSpanIterator const& that) const
    {
        return static_cast<difference_type>(idx) - static_cast<difference_type>(that.idx);
    }

private:
    Point<T, dim> start{};
    std::uint32_t size = 0;
    T wrap_lower{};
    T wrap_upper{};
    std::uint32_t idx = 0;
};


template<typename T, std::size_t dim>
auto make_box_spans(Box<T, dim> const& box)
{
    using Iterator = BoxSpanIterator<T, dim>;

    std::uint32_t n_spans = Iterator::extent(box, dim - 1) > 0 ? 1 : 0;
    for (std::size_t d = 0; d + 1 < dim; ++d)
        n_spans *= Iterator::extent(box, d);

    return std::ranges::subrange{Iterator{box, 0}, Iterator{box, n_spans}};
}


template<typename T, std::size_t dim>
void for_each_point(Box<T, dim> const& box, auto&& fn, auto&&... args)
{
    for (auto const& [start, size] : make_box_spans(box))
    {
        auto point = start;
        for (T i = start[dim - 1], end = i + size; i < end; ++i)
        {
            point[dim - 1] = i;
            fn(point, args...);
        }
    }
}


} // namespace PHARE::core


#endif //  PHARE_CORE_UTILITIES_BOX_BOX_SPANS_HPP
