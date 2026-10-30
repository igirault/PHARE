#ifndef PHARE_CORE_UTILITIES_BOX_BOX_SLABS_HPP
#define PHARE_CORE_UTILITIES_BOX_BOX_SLABS_HPP

#include "core/utilities/box/box.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>


namespace PHARE::core
{

/*
Span-wise iteration over an N dimensional Box.

Instead of visiting the box point by point, the box is decomposed into:
  - slabs: one per index of the slowest-varying dimension (3D only,
           1D/2D boxes are a single slab)
  - spans: within a slab, one per index of the second-fastest dimension;
           each span is the contiguous run of cells along the last
           (fastest-varying) dimension

    for (auto const& slab : make_box_slabs(box))   // 3D: one slab per x index
        for (auto const& [start, size] : slab)    // one span per y index
            ...  // span covers cells [start, start + size) along z

This gives callers whole contiguous spans to work with (memcpy, vectorizable
inner loops) rather than a callback per point.
*/


template<typename T, std::size_t dim>
auto make_box_slabs(Box<T, dim> const& box, auto const& make_span)
{
    auto const indices = [&](std::size_t const d) {
        return std::views::iota(box.lower[d], std::max<T>(box.lower[d], box.upper[d] + 1));
    };
    auto const single = std::views::iota(T{0}, T{1});
    auto const slabs  = [&] {
        if constexpr (dim == 3)
            return indices(0);
        else
            return single;
    }();
    auto const spans = [&] {
        if constexpr (dim > 1)
            return indices(dim - 2);
        else
            return single;
    }();
    std::uint32_t const size = indices(dim - 1).size();

    return slabs | std::views::transform([=, lower = box.lower](T const slab) {
               return spans | std::views::transform([=](T const span) {
                          auto start = lower;
                          if constexpr (dim == 3)
                              start[0] = slab;
                          if constexpr (dim > 1)
                              start[dim - 2] = span;
                          return make_span(start, size);
                      });
           });
}

template<typename T, std::size_t dim>
auto make_box_slabs(Box<T, dim> const& box)
{
    return make_box_slabs(
        box, [](auto const& start, std::uint32_t const size) { return std::pair{start, size}; });
}


/** @brief Calls fn(point, args...) for every point of `box`, walking it span-wise.
 *
 * `point` is a copy of the span's start advanced along the fastest-varying dimension,
 * which unlike advancing the start in place lets the compiler vectorize the span loop
 * (2D points otherwise stay packed in one 64 bit register); it is only valid for the
 * duration of the call.
 * `args` are passed to every call as lvalues, never forwarded: fn runs once per
 * point, so a by-value parameter could otherwise move from them on the first one.
 */
template<typename T, std::size_t dim>
void for_each_point(Box<T, dim> const& box, auto&& fn, auto&&... args)
{
    for (auto const& slab : make_box_slabs(box))
        for (auto const& [start, size] : slab)
        {
            // `<` (not `!=`) tells the compiler the 32 bit index cannot wrap,
            // so field offsets stay affine and the loop can be vectorized
            auto point = start;
            for (T i = start[dim - 1], end = i + size; i < end; ++i)
            {
                point[dim - 1] = i;
                fn(point, args...);
            }
        }
}


} // namespace PHARE::core


#endif //  PHARE_CORE_UTILITIES_BOX_BOX_SLABS_HPP
