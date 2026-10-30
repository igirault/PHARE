#ifndef PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP
#define PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP


#include "core/utilities/span.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <utility>


namespace PHARE::core
{

template<typename Array_t>
using FieldSpan_t
    = Span<std::conditional_t<std::is_const_v<Array_t>, typename Array_t::value_type const,
                              typename Array_t::value_type>>;


template<typename Array_t, std::size_t dim>
auto make_field_box_spans(Box<std::uint32_t, dim> const& box, Array_t& arr)
{
    static_assert(Array_t::is_c_ordered, "field box spans require a C ordered array");
    return make_box_spans(box) | std::views::transform([&arr](auto const& span) {
               auto const& [start, size] = span;
               return FieldSpan_t<Array_t>{&arr(start), size};
           });
}

template<typename Array_t, std::size_t dim>
auto make_field_box_point_spans(Box<std::uint32_t, dim> const& box, Array_t& arr)
{
    static_assert(Array_t::is_c_ordered, "field box spans require a C ordered array");
    return make_box_spans(box) | std::views::transform([&arr](auto const& span) {
               auto const& [start, size] = span;
               return std::pair{FieldSpan_t<Array_t>{&arr(start), size}, start};
           });
}

} // namespace PHARE::core


#endif //  PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP
