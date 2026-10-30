#ifndef PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP
#define PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP

#include "core/utilities/span.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_slabs.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace PHARE::core
{

template<typename Array_t>
using FieldSpan_t
    = Span<std::conditional_t<std::is_const_v<Array_t>, typename Array_t::value_type const,
                              typename Array_t::value_type>>;


template<typename Array_t, std::size_t dim>
auto make_field_box_slabs(Box<std::uint32_t, dim> const& box, Array_t& arr)
{
    static_assert(Array_t::is_c_ordered, "field box slabs require a C ordered array");
    return make_box_slabs(box, [&arr](auto const& start, std::uint32_t const size) {
        return FieldSpan_t<Array_t>{&arr(start), size};
    });
}

template<typename Array_t, std::size_t dim>
auto make_field_box_point_slabs(Box<std::uint32_t, dim> const& box, Array_t& arr)
{
    static_assert(Array_t::is_c_ordered, "field box slabs require a C ordered array");
    return make_box_slabs(box, [&arr](auto const& start, std::uint32_t const size) {
        return std::pair{FieldSpan_t<Array_t>{&arr(start), size}, start};
    });
}

} // namespace PHARE::core


#endif //  PHARE_CORE_DATA_FIELD_FIELD_BOX_SPANS_HPP
