#ifndef PHARE_CORE_DATA_FIELD_FIELD_BOX_HPP
#define PHARE_CORE_DATA_FIELD_FIELD_BOX_HPP

#include "core/utilities/types.hpp"
#include "core/utilities/box/box.hpp"
#include "core/data/field/field_box_spans.hpp"

#include <vector>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace PHARE::core
{

template<typename Field_t>
class FieldBox
{
    using value_type = std::decay_t<typename Field_t::value_type>;

public:
    auto constexpr static dimension = Field_t::dimension;

    Field_t& field;
    Box<int, dimension> amr_ghost_box;
    Box<std::uint32_t, dimension> lcl_box;

    template<typename GridLayout_t>
    FieldBox(Field_t& field_, GridLayout_t const& layout)
        : field{field_}
        , amr_ghost_box{layout.AMRGhostBoxFor(field.physicalQuantity())}
        , lcl_box{layout.ghostBoxFor(field)}
    {
    }

    template<typename GridLayout_t>
    FieldBox(Field_t& field_, GridLayout_t const& layout,
             Box<std::uint32_t, dimension> const& selection)
        : field{field_}
        , amr_ghost_box{layout.AMRGhostBoxFor(field.physicalQuantity())}
        , lcl_box{selection}
    {
    }

    template<typename GridLayout_t>
    FieldBox(Field_t& field_, GridLayout_t const& layout, Box<int, dimension> const& selection)
        : field{field_}
        , amr_ghost_box{layout.AMRGhostBoxFor(field.physicalQuantity())}
        , lcl_box{layout.AMRToLocal(selection)}
    {
    }


    template<typename Operator>
    void set_from(std::vector<value_type> const& vec, std::size_t seek = 0);

    void append_to(std::vector<value_type>& vec);
};


// per-element Operator (e.g. SetEqual, PlusEquals, SetMax, see types.hpp): no memcpy, which
//  is only valid for trivially copyable value types and is an out-of-line call that costs
//  more than the copy on the short spans of ghost slabs thin along the fastest dimension
template<typename Operator, typename T>
void operate_on_span(auto const& dst, T const* src_data)
{
    for (std::size_t i = 0; i < dst.size(); ++i, ++src_data)
        Operator{dst[i]}(*src_data);
}



// applies Operator span-by-span between two same-shaped FieldBoxes (see
// core/utilities/box/box_spans.hpp); used to add/copy/max one field's local
// region into another's, e.g. periodic boundary data exchange
template<typename Operator>
void operate_on_fields(auto& dst, auto const& src)
{
    assert(dst.lcl_box.shape() == src.lcl_box.shape());
    auto const d_spans = make_field_box_spans(dst.lcl_box, dst.field);
    auto const s_spans = make_field_box_spans(src.lcl_box, src.field);

    auto s_span = s_spans.begin();
    for (auto const& d_span : d_spans)
        operate_on_span<Operator>(d_span, (*s_span++).data());
}




template<typename Field_t>
template<typename Operator>
void FieldBox<Field_t>::set_from(std::vector<value_type> const& vec, std::size_t seek)
{
    for (auto const& span : make_field_box_spans(lcl_box, field))
    {
        operate_on_span<Operator>(span, vec.data() + seek);
        seek += span.size();
    }
}


template<typename Field_t>
void FieldBox<Field_t>::append_to(std::vector<value_type>& vec)
{
    // reserve vec before use!
    std::size_t seek = vec.size();
    vec.resize(vec.size() + lcl_box.size());

    for (auto const& span : make_field_box_spans(lcl_box, field))
    {
        // element-wise for the same reasons as operate_on_span
        auto* const dst = vec.data() + seek;
        for (std::size_t i = 0; i < span.size(); ++i)
            dst[i] = span[i];
        seek += span.size();
    }
}

} // namespace PHARE::core


#endif
