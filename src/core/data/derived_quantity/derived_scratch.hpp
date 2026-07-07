#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP

#include "core/data/derived_quantity/centering.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace PHARE::core
{
namespace detail
{
    template<std::size_t dim>
    std::size_t product(std::array<std::uint32_t, dim> const& shape)
    {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                               std::multiplies<std::size_t>{});
    }
} // namespace detail

/** Build a centering-correct scalar view over an all-primal SAMRAI-backed
 *  scratch field. The backing buffer is the most demanding case (all-primal),
 *  so any centering shape fits inside its per-patch allocation. The view
 *  aliases the backing memory; contents are transient per (patch, quantity). */
template<typename PhysicalQuantity, typename Field_t, typename GridLayout>
Field_t derived_scalar_view(Field_t& backing, ScalarCentering const centering,
                            GridLayout const& layout)
{
    auto const qty   = scalar_qty<PhysicalQuantity>(centering);
    auto const shape = layout.allocSize(qty);
    assert(backing.isUsable());
    // always-on: an over-sized quantity would alias past the all-primal backing
    if (detail::product<GridLayout::dimension>(shape) > backing.size())
        throw std::runtime_error("derived scalar scratch overflow: quantity exceeds all-primal "
                                 "backing allocation");
    return Field_t{backing.name(), qty, backing.data(), shape};
}

/** Same for vectors: each component view lives inside the corresponding
 *  all-primal backing component (no cross-component packing). */
template<typename PhysicalQuantity, typename VecField_t, typename GridLayout>
VecField_t derived_vector_view(VecField_t& backing, VectorCentering const centering,
                               GridLayout const& layout)
{
    using Field_t = typename VecField_t::field_type;

    auto const vqty = vector_qty<PhysicalQuantity>(centering);
    auto const qtys = PhysicalQuantity::componentsQuantities(vqty);

    VecField_t vf{backing.name(), vqty};
    for (std::size_t i = 0; i < 3; ++i)
    {
        auto const shape = layout.allocSize(qtys[i]);
        assert(backing[i].isUsable());
        // always-on: an over-sized component would alias past the all-primal backing
        if (detail::product<GridLayout::dimension>(shape) > backing[i].size())
            throw std::runtime_error(
                "derived vector scratch overflow: component exceeds all-primal "
                "backing allocation");
        Field_t component{vf[i].name(), qtys[i], backing[i].data(), shape};
        vf[i].setBuffer(&component);
    }
    return vf;
}

/** Zero a scalar scratch view before compute. The backing scratch is reused
 *  across patches and quantities and is never cleared, so a computer that only
 *  fills part of the view (e.g. curl/laplacian operators that shrink the
 *  evaluation box) would otherwise leave stale cross-patch data in the ghost
 *  layers that get written to file. */
template<typename Field_t>
void zero_scalar_view(Field_t& view)
{
    std::fill(view.data(), view.data() + view.size(), 0.0);
}

/** Same, per component, for a vector scratch view. */
template<typename VecField_t>
void zero_vector_view(VecField_t& view)
{
    for (std::size_t i = 0; i < 3; ++i)
        std::fill(view[i].data(), view[i].data() + view[i].size(), 0.0);
}

} // namespace PHARE::core

#endif
