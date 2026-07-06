#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP

#include "core/data/derived_quantity/centering.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>

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
    assert(detail::product<GridLayout::dimension>(shape) <= backing.size());
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
        assert(detail::product<GridLayout::dimension>(shape) <= backing[i].size());
        Field_t component{vf[i].name(), qtys[i], backing[i].data(), shape};
        vf[i].setBuffer(&component);
    }
    return vf;
}

} // namespace PHARE::core

#endif
