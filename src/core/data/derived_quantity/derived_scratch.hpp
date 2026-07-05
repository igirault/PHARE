#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_SCRATCH_HPP

#include "core/data/derived_quantity/centering.hpp"
#include "core/utilities/types.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

namespace PHARE::core
{
/** Transient scratch memory for derived-quantity outputs. One raw block, grown
 *  lazily to the most demanding request; Field/VecField views are built per
 *  patch over it. Deliberately NOT a SAMRAI resource: values never survive the
 *  patch visit, so ResourcesManager/setOnPatch machinery is unnecessary. */
template<typename VecField_t, typename PhysicalQuantity>
class DerivedScratch
{
public:
    using Field_t                   = typename VecField_t::field_type;
    static constexpr auto dimension = Field_t::dimension;

    template<typename GridLayout>
    Field_t scalar(ScalarCentering const centering, GridLayout const& layout)
    {
        auto const qty   = scalar_qty<PhysicalQuantity>(centering);
        auto const shape = layout.allocSize(qty);
        ensure_(product_(shape));
        return Field_t{"PHARE_derived_scratch", qty, mem_.data(), shape};
    }

    template<typename GridLayout>
    VecField_t vector(VectorCentering const centering, GridLayout const& layout)
    {
        auto const qty  = vector_qty<PhysicalQuantity>(centering);
        auto const qtys = PhysicalQuantity::componentsQuantities(qty);

        VecField_t vf{"PHARE_derived_scratch_vec", qty};

        std::array<std::array<std::uint32_t, dimension>, 3> shapes;
        std::size_t total = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            shapes[i] = layout.allocSize(qtys[i]);
            total += product_(shapes[i]);
        }
        ensure_(total);

        std::size_t offset = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            Field_t component{vf[i].name(), qtys[i], mem_.data() + offset, shapes[i]};
            vf[i].setBuffer(&component);
            offset += product_(shapes[i]);
        }
        return vf;
    }

    template<std::size_t rank, typename GridLayout, typename Centering>
    auto view(Centering const centering, GridLayout const& layout)
    {
        static_assert(rank <= 1, "tensor scratch not implemented");
        if constexpr (rank == 0)
            return scalar(centering, layout);
        else
            return vector(centering, layout);
    }

private:
    static std::size_t product_(std::array<std::uint32_t, dimension> const& shape)
    {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                               std::multiplies<std::size_t>{});
    }

    void ensure_(std::size_t const n)
    {
        if (mem_.size() < n)
            mem_.resize(n);
    }

    std::vector<double> mem_;
};

} // namespace PHARE::core

#endif
