#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/divergence_free_transverse_common.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"

#include <cstddef>

namespace PHARE::core
{
/**
 * @brief Boundary condition for the magnetic field B that enforces zero normal derivative on the
 * transverse components and sets the normal component so that the numerical divergence of B is
 * zero.
 *
 * On the transverse components the ghost value mirrors the first interior value (zero normal
 * gradient): B(index) = B(mirror). The normal component is then set so that div B = 0.
 *
 * @warning Only valid for vector fields with the same centering as the magnetic field.
 *
 * @tparam VecFieldT Type of the vector field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class FieldDivergenceFreeTransverseNeumannBoundaryCondition
    : public IFieldBoundaryCondition<VecFieldT, GridLayoutT>
{
public:
    using Super                = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type           = Super::field_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static_assert(
        N == 3,
        "Divergence-free transverse Neumann boundary condition only applies to vector fields.");

    FieldDivergenceFreeTransverseNeumannBoundaryCondition() = default;

    FieldDivergenceFreeTransverseNeumannBoundaryCondition(
        FieldDivergenceFreeTransverseNeumannBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseNeumannBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition(
        FieldDivergenceFreeTransverseNeumannBoundaryCondition&&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseNeumannBoundaryCondition&&)
        = default;

    virtual ~FieldDivergenceFreeTransverseNeumannBoundaryCondition() = default;

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::DivergenceFreeTransverseNeumann;
    }

    void apply(VecFieldT& vecField, BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        size_t const iNormal      = static_cast<size_t>(direction);

        if (iNormal >= dimension)
            return;

        auto fields = vecField.components();

        assert(gridLayout.centering(vecField) == gridLayout.centering(tensor_quantity_type::B));

        // transverse components: zero normal gradient, B(index) = B(mirror)
        for_N<N>([&](auto iTransverse) {
            if (static_cast<size_t>(iTransverse) != iNormal)
            {
                field_type& Bc = std::get<iTransverse>(fields);

                QtyCentering const centering = GridLayoutT::centering(
                    Bc.physicalQuantity())[static_cast<size_t>(direction)];
                auto fieldBox = gridLayout.toFieldBox(localGhostBox, Bc.physicalQuantity());

                for (_index const& index : fieldBox)
                {
                    _index const mirrorIndex
                        = gridLayout.boundaryMirrored(direction, side, centering, index);
                    Bc(index) = Bc(mirrorIndex);
                }
            }
        });

        // set the normal component so the discrete divergence of B is zero, given the transverse
        // ghosts filled above (shared with the transverse-Dirichlet condition).
        applyDivergenceFreeNormalComponent<dimension>(fields, iNormal, side, gridLayout,
                                                       localGhostBox);
    }

private:
    using _index = Point<std::uint32_t, dimension>;

}; // class FieldDivergenceFreeTransverseNeumannBoundaryCondition

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
