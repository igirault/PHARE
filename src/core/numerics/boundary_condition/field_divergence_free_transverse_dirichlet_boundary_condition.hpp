#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_DIRICHLET_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_DIRICHLET_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/divergence_free_transverse_common.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"

#include "initializer/data_provider.hpp"

#include <array>
#include <cstddef>

namespace PHARE::core
{
/**
 * @brief Boundary condition for vector fields that imposes a value on tangential
 * components and sets the normal component so that numerical divergence is zero.
 *
 * The imposed tangential value can be either a compile-time constant (per component) or a
 * user function of space and time, evaluated at the current time (@c ctx.time): each
 * tangential component is delegated to a scalar @c FieldDirichletBoundaryCondition, which
 * supports both. The normal component is then recomputed from the (possibly
 * time-dependent) tangential ghost values so that discrete divergence stays zero, so it
 * inherits the time dependence automatically.
 *
 * @warning Only valid for vector fields with the same centering as the magnetic field.
 *
 * @tparam VecFieldT Type of the vector field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class FieldDivergenceFreeTransverseDirichletBoundaryCondition
    : public IFieldBoundaryCondition<VecFieldT, GridLayoutT>
{
public:
    using Super                = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type           = Super::field_type;
    using value_type           = field_type::value_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static_assert(N == 3,
                  "Divergence-free transverse Dirichlet boundary condition only applies to vector "
                  "fields.");

    FieldDivergenceFreeTransverseDirichletBoundaryCondition() = default;

    FieldDivergenceFreeTransverseDirichletBoundaryCondition(value_type value)
    {
        for (size_t i = 0; i < N; ++i)
            scalar_dirichlet_conditions_[i] = _scalar_dirichlet_bc_type{value};
    }

    FieldDivergenceFreeTransverseDirichletBoundaryCondition(
        std::array<value_type, N> const& values)
    {
        for (size_t i = 0; i < N; ++i)
            scalar_dirichlet_conditions_[i] = _scalar_dirichlet_bc_type{values[i]};
    }

    // Function-based tangential values: one space-time function f(x[,y[,z]], t) per
    // component, evaluated at ctx.time. Drives a time-varying prescribed field (e.g. an
    // inflow B(t) during IMF turning) while keeping the normal component divergence-free.
    FieldDivergenceFreeTransverseDirichletBoundaryCondition(
        std::array<initializer::SpaceTimeFunction<dimension>, N> const& fns)
    {
        for (size_t i = 0; i < N; ++i)
            scalar_dirichlet_conditions_[i] = _scalar_dirichlet_bc_type{fns[i]};
    }

    FieldDivergenceFreeTransverseDirichletBoundaryCondition(
        FieldDivergenceFreeTransverseDirichletBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseDirichletBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseDirichletBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseDirichletBoundaryCondition(
        FieldDivergenceFreeTransverseDirichletBoundaryCondition&&)
        = default;
    FieldDivergenceFreeTransverseDirichletBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseDirichletBoundaryCondition&&)
        = default;

    virtual ~FieldDivergenceFreeTransverseDirichletBoundaryCondition() = default;

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::DivergenceFreeTransverseDirichlet;
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

        // handle transverse components with Dirichlet
        for_N<N>([&](auto iTransverse) {
            if (static_cast<size_t>(iTransverse) != iNormal)
            {
                field_type& tField = std::get<iTransverse>(fields);
                scalar_dirichlet_conditions_[iTransverse].apply(tField, boundaryLocation,
                                                                localGhostBox, gridLayout, ctx);
            }
        });

        // set the normal component so the discrete divergence of B is zero, given the transverse
        // ghosts filled above (shared with the transverse-Neumann condition).
        applyDivergenceFreeNormalComponent<dimension>(fields, iNormal, side, gridLayout,
                                                       localGhostBox);
    }

private:
    using _scalar_dirichlet_bc_type = FieldDirichletBoundaryCondition<field_type, GridLayoutT>;
    using _index                    = Point<std::uint32_t, dimension>;

    std::array<_scalar_dirichlet_bc_type, N> scalar_dirichlet_conditions_;

}; // class FieldDivergenceFreeTransverseDirichletBoundaryCondition

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_DIRICHLET_BOUNDARY_CONDITION_HPP
