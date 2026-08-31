#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_BOUNDARY_CONDITION_CONTEXT_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_BOUNDARY_CONDITION_CONTEXT_HPP

#include "core/data/patch_field_accessor.hpp"

namespace PHARE::core
{

/**
 * @brief Bundle of context data passed to outer (physical-edge) boundary condition appliers.
 *
 * Exposes the current substage state (`accessor_new`) plus the simulation time.
 *
 * @tparam FieldT             Scalar field type.
 * @tparam PhysicalQuantityT  Quantity traits (e.g. MHDQuantity, HybridQuantity).
 */
template<typename FieldT, typename PhysicalQuantityT>
struct BoundaryConditionContext
{
    using accessor_type = IPatchFieldAccessor<FieldT, PhysicalQuantityT>;

    accessor_type const& accessor_new;
    double time;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_BOUNDARY_CONDITION_CONTEXT_HPP
