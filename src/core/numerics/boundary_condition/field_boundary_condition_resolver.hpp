#ifndef PHARE_CORE_NUMERICS_FIELD_BOUNDARY_CONDITION_RESOLVER
#define PHARE_CORE_NUMERICS_FIELD_BOUNDARY_CONDITION_RESOLVER

#include "core/data/tensorfield/tensorfield_traits.hpp"
#include "core/data/vecfield/vecfield_traits.hpp"
#include "core/numerics/boundary_condition/field_antisymmetric_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_divergence_free_transverse_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_divergence_free_transverse_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_none_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_symmetric_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_total_energy_from_pressure_boundary_condition.hpp"

namespace PHARE::core
{

template<FieldBoundaryConditionType type>
struct FieldBoundaryConditionSelector;

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::None>
{
    template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename StateT>
    using type = FieldNoneBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::Dirichlet>
{
    template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename StateT>
    using type = FieldDirichletBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::AntiSymmetric>
{
    template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename StateT>
    using type = FieldAntiSymmetricBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::Symmetric>
{
    template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename StateT>
    using type = FieldSymmetricBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::Neumann>
{
    template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename StateT>
    using type = FieldNeumannBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>
{
    // only makes sense for a vector field
    template<IsVecField VecFieldT, typename GridLayoutT, typename StateT>
    using type
        = FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::DivergenceFreeTransverseDirichlet>
{
    // only makes sense for a vector field
    template<IsVecField VecFieldT, typename GridLayoutT, typename StateT>
    using type
        = FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecFieldT, GridLayoutT, StateT>;
};

template<>
struct FieldBoundaryConditionSelector<FieldBoundaryConditionType::TotalEnergyFromPressure>
{
    // only makes sense for a scalar field
    template<IsField FieldT, typename GridLayoutT, typename StateT>
    using type = FieldTotalEnergyFromPressureBoundaryCondition<FieldT, GridLayoutT, StateT>;
};

template<FieldBoundaryConditionType type, IsScalarOrTensorField ScalarOrTensorFieldT,
         typename GridLayoutT, typename StateT>
using FieldBoundaryCondition
    = FieldBoundaryConditionSelector<type>::template type<ScalarOrTensorFieldT, GridLayoutT,
                                                          StateT>;

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_FIELD_BOUNDARY_CONDITION_RESOLVER
