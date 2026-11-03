#ifndef PHARE_CORE_INNER_BOUNDARY_FIELD_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_INNER_BOUNDARY_FIELD_INNER_BOUNDARY_CONDITION_HPP

#include "core/data/field/field_traits.hpp"
#include "core/data/tensorfield/tensorfield_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_geometry.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/numerics/interpolator/field_at_point.hpp"

namespace PHARE::core
{

enum class FieldInnerBoundaryConditionType { None, Dirichlet, Neumann, TotalEnergyFromPressure };

template<IsScalarOrTensorField ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
class FieldInnerBoundaryCondition
{
public:
    static constexpr bool is_scalar   = IsField<ScalarOrTensorFieldT>;
    static constexpr size_t dimension = GridLayoutT::dimension;
    static constexpr size_t N = NumberOfComponentsSelector<ScalarOrTensorFieldT, is_scalar>::value;

    using This = FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    using inner_boundary_type    = InnerBoundaryGeometry<dimension>;
    using field_type             = FieldTypeSelector<ScalarOrTensorFieldT, is_scalar>::type;
    using physical_quantity_type = layout_quantity_t<GridLayoutT>;
    using vecfield_type          = VecField<field_type, physical_quantity_type>;
    using tensor_quantity_type
        = PhysicalQuantityTypeSelector<ScalarOrTensorFieldT, is_scalar>::type;
    using inner_boundary_mesh_data_type = InnerBoundaryMeshData<dimension, physical_quantity_type>;
    using ghost_elem_data_type          = inner_boundary_mesh_data_type::ghost_elem_data_type;
    using state_type                    = PhysicalStateT;
    using context_type                  = InnerBCContext<PhysicalStateT>;
    using interpolator_type             = FieldAtPoint<dimension, 1>;

    FieldInnerBoundaryCondition()          = default;
    virtual ~FieldInnerBoundaryCondition() = default;

    FieldInnerBoundaryCondition(This const&) = delete;
    FieldInnerBoundaryCondition(This&&)      = delete;
    This& operator=(This const&)             = delete;
    This& operator=(This&&)                  = delete;

    virtual FieldInnerBoundaryConditionType getType() const = 0;

    virtual int priority() const { return 0; }

    virtual bool fillsNonInterpolableGhosts() const { return false; }

    virtual void apply(ScalarOrTensorFieldT& scalarOrTensorField, GridLayoutT const& layout,
                       inner_boundary_mesh_data_type const& boundaryMeshData,
                       context_type const& ctx)
        = 0;

protected:
    interpolator_type interpolator_;
};

} // namespace PHARE::core

#endif
