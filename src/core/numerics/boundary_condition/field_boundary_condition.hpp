#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/field/field_traits.hpp"
#include "core/data/patch_field_accessor.hpp"
#include "core/data/tensorfield/tensorfield_traits.hpp"
#include "core/utilities/box/box.hpp"

#include <tuple>

namespace PHARE::core
{

/**
 * @brief Supported types of field boundary conditions.
 *
 */
enum class FieldBoundaryConditionType : int {
    None,
    Dirichlet,
    AntiSymmetric,
    Symmetric,
    Neumann,
    DivergenceFreeTransverseNeumann,
    DivergenceFreeTransverseDirichlet,
    TotalEnergyFromPressure
};

/** @brief Context data passed to boundary conditions */
template<typename FieldT, typename PhysicalQuantityT>
struct BoundaryConditionContext
{
    using patch_field_accessor_type = IPatchFieldAccessor<FieldT, PhysicalQuantityT>;

    patch_field_accessor_type const& accessor_new;
    double time;
};

/**
 * @brief Interface for applying boundary conditions to scalar or tensor fields.
 *
 * Concrete field boundary conditions are provided by implementating this interface.
 *
 * @tparam ScalarOrTensorFieldT The type of the scalarOrTensorField (must satisfy IsField or
 * IsTensorField).
 * @tparam GridLayoutT The grid layout type .
 *
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
    requires(IsField<ScalarOrTensorFieldT> || IsTensorField<ScalarOrTensorFieldT>)
class IFieldBoundaryCondition
{
public:
    static constexpr bool is_scalar   = IsField<ScalarOrTensorFieldT>;
    static constexpr size_t dimension = GridLayoutT::dimension;
    static constexpr size_t N = NumberOfComponentsSelector<ScalarOrTensorFieldT, is_scalar>::value;

    using This = IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>;
    // the quantity category (HybridQuantity / MHDQuantity) is carried by the layout options
    using physical_quantity_type = typename decltype(GridLayoutT::options.field_options)::Quantity;
    using tensor_quantity_type
        = PhysicalQuantityTypeSelector<ScalarOrTensorFieldT, is_scalar>::type;
    using field_type                = FieldTypeSelector<ScalarOrTensorFieldT, is_scalar>::type;
    using context_type              = BoundaryConditionContext<field_type, physical_quantity_type>;
    using patch_field_accessor_type = context_type::patch_field_accessor_type;

    /** @brief Return the type of the boundary condition. */
    virtual FieldBoundaryConditionType getType() const = 0;

    virtual ~IFieldBoundaryCondition() = default;

    /**
     * @brief Whether this condition can be applied with the state currently reachable through
     * @p ctx.
     *
     * This query is necessary, because on tempory regrid patches, coupled field boundary
     * conditions might not be applicable and require a fallback strategy
     */
    virtual bool canApply(context_type const& /*ctx*/) const { return true; }


    /**
     * @brief Enforce the boundary condition on the provided scalar/tensor @p scalarOrTensorField,
     * by filling accordingly the ghost cells contained in the local box @p localGhostBox, at the
     * physical time carried by @p ctx, and considering that the boundary is located at
     * @p boundaryLocation.
     *
     * @param scalarOrTensorField The scalar or tensor to which we apply the boundary condition.
     * @param boundaryLocation The location of the physical boundary.
     * @param localGhostBox The box containing the ghost cells/nodes to fill.
     * @param gridLayout The grid layout.
     * @param ctx Bundle of context data: accessor to the current substage state and the
     *            simulation time. BCs read siblings through `ctx.accessor_new` and use `ctx.time`.
     */
    virtual void apply(ScalarOrTensorFieldT& scalarOrTensorField,
                       BoundaryLocation const boundaryLocation,
                       Box<std::uint32_t, dimension> const& localGhostBox,
                       GridLayoutT const& gridLayout, context_type const& ctx) = 0;

protected:
    /** @brief Unwrap a scalar-or-tensor field into a tuple of its scalar component fields */
    static auto asComponentTuple(ScalarOrTensorFieldT& scalarOrTensorField)
    {
        if constexpr (is_scalar)
            return std::make_tuple(scalarOrTensorField);
        else
            return scalarOrTensorField.components();
    }
};

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP
