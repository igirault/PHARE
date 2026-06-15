#ifndef PHARE_CORE_INNER_BOUNDARY_FIELD_DIRICHLET_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_INNER_BOUNDARY_FIELD_DIRICHLET_INNER_BOUNDARY_CONDITION_HPP

#include "core/inner_boundary/field_inner_boundary_condition.hpp"

namespace PHARE::core
{
template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
class FieldDirichletInnerBoundaryCondition
    : public FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>
{
public:
    using Super = FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    using value_type                    = Super::field_type::value_type;
    using field_type                    = Super::field_type;
    using inner_boundary_mesh_data_type = Super::inner_boundary_mesh_data_type;
    using ghost_elem_data_type          = Super::ghost_elem_data_type;
    using inner_boundary_type           = Super::inner_boundary_type;
    using interpolator_type             = Super::interpolator_type;
    using state_type                    = Super::state_type;
    using context_type                  = Super::context_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    enum class ExtrapolationType { Constant, Linear };

    FieldDirichletInnerBoundaryCondition() = default;

    FieldDirichletInnerBoundaryCondition(value_type value, ExtrapolationType extrapolationType
                                                           = ExtrapolationType::Linear)
        : values_{value}
        , extrapolation_type_(extrapolationType)
    {
    }

    FieldDirichletInnerBoundaryCondition(std::array<value_type, N> values,
                                         ExtrapolationType extrapolationType
                                         = ExtrapolationType::Linear)
        : values_{values}
        , extrapolation_type_(extrapolationType)
    {
    }

    FieldInnerBoundaryConditionType getType() const override
    {
        return FieldInnerBoundaryConditionType::Dirichlet;
    }

    void setExtrapolationType(ExtrapolationType extrapolationType)
    {
        extrapolation_type_ = extrapolationType;
    }

    void apply(ScalarOrTensorFieldT& scalarOrTensorField, GridLayoutT const& layout,
               inner_boundary_mesh_data_type const& boundaryMeshData,
               context_type const& ctx) override
    {
        auto fields = [&]() {
            if constexpr (is_scalar)
                return std::make_tuple(scalarOrTensorField);
            else
                return scalarOrTensorField.components();
        }();

        auto extrapolate = (extrapolation_type_ == ExtrapolationType::Linear)
                               ? [](value_type v, value_type m) { return 2.0 * v - m; }
                               : [](value_type v, value_type /*m*/) { return v; };

        for_N<N>([&](auto i) {
            auto& field            = std::get<i>(fields);
            auto const centering   = GridLayoutT::centering(field);
            auto const& ghostElems = boundaryMeshData.getGhostDataFromCentering(centering);

            for (ghost_elem_data_type const& ghostElem : ghostElems)
            {
                // WARNING: when the mirror is not interpolable, the ghost cell is left
                // untouched. This may be the cause of issues — TBD. If so, a lower-order
                // interpolation could be applied instead. See
                // GhostElemData::mirrorIsInterpolable.
                if (!ghostElem.mirrorIsInterpolable)
                    continue;
                double mirrorValue     = this->interpolator_(layout, field, ghostElem.mirrorPoint);
                field(ghostElem.index) = extrapolate(values_[i], mirrorValue);
            }
        });
    }

private:
    std::array<value_type, N> values_{0};
    ExtrapolationType extrapolation_type_{ExtrapolationType::Linear};
};

} // namespace PHARE::core
#endif // PHARE_CORE_INNER_BOUNDARY_FIELD_DIRICHLET_INNER_BOUNDARY_CONDITION_HPP
