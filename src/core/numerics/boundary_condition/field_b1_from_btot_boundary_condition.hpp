#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_B1_FROM_BTOT_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_B1_FROM_BTOT_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "initializer/data_provider.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace PHARE::core
{
/**
 * @brief Boundary condition for the magnetic perturbation B1 that imposes a prescribed value
 * on the *total* field B = B0 + B1.
 *
 * A divergence-free transverse Dirichlet on the *total* field: on the transverse components the
 * ghost values of
 * B1 are written so that B = B0 + B1 linearly extrapolates to the constant @c Btot_, with the
 * spatially-varying background B0 read from the current-state accessor. The normal component is
 * set so that the numerical divergence of B1 is zero; since the background B0 is itself
 * divergence-free, div B = 0 is preserved.
 *
 * This is the recipe used by inflow boundaries when the user prescribes the total magnetic field
 * rather than its perturbation.
 *
 * @warning Only valid for vector fields with the same centering as the magnetic field (B0 is read
 * at the same indices as B1).
 *
 * @tparam VecFieldT Type of the vector field.
 * @tparam GridLayoutT Grid layout configuration.
 */
template<typename VecFieldT, typename GridLayoutT>
class FieldB1FromBtotBoundaryCondition : public IFieldBoundaryCondition<VecFieldT, GridLayoutT>
{
public:
    using Super                = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type           = Super::field_type;
    using value_type           = field_type::value_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static_assert(N == 3, "B1-from-Btot boundary condition only applies to vector fields.");

    FieldB1FromBtotBoundaryCondition() = default;

    FieldB1FromBtotBoundaryCondition(value_type value)
    {
        for (size_t i = 0; i < N; ++i)
            Btot_[i] = value;
    }

    FieldB1FromBtotBoundaryCondition(std::array<value_type, N> const& values)
        : Btot_{values}
    {
    }

    // Time-varying total field B(t): one function per component, evaluated (uniform in space)
    // at ctx.time on each apply. Used for a rotating inflow field (IMF turning).
    FieldB1FromBtotBoundaryCondition(std::array<initializer::TimeFunction<dimension>, N> fns)
        : hasFn_{true}
    {
        for (size_t i = 0; i < N; ++i)
            fn_[i] = std::move(fns[i]);
    }

    FieldB1FromBtotBoundaryCondition(FieldB1FromBtotBoundaryCondition const&)            = default;
    FieldB1FromBtotBoundaryCondition& operator=(FieldB1FromBtotBoundaryCondition const&) = default;
    FieldB1FromBtotBoundaryCondition(FieldB1FromBtotBoundaryCondition&&)                 = default;
    FieldB1FromBtotBoundaryCondition& operator=(FieldB1FromBtotBoundaryCondition&&)      = default;

    virtual ~FieldB1FromBtotBoundaryCondition() = default;

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::B1FromBtot;
    }

    void apply(VecFieldT& vecField, BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        size_t const iNormal      = static_cast<size_t>(direction);

        auto B1fields = vecField.components();

        assert(gridLayout.centering(vecField) == gridLayout.centering(tensor_quantity_type::B1));

        // total field to impose: a constant, or a time function B(t) evaluated (uniform in
        // space) at the current time.
        std::array<value_type, N> Btot = Btot_;
        if (hasFn_)
            for (size_t i = 0; i < N; ++i)
                Btot[i] = evalUniform_(i, ctx.time);

        // background field B0, read at the same indices as B1 (co-located, same centering)
        auto B0vec    = ctx.accessor_new.getVecField(tensor_quantity_type::B0);
        auto B0fields = B0vec.components();

        // transverse components: Dirichlet on the total field B = B0 + B1, written into B1
        for_N<N>([&](auto iTransverse) {
            if (static_cast<size_t>(iTransverse) != iNormal)
            {
                field_type& B1c        = std::get<iTransverse>(B1fields);
                field_type const& B0c  = std::get<iTransverse>(B0fields);
                value_type const Btoti = Btot[iTransverse];

                QtyCentering const centering = GridLayoutT::centering(
                    B1c.physicalQuantity())[static_cast<size_t>(direction)];
                auto fieldBox = gridLayout.toFieldBox(localGhostBox, B1c.physicalQuantity());

                for (_index const& index : fieldBox)
                {
                    _index const mirrorIndex
                        = gridLayout.boundaryMirrored(direction, side, centering, index);
                    // total field Dirichlet to Btoti, solved for the B1 ghost value:
                    //   B(index) = (mirror==index) ? Btoti : 2*Btoti - B(mirror)
                    //   B1(index) = B(index) - B0(index)
                    // the mirror only flips the normal component, so mirrorIndex == index is
                    // equivalent to comparing that component (and avoids a dynamic [] index that
                    // trips a -Warray-bounds false positive in 1D).
                    B1c(index)
                        = (mirrorIndex == index)
                              ? Btoti - B0c(index)
                              : 2.0 * Btoti - B0c(mirrorIndex) - B1c(mirrorIndex) - B0c(index);
                }
            }
        });

        // normal component: set so that the numerical divergence of B1 is zero (operates on B1).
        field_type& nField = [&]() -> field_type& {
            switch (iNormal)
            {
                case 0: return std::get<0>(B1fields);
                case 1: return std::get<1>(B1fields);
                default: return std::get<2>(B1fields);
            }
        }();

        auto apply_loop = [&](auto begin, auto end) {
            for (auto it = begin; it != end; ++it)
            {
                _index const& index = *it;

                double transverseDiv = 0.0;
                for_N<dimension>([&](auto iTransverse) {
                    if (static_cast<size_t>(iTransverse) != iNormal)
                    {
                        field_type& tField       = std::get<iTransverse>(B1fields);
                        _index const upper_index = index.neighbor(iTransverse, 1);
                        transverseDiv += tField(upper_index) - tField(index);
                    }
                });

                if (side == Side::Upper)
                {
                    _index const index_to_set      = index.neighbor(iNormal, 1);
                    _index const index_already_set = index;
                    nField(index_to_set)           = nField(index_already_set) - transverseDiv;
                }
                else
                {
                    _index const index_to_set      = index;
                    _index const index_already_set = index.neighbor(iNormal, 1);
                    nField(index_to_set)           = nField(index_already_set) + transverseDiv;
                }
            }
        };

        if (side == Side::Upper)
            apply_loop(localGhostBox.begin(), localGhostBox.end());
        else
            apply_loop(localGhostBox.rbegin(), localGhostBox.rend());
    }

private:
    using _index = Point<std::uint32_t, dimension>;

    // Evaluate component i of the (space-uniform) time function at time t. Passes a single
    // dummy coordinate per dimension and reads the first returned node value.
    value_type evalUniform_(size_t i, double t) const
    {
        std::vector<double> const one(1, 0.0);
        std::shared_ptr<Span<double>> s;
        if constexpr (dimension == 1)
            s = (*fn_[i])(one, t);
        else if constexpr (dimension == 2)
            s = (*fn_[i])(one, one, t);
        else
            s = (*fn_[i])(one, one, one, t);
        return (*s)[0];
    }

    std::array<value_type, N> Btot_{0};
    std::array<std::optional<initializer::TimeFunction<dimension>>, N> fn_{};
    bool hasFn_ = false;

}; // class FieldB1FromBtotBoundaryCondition

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_B1_FROM_BTOT_BOUNDARY_CONDITION_HPP
