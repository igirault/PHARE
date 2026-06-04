#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"

#include <cstddef>

namespace PHARE::core
{
/**
 * @brief Outflow boundary condition for the magnetic perturbation B1: zero normal derivative on the
 * transverse B1 components, with the normal component set so that the numerical divergence of B1 is
 * zero.
 *
 * The evolved/transported magnetic quantity is the perturbation B1; the background B0 (e.g. a
 * dipole or IMF) is analytic and is filled into the ghost cells at its exact value there
 * (independently of this BC). Enforcing zero normal gradient on B1 — rather than on the total
 * B = B0 + B1 — is what makes the scheme well-balanced w.r.t. a non-uniform static B0: at rest
 * B1 = 0 everywhere, so B1(ghost) = B1(mirror) = 0 and no spurious B1 (hence no spurious EMF /
 * Lorentz force) is injected at the boundary. Mirroring the *total* field instead would write
 * B0(mirror) - B0(index) into the B1 ghost, i.e. the discrete dipole gradient, seeding a spurious
 * perturbation from rest. Since B0 is divergence-free, div B1 = 0 implies div B = 0.
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
               Super::boundary_condition_context_type const& /*ctx*/) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        size_t const iNormal      = static_cast<size_t>(direction);

        auto fields = vecField.components();

        assert(gridLayout.centering(vecField) == gridLayout.centering(tensor_quantity_type::B1));

        // transverse components: zero normal gradient on the perturbation B1 (B0 is filled
        // analytically into the ghosts elsewhere). Mirroring B1 keeps the rest state B1 = 0
        // exactly, which is what makes the boundary well-balanced w.r.t. a non-uniform B0.
        //   B1(index) = B1(mirror)
        for_N<N>([&](auto iTransverse) {
            if (static_cast<size_t>(iTransverse) != iNormal)
            {
                field_type& B1c = std::get<iTransverse>(fields);

                QtyCentering const centering = GridLayoutT::centering(
                    B1c.physicalQuantity())[static_cast<size_t>(direction)];
                auto fieldBox = gridLayout.toFieldBox(localGhostBox, B1c.physicalQuantity());

                for (_index const& index : fieldBox)
                {
                    _index const mirrorIndex
                        = gridLayout.boundaryMirrored(direction, side, centering, index);
                    B1c(index) = B1c(mirrorIndex);
                }
            }
        });

        // handle normal component: iterate ghost cells closest-to-farthest from physical domain
        field_type& nField = [&]() -> field_type& {
            switch (iNormal)
            {
                case 0: return std::get<0>(fields);
                case 1: return std::get<1>(fields);
                default: return std::get<2>(fields);
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
                        field_type& tField       = std::get<iTransverse>(fields);
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

}; // class FieldDivergenceFreeTransverseNeumannBoundaryCondition

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
