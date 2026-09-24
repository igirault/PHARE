#ifndef PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP
#define PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP

#include "amr/data/field/field_data_traits.hpp"
#include "amr/data/tensorfield/tensor_field_data_traits.hpp"

#include "core/boundary/boundary_defs.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"

#include "SAMRAI/geom/CartesianPatchGeometry.h"
#include "SAMRAI/hier/BoundaryBox.h"
#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/hier/PatchGeometry.h"
#include "SAMRAI/tbox/Dimension.h"
#include "SAMRAI/xfer/RefinePatchStrategy.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace PHARE::amr
{

/**
 * @brief Strategy for filling physical boundary conditions and customizing patch refinment.
 *
 * This class implements the SAMRAI::xfer::RefinePatchStrategy interface to
 * specify how physical boundary conditions must be enforced for patches that touch
 * the domain boundaries. Refinement customization via preprocessRefine and postProcessRefine is
 * deferred to child classes.
 *
 * Each refiner is expected to hold an instance of this class, but all these instances will point to
 * the same common boundary manager.
 *
 * @tparam ResMan The resources manager type.
 * @tparam ScalarOrTensorFieldDataT The data type for fields or tensor fields.
 * @tparam BoundaryManagerT Manager responsible for providing boundary condition objects.
 */
template<typename ResMan, typename ScalarOrTensorFieldDataT, typename BoundaryManagerT>
    requires(IsFieldData<ScalarOrTensorFieldDataT> || IsTensorFieldData<ScalarOrTensorFieldDataT>)
class FieldRefinePatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
public:
    static constexpr bool is_scalar   = IsFieldData<ScalarOrTensorFieldDataT>;
    static constexpr bool is_tensor   = !is_scalar;
    static constexpr size_t dimension = ScalarOrTensorFieldDataT::dimension;

    using field_geometry_type    = FieldGeometrySelector<ScalarOrTensorFieldDataT, is_scalar>::type;
    using gridlayout_type        = ScalarOrTensorFieldDataT::gridlayout_type;
    using grid_type              = ScalarOrTensorFieldDataT::grid_type;
    using field_type             = grid_type::field_type;
    using physical_quantity_type = BoundaryManagerT::physical_quantity_type;
    using vectorfield_type       = core::VecField<field_type, physical_quantity_type>;
    using scalar_or_tensor_field_type
        = ScalarOrTensorFieldSelector<ScalarOrTensorFieldDataT, is_scalar>::type;
    using scalar_quantity_type = physical_quantity_type::Scalar;
    using vector_quantity_type = physical_quantity_type::Vector;

    using patch_geometry_type           = SAMRAI::hier::PatchGeometry;
    using cartesian_patch_geometry_type = SAMRAI::geom::CartesianPatchGeometry;

    using boundary_type = BoundaryManagerT::boundary_type;
    using state_type    = BoundaryManagerT::state_type;
    using boundary_condition_type
        = core::IFieldBoundaryCondition<scalar_or_tensor_field_type, gridlayout_type, state_type>;
    using boundary_condition_context_type = boundary_condition_type::context_type;

    /**
     * @brief Constructor.
     * @param resources_manager Simulation resources manager.
     * @param boundary_manager Manager handling boundary conditions.
     */
    FieldRefinePatchStrategy(ResMan& resourcesManager, BoundaryManagerT& boundaryManager)
        : rm_{resourcesManager}
        , boundaryManager_{boundaryManager}
        , data_id_{-1}
    {
    }

    /**
     * @brief Check that the patch data identifier is registered.
     */
    void assertIDsSet() const
    {
        assert(data_id_ >= 0 && "FieldRefinePatchStrategy: IDs must be registered before use");
    }

    /**
     * @brief Register the SAMRAI patch data identifier.
     * @param field_id Integer ID from the SAMRAI variable database.
     * @param state view of the sub-state bound to the patch and exposed to BC appliers; null if
     * none.
     */
    void registerIDs(int const field_id, std::shared_ptr<state_type> state = nullptr)
    {
        data_id_  = field_id;
        state_    = std::move(state);
        stateIds_ = state_ ? rm_.getIDs(*state_) : std::vector<int>{};
    }

    /**
     * @brief Apply physical boundary conditions via SAMRAI callback.
     *
     * Iterate over patch boundaries that touch a physical domain boundary and apply the appropriate
     * PHARE boundary condition to ghost regions.
     *
     * @param patch The fine patch being refined.
     * @param fill_time Simulation time for BC application.
     * @param ghost_width_to_fill Width of ghost cell layer to be filled.
     */
    void
    setPhysicalBoundaryConditions(SAMRAI::hier::Patch& patch, double const fill_time,
                                  SAMRAI::hier::IntVector const& /*ghost_width_to_fill*/) override
    {
        if (!state_ || !patch.inHierarchy())
        {
            applyBoundaryConditions_(patch, fill_time, nullptr);
            return;
        }
        if (!std::all_of(stateIds_.begin(), stateIds_.end(),
                         [&](int const id) { return patch.checkAllocated(id); }))
            throw std::runtime_error(
                "FieldRefinePatchStrategy: sub-state not fully allocated on a hierarchy patch");
        auto _ = rm_.setOnPatch(patch, *state_);
        applyBoundaryConditions_(patch, fill_time, state_.get());
    }


    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 1};
    }


    void preprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                          SAMRAI::hier::Box const& fine_box,
                          SAMRAI::hier::IntVector const& ratio) override
    {
    }


    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                           SAMRAI::hier::Box const& fine_box,
                           SAMRAI::hier::IntVector const& ratio) override
    {
    }


protected:
    void applyBoundaryConditions_(SAMRAI::hier::Patch& patch, double const fill_time,
                                  state_type* state)
    {
        gridlayout_type const& gridLayout = ScalarOrTensorFieldDataT::getLayout(patch, data_id_);

        /// @todo SAMRAI does not pass a `ghost_width_to_fill` consistent with the layout's
        /// field ghost width beyond L0, so we ignore the argument and refill the whole ghost
        /// layer. Making it consistent would require overriding getRefineOpStencilWidth,
        /// deferred to avoid perturbing the always-return-1 interpolation stencil.
        SAMRAI::hier::IntVector const ghost_width_to_fill{
            static_cast<SAMRAI::tbox::Dimension>(static_cast<int>(dimension)),
            static_cast<int>(gridLayout.options.field_ghost_width)};

        // no check this is a valid cast
        std::shared_ptr<cartesian_patch_geometry_type> patchGeom
            = std::static_pointer_cast<cartesian_patch_geometry_type>(patch.getPatchGeometry());

        auto scalarOrTensorField = [&]() {
            if constexpr (is_scalar)
            {
                return *(&(ScalarOrTensorFieldDataT::getField(patch, data_id_)));
            }
            else
            {
                return ScalarOrTensorFieldDataT::getTensorField(patch, data_id_);
            };
        }();

        // wrap the current sub-state (null if unavailable) and time into a single struct
        boundary_condition_context_type const ctx{state, fill_time};

        // must be retrieved to pass as argument to patchGeom->getBoundaryFillBox later
        SAMRAI::hier::Box const& patch_box = patch.getBox();

        // iterations on potential boundary codimensions in [[1, dim]]
        core::for_N<dimension>([&](auto tag) {
            constexpr auto codim = tag.value + 1;

            // find all boundaries with the current codimension
            std::vector<SAMRAI::hier::BoundaryBox> const& boundaries
                = patchGeom->getCodimensionBoundaries(static_cast<int>(codim));

            // iterate on all found boundaries of given codimension
            for (SAMRAI::hier::BoundaryBox const& bBox : boundaries)
            {
                // retrieve the localBox of ghost that must be filled
                SAMRAI::hier::Box samraiBoxToFill
                    = patchGeom->getBoundaryFillBox(bBox, patch_box, ghost_width_to_fill);
                auto localBox = gridLayout.AMRToLocal(phare_box_from<dimension>(samraiBoxToFill));

                // get location of the currently treated boundary
                auto const currentBoundaryLocation
                    = static_cast<core::CodimNBoundaryLocation<codim>>(bBox.getLocationIndex());

                // get the "master" 1-codimensional boundary that applies at the currently treated
                // boundary: for instance corner in 2D belongs to two different 1-codimensional
                // boundaries (edges), so two boundary conditions compete there. The responsability
                // of choosing which boundary condition prevails there is on the boundaryManager.
                // If the current boundary is itself 1-codimensional, then masterBoundaryLocation =
                // currentBoundaryLocation.
                core::BoundaryLocation const masterBoundaryLocation
                    = boundaryManager_.getMasterBoundaryLocation(currentBoundaryLocation);
                auto* const masterBoundary = boundaryManager_.getBoundary(masterBoundaryLocation);
                if (!masterBoundary)
                    throw std::runtime_error("Boundary not found.");

                // get the boundary condition for the current physical quantity.
                std::shared_ptr<boundary_condition_type> bc
                    = masterBoundary->getFieldCondition(scalarOrTensorField.physicalQuantity());
                if (!bc)
                    throw std::runtime_error("Field boundary condition not found.");

                // if possible, apply the retained boundary condition, as if the current boundary
                // was belonging to the 1-codimensional master boundary; this essentially defines
                // which Cartesian direction is considered to be the normal one. Again, if the
                // current boundary is itself 1-codimensional, the master boundary is just the
                // current boundary.
                //
                // Why are there situations where the boundary condition cannot be applied:
                // SAMRAI can call this on temporary, single-quantity patches it builds for
                // cross-level (coarse->fine) interpolation, on temporary levels that are not in the
                // hierarchy (patch.inHierarchy() is false). PHARE's coupled field conditions need
                // extra fields off the patch than the quantity it applies to. For instance energy
                // BC usually requires knowing rho/P/rhoV/B; those extra fields are not allocated
                // on the interpolation temp patches dedicated to the total energy.
                // `bc->canApply(ctx)` reports if the boundary condition is not appplicable because
                // some quantities are missing. Simple uncoupled boundary conditions will always
                // be applicable. But for coupled ones, this might not be the case, yet
                // temporary-patch ghosts cells at physical boundaries cannot be left as NaNs and
                // should be assigned a meaningful value. Otherwise it propagates to the final patch
                // resulting from the regrid (tried to vibe-circumvent this, but did not succeed).
                // In this situation, we therefore fall back to a Neumann boundary condition, that
                // requires no other fields than the quantity itself. Maybe a better solution exists
                // to this.
                if (bc->canApply(ctx))
                {
                    bc->apply(scalarOrTensorField, masterBoundaryLocation, localBox, gridLayout,
                              ctx);
                }
                else
                {
                    // leave a clear trace in the log when and where this scenario occured
                    PHARE_LOG_LINE_SS(
                        "Neumann fallback triggered in setPhysicalBoundaryConditions"
                        << " | field=" << scalarOrTensorField.name() << " | quantity="
                        << static_cast<int>(scalarOrTensorField.physicalQuantity())
                        << " | codim=" << static_cast<int>(codim) << " | currentBoundaryLocation="
                        << static_cast<int>(currentBoundaryLocation) << " | masterBoundaryLocation="
                        << static_cast<int>(masterBoundaryLocation) << " | fill_time=" << fill_time
                        << " | patch_box=" << patch_box << " | localBox=" << localBox);
                    core::FieldNeumannBoundaryCondition<scalar_or_tensor_field_type,
                                                        gridlayout_type, state_type>
                        neumannFallback;
                    neumannFallback.apply(scalarOrTensorField, masterBoundaryLocation, localBox,
                                          gridLayout, ctx);
                }
            }
        });
    }

    ResMan& rm_;
    BoundaryManagerT& boundaryManager_;
    int data_id_;
    std::shared_ptr<state_type> state_;
    std::vector<int> stateIds_;
};

} // namespace PHARE::amr

#endif // PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP
