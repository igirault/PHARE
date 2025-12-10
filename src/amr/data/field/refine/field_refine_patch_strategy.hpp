#ifndef PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP
#define PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP

#include "SAMRAI/geom/CartesianPatchGeometry.h"
#include "SAMRAI/hier/PatchGeometry.h"
#include "core/boundary/boundary_defs.hpp"
#include "core/boundary/boundary.hpp"
#include "core/utilities/constants.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "amr/data/field/field_data_traits.hpp"
#include "amr/data/tensorfield/tensor_field_data_traits.hpp"

#include "SAMRAI/xfer/RefinePatchStrategy.h"
#include "SAMRAI/hier/BoundaryBox.h"
#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/hier/PatchGeometry.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include <cassert>
#include <memory>
#include <stdexcept>

namespace PHARE::amr
{
using core::dirX;
using core::dirY;
using core::dirZ;

/**
 * @brief Strategy for filling physical boundary conditions and customizing patch refinment.
 *
 * This class implements the SAMRAI::xfer::RefinePatchStrategy interface to
 * specify how physical boundary conditions must be enforced for patches that touch
 * the domain boundaries. Refinement customization is deferred to child classes.
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
    static constexpr bool is_scalar = IsFieldData<ScalarOrTensorFieldDataT>;
    static constexpr bool is_tensor = !is_scalar;

    using field_geometry_type = FieldGeometrySelector<ScalarOrTensorFieldDataT, is_scalar>::type;
    using gridlayout_type     = ScalarOrTensorFieldDataT::gridlayout_type;
    using grid_type           = ScalarOrTensorFieldDataT::grid_type;
    using field_type          = grid_type::field_type;
    using scalar_or_tensor_field_type
        = ScalarOrTensorFieldSelector<ScalarOrTensorFieldDataT, is_scalar>::type;

    using patch_geometry_type           = SAMRAI::hier::PatchGeometry;
    using cartesian_patch_geometry_type = SAMRAI::geom::CartesianPatchGeometry;

    using boundary_type = BoundaryManagerT::boundary_type;
    using boundary_condition_type
        = core::IFieldBoundaryCondition<scalar_or_tensor_field_type, gridlayout_type>;

    static constexpr std::size_t dimension = ScalarOrTensorFieldDataT::dimension;

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
     */
    void registerIDs(int const field_id) { data_id_ = field_id; }

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
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch& patch, double const fill_time,
                                       SAMRAI::hier::IntVector const& ghost_width_to_fill) override
    {
        gridlayout_type const& gridLayout = ScalarOrTensorFieldDataT::getLayout(patch, data_id_);

        // consistency check on the number of ghosts
        // SAMRAI::hier::IntVector dataGhostWidths = patchData->getGhostCellWidth();
        if (ghost_width_to_fill != gridLayout.nbrGhosts())
            throw std::runtime_error("Error - inconsistent ghost cell widths");

        // no check this is a valid cast
        std::shared_ptr<cartesian_patch_geometry_type> patchGeom
            = std::static_pointer_cast<cartesian_patch_geometry_type>(patch.getPatchGeometry());

        std::vector<SAMRAI::hier::BoundaryBox> const& boundaries
            = patchGeom->getCodimensionBoundaries(static_cast<int>(core::BoundaryCodim::One));

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

        // must be retrieved to pass as argument to patchGeom->getBoundaryFillBox later
        SAMRAI::hier::Box const& patch_box = patch.getBox();

        for (SAMRAI::hier::BoundaryBox const& bBox : boundaries)
        {
            // Boundary definitions in PHARE matches those of SAMRAI
            core::BoundaryLocation const bLoc
                = static_cast<core::BoundaryLocation>(bBox.getLocationIndex());

            SAMRAI::hier::Box samraiBoxToFill
                = patchGeom->getBoundaryFillBox(bBox, patch_box, ghost_width_to_fill);
            auto localBox = gridLayout.AMRToLocal(phare_box_from<dimension>(samraiBoxToFill));

            std::shared_ptr<boundary_type> boundary = boundaryManager_.getBoundary(bLoc);
            if (!boundary)
                throw std::runtime_error("boundary not found.");
            std::shared_ptr<boundary_condition_type> bc
                = boundary->getFieldCondition(scalarOrTensorField.physicalQuantity());
            if (!bc)
                throw std::runtime_error("boundary condition not found.");

            bc->apply(scalarOrTensorField, bLoc, localBox, gridLayout, fill_time);
        };
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


    static auto isNewFineFace(auto const& amrIdx, auto const dir) {}


protected:
    /// Reference to the resources manager.
    ResMan& rm_;
    /// Reference to the boundary manager.
    BoundaryManagerT& boundaryManager_;
    /// SAMRAI patch data identifier.
    int data_id_;
};

} // namespace PHARE::amr

#endif // PHARE_AMR_FIELD_REFINE_PATCH_STRATEGY_HPP
