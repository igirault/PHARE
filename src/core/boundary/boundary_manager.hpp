#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_MANAGER
#define PHARE_CORE_BOUNDARY_BOUNDARY_MANAGER

#include "core/boundary/boundary.hpp"
#include "core/boundary/boundary_defs.hpp"
#include "core/boundary/boundary_factory.hpp"
#include "core/data/field/field_traits.hpp"
#include "core/data/grid/gridlayout_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_factory.hpp"

#include "initializer/data_provider.hpp"

#include <concepts>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace PHARE::core
{
/**
 * @brief Manage the lifecycle and retrieval of physical boundary conditions.
 *
 * Store and provide access to boundary condition objects for both
 * scalar and vector fields based on the boundary location and physical quantity.
 *
 * @tparam PhysicalQuantityT Type defining scalar and vector quantities (MHDQuantity or
 * HybridQuantity).
 * @tparam FieldT The scalar field type.
 * @tparam GridLayoutT The grid layout type.
 */
template<typename PhysicalQuantityT, IsField FieldT, IsGridLayout GridLayoutT>
class BoundaryManager
{
public:
    using boundary_type         = Boundary<PhysicalQuantityT, FieldT, GridLayoutT>;
    using boundary_factory_type = BoundaryFactory<PhysicalQuantityT, FieldT, GridLayoutT>;
    using scalar_quantity_type  = FieldT::physical_quantity_type;
    static_assert(std::same_as<scalar_quantity_type, typename PhysicalQuantityT::Scalar>);
    using vector_field_type     = VecField<FieldT, PhysicalQuantityT>;
    using scalar_condition_type = IFieldBoundaryCondition<FieldT, GridLayoutT>;
    using vector_condition_type = IFieldBoundaryCondition<vector_field_type, GridLayoutT>;

    BoundaryManager() = delete;

    /**
     * @brief Constructor. Register boundary conditions based on inputfile data.
     * @param dict Configuration dictionary.
     * @param scalar_quantities List of scalar quantities to manage.
     * @param vector_quantities List of vector quantities to manage.
     */
    BoundaryManager(PHARE::initializer::PHAREDict const& dict,
                    std::vector<typename PhysicalQuantityT::Scalar> const& scalarQuantities,
                    std::vector<typename PhysicalQuantityT::Vector> const& vectorQuantities)
    {
        dict.visit(cppdict::visit_all_nodes,
                   [&](std::string const& locationName, initializer::PHAREDict::data_t _) {
                       /// @todo I don't do anything with the second argument because it cannot be
                       /// transformed back into a dict. Maybe add the corresponding constructor to
                       /// cppdict, or add the possibility to have a lambda with the second arg
                       /// being a dict ?
                       BoundaryLocation location = getBoundaryLocationFromString(locationName);
                       boundaries_[location]     = boundary_factory_type::create(
                           location, dict[locationName], scalarQuantities, vectorQuantities);
                   });
    }


    /**
     * @brief Retrieve the boundary for a specific location.
     *
     * @param location The location of the desired boundary.
     * @return Shared pointer to the matching boundary, or nullptr if not found.
     *
     */
    std::shared_ptr<boundary_type> getBoundary(BoundaryLocation location) const
    {
        auto it = boundaries_.find(location);
        return (it != boundaries_.end()) ? it->second : nullptr;
    }


private:
    using _boundary_map_type = std::unordered_map<BoundaryLocation, std::shared_ptr<boundary_type>>;

    /**
     * @brief Utility struct to group scalar and vector quantities together
     *
     */
    struct SimulationMenu
    {
        std::vector<typename PhysicalQuantityT::Scalar> const& scalars;
        std::vector<typename PhysicalQuantityT::Vector> const& vectors;
    };

    /**
     * @brief List of boundaries mapped by their location
     *
     */
    _boundary_map_type boundaries_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_BOUNDARY_BOUNDARY_MANAGER
