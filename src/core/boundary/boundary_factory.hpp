#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_FACTORY
#define PHARE_CORE_BOUNDARY_BOUNDARY_FACTORY

#include "core/boundary/boundary.hpp"
#include "core/boundary/boundary_defs.hpp"
#include "core/data/field/field_traits.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_factory.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"

#include "initializer/data_provider.hpp"
#include "initializer/dict_utils.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace PHARE::core
{

/**
 * @brief Concept that detects whether a physical quantity type carries the conserved-variable set
 * required by super-magnetofast inflow boundary conditions (momentum vector @c rhoV and total
 * energy @c Etot). Satisfied by MHDQuantity, not by HybridQuantity.
 */
template<typename T>
concept HasInflowQuantities = requires {
    { T::Vector::rhoV };
    { T::Scalar::Etot };
};

/**
 * @brief Contains all the recipes to create a boundary object according to the desired
 * type of physical boundary (reflective, open, ...). It can extracts all the necessary data from
 * the input data dict associated to the boundary (value of physical quantities on the boundary for
 * an Inflow condition for instance), and create the right boundary conditions associated to each
 * physical quantity that requires one.
 *
 * @tparam PhysicalQuantityT The model category of physical quantities (MHDQuantity or
 * HybridQuantity).
 * @tparam FieldT The type for scalar fields.
 * @tparam GridLayoutT The type for the grid layout.
 */
template<typename PhysicalQuantityT, IsField FieldT, typename GridLayoutT>
class BoundaryFactory
{
public:
    using boundary_type             = Boundary<PhysicalQuantityT, FieldT, GridLayoutT>;
    using boundary_ptr_type         = std::unique_ptr<boundary_type>;
    using scalar_quantity_list_type = std::vector<typename PhysicalQuantityT::Scalar>;
    using vector_quantity_list_type = std::vector<typename PhysicalQuantityT::Vector>;

    static constexpr std::size_t dimension = GridLayoutT::dimension;

    BoundaryFactory() = delete;

    /**
     * @brief Create a boundary with the type indicated in the input dict, and register to it all
     * corresponding field boundary conditions.
     *
     * @param location The location of the boundary.
     * @param dict Input dictionnary related to the boundary.
     * @param scalars Scalar quantities for which it is necessary to register a field boundary
     *                condition.
     * @param vectors Vector quantities for which it is necessary to register a field boundary
     *                condition.
     * @param gamma Heat capacity ratio, used by the EOS-dependent boundary conditions
     *              (inflow and open) to convert pressure to total energy.
     *
     * @return A unique pointer to the created @c Boundary object.
     */
    static boundary_ptr_type create(BoundaryLocation location, initializer::PHAREDict dict,
                                    scalar_quantity_list_type const& scalars,
                                    vector_quantity_list_type const& vectors,
                                    double const gamma = 0.0)
    {
        std::string typeName = dict["type"].to<std::string>();
        BoundaryType type    = getBoundaryTypeFromString(typeName);
        _model_menu_type const quantities{scalars, vectors};
        initializer::PHAREDict const data
            = (dict.contains("data")) ? dict["data"] : initializer::PHAREDict{};

        // initialize the boundary
        boundary_ptr_type boundary = std::make_unique<boundary_type>(type, location);

        // register the right boundary condition per physical quantity following the boundary type
        switch (type)
        {
            case BoundaryType::None: register_none_conditions_(boundary, quantities); break;
            case BoundaryType::Reflective:
                register_reflective_conditions_(boundary, quantities);
                break;
            case BoundaryType::SuperMagnetofastInflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_inflow_conditions_(boundary, data, quantities, gamma);
                else
                    throw std::runtime_error(
                        "SuperMagnetofastInflow boundary type is not supported for this physical "
                        "model.");
                break;
            case BoundaryType::Open:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_open_conditions_(boundary, quantities, gamma);
                else
                    throw std::runtime_error(
                        "'" + typeName
                        + "' boundary type is not supported for this physical model.");
                break;
            default: throw std::runtime_error("Boundary type not implemented.");
        }
        return boundary;
    }

private:
    /** @brief Utility struct to group scalar and vector quantities together */
    struct _model_menu_type
    {
        scalar_quantity_list_type const& scalars;
        vector_quantity_list_type const& vectors;
    };

    /** @brief Build the shared B sub-BC used by compound conditions (e.g.
     * TotalEnergyFromPressure) to provide ghost values of the prescribed total field.
     * It is a plain Dirichlet on the total field, used only inside the energy computation;
     * the magnetic field's own ghost condition is a separate divergence-free transverse
     * Dirichlet (see register_inflow_conditions_). */
    template<typename VecFieldT, typename VectorBcType>
    static std::shared_ptr<VectorBcType> make_inflow_B_bc_(std::array<double, 3> const& B)
    {
        return std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, VecFieldT,
                                                  GridLayoutT>(B)};
    }

    /** @brief Register no-op (None) conditions so a "none" boundary leaves ghosts untouched
     * rather than falling through to another type or throwing "condition not found". */
    static void register_none_conditions_(boundary_ptr_type& boundary,
                                          _model_menu_type const& quantities)
    {
        for (auto const quantity : quantities.scalars)
            boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(quantity);
        for (auto const quantity : quantities.vectors)
            boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(quantity);
    }

    /** @brief Register boundary conditions to make a reflective boundary */
    static void register_reflective_conditions_(boundary_ptr_type& boundary,
                                                _model_menu_type const& quantities)
    {
        for (auto const quantity : quantities.scalars)
        {
            boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                quantity);
        }
        for (auto const quantity : quantities.vectors)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Vector::B):
                    // Fill outside-domain B ghosts with a divergence-free transverse Neumann
                    // extrapolation of the interior field (Faraday runs on the interior box only,
                    // so the ghost B must be provided by this condition rather than CT).
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>(quantity);
                    break;
                case (PhysicalQuantityT::Vector::J):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::AntiSymmetric>(quantity);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::AntiSymmetric>(quantity);
                    break;
                default:
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Symmetric>(
                            quantity);
                    break;
            }
        }
    }

    /** @brief Register boundary conditions to make an open boundary */
    static void register_open_conditions_(boundary_ptr_type& boundary,
                                          _model_menu_type const& quantities, double const gamma)
    {
        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, FieldT,
                                                  GridLayoutT>()};
        auto P_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, FieldT,
                                                  GridLayoutT>()};
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, VecFieldT,
                                                  GridLayoutT>()};
        auto B_bc = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
            FieldBoundaryConditionType::DivergenceFreeTransverseNeumann, VecFieldT, GridLayoutT>()};

        // The rho / rhoV / B sub-BCs built above for the energy reconstruction are stateless
        // extrapolations, so they double as the quantities' own main conditions: register them
        // by pointer rather than building a second identical object (matches the inflow builder).
        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary->registerFieldCondition(quantity, rho_bc);
                    break;
                case (PhysicalQuantityT::Scalar::Etot):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, gamma);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
            }
        }
        for (auto const quantity : quantities.vectors)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Vector::rhoV):
                    boundary->registerFieldCondition(quantity, rhoV_bc);
                    break;
                case (PhysicalQuantityT::Vector::B):
                    boundary->registerFieldCondition(quantity, B_bc);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }
    }

    /** @brief Register boundary conditions to make a super-magnetofast inflow boundary.
     *
     *  Every prescribable quantity is imposed: Dirichlet ρ, the momentum ρv = ρ·v, a Dirichlet
     *  pressure feeding the total-energy reconstruction, the prescribed-B energy sub-BC, and a
     *  divergence-free transverse Dirichlet B ghost condition (E is left None).
     *
     *  Only available for physical quantity types carrying conserved MHD variables. */
    static void register_inflow_conditions_(boundary_ptr_type& boundary,
                                            initializer::PHAREDict const& data,
                                            _model_menu_type const& quantities, double const gamma)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!(gamma > 1.0))
            throw std::runtime_error("BoundaryFactory: a heat capacity ratio > 1 is required for "
                                     "SuperMagnetofastInflow boundaries, got "
                                     + std::to_string(gamma) + ".");

        if (!data.contains("B"))
            throw std::runtime_error(
                "BoundaryFactory: SuperMagnetofastInflow requires the magnetic field 'B'.");

        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        // density read once; reused by rho_bc and by the momentum ρv = ρ·v
        double const rho = data["density"].template to<double>();

        // --- prescribed pressure, only used by the energy reconstruction ---
        auto P_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(
                data["pressure"].template to<double>())};

        // --- scalar Dirichlet sub/main BC: rho ---
        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(rho)};

        // --- momentum rhoV = rho * v ---
        auto const v = initializer::parseDimXYZType<double, 3>(data, "velocity");
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, VecFieldT,
                                                  GridLayoutT>(vToRhoV(rho, v))};

        // --- magnetic field B: two conditions built from the same prescribed inflow field ---
        //  * B_bc: plain-Dirichlet total field, used only as a sub-BC of the energy
        //    reconstruction (TotalEnergyFromPressure) to provide ghost B values.
        //  * B_main: the field's own ghost condition, a divergence-free transverse Dirichlet.
        //    It sets the transverse ghost components to the prescribed field while keeping the
        //    normal face divergence free. Being value-prescribed (not extrapolated from the
        //    interior) it also produces valid ghosts at regrid/init, when the fine interior is
        //    not yet available, so no separate regrid fallback condition is needed.
        auto const B = initializer::parseDimXYZType<double, 3>(data, "B");
        auto B_bc    = make_inflow_B_bc_<VecFieldT, VectorBcType>(B);
        auto B_main  = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
            FieldBoundaryConditionType::DivergenceFreeTransverseDirichlet, VecFieldT, GridLayoutT>(
            B)};

        // energy sub-BC needs rho/rhoV/B as ghost providers; the same rho_bc / rhoV_bc objects
        // also serve as the quantities' own main Dirichlet conditions (registered by pointer).
        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary->registerFieldCondition(quantity, rho_bc);
                    break;
                case (PhysicalQuantityT::Scalar::Etot):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, gamma);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }

        for (auto const quantity : quantities.vectors)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Vector::rhoV):
                    boundary->registerFieldCondition(quantity, rhoV_bc);
                    break;
                case (PhysicalQuantityT::Vector::B):
                    boundary->registerFieldCondition(quantity, B_main);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_BOUNDARY_BOUNDARY_FACTORY
