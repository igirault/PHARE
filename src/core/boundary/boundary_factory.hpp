#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_FACTORY
#define PHARE_CORE_BOUNDARY_BOUNDARY_FACTORY

#include "core/boundary/boundary.hpp"
#include "core/boundary/boundary_defs.hpp"
#include "core/data/field/field_traits.hpp"
#include "core/data/grid/gridlayout_traits.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_factory.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
#include "core/numerics/thermo/thermo.hpp"

#include "initializer/data_provider.hpp"
#include "initializer/dict_utils.hpp"

#include <memory>
#include <stdexcept>
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
    { T::Scalar::Etot1 };
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
template<typename PhysicalQuantityT, IsField FieldT, IsGridLayout GridLayoutT>
class BoundaryFactory
{
public:
    using boundary_type             = Boundary<PhysicalQuantityT, FieldT, GridLayoutT>;
    using boundary_ptr_type         = std::unique_ptr<boundary_type>;
    using scalar_quantity_list_type = std::vector<typename PhysicalQuantityT::Scalar>;
    using vector_quantity_list_type = std::vector<typename PhysicalQuantityT::Vector>;

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
     * @param thermo Optional thermodynamic model used by EOS-dependent boundary conditions
     *               (e.g. inflow). May be nullptr for boundary types that do not require an EOS.
     *
     * @return A unique pointer to the created @c Boundary object.
     */
    static boundary_ptr_type create(BoundaryLocation location, initializer::PHAREDict dict,
                                    scalar_quantity_list_type const& scalars,
                                    vector_quantity_list_type const& vectors,
                                    std::shared_ptr<Thermo> thermo = nullptr)
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
            case BoundaryType::None:
                // do nothing
            case BoundaryType::Reflective:
                register_reflective_conditions_(boundary, data, quantities);
                break;
            case BoundaryType::SuperMagnetofastInflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_super_magnetofast_inflow_conditions_(boundary, data, quantities,
                                                                  thermo);
                else
                    throw std::runtime_error(
                        "SuperMagnetofastInflow boundary type is not supported for this physical "
                        "model.");
                break;
            case BoundaryType::SuperMagnetofastOutflow:
            case BoundaryType::Open:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_open_conditions_(boundary, data, quantities, thermo);
                else
                    throw std::runtime_error(
                        "FreePressureInflow boundary type is not supported for this physical "
                        "model.");
                break;
                break;
            case BoundaryType::FreePressureInflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_free_pressure_inflow_conditions_(boundary, data, quantities, thermo);
                else
                    throw std::runtime_error(
                        "FreePressureInflow boundary type is not supported for this physical "
                        "model.");
                break;
            case BoundaryType::FixedPressureOutflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_fixed_pressure_outflow_conditions_(boundary, data, quantities, thermo);
                else
                    throw std::runtime_error(
                        "FixedPressureOutflow boundary type is not supported for this physical "
                        "model.");
                break;
            case BoundaryType::AdaptiveOutflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_adaptive_outflow_conditions_(boundary, data, quantities, thermo);
                else
                    throw std::runtime_error(
                        "AdaptiveOutflow boundary type is not supported for this physical "
                        "model.");
                break;
            case BoundaryType::NonReflectingHydroSubsonicOutflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_non_reflecting_hydro_subsonic_outflow_conditions_(boundary, data,
                                                                               quantities, thermo);
                else
                    throw std::runtime_error(
                        "NonReflectingHydroSubsonicOutflow boundary type is not supported for "
                        "this physical model.");
                break;
            case BoundaryType::NonReflectingHydroSubsonicInflow:
                if constexpr (HasInflowQuantities<PhysicalQuantityT>)
                    register_non_reflecting_hydro_subsonic_inflow_conditions_(boundary, data,
                                                                              quantities, thermo);
                else
                    throw std::runtime_error(
                        "NonReflectingHydroSubsonicInflow boundary type is not supported for "
                        "this physical model.");
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

    /** @brief Ideal motional electric field E = -v x B prescribed at an inflow boundary.
     *  With v and the total field B both uniform, E is uniform too, so its tangential
     *  part has zero edge-EMF variation and the constrained transport keeps the boundary
     *  normal magnetic field frozen (field lines stay tangent to the inlet). */
    static std::array<double, 3> inflow_motional_E_(std::array<double, 3> const& v,
                                                    std::array<double, 3> const& B)
    {
        return {-(v[1] * B[2] - v[2] * B[1]), -(v[2] * B[0] - v[0] * B[2]),
                -(v[0] * B[1] - v[1] * B[0])};
    }

    /** @brief Build the shared B1 sub-BC used by compound conditions (e.g.
     * TotalEnergyFromPressure) to fill B1 = B - B0 from the prescribed total field
     * (@c FieldB1FromBtotBoundaryCondition). */
    template<typename VecFieldT, typename VectorBcType>
    static std::shared_ptr<VectorBcType> make_inflow_B_bc_(std::array<double, 3> const& B)
    {
        return std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::B1FromBtot,
                                                  VecFieldT, GridLayoutT>(B)};
    }

    /** @brief Register boundary conditions to make a reflective boundary */
    static void register_reflective_conditions_(boundary_ptr_type& boundary,
                                                PHARE::initializer::PHAREDict const& data,
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
                case (PhysicalQuantityT::Vector::B1):
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
                                          initializer::PHAREDict const& data,
                                          _model_menu_type const& quantities,
                                          std::shared_ptr<Thermo> thermo)
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
        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, thermo);
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
                case (PhysicalQuantityT::Vector::B1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>(quantity);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
            }
        }
    }

    /** @brief Register boundary conditions to make a super-magnetofast inflow boundary.
     *
     *  The total magnetic field is prescribed via @c data["B"]. The boundary field is driven
     *  through the constrained transport from the motional electric field E = -v x B, imposed
     *  as a full Dirichlet on E; B1 is left free (None) in the ghosts. With v and B uniform, E
     *  is uniform, so its tangential part has zero edge-EMF variation and the boundary normal
     *  field stays frozen (field lines tangent to the inlet). Etot1 is derived from the
     *  prescribed pressure through @c FieldTotalEnergyFromPressureBoundaryCondition, which
     *  fills B1 = B - B0 internally for its magnetic term.
     *
     *  Prescribing the perturbation @c data["B1"] is not supported yet (it needs an
     *  E = -v x (B0 + B1) condition); passing it throws.
     *
     *  Only available for physical quantity types carrying conserved MHD variables. */
    static void register_super_magnetofast_inflow_conditions_(boundary_ptr_type& boundary,
                                                              initializer::PHAREDict const& data,
                                                              _model_menu_type const& quantities,
                                                              std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for SuperMagnetofastInflow "
                "boundaries but none was provided.");

        double const p   = data["pressure"].to<double>();
        double const rho = data["density"].to<double>();
        auto const v     = initializer::parseDimXYZType<double, 3>(data, "velocity");
        auto const rhoV  = vToRhoV(rho, v);

        if (!data.contains("B"))
            throw std::runtime_error(
                "BoundaryFactory: SuperMagnetofastInflow requires the total magnetic field 'B'; "
                "the 'B1' perturbation inflow is not supported yet (it needs an E = -v x (B0 + "
                "B1) electric-field condition that is not implemented).");
        auto const B = initializer::parseDimXYZType<double, 3>(data, "B");
        auto const E = inflow_motional_E_(v, B);

        // The boundary magnetic field is driven through the constrained transport from the
        // prescribed motional electric field E = -v x B (full Dirichlet on E), not by pinning
        // B1 in the ghosts (B1 = None). Etot1 is derived from the prescribed pressure via the
        // compound energy BC, which fills B1 = B - B0 internally for its magnetic term.
        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(rho)};
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, VecFieldT,
                                                  GridLayoutT>(rhoV)};
        auto B_bc = make_inflow_B_bc_<VecFieldT, VectorBcType>(B);
        auto P_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(p)};

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, rho);
                    break;
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, thermo);
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
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, rhoV);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, E);
                    break;
                case (PhysicalQuantityT::Vector::B1):
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

    /**
     * @brief Register boundary conditions for a free-pressure inflow boundary.
     *
     * Like @c SuperMagnetofastInflow for ρ, ρv, and B (Dirichlet ρ/ρv, total field 'B'
     * driven through E = -v x B), but with a Neumann condition on pressure instead of a
     * prescribed value. The energy ghost values are derived from the Neumann pressure via
     * the EOS by @c FieldTotalEnergyFromPressureBoundaryCondition.
     *
     * Only available for physical quantity types carrying conserved MHD variables.
     */
    static void register_free_pressure_inflow_conditions_(boundary_ptr_type& boundary,
                                                          initializer::PHAREDict const& data,
                                                          _model_menu_type const& quantities,
                                                          std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for FreePressureInflow "
                "boundaries but none was provided.");

        double const rho = data["density"].to<double>();
        auto const v     = initializer::parseDimXYZType<double, 3>(data, "velocity");
        auto const rhoV  = vToRhoV(rho, v);

        if (!data.contains("B"))
            throw std::runtime_error(
                "BoundaryFactory: FreePressureInflow requires the total magnetic field 'B'; the "
                "'B1' perturbation inflow is not supported yet (it needs an E = -v x (B0 + B1) "
                "electric-field condition that is not implemented).");
        auto const B = initializer::parseDimXYZType<double, 3>(data, "B");
        auto const E = inflow_motional_E_(v, B);

        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        // Build sub-BCs shared by the energy compound BC
        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(rho)};
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, VecFieldT,
                                                  GridLayoutT>(rhoV)};
        auto B_bc = make_inflow_B_bc_<VecFieldT, VectorBcType>(B);
        auto P_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, FieldT,
                                                  GridLayoutT>()};

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, rho);
                    break;
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, thermo);
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
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, rhoV);
                    break;
                case (PhysicalQuantityT::Vector::E):
                    boundary
                        ->template registerFieldCondition<FieldBoundaryConditionType::Dirichlet>(
                            quantity, E);
                    break;
                case (PhysicalQuantityT::Vector::B1):
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

    /**
     * @brief Register boundary conditions for a fixed-pressure outflow boundary.
     *
     * ρ, ρv, and B use Neumann (zero-gradient) conditions. The pressure uses a Dirichlet
     * condition and the total energy ghost values are derived from the resulting ghost
     * pressure via @c FieldTotalEnergyFromPressureBoundaryCondition.
     *
     * Only available for physical quantity types carrying conserved MHD variables.
     */
    static void register_fixed_pressure_outflow_conditions_(boundary_ptr_type& boundary,
                                                            initializer::PHAREDict const& data,
                                                            _model_menu_type const& quantities,
                                                            std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for FixedPressureOutflow "
                "boundaries but none was provided.");

        double const pressure = data["pressure"].to<double>();

        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, FieldT,
                                                  GridLayoutT>()};
        auto P_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Dirichlet, FieldT,
                                                  GridLayoutT>(pressure)};
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, VecFieldT,
                                                  GridLayoutT>()};
        auto B_bc = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
            FieldBoundaryConditionType::DivergenceFreeTransverseNeumann, VecFieldT, GridLayoutT>()};

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, thermo);
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
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
                case (PhysicalQuantityT::Vector::B1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>(quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }
    }


    /**
     * @brief Register boundary conditions for an adaptive outflow boundary.
     *
     * ρ, ρv, and B use Neumann (zero-gradient) conditions, exactly like a fixed-pressure
     * outflow. The pressure uses @c FieldAdaptiveOutflowPressureBoundaryCondition, which
     * decides per tangential ghost-column between zero-gradient (super-magnetofast) and
     * Dirichlet at the target pressure (sub-fast); the total energy ghost values are then
     * derived from that ghost pressure via @c
     * FieldTotalEnergyFromPressureBoundaryCondition.
     *
     * Only available for physical quantity types carrying conserved MHD variables.
     */
    static void register_adaptive_outflow_conditions_(boundary_ptr_type& boundary,
                                                      initializer::PHAREDict const& data,
                                                      _model_menu_type const& quantities,
                                                      std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for AdaptiveOutflow "
                "boundaries but none was provided.");

        double const pressure = data["pressure"].to<double>();

        using VecFieldT    = VecField<FieldT, PhysicalQuantityT>;
        using ScalarBcType = IFieldBoundaryCondition<FieldT, GridLayoutT>;
        using VectorBcType = IFieldBoundaryCondition<VecFieldT, GridLayoutT>;

        auto rho_bc = std::shared_ptr<ScalarBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, FieldT,
                                                  GridLayoutT>()};
        auto P_bc    = std::shared_ptr<ScalarBcType>{FieldBoundaryConditionFactory::create<
               FieldBoundaryConditionType::AdaptiveOutflowPressure, FieldT, GridLayoutT>(pressure,
                                                                                         thermo)};
        auto rhoV_bc = std::shared_ptr<VectorBcType>{
            FieldBoundaryConditionFactory::create<FieldBoundaryConditionType::Neumann, VecFieldT,
                                                  GridLayoutT>()};
        auto B_bc = std::shared_ptr<VectorBcType>{FieldBoundaryConditionFactory::create<
            FieldBoundaryConditionType::DivergenceFreeTransverseNeumann, VecFieldT, GridLayoutT>()};

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::TotalEnergyFromPressure>(
                        quantity, rho_bc, rhoV_bc, B_bc, P_bc, thermo);
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
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
                case (PhysicalQuantityT::Vector::B1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>(quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }
    }


    /** @brief Register boundary conditions for a HD characteristic subsonic outflow with
     * target exit pressure (LODI / Poinsot-Lele).
     *
     *  A single BC type @c NonReflectingHydroSubsonicOutflow is registered for ρ, ρv, and
     *  Etot1; each invocation writes the ghost values of the field it was called for using
     *  the same LODI relations. B uses DivFreeNeumann (HD: B = 0). The pressure field, when
     *  present in the quantity list, is given a Neumann placeholder — it is read from the
     *  previous-substage state via the BC context, not from the current-state ghosts.
     */
    static void register_non_reflecting_hydro_subsonic_outflow_conditions_(
        boundary_ptr_type& boundary, initializer::PHAREDict const& data,
        _model_menu_type const& quantities, std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for "
                "NonReflectingHydroSubsonicOutflow boundaries but none was provided.");

        double const pressure     = data["pressure"].to<double>();
        double const sigma        = data["sigma"].to<double>();
        double const length_scale = data["length_scale"].to<double>();

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::NonReflectingHydroSubsonicOutflow>(
                        quantity, pressure, sigma, length_scale, thermo);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
            }
        }

        for (auto const quantity : quantities.vectors)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Vector::rhoV):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::NonReflectingHydroSubsonicOutflow>(
                        quantity, pressure, sigma, length_scale, thermo);
                    break;
                case (PhysicalQuantityT::Vector::B1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::DivergenceFreeTransverseNeumann>(quantity);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::None>(
                        quantity);
                    break;
            }
        }
    }


    /** @brief Register boundary conditions for a HD characteristic non-reflecting subsonic
     *  inflow (LODI / Poinsot-Lele soft inflow).
     *
     *  A single BC type @c NonReflectingHydroSubsonicInflow is registered for ρ, ρv, and
     *  Etot1; each invocation writes the ghost values of the field it was called for using
     *  the same LODI relations. This BC is hydro-only: it carries no magnetic eigenmodes and
     *  no magnetic field — B1 and E are left free (None) and it must not be used with a
     *  non-zero magnetic field. Pressure is left free — its evolution is set entirely by the
     *  outgoing acoustic wave coming from the interior.
     */
    static void register_non_reflecting_hydro_subsonic_inflow_conditions_(
        boundary_ptr_type& boundary, initializer::PHAREDict const& data,
        _model_menu_type const& quantities, std::shared_ptr<Thermo> thermo)
        requires HasInflowQuantities<PhysicalQuantityT>
    {
        if (!thermo)
            throw std::runtime_error(
                "BoundaryFactory: a Thermo object is required for "
                "NonReflectingHydroSubsonicInflow boundaries but none was provided.");

        double const rho_target = data["density"].to<double>();
        auto const v_target     = initializer::parseDimXYZType<double, 3>(data, "velocity");
        double const relax_velocity_n = data["relax_velocity_n"].to<double>();
        double const relax_velocity_t = data["relax_velocity_t"].to<double>();
        double const relax_density    = data["relax_density"].to<double>();

        for (auto const quantity : quantities.scalars)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Scalar::rho):
                case (PhysicalQuantityT::Scalar::Etot1):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::NonReflectingHydroSubsonicInflow>(
                        quantity, rho_target, v_target, relax_velocity_n, relax_velocity_t,
                        relax_density, thermo);
                    break;
                default:
                    boundary->template registerFieldCondition<FieldBoundaryConditionType::Neumann>(
                        quantity);
                    break;
            }
        }

        for (auto const quantity : quantities.vectors)
        {
            switch (quantity)
            {
                case (PhysicalQuantityT::Vector::rhoV):
                    boundary->template registerFieldCondition<
                        FieldBoundaryConditionType::NonReflectingHydroSubsonicInflow>(
                        quantity, rho_target, v_target, relax_velocity_n, relax_velocity_t,
                        relax_density, thermo);
                    break;
                case (PhysicalQuantityT::Vector::B1):
                    // Hydro-only LODI: no magnetic eigenmodes and no magnetic field carried.
                    // B1 is left free (None); E is unconstrained too. This BC must not be used
                    // with a non-zero magnetic field.
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
