#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_CONDITION_FACTORY_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_CONDITION_FACTORY_HPP

#include "core/data/field/field_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/inner_boundary/field_dirichlet_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_neumann_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_none_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_total_energy_from_pressure_inner_boundary_condition.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PHARE::core
{
enum class InnerBoundaryConditionType { FixedState, FixedDensity };

inline InnerBoundaryConditionType getInnerBoundaryConditionTypeFromString(std::string const& name)
{
    static std::unordered_map<std::string, InnerBoundaryConditionType> const typeMap{
        {"fixed-state", InnerBoundaryConditionType::FixedState},
        {"fixed-density", InnerBoundaryConditionType::FixedDensity},
    };
    auto it = typeMap.find(name);
    if (it == typeMap.end())
        throw std::runtime_error("Unknown inner boundary condition type: " + name);
    return it->second;
}

struct FixedStateValues
{
    double density;
    double pressure;
};

template<typename PhysicalQuantityT, IsField FieldT, typename GridLayoutT, typename PhysicalStateT>
class InnerBoundaryConditionFactory
{
public:
    using vecfield_type  = VecField<FieldT, PhysicalQuantityT>;
    using scalar_qty     = PhysicalQuantityT::Scalar;
    using vector_qty     = PhysicalQuantityT::Vector;
    using scalar_bc_type = FieldInnerBoundaryCondition<FieldT, GridLayoutT, PhysicalStateT>;
    using vector_bc_type = FieldInnerBoundaryCondition<vecfield_type, GridLayoutT, PhysicalStateT>;
    using scalar_bc_map_type = std::unordered_map<scalar_qty, std::unique_ptr<scalar_bc_type>>;
    using vector_bc_map_type = std::unordered_map<vector_qty, std::unique_ptr<vector_bc_type>>;

    InnerBoundaryConditionFactory() = delete;

    static void create(InnerBoundaryConditionType type,
                       std::vector<scalar_qty> const& scalarQuantities,
                       std::vector<vector_qty> const& vectorQuantities,
                       scalar_bc_map_type& scalarBCs, vector_bc_map_type& vectorBCs, double gamma,
                       FixedStateValues const& fixedState)
    {
        switch (type)
        {
            case InnerBoundaryConditionType::FixedState:
                registerFixedState_(scalarQuantities, vectorQuantities, scalarBCs, vectorBCs, gamma,
                                    fixedState);
                break;
            case InnerBoundaryConditionType::FixedDensity:
                registerFixedDensity_(scalarQuantities, vectorQuantities, scalarBCs, vectorBCs,
                                      gamma, fixedState);
                break;
            default: throw std::runtime_error("InnerBoundaryConditionFactory: unknown type");
        }
    }

private:
    template<typename ScalarOrTensorFieldT>
    using None = FieldNoneInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    template<typename ScalarOrTensorFieldT>
    using Neumann
        = FieldNeumannInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    template<typename ScalarOrTensorFieldT>
    using Dirichlet
        = FieldDirichletInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    template<typename ScalarOrTensorFieldT>
    using TotalEnergyFromPressure
        = FieldTotalEnergyFromPressureInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT,
                                                             PhysicalStateT>;

    static constexpr auto constant = Dirichlet<FieldT>::ExtrapolationType::Constant;

    static void registerFixedDensity_(std::vector<scalar_qty> const& scalars,
                                      std::vector<vector_qty> const& vectors,
                                      scalar_bc_map_type& scalarBCs, vector_bc_map_type& vectorBCs,
                                      double gamma, FixedStateValues const& fixedState)
    {
        registerFixedState_(scalars, vectors, scalarBCs, vectorBCs, gamma, fixedState);
        for (auto const qty : scalars)
            if (qty == PhysicalQuantityT::Scalar::Etot1)
                scalarBCs[qty] = std::make_unique<TotalEnergyFromPressure<FieldT>>(
                    std::make_unique<Neumann<FieldT>>(), gamma);
    }

    static void registerFixedState_(std::vector<scalar_qty> const& scalars,
                                    std::vector<vector_qty> const& vectors,
                                    scalar_bc_map_type& scalarBCs, vector_bc_map_type& vectorBCs,
                                    double gamma, FixedStateValues const& fixedState)
    {
        for (auto const qty : scalars)
        {
            if (qty == PhysicalQuantityT::Scalar::Etot1)
                scalarBCs[qty] = std::make_unique<TotalEnergyFromPressure<FieldT>>(
                    std::make_unique<Dirichlet<FieldT>>(fixedState.pressure, constant), gamma);
            else if (qty == PhysicalQuantityT::Scalar::rho)
                scalarBCs[qty] = std::make_unique<Dirichlet<FieldT>>(fixedState.density, constant);
            else
                scalarBCs[qty] = std::make_unique<Neumann<FieldT>>();
        }

        for (auto const qty : vectors)
        {
            if (qty == PhysicalQuantityT::Vector::B1)
                vectorBCs[qty] = std::make_unique<None<vecfield_type>>();
            else if (qty == PhysicalQuantityT::Vector::rhoV || qty == PhysicalQuantityT::Vector::E)
                vectorBCs[qty] = std::make_unique<Dirichlet<vecfield_type>>(
                    0.0, Dirichlet<vecfield_type>::ExtrapolationType::Constant);
            else
                vectorBCs[qty] = std::make_unique<Neumann<vecfield_type>>();
        }
    }
};

} // namespace PHARE::core

#endif
