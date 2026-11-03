#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MANAGER_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MANAGER_HPP

#include "core/data/field/field_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_condition_factory.hpp"
#include "core/inner_boundary/inner_boundary_factory.hpp"
#include "core/inner_boundary/inner_boundary_geometry.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"

#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"

#include "initializer/data_provider.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace PHARE::core
{

template<typename PhysicalQuantityT, IsField FieldT, typename GridLayoutT, typename PhysicalStateT>
class InnerBoundaryManager
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using vecfield_type   = VecField<FieldT, PhysicalQuantityT>;
    using scalar_qty      = PhysicalQuantityT::Scalar;
    using vector_qty      = PhysicalQuantityT::Vector;
    using geometry_type   = InnerBoundaryGeometry<dimension>;
    using mesh_data_type  = InnerBoundaryMeshData<dimension, PhysicalQuantityT>;
    using classifier_type = InnerBoundaryMeshClassifier<dimension, GridLayoutT, PhysicalQuantityT>;
    using context_type    = InnerBCContext<PhysicalStateT>;
    using scalar_bc_type  = FieldInnerBoundaryCondition<FieldT, GridLayoutT, PhysicalStateT>;
    using vector_bc_type  = FieldInnerBoundaryCondition<vecfield_type, GridLayoutT, PhysicalStateT>;
    using scalar_bc_map_type = std::unordered_map<scalar_qty, std::unique_ptr<scalar_bc_type>>;
    using vector_bc_map_type = std::unordered_map<vector_qty, std::unique_ptr<vector_bc_type>>;
    using factory_type
        = InnerBoundaryConditionFactory<PhysicalQuantityT, FieldT, GridLayoutT, PhysicalStateT>;

    InnerBoundaryManager() = delete;

    struct SafeStateValues
    {
        double density{1.};
        double pressure{1.};
        std::array<double, 3> velocity{0., 0., 0.};
        std::array<double, 3> B1{0., 0., 0.};
    };

    InnerBoundaryManager(std::unique_ptr<geometry_type> geometry,
                         InnerBoundaryConditionType conditionType,
                         std::vector<scalar_qty> const& scalarQuantities,
                         std::vector<vector_qty> const& vectorQuantities, double gamma,
                         FixedStateValues const& fixedState, SafeStateValues const& safe = {})
        : geometry_{std::move(geometry)}
        , meshData_{geometry_->name()}
        , gamma_{gamma}
    {
        factory_type::create(conditionType, scalarQuantities, vectorQuantities, scalarBCs_,
                             vectorBCs_, gamma_, fixedState);
        buildSafeState_(safe);
    }

    static std::unique_ptr<InnerBoundaryManager>
    create(initializer::PHAREDict const& dict, std::vector<scalar_qty> const& scalarQuantities,
           std::vector<vector_qty> const& vectorQuantities, double gamma)
    {
        if (!dict.contains("inner_boundary"))
            return nullptr;

        auto geometry = InnerBoundaryFactory<dimension>::create(dict);
        if (!geometry)
            return nullptr;

        auto const& ibDict  = dict["inner_boundary"];
        auto const typeName = ibDict["condition_type"].template to<std::string>();
        auto const condType = getInnerBoundaryConditionTypeFromString(typeName);

        if (!ibDict.contains("density") || !ibDict.contains("pressure"))
            throw std::runtime_error("inner_boundary: fixed-state requires density and pressure");
        FixedStateValues const fixedState{ibDict["density"].template to<double>(),
                                          ibDict["pressure"].template to<double>()};

        SafeStateValues safe{};
        if (ibDict.contains("inactive_safe_state"))
        {
            auto const& ss = ibDict["inactive_safe_state"];
            safe.density   = cppdict::get_value(ss, "density", 1.0);
            safe.pressure  = cppdict::get_value(ss, "pressure", 1.0);
            safe.velocity  = {cppdict::get_value(ss, "velocity/x", 0.0),
                              cppdict::get_value(ss, "velocity/y", 0.0),
                              cppdict::get_value(ss, "velocity/z", 0.0)};
            safe.B1 = {cppdict::get_value(ss, "B1/x", 0.0), cppdict::get_value(ss, "B1/y", 0.0),
                       cppdict::get_value(ss, "B1/z", 0.0)};
        }

        return std::make_unique<InnerBoundaryManager>(std::move(geometry), condType,
                                                      scalarQuantities, vectorQuantities, gamma,
                                                      fixedState, safe);
    }

    void classify(GridLayoutT const& layout)
    {
        auto classifier = classifier_type::withDefaults(*geometry_, layout);
        classifier(layout, meshData_);
    }

    void applyBC(FieldT& field, GridLayoutT const& layout, context_type const& ctx)
    {
        auto it = scalarBCs_.find(field.physicalQuantity());
        if (it == scalarBCs_.end())
            return;
        it->second->apply(field, layout, meshData_, ctx);
    }

    void applyBC(vecfield_type& vecfield, GridLayoutT const& layout, context_type const& ctx)
    {
        auto it = vectorBCs_.find(vecfield.physicalQuantity());
        if (it == vectorBCs_.end())
            return;
        it->second->apply(vecfield, layout, meshData_, ctx);
    }

    void applyToMoments(GridLayoutT const& layout, context_type const& ctx)
    {
        auto& state = ctx.statenew;
        std::array<std::pair<int, std::function<void()>>, 3> tasks{{
            scalarTask_(state.rho, layout, ctx),
            vectorTask_(state.rhoV, layout, ctx),
            scalarTask_(state.Etot1, layout, ctx),
        }};

        std::stable_sort(tasks.begin(), tasks.end(),
                         [](auto const& a, auto const& b) { return a.first < b.first; });

        for (auto const& [prio, run] : tasks)
            if (run)
                run();
    }

    void setSafeState(FieldT& field, GridLayoutT const& layout,
                      ElemStatus status = ElemStatus::Inactive)
    {
        auto it = safeScalars_.find(field.physicalQuantity());
        if (it == safeScalars_.end())
            throw std::runtime_error(
                "InnerBoundaryManager::setSafeState: no safe value for this scalar quantity");

        double const value   = it->second;
        auto const centering = layout.centering(field.physicalQuantity());
        auto& statusField    = meshData_.getStatusFieldFromCentering(centering);
        layout.evalOnGhostBox(field, [&](auto&... args) {
            auto const idx = MeshIndex<dimension>{args...};
            if (statusField(idx) == toDouble(status))
                field(idx) = value;
        });
    }

    void setSafeState(vecfield_type& vecfield, GridLayoutT const& layout,
                      ElemStatus status = ElemStatus::Inactive)
    {
        auto it = safeVectors_.find(vecfield.physicalQuantity());
        if (it == safeVectors_.end())
            throw std::runtime_error(
                "InnerBoundaryManager::setSafeState: no safe value for this vector quantity");

        auto const& values = it->second;
        auto comps         = vecfield.components();
        for_N<3>([&](auto ci) {
            constexpr auto c     = ci();
            auto& comp           = std::get<c>(comps);
            auto const centering = layout.centering(comp.physicalQuantity());
            auto& statusField    = meshData_.getStatusFieldFromCentering(centering);
            layout.evalOnGhostBox(comp, [&](auto&... args) {
                auto const idx = MeshIndex<dimension>{args...};
                if (statusField(idx) == toDouble(status))
                    comp(idx) = values[c];
            });
        });
    }

    NO_DISCARD mesh_data_type& getMeshData() { return meshData_; }

    NO_DISCARD mesh_data_type const& getMeshData() const { return meshData_; }

    NO_DISCARD geometry_type const& getGeometry() const { return *geometry_; }

    NO_DISCARD bool isUsable() const { return meshData_.isUsable(); }

    NO_DISCARD bool isSettable() const { return meshData_.isSettable(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(meshData_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(meshData_); }

private:
    std::pair<int, std::function<void()>> scalarTask_(FieldT& field, GridLayoutT const& layout,
                                                      context_type const& ctx)
    {
        auto it = scalarBCs_.find(field.physicalQuantity());
        if (it == scalarBCs_.end())
            return {0, {}};
        auto* bc = it->second.get();
        return {bc->priority(),
                [this, bc, &field, &layout, &ctx]() { bc->apply(field, layout, meshData_, ctx); }};
    }

    std::pair<int, std::function<void()>>
    vectorTask_(vecfield_type& vecfield, GridLayoutT const& layout, context_type const& ctx)
    {
        auto it = vectorBCs_.find(vecfield.physicalQuantity());
        if (it == vectorBCs_.end())
            return {0, {}};
        auto* bc = it->second.get();
        return {bc->priority(), [this, bc, &vecfield, &layout, &ctx]() {
                    bc->apply(vecfield, layout, meshData_, ctx);
                }};
    }

    void buildSafeState_(SafeStateValues const& s)
    {
        using Scalar = scalar_qty;
        using Vector = vector_qty;

        safeScalars_[Scalar::rho] = s.density;
        safeScalars_[Scalar::P]   = s.pressure;

        safeScalars_[Scalar::Etot1]
            = eosPToEtot1(gamma_, s.density, s.velocity[0], s.velocity[1], s.velocity[2], s.B1[0],
                          s.B1[1], s.B1[2], s.pressure);

        safeVectors_[Vector::V] = s.velocity;
        safeVectors_[Vector::rhoV]
            = {s.density * s.velocity[0], s.density * s.velocity[1], s.density * s.velocity[2]};
        safeVectors_[Vector::B1] = s.B1;
    }

    std::unique_ptr<geometry_type> geometry_;
    mesh_data_type meshData_;
    double gamma_;
    scalar_bc_map_type scalarBCs_;
    vector_bc_map_type vectorBCs_;
    std::unordered_map<scalar_qty, double> safeScalars_;
    std::unordered_map<vector_qty, std::array<double, 3>> safeVectors_;
};

} // namespace PHARE::core

#endif
