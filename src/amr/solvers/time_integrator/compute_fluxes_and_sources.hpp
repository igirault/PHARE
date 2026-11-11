#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_AND_SOURCES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_AND_SOURCES_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "initializer/data_provider.hpp"

#include <cassert>
#include <stdexcept>

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel>
class ComputeFluxesAndSources
{
    using level_t = MHDModel::level_t;
    // using Layout        = MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using Ampere_t = Dispatchers_t::Ampere_t;

    using FVMethod_t     = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = FVMethod_t::info_type;

    constexpr static auto Hall = FVMethod_t::Hall;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t     = Dispatchers_t::template ConstrainedTransport_t<Rec>;
    using ConstrainedTransportInfo_t = ConstrainedTransport_t::info_type;

    using PoyntingCorrection_t = Dispatchers_t::template PoyntingCorrection_t<Rec>;

    using NonIdealElectricField_t = Dispatchers_t::template NonIdealElectricField_t<Hall>;

    using ToPrimitiveConverter_t    = Dispatchers_t::ToPrimitiveConverter_t;
    using ToConservativeConverter_t = Dispatchers_t::ToConservativeConverter_t;
    using ExternalFieldSource_t     = Dispatchers_t::ExternalFieldSource_t;

    using VecField    = MHDModel::vecfield_type;
    using Equations_t = FVMethod_t::Equations_t;


public:
    ComputeFluxesAndSources(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , to_primitive_gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
        , to_conservative_gamma_{dict["to_conservative"]["heat_capacity_ratio"]}
        , needsCurrent_{Hall || fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()}
    {
        // eta/nu are duplicated into both info structs
        assert(fVMethodInfo_.eta == constrainedTransportInfo_.eta
               && fVMethodInfo_.nu == constrainedTransportInfo_.nu);

        // hyper-resistivity is only meaningful alongside the Hall term
        if constexpr (!Hall)
            if (fVMethodInfo_.isHyperResistive())
                throw std::runtime_error(
                    "Error - hyper-resistivity (nu > 0) requires the Hall term to be enabled");
    }

    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& sources, auto& bc,
                    level_t& level, double const newTime)
    {
        ToPrimitiveConverter_t{level, model}(state, to_primitive_gamma_, model.positivityFloors);
        TimeSetter{level, model, newTime}(state.rho, state.rhoV, state.Etot1, state.V, state.P);

        if (needsCurrent_)
        {
            Ampere_t{level, model}(state.B1, state.J);
            TimeSetter{level, model, newTime}(state.B1, state.J);
        }

        if (isNonIdeal_)
            NonIdealElectricField_t{level, model, fVMethodInfo_}(non_ideal_electric_state_, state);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, state, fluxes);
        TimeSetter{level, model, newTime}(state.rho, state.V, state.P, state.J);

        // unecessary if we decide to store both primitive and conservative variables
        ToConservativeConverter_t{level, model}(state, to_conservative_gamma_);
        TimeSetter{level, model, newTime}(state.rho, state.V, state.P, state.rhoV, state.Etot1);

        ConstrainedTransport_t{level, model,
                               constrainedTransportInfo_}(ct_, non_ideal_electric_state_, state);

        if (model.hasInnerBoundary())
            applyInnerBoundaryToElectricField_(model, state, level, newTime);

        bc.fillElectricGhosts(state.E, level, newTime);
        ExternalFieldSource_t{level, model}(state, sources);

        PoyntingCorrection_t{level, model}(ct_, state, fluxes);
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(ct_);
        model.resourcesManager->registerResources(non_ideal_electric_state_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(ct_, patch, allocateTime);
        model.resourcesManager->allocate(non_ideal_electric_state_, patch, allocateTime);
    }

private:
    void applyInnerBoundaryToElectricField_(MHDModel& model, auto& state, level_t& level,
                                            double const newTime)
    {
        using state_type = std::decay_t<decltype(state)>;
        using GridLayout = MHDModel::gridlayout_type;
        auto& ibm        = *model.innerBoundaryManager;
        amr::visitLevel<GridLayout>(
            level, *model.resourcesManager,
            [&](auto const& layout, auto&&, auto&&) {
                ibm.applyBC(state.E, layout,
                            core::InnerBCContext<state_type>{state, state, newTime, 0.});
                auto const& meshData = ibm.getMeshData();
                for (auto const c : {core::Component::X, core::Component::Y, core::Component::Z})
                {
                    auto& Ec           = state.E(c);
                    auto const& status = meshData.getStatusFieldFromCentering(layout.centering(Ec));
                    layout.evalOnBox(Ec, [&](auto&... args) {
                        core::MeshIndex<GridLayout::dimension> const idx{args...};
                        if (status(idx) == core::toDouble(core::ElemStatus::Inactive))
                            Ec(idx) = 0.;
                    });
                }
            },
            ibm, state);
    }

    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    bool const isNonIdeal_{Hall || fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()};

    core::UpwindConstrainedTransportState<VecField> ct_{};
    core::NonIdealElectricFieldState<VecField> non_ideal_electric_state_{isNonIdeal_};
    double to_primitive_gamma_;
    double to_conservative_gamma_;
    bool needsCurrent_;
};
} // namespace PHARE::solver

#endif
