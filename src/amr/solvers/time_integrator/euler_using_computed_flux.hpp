#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
class EulerUsingComputedFlux
{
    using level_t                     = MHDModel::level_t;
    using gridlayout_type             = MHDModel::gridlayout_type;
    using state_type                  = MHDModel::state_type;
    using resources_manager_type      = MHDModel::resources_manager_type;
    using inner_boundary_manager_type = MHDModel::inner_boundary_manager_type;

    using Dispatchers_t = Dispatchers<gridlayout_type>;

    using FiniteVolumeEuler_t = Dispatchers_t::FiniteVolumeEuler_t;
    using Faraday_t           = Dispatchers_t::Faraday_t;

public:
    EulerUsingComputedFlux() {}

    // we provide dt here because we sometimes need it to be different from newTime-currentTime, for
    // example in the case of some rk integration methods
    void operator()(MHDModel& model, auto& state, auto& statenew, auto& E, auto& fluxes, auto& bc,
                    level_t& level, double const newTime, double const dt)
    {
        fv_euler_(level, model, newTime, state, statenew, fluxes, dt);

        resources_manager_type& rm = *model.resourcesManager;

        // Background-field energy source: when B0 is not numerically curl-free,
        // the reduced energy E1 gains a source +E·j0 (j0 = curl(B0)) that the
        // continuous derivation drops assuming curl(B0) = 0. E and j0 (= state.J0)
        // are co-located on edges; project both to the cell centre and accumulate.
        amr::visitLevel<gridlayout_type>(
            level, rm,
            [&](auto& layout, auto&&, auto&&) {
                using Layout = gridlayout_type;
                auto const& Ex  = E(core::Component::X);
                auto const& Ey  = E(core::Component::Y);
                auto const& Ez  = E(core::Component::Z);
                auto const& j0x = state.J0(core::Component::X);
                auto const& j0y = state.J0(core::Component::Y);
                auto const& j0z = state.J0(core::Component::Z);

                layout.evalOnBox(statenew.Etot1, [&](auto&... args) {
                    auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};

                    auto const Exc = Layout::template project<Layout::edgeXToCellCenter>(Ex, idx);
                    auto const Eyc = Layout::template project<Layout::edgeYToCellCenter>(Ey, idx);
                    auto const Ezc = Layout::template project<Layout::edgeZToCellCenter>(Ez, idx);

                    auto const j0xc = Layout::template project<Layout::edgeXToCellCenter>(j0x, idx);
                    auto const j0yc = Layout::template project<Layout::edgeYToCellCenter>(j0y, idx);
                    auto const j0zc = Layout::template project<Layout::edgeZToCellCenter>(j0z, idx);

                    statenew.Etot1(idx) += dt * (Exc * j0xc + Eyc * j0yc + Ezc * j0zc);
                });
            },
            state, statenew, E);

        faraday_(level, model, state, E, statenew, dt);

        // Update external magnetic field
        amr::visitLevel<gridlayout_type>(
            level, rm,
            [&](auto& layout, auto&&, auto&&) {
                statenew.updateExternalMagneticField(layout, newTime);
            },
            statenew);

        // Pin the inner-boundary inactive region to a canonical safe physical state.
        // We cannot simply copy the previous-step values into `statenew` because some
        // integrators (e.g. TVDRK3) reuse a single buffer for both `state` and `statenew`;
        // in that aliased case the copy is a self-assignment and the drift from
        // `fv_euler_` / `faraday_` persists. Recomputing the safe state every substep is
        // both alias-safe and matches what `mhd_level_initializer` does at init.
        if (model.hasInnerBoundary())
        {
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;

            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    auto& meshData = ibm.getMeshData();

                    // Zero inactive face-centered B1 first: the cell-centered safe reset
                    // recomputes Etot1 from the projected B at the cell centre, so any
                    // stale face value would leak into the energy.
                    auto resetB = [&](auto component) {
                        auto centering
                            = layout.centering(statenew.B1(component).physicalQuantity());
                        auto& faceStatus = meshData.getStatusFieldFromCentering(centering);
                        layout.evalOnBox(statenew.B1(component), [&](auto&... args) {
                            auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                            if (faceStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                                statenew.B1(component)(idx) = 0.0;
                        });
                    };
                    resetB(core::Component::X);
                    resetB(core::Component::Y);
                    resetB(core::Component::Z);

                    // Pin inactive cell-centered conservatives to (rho=1, P=1, V=0).
                    auto& cellStatus = meshData.cellStatusField();
                    layout.evalOnGhostBox(statenew.rho, [&](auto&... args) {
                        auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                        if (cellStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                            statenew.safeResetInactiveCell(idx, layout, *model.thermo);
                    });
                },
                ibm, statenew);
        }

        bc.fillMagneticGhosts(statenew.B1, level, newTime);
        bc.fillMomentsGhosts(statenew, level, newTime, dt);

        // Application of inner boundary conditions
        if (model.hasInnerBoundary())
        {
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;
            core::InnerBCContext<state_type> ctx{statenew, state, newTime, dt};

            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    // Apply inner boundary conditions on the moments (priority-ordered: Etot1 last).
                    ibm.applyToMoments(layout, ctx);
                },
                ibm, state, statenew);
        }
    }

private:
    FiniteVolumeEuler_t fv_euler_;
    Faraday_t faraday_;
};
} // namespace PHARE::solver

#endif
