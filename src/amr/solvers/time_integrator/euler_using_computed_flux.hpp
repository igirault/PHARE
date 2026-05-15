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

        faraday_(level, model, state, E, statenew, dt);

        resources_manager_type& rm = *model.resourcesManager;

        // Update external magnetic field
        amr::visitLevel<gridlayout_type>(
            level, rm,
            [&](auto& layout, auto&&, auto&&) {
                statenew.updateExternalMagneticField(layout, newTime);
            },
            statenew);

        // Assign a safe physical state in the inner boundary inactive region
        if (model.hasInnerBoundary())
        {
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;

            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    // Restore cell-centered quantities for inactive cells
                    auto& meshData   = ibm.getMeshData();
                    auto& cellStatus = meshData.cellStatusField();
                    layout.evalOnGhostBox(statenew.rho, [&](auto&... args) {
                        auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                        if (cellStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                        {
                            statenew.rho(idx)   = state.rho(idx);
                            statenew.Etot1(idx) = state.Etot1(idx);
                            statenew.rhoV(core::Component::X)(idx)
                                = state.rhoV(core::Component::X)(idx);
                            statenew.rhoV(core::Component::Y)(idx)
                                = state.rhoV(core::Component::Y)(idx);
                            statenew.rhoV(core::Component::Z)(idx)
                                = state.rhoV(core::Component::Z)(idx);
                        }
                    });

                    // Restore face-centered B for inactive face elements
                    auto restoreB = [&](auto component) {
                        auto centering
                            = layout.centering(statenew.B1(component).physicalQuantity());
                        auto& faceStatus = meshData.getStatusFieldFromCentering(centering);
                        layout.evalOnBox(statenew.B1(component), [&](auto&... args) {
                            auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                            if (faceStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                                statenew.B1(component)(idx) = state.B1(component)(idx);
                        });
                    };
                    restoreB(core::Component::X);
                    restoreB(core::Component::Y);
                    restoreB(core::Component::Z);
                },
                ibm, state, statenew);
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
                    // Apply inner boundary conditions
                    ibm.applyBC(statenew.B1, layout, ctx);
                    ibm.applyBC(statenew.rhoV, layout, ctx);
                    ibm.applyBC(statenew.rho, layout, ctx);
                    ibm.applyBC(statenew.Etot1, layout, ctx);
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
