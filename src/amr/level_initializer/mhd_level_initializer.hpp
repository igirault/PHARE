#ifndef PHARE_AMR_MHD_LEVEL_INITIALIZER_HPP
#define PHARE_AMR_MHD_LEVEL_INITIALIZER_HPP

#include "amr/level_initializer/level_initializer.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/resources_manager/amr_utils.hpp"

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"
#include "core/logger.hpp"
#include "core/utilities/index/index.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
class MHDLevelInitializer : public LevelInitializer<typename MHDModel::amr_types>
{
    using amr_types                    = MHDModel::amr_types;
    using hierarchy_t                  = amr_types::hierarchy_t;
    using level_t                      = amr_types::level_t;
    using patch_t                      = amr_types::patch_t;
    using IPhysicalModelT              = IPhysicalModel<amr_types>;
    using IMessengerT                  = amr::IMessenger<IPhysicalModelT>;
    using gridlayout_type              = MHDModel::gridlayout_type;
    using state_type                   = MHDModel::state_type;
    using resources_manager_type       = MHDModel::resources_manager_type;
    using inner_boundary_manager_type  = MHDModel::inner_boundary_manager_type;
    static constexpr auto dimension    = gridlayout_type::dimension;
    static constexpr auto interp_order = gridlayout_type::interp_order;

    inline bool isRootLevel(int levelNumber) const { return levelNumber == 0; }

public:
    MHDLevelInitializer() = default;

    void initialize(std::shared_ptr<hierarchy_t> const& hierarchy, int levelNumber,
                    std::shared_ptr<level_t> const& oldLevel, IPhysicalModelT& model,
                    amr::IMessenger<IPhysicalModelT>& messenger, double initDataTime,
                    bool isRegridding) override
    {
        auto& mhdModel = static_cast<MHDModel&>(model);
        auto& level    = amr_types::getLevel(*hierarchy, levelNumber);

        if (isRegridding)
        {
            PHARE_LOG_LINE_STR("regriding level " + std::to_string(levelNumber));
            PHARE_LOG_START(3, "mhdLevelInitializer::initialize : regriding block");
            messenger.regrid(hierarchy, levelNumber, oldLevel, model, initDataTime);
            model.updateExternalFields(level, initDataTime);
            PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : regriding block");
        }
        else
        {
            if (isRootLevel(levelNumber))
            {
                PHARE_LOG_START(3, "mhdLevelInitializer::initialize : root level init");
                model.initialize(level);
                messenger.fillRootGhosts(model, level, initDataTime);
                PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : root level init");
            }
            else
            {
                PHARE_LOG_START(3, "mhdLevel Initializer::initialize : initlevel");
                messenger.initLevel(model, level, initDataTime);
                model.updateExternalFields(level, initDataTime);
                PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : initlevel");
            }
        }

        // Fill the moment ghosts (rho/rhoV/Etot1) of a freshly (re)created refined level so its
        // physical-boundary ghosts carry their boundary-condition values before the first advance's
        // flux consumes them. resetGhosts=false: do NOT NaN-stamp first, so the interior/coarse-fine
        // ghosts just set by the init refiners (and the outermost layers the boundary fill does not
        // reach) keep their values instead of being poisoned with NaN. B is handled by the magnetic
        // (re)grid fill.
        if (!isRootLevel(levelNumber))
        {
            auto& mhdMessenger = static_cast<amr::MHDMessenger<MHDModel>&>(messenger);
            mhdMessenger.fillMomentsGhosts(mhdModel.state, level, initDataTime, 0.0,
                                           /*resetGhosts=*/false);
        }

        // Inner-boundary setup (classify + safe state + moment BCs). Shared with the restart path
        // (MHDModel::reinitializeAfterRestart) so both fresh and restored runs establish it.
        mhdModel.setupInnerBoundaryState(level, initDataTime);
    }
};

} // namespace PHARE::solver


#endif
