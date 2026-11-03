#ifndef PHARE_GHOST_ELEM_RESOURCE_HPP
#define PHARE_GHOST_ELEM_RESOURCE_HPP

#include "amr/data/inner_boundary/ghost_elem_data.hpp"
#include "amr/data/inner_boundary/ghost_elem_variable.hpp"

namespace PHARE::amr
{

template<typename ResourcesUser>
struct UserGhostElemType
{
    static constexpr auto dimension = ResourcesUser::dimension;

    using variable_type   = GhostElemVariable<dimension>;
    using patch_data_type = GhostElemPatchData<dimension>;
};

} // namespace PHARE::amr

#endif
