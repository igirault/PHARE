#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BC_CONTEXT_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BC_CONTEXT_HPP

namespace PHARE::core
{

template<typename PhysicalStateT>
struct InnerBCContext
{
    PhysicalStateT& statenew;
    PhysicalStateT const& state;
    double time{0.};
    double dt{0.};
};

} // namespace PHARE::core

#endif
