#if !defined(PHARE_SIM_STR) || !defined(PHARE_SIM_ID)
#error // needs PHARE_SIM_STR! like "1, 1, 1"
#endif // PHARE_SIM_STR

#include "python3/cpp_simulator_options.hpp"

#include "simulator/solver_mhd_factory_impl.hpp"


template std::unique_ptr<PHARE::solver::ISolver<PHARE::amr::SAMRAI_Types>>
PHARE::solver::make_mhd_solver<PHARE::pydata::resolve_simulator_options()>(
    PHARE::initializer::PHAREDict const&);
