#ifndef PHARE_PYTHON_CPP_SIMULATOR_OPTIONS_HPP
#define PHARE_PYTHON_CPP_SIMULATOR_OPTIONS_HPP


#ifndef PHARE_SIM_STR
#define PHARE_SIM_STR 1, 1, 2 // mostly for clangformat - errors in cpp file if define is missing
#endif

#include "simulator/simulator_def.hpp"


namespace PHARE::pydata
{


auto static constexpr resolve_simulator_options()
{
    using namespace PHARE::MHDOpts;
    return SimOpts{PHARE_SIM_STR};
}


} // namespace PHARE::pydata


#endif /*PHARE_PYTHON_CPP_SIMULATOR_OPTIONS_HPP*/
