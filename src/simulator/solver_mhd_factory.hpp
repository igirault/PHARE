#ifndef PHARE_SIMULATOR_SOLVER_MHD_FACTORY_HPP
#define PHARE_SIMULATOR_SOLVER_MHD_FACTORY_HPP

#include "phare_mpi.hpp" // IWYU pragma: keep

#include "amr/solvers/solver.hpp"
#include "amr/types/amr_types.hpp"

#include "initializer/data_provider.hpp"

#include <memory>

namespace PHARE::solver
{

template<auto opts>
    requires(opts.mhd_enabled)
std::unique_ptr<ISolver<amr::SAMRAI_Types>> make_mhd_solver(initializer::PHAREDict const& dict);

} // namespace PHARE::solver

#endif // PHARE_SIMULATOR_SOLVER_MHD_FACTORY_HPP
