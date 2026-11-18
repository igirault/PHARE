#ifndef PHARE_SIMULATOR_SOLVER_MHD_FACTORY_IMPL_HPP
#define PHARE_SIMULATOR_SOLVER_MHD_FACTORY_IMPL_HPP

#include "phare_solver.hpp"

#include "amr/solvers/mhd_resolver.hpp"
#include "amr/solvers/solver_mhd.hpp"

#include "simulator/solver_mhd_factory.hpp"

#include <memory>

namespace PHARE::solver
{

template<auto opts>
    requires(opts.mhd_enabled)
std::unique_ptr<ISolver<amr::SAMRAI_Types>> make_mhd_solver(initializer::PHAREDict const& dict)
{
    using Model_t  = PHARE_Types<opts>::MHD::Model_t;
    using Solver_t = SolverMHD<Model_t, amr::SAMRAI_Types,
                               typename MHDResolver<opts, Model_t>::MHDTimeStepper_t>;

    return std::make_unique<Solver_t>(dict);
}

} // namespace PHARE::solver

#endif // PHARE_SIMULATOR_SOLVER_MHD_FACTORY_IMPL_HPP
