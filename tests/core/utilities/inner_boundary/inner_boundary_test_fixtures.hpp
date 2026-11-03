#ifndef PHARE_TESTS_CORE_UTILITIES_INNER_BOUNDARY_INNER_BOUNDARY_TEST_FIXTURES_HPP
#define PHARE_TESTS_CORE_UTILITIES_INNER_BOUNDARY_INNER_BOUNDARY_TEST_FIXTURES_HPP

#include "core/data/grid/gridlayout.hpp"
#include "core/models/options/mhd_options.hpp"
#include "simulator/simulator_def.hpp"

namespace PHARE::core
{
inline constexpr PHARE::SimOpts ibTestSimOptions2D{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::Constant,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::None,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<ibTestSimOptions2D> ibTestFieldOptions2D{};

using IBTestGridLayout2D = GridLayout<PHARE::MHDOptions<ibTestFieldOptions2D>{}>;

} // namespace PHARE::core

#endif
