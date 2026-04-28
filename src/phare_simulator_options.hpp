#ifndef PHARE_SIMULATOR_OPTIONS_HPP
#define PHARE_SIMULATOR_OPTIONS_HPP

#include "core/utilities/meta/meta_utilities.hpp"
#include "core/utilities/meta/enum.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace PHARE
{

// if mhd is off, axes stay at MHDOff and no selector specializations exist for it (canary)
namespace MHDOpts
{
    enum class TimeIntegratorType : uint8_t { MHDOff, Euler, TVDRK2, TVDRK3, SSPRK4_5, count };
    enum class ReconstructionType : uint8_t { MHDOff, Constant, Linear, WENO3, WENOZ, MP5, count };
    enum class SlopeLimiterType : uint8_t { MHDOff, None, VanLeer, MinMod, count };
    enum class RiemannSolverType : uint8_t { MHDOff, Rusanov, HLL, HLLD, count };
} // namespace MHDOpts

namespace core
{
    template<>
    struct EnumTraits<MHDOpts::TimeIntegratorType>
    {
        static constexpr std::string_view label = "time integrator type";
        static constexpr std::array names{
            enumEntry("euler", MHDOpts::TimeIntegratorType::Euler),
            enumEntry("tvdrk2", MHDOpts::TimeIntegratorType::TVDRK2),
            enumEntry("tvdrk3", MHDOpts::TimeIntegratorType::TVDRK3),
            enumEntry("ssprk4_5", MHDOpts::TimeIntegratorType::SSPRK4_5),
        };
    };

    template<>
    struct EnumTraits<MHDOpts::ReconstructionType>
    {
        static constexpr std::string_view label = "reconstruction type";
        static constexpr std::array names{
            enumEntry("constant", MHDOpts::ReconstructionType::Constant),
            enumEntry("linear", MHDOpts::ReconstructionType::Linear),
            enumEntry("weno3", MHDOpts::ReconstructionType::WENO3),
            enumEntry("wenoz", MHDOpts::ReconstructionType::WENOZ),
            enumEntry("mp5", MHDOpts::ReconstructionType::MP5),
        };
    };

    template<>
    struct EnumTraits<MHDOpts::SlopeLimiterType>
    {
        static constexpr std::string_view label = "slope limiter type";
        static constexpr std::array names{
            enumEntry("none", MHDOpts::SlopeLimiterType::None),
            enumEntry("vanleer", MHDOpts::SlopeLimiterType::VanLeer),
            enumEntry("minmod", MHDOpts::SlopeLimiterType::MinMod),
        };
    };

    template<>
    struct EnumTraits<MHDOpts::RiemannSolverType>
    {
        static constexpr std::string_view label = "riemann solver type";
        static constexpr std::array names{
            enumEntry("rusanov", MHDOpts::RiemannSolverType::Rusanov),
            enumEntry("hll", MHDOpts::RiemannSolverType::HLL),
            enumEntry("hlld", MHDOpts::RiemannSolverType::HLLD),
        };
    };
} // namespace core

struct SimOpts
{
    std::size_t dimension    = 1;
    std::size_t interp_order = 1;

    std::size_t nbRefinedPart = core::defaultNbrRefinedParts(dimension, interp_order);

    MHDOpts::ReconstructionType reconstruction_type = MHDOpts::ReconstructionType::MHDOff;
    MHDOpts::SlopeLimiterType slope_limiter_type    = MHDOpts::SlopeLimiterType::MHDOff;
    MHDOpts::RiemannSolverType riemann_solver_type  = MHDOpts::RiemannSolverType::MHDOff;
    bool Hall                                       = false;

    // derived — two independent axes, never one as the other's negation
    bool hybrid_enabled = interp_order > 0;
    bool mhd_enabled    = reconstruction_type != MHDOpts::ReconstructionType::MHDOff;

    // all four MHD axes must agree on off-ness; a member fn keeps SimOpts an aggregate
    constexpr bool mhd_axes_consistent() const
    {
        bool const mhd_is_off = !mhd_enabled;
        return mhd_is_off == (slope_limiter_type == MHDOpts::SlopeLimiterType::MHDOff)
               && mhd_is_off == (riemann_solver_type == MHDOpts::RiemannSolverType::MHDOff);
    }
};

template<SimOpts opts>
inline constexpr bool has_hybrid_v = opts.hybrid_enabled;

template<SimOpts opts>
inline constexpr bool has_mhd_v = opts.mhd_enabled;

} // namespace PHARE

#endif // PHARE_SIMULATOR_OPTIONS_HPP
