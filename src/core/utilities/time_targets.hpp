#ifndef PHARE_CORE_UTILITIES_TIME_TARGETS_HPP
#define PHARE_CORE_UTILITIES_TIME_TARGETS_HPP

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>


namespace PHARE::core
{

enum class TimeStepType : std::uint8_t { constant, adaptive };


struct TimeAnchor
{
    double time;
    int step;
};


struct TimeRecord
{
    TimeStepType mode;
    double dt;
    std::optional<TimeAnchor> anchor;
};


inline double time_at_step(TimeAnchor const& anchor, int const step, double const dt)
{
    return anchor.time + static_cast<double>(step - anchor.step) * dt;
}


inline double adaptive_target(double const t, double const dt_stable, double const final_time)
{
    auto const next = t + dt_stable;
    if (next >= final_time or final_time - next < 1e-6 * dt_stable)
        return final_time;
    return next;
}


inline double increment_landing_on(double const t, double const target)
{
    if (!(target > t))
    {
        std::stringstream ss;
        ss << std::setprecision(17) << "increment_landing_on: target " << target
           << " is not after t = " << t;
        throw std::runtime_error(ss.str());
    }

    auto constexpr inf = std::numeric_limits<double>::infinity();

    auto const solve = [&](double const goal) -> std::optional<double> {
        auto d = goal - t;
        for (int iteration = 0; iteration < 8; ++iteration)
        {
            auto const landed = t + d;
            if (landed == goal)
                return d;
            auto const corrected = d + (goal - landed);
            d = corrected == d ? std::nextafter(d, landed < goal ? inf : -inf) : corrected;
        }
        return std::nullopt;
    };

    if (auto const d = solve(target))
        return *d;
    if (auto const d = solve(std::nextafter(target, inf)))
        return *d;

    std::stringstream ss;
    ss << std::setprecision(17) << "increment_landing_on: cannot land on " << target
       << " from t = " << t;
    throw std::runtime_error(ss.str());
}


inline TimeAnchor resolve_anchor(TimeStepType const mode, double const dt,
                                 std::optional<TimeRecord> const& previous, double const start_time,
                                 double const restored_time, int const restored_step)
{
    if (!previous)
        return {start_time, 0};

    if (mode == TimeStepType::constant and previous->mode == TimeStepType::constant
        and previous->dt == dt)
        return previous->anchor.value_or(TimeAnchor{start_time, 0});

    return {restored_time, restored_step};
}


} // namespace PHARE::core

#endif /* PHARE_CORE_UTILITIES_TIME_TARGETS_HPP */
