#ifndef PHARE_CORE_UTILITIES_SPACE_TIME_FUNCTION_HPP
#define PHARE_CORE_UTILITIES_SPACE_TIME_FUNCTION_HPP

#include "core/utilities/span.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace PHARE::core
{

/**
 * @brief A read-only view of one coordinate array handed to a SpaceTimeFunction.
 *
 * Deliberately not a Span<double>: that one is registered as a python class (see
 * cpp_etc.cpp) so that it can be returned through a shared_ptr holder, and pybind11
 * forbids a type to have both a class registration and a custom type caster. The caster
 * giving this one to python as a zero-copy numpy view lives in python3/pybind_def.hpp.
 */
struct CoordinateSpan
{
    double const* ptr = nullptr;
    std::size_t size  = 0;
};


/**
 * @brief A user function of space and time, f(x[, y[, z]], t).
 */
template<typename ReturnType, std::size_t dim>
struct SpaceTimeFunctionHelper
{
};

template<>
struct SpaceTimeFunctionHelper<double, 1>
{
    using return_type = std::shared_ptr<Span<double>>;
    using param_type  = CoordinateSpan const&;
    using type        = std::function<return_type(param_type, double)>;
};

template<>
struct SpaceTimeFunctionHelper<double, 2>
{
    using return_type = std::shared_ptr<Span<double>>;
    using param_type  = CoordinateSpan const&;
    using type        = std::function<return_type(param_type, param_type, double)>;
};

template<>
struct SpaceTimeFunctionHelper<double, 3>
{
    using return_type = std::shared_ptr<Span<double>>;
    using param_type  = CoordinateSpan const&;
    using type        = std::function<return_type(param_type, param_type, param_type, double)>;
};

template<std::size_t dim>
using SpaceTimeFunction = typename SpaceTimeFunctionHelper<double, dim>::type;

} // namespace PHARE::core

#endif // PHARE_CORE_UTILITIES_SPACE_TIME_FUNCTION_HPP
