#ifndef PHARE_CORE_NUMERICS_PRIMITE_CONSERVATIVE_CONVERTER_POSITIVITY_FLOORS_HPP
#define PHARE_CORE_NUMERICS_PRIMITE_CONSERVATIVE_CONVERTER_POSITIVITY_FLOORS_HPP

#include "initializer/data_provider.hpp"

#include <cstddef>

namespace PHARE::core
{
struct PositivityFloors
{
    double density{0.};
    double pressure{0.};

    bool any() const { return density > 0. or pressure > 0.; }

    static PositivityFloors FROM(initializer::PHAREDict const& toPrimitiveDict)
    {
        return {cppdict::get_value(toPrimitiveDict, "density_floor", 0.),
                cppdict::get_value(toPrimitiveDict, "pressure_floor", 0.)};
    }
};

struct ClipCounts
{
    std::size_t density{0};
    std::size_t pressure{0};

    ClipCounts& operator+=(ClipCounts const& other)
    {
        density += other.density;
        pressure += other.pressure;
        return *this;
    }
};

} // namespace PHARE::core

#endif
