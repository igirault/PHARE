#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_DEFS_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_DEFS_HPP

#include "core/utilities/point/point.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>

namespace PHARE::core
{
template<typename GridLayoutT>
using layout_quantity_t = std::decay_t<decltype(GridLayoutT::options)>::FieldOptions::Quantity;


enum class ElemStatus : std::uint8_t { Fluid, Cut, Ghost, Inactive };

enum class NearBodyScheme : std::uint8_t { None, FirstOrder, SecondOrder };

inline constexpr double toDouble(ElemStatus s)
{
    return static_cast<double>(static_cast<std::uint8_t>(s));
}

template<std::size_t dim>
struct GhostElemData
{
    Point<std::uint32_t, dim> index;
    Point<double, dim> mirrorPoint;
    Point<double, dim> normal;
    bool mirrorIsInterpolable;
};

enum class InnerBoundaryShape { Sphere };

inline InnerBoundaryShape getInnerBoundaryShapeFromString(std::string const& name)
{
    static std::unordered_map<std::string, InnerBoundaryShape> const shapeMap_{
        {"sphere", InnerBoundaryShape::Sphere}};

    auto it = shapeMap_.find(name);
    if (it == shapeMap_.end())
    {
        throw std::runtime_error("Unknown inner boundary shape " + name);
    }
    return it->second;
}

} // namespace PHARE::core

#endif
