#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_DEFS_HPP
#define PHARE_CORE_BOUNDARY_BOUNDARY_DEFS_HPP

#include "core/data/grid/gridlayoutdefs.hpp"

#include "unordered_map"

namespace PHARE::core
{
/**
 * @brief Physical behavior of a boundary.
 */
enum class BoundaryType { None, Reflective, Inflow, Outflow, Open };

/*
 * @brief Possible codimension of a boundary.
 */
enum class BoundaryCodim { One = 1, Two = 2, Three = 3 };

//@{
//! @name Definitions for boundary array sizes in 1d, 2d, or 3d:
int const NUM_1D_NODES = 2;

int const NUM_2D_EDGES = 4;
int const NUM_2D_NODES = 4;

int const NUM_3D_FACES = 6;
int const NUM_3D_EDGES = 12;
int const NUM_3D_NODES = 8;
//@}

/**
 * @brief Possible locations of 1-codimensional boundary (a face in 3D, an edge in 2D, an extremity
 * in 1D).
 */
enum class BoundaryLocation {
    XLower = 0,
    XUpper = 1,
    YLower = 2,
    YUpper = 3,
    ZLower = 4,
    ZUpper = 5
};

/// @brief Return the side of a boundary location.
/// @param boundaryLoc The boundary location.
/// @return The boundary side.
constexpr Side getSide(BoundaryLocation boundaryLoc)
{
    switch (boundaryLoc)
    {
        case BoundaryLocation::XLower:
        case BoundaryLocation::YLower:
        case BoundaryLocation::ZLower: return Side::Lower; break;

        case BoundaryLocation::XUpper:
        case BoundaryLocation::YUpper:
        case BoundaryLocation::ZUpper: return Side::Upper; break;

        default: throw std::runtime_error("Invalid BoundaryLocation.");
    }
};

/// @brief Return the direction of a boundary location.
/// @param boundaryLoc The boundary location.
/// @return The boundary direction.
constexpr Direction getDirection(BoundaryLocation boundaryLoc)
{
    switch (boundaryLoc)
    {
        case BoundaryLocation::XLower:
        case BoundaryLocation::XUpper: return Direction::X; break;

        case BoundaryLocation::YLower:
        case BoundaryLocation::YUpper: return Direction::Y; break;

        case BoundaryLocation::ZLower:
        case BoundaryLocation::ZUpper: return Direction::Z; break;

        default: throw std::runtime_error("Invalid BoundaryLocation.");
    }
};

/*
 * @brief Possible locations of a 2-codimensional boundary (an edge in 3D, a corner in 2D)
 */
enum class Codim2BoundaryLocation {
    XLower_YLower = 0,
    XHI_YLower    = 1,
    XLower_YUpper = 2,
    XHI_YUpper    = 3
};

/*
 * @brief Possible locations of a 3-codimensional boundary (a corner in 3D)
 */
enum class Codim3BoundaryLocation {
    XLower_YLower_ZLower = 0,
    XHI_YLower_ZLower    = 1,
    XLower_YUpper_ZLower = 2,
    XHI_YUpper_ZLower    = 3,
    XLower_YLower_ZUpper = 4,
    XHI_YLower_ZUpper    = 5,
    XLower_YUpper_ZUpper = 6,
    XHI_YUpper_ZUpper    = 7
};

/**
 * @brief Get the BoundaryType from input keyword, and throw and error if the keyword does not
 * correspond to any known boundary type.
 */
inline BoundaryType getBoundaryTypeFromString(std::string const& name)
{
    static std::unordered_map<std::string, BoundaryType> const typeMap_ = {
        {"none", BoundaryType::None},       {"open", BoundaryType::Open},
        {"inflow", BoundaryType::Inflow},   {"reflective", BoundaryType::Reflective},
        {"outflow", BoundaryType::Outflow},
    };

    auto it = typeMap_.find(name);
    if (it == typeMap_.end())
        throw std::runtime_error("Wrong boundary type name = " + name);
    return it->second;
}

/**
 * @brief Get the BoundaryType from input keyword, and throw and error if the keyword foes not
 * correspond to any known boundary type.
 */
inline BoundaryLocation getBoundaryLocationFromString(std::string const& name)
{
    static std::unordered_map<std::string, BoundaryLocation> const typeMap_ = {
        {"xlower", BoundaryLocation::XLower}, {"xupper", BoundaryLocation::XUpper},
        {"ylower", BoundaryLocation::YLower}, {"yupper", BoundaryLocation::YUpper},
        {"zlower", BoundaryLocation::ZLower}, {"zupper", BoundaryLocation::ZUpper},
    };

    auto it = typeMap_.find(name);
    if (it == typeMap_.end())
        throw std::runtime_error("Wrong boundary location name = " + name);
    return it->second;
}

} // namespace PHARE::core

#endif /* PHARE_CORE_BOUNDARY_BOUNDARY_DEFS_HPP */
