#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_DEFS_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_DEFS_HPP

#include "core/utilities/point/point.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace PHARE::core
{

/**
 * @brief Status of a mesh element (cell, face, edge, or node) relative to an inner boundary.
 *
 * - **Fluid**    — element lies entirely in the fluid domain.
 * - **Cut**      — element straddles the boundary surface.
 * - **Ghost**    — element lies inside the body but is used to enforce the boundary condition
 *                  by mirror-point interpolation from the fluid side.
 * - **Inactive** — element lies entirely inside the body and plays no role in the solver.
 */
enum class ElemStatus : std::uint8_t { Fluid, Cut, Ghost, Inactive };

/// Convert an ElemStatus value to its double encoding for field storage.
inline constexpr double toDouble(ElemStatus s)
{
    return static_cast<double>(static_cast<std::uint8_t>(s));
}


/**
 * @brief Precomputed per-ghost-element data used by the BC applier every time step.
 *
 * Storing these avoids recomputing expensive boundary queries (normal, symmetric)
 * once per field per ghost element per time step.
 *
 * @note `mirrorIsInterpolable` is `false` when the full mirror-point interp support (2 consecutive
 * grid values per direction) does not fit inside the allocated field extent. In that case the BC
 * still samples the field at `interpPoint` — the farthest interpolable point on the outward normal
 * — and extrapolates to the ghost using the `phiGhost`/`phiInterp` lever arm, so the ghost is only
 * skipped in the genuinely irreducible case (`interpValid == false`).
 * */
template<std::size_t dim>
struct GhostElemData
{
    Point<std::uint32_t, dim> index; ///< Local array index of the ghost element.
    Point<double, dim> mirrorPoint;  ///< Physical coords of the symmetric point in the fluid.
    Point<double, dim> normal;       ///< Unit outward normal at the boundary (ghost → mirror).
    bool mirrorIsInterpolable;       ///< True iff the full mirror-point interp support fits.

    /// Farthest interpolable sample point on the outward-normal ray between the surface and the
    /// mirror. Equals @ref mirrorPoint whenever the mirror itself is interpolable; otherwise it is
    /// pulled back toward the surface to the last point whose order-1 interp support fits locally.
    /// BC appliers sample the field here and extrapolate to the ghost using the signed-distance
    /// lever arm (@ref phiGhost / @ref phiInterp), so a far-off-patch mirror no longer forces the
    /// ghost to be skipped.
    Point<double, dim> interpPoint;
    double phiGhost;  ///< signedDistance(ghost) < 0 (ghost is inside the body).
    double phiInterp; ///< signedDistance(interpPoint) > 0 (sample is in the fluid).
    bool interpValid; ///< True iff a usable interpPoint (phiInterp above the floor) exists.
};


enum class InnerBoundaryShape { Plane, Sphere };

inline InnerBoundaryShape getInnerBoundaryShapeFromString(std::string const& name)
{
    static std::unordered_map<std::string, InnerBoundaryShape> const shapeMap_{
        {"plane", InnerBoundaryShape::Plane}, {"sphere", InnerBoundaryShape::Sphere}};

    auto it = shapeMap_.find(name);
    if (it == shapeMap_.end())
    {
        throw std::runtime_error("Unknown inner boundary shape " + name);
    }
    return it->second;
}

} // namespace PHARE::core


#endif // PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_DEFS_HPP
