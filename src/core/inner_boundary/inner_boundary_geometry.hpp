#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_GEOMETRY_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_GEOMETRY_HPP

#include "core/inner_boundary/inner_boundary_defs.hpp"
#include "core/utilities/point/point.hpp"

#include <limits>

namespace PHARE::core
{

template<std::size_t dim>
class InnerBoundaryGeometry
{
public:
    using point_type = Point<double, dim>;

    InnerBoundaryGeometry(std::string name)
        : name_{name}
    {
    }

    virtual ~InnerBoundaryGeometry() = default;

    std::string name() const { return name_; }

    InnerBoundaryShape shape() const { return shape_; }

    virtual double signedDistance(point_type const& point) const = 0;

    virtual point_type normal(point_type const& point) const = 0;

    virtual double characteristicLength() const { return std::numeric_limits<double>::infinity(); }

    virtual point_type project(point_type const& point) const
    {
        auto n = normal(point);
        return point - n * signedDistance(point);
    }

    virtual point_type symmetric(point_type const& point) const
    {
        auto projected = project(point);
        return projected * 2. - point;
    }

private:
    InnerBoundaryShape shape_;
    std::string name_;
};
} // namespace PHARE::core

#endif
