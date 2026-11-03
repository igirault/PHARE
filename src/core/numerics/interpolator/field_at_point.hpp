#ifndef PHARE_CORE_NUMERICS_INTERPOLATOR_FIELD_AT_POINT_HPP
#define PHARE_CORE_NUMERICS_INTERPOLATOR_FIELD_AT_POINT_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "core/numerics/interpolator/interpolator.hpp"
#include "core/utilities/point/point.hpp"

namespace PHARE::core
{

template<std::size_t dim, std::size_t interpOrder>
class FieldAtPoint : private Interpolator<dim, interpOrder>
{
    using Base = Interpolator<dim, interpOrder>;

public:
    template<typename GridLayout>
    static bool pointIsInterpolable(GridLayout const& layout, Point<double, dim> const& point,
                                    std::array<QtyCentering, dim> const& centerings)
    {
        auto const& dx     = layout.meshSize();
        auto const& amrBox = layout.AMRBox();
        auto const nGhosts = static_cast<int>(GridLayout::options.field_ghost_width);
        for (auto d = 0u; d < dim; ++d)
        {
            int const margin = (centerings[d] == QtyCentering::dual) ? 1 : 0;
            int const iCell  = static_cast<int>(std::floor(point[d] / dx[d]));
            if (iCell < amrBox.lower[d] - nGhosts + margin
                || iCell > amrBox.upper[d] + nGhosts - margin)
                return false;
        }
        return true;
    }

    template<auto quantity, typename GridLayout, typename Field>
    double operator()(GridLayout const& layout, Field const& field,
                      Point<double, dim> const& physPoint)
    {
        auto const& dx = layout.meshSize();

        Point<int, dim> iCell;
        std::array<double, dim> delta;
        for (auto d = 0u; d < dim; ++d)
        {
            double const normalizedPos = physPoint[d] / dx[d];
            iCell[d]                   = static_cast<int>(std::floor(normalizedPos));
            delta[d]                   = normalizedPos - iCell[d];
        }

        this->template indexAndWeights_<QtyCentering, QtyCentering::dual>(layout, iCell, delta);
        this->template indexAndWeights_<QtyCentering, QtyCentering::primal>(layout, iCell, delta);

        auto const indexWeights
            = std::forward_as_tuple(this->dual_startIndex_, this->dual_weights_,
                                    this->primal_startIndex_, this->primal_weights_);

        return this->meshToParticle_.template operator()<GridLayout, quantity>(field, indexWeights);
    }

    template<typename GridLayout, typename Field>
    double operator()(GridLayout const& layout, Field const& field,
                      Point<double, dim> const& physPoint)
    {
        auto const centering = GridLayout::centering(field);

        auto const& dx = layout.meshSize();

        Point<int, dim> iCell;
        std::array<double, dim> delta;
        for (auto d = 0u; d < dim; ++d)
        {
            double const normalizedPos = physPoint[d] / dx[d];
            iCell[d]                   = static_cast<int>(std::floor(normalizedPos));
            delta[d]                   = normalizedPos - iCell[d];
        }

        this->template indexAndWeights_<QtyCentering, QtyCentering::dual>(layout, iCell, delta);
        this->template indexAndWeights_<QtyCentering, QtyCentering::primal>(layout, iCell, delta);

        auto startFor = [&](std::size_t d) -> std::uint32_t {
            return centering[d] == QtyCentering::primal ? this->primal_startIndex_[d]
                                                        : this->dual_startIndex_[d];
        };
        auto weightsFor = [&](std::size_t d) -> auto const& {
            return centering[d] == QtyCentering::primal ? this->primal_weights_[d]
                                                        : this->dual_weights_[d];
        };

        if constexpr (dim == 1)
        {
            auto const s0       = startFor(0);
            auto const& w0      = weightsFor(0);
            double fieldAtPoint = 0.;
            for (auto i0 = 0u; i0 < w0.size(); ++i0)
                fieldAtPoint += field(s0 + i0) * w0[i0];
            return fieldAtPoint;
        }
        else if constexpr (dim == 2)
        {
            auto const s0       = startFor(0);
            auto const s1       = startFor(1);
            auto const& w0      = weightsFor(0);
            auto const& w1      = weightsFor(1);
            double fieldAtPoint = 0.;
            for (auto i0 = 0u; i0 < w0.size(); ++i0)
                for (auto i1 = 0u; i1 < w1.size(); ++i1)
                    fieldAtPoint += field(s0 + i0, s1 + i1) * w0[i0] * w1[i1];
            return fieldAtPoint;
        }
        else
        {
            auto const s0       = startFor(0);
            auto const s1       = startFor(1);
            auto const s2       = startFor(2);
            auto const& w0      = weightsFor(0);
            auto const& w1      = weightsFor(1);
            auto const& w2      = weightsFor(2);
            double fieldAtPoint = 0.;
            for (auto i0 = 0u; i0 < w0.size(); ++i0)
                for (auto i1 = 0u; i1 < w1.size(); ++i1)
                    for (auto i2 = 0u; i2 < w2.size(); ++i2)
                        fieldAtPoint += field(s0 + i0, s1 + i1, s2 + i2) * w0[i0] * w1[i1] * w2[i2];
            return fieldAtPoint;
        }
    }
};

} // namespace PHARE::core

#endif
