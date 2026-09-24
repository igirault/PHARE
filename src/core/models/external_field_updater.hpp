#ifndef PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER
#define PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER

#include "core/def.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/user/user_field_updater.hpp"
#include "core/models/external_field.hpp"
#include "core/utilities/point/point.hpp"

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>

namespace PHARE::core
{


/**
 * @brief Interface of the external field updaters.
 *
 * @tparam VecFieldT vecfield implementation
 * @tparam GridLayoutT grid layout implementation
 *
 * Implementations provide the vector potential @f$\mathbf{A}_0@f$ of the external field, and
 * its time derivative when the field is time dependent. The external field itself is obtained
 * here as @f$\mathbf{B}_0 = \nabla\times\mathbf{A}_0@f$, so that it is divergence free by
 * construction at the discrete level.
 */
template<typename VecFieldT, typename GridLayoutT>
class IExternalFieldUpdater
{
public:
    using vecfield_type       = VecFieldT;
    using field_type          = VecFieldT::field_type;
    using value_type          = field_type::value_type;
    using tensor_type         = VecFieldT::tensor_t;
    using component_type      = VecFieldT::component_type;
    using point_type          = Point<double, GridLayoutT::dimension>;
    using external_field_type = ExternalField<VecFieldT>;

    static constexpr std::size_t dimension = GridLayoutT::dimension;

    virtual ~IExternalFieldUpdater() = default;

    /**
     * @brief fill an external field on a new patch: fill its coordinates cache when the
     * updater needs one, then compute B0 and dB0/dt
     *
     * @param externalField the external field to fill, whose E-centered scratch vecfield holds
     * the vector potential on output
     * @param layout the grid layout of the patch
     * @param time the current time
     */
    void initialize(external_field_type& externalField, GridLayoutT const& layout, double time)
    {
        if (needsCoordinates())
            UserFieldUpdater::fillCoordinates(layout, externalField.coordinates());
        compute_(externalField, layout, time);
    }

    /**
     * @brief advance an external field to a new time: recompute B0 and dB0/dt if the field is
     * time dependent, no-op otherwise
     */
    void update(external_field_type& externalField, GridLayoutT const& layout, double time)
    {
        if (isTimeDependent())
            compute_(externalField, layout, time);
    }

    /**
     * @brief fill the vector potential, respectively its time derivative
     *
     * @param coordinates node coordinates cache of the external field (see
     * ExternalField::coordinates()), empty when the updater does not need one
     */
    void virtual computePotential(vecfield_type& a0, double time, GridLayoutT const& layout,
                                  std::span<vecfield_type const> coordinates)               = 0;
    void virtual computePotentialTimeDerivative(vecfield_type& da0_dt, double time,
                                                GridLayoutT const& layout,
                                                std::span<vecfield_type const> coordinates) = 0;

    NO_DISCARD bool virtual isTimeDependent() const  = 0;
    NO_DISCARD bool virtual needsCoordinates() const = 0;

protected:
    //! compute B0 and dB0/dt, as the curl of the vector potential and of its time derivative
    virtual void compute_(external_field_type& externalField, GridLayoutT const& layout,
                          double time)
    {
        auto const coordinates = std::as_const(externalField).coordinates();

        computePotential(externalField.scratch, time, layout, coordinates);
        curlOnGhostBox_(externalField.B0, externalField.scratch, layout);
        if (isTimeDependent())
        {
            computePotentialTimeDerivative(externalField.scratch, time, layout, coordinates);
            curlOnGhostBox_(externalField.dB0dt, externalField.scratch, layout);
        }
        else
        {
            externalField.dB0dt.zero();
        }
    }

private:
    void curlOnGhostBox_(vecfield_type& out, vecfield_type const& in, GridLayoutT const& layout)
    {
        auto& outX = out(component_type::X);
        auto& outY = out(component_type::Y);
        auto& outZ = out(component_type::Z);

        auto& inX = in(component_type::X);
        auto& inY = in(component_type::Y);
        auto& inZ = in(component_type::Z);


        layout.evalOnGhostBox(outX, [&](auto&... args) {
            if constexpr (dimension == 1)
                outX(args...) = 0.0;
            else if constexpr (dimension == 2)
                outX(args...) = layout.template deriv<Direction::Y>(inZ, {args...});
            else
                outX(args...) = layout.template deriv<Direction::Y>(inZ, {args...})
                                - layout.template deriv<Direction::Z>(inY, {args...});
        });
        layout.evalOnGhostBox(outY, [&](auto&... args) {
            if constexpr (dimension == 3)
                outY(args...) = layout.template deriv<Direction::Z>(inX, {args...})
                                - layout.template deriv<Direction::X>(inZ, {args...});
            else
                outY(args...) = -layout.template deriv<Direction::X>(inZ, {args...});
        });
        layout.evalOnGhostBox(outZ, [&](auto&... args) {
            if constexpr (dimension == 1)
                outZ(args...) = layout.template deriv<Direction::X>(inY, {args...});
            else
                outZ(args...) = layout.template deriv<Direction::X>(inY, {args...})
                                - layout.template deriv<Direction::Y>(inX, {args...});
        });
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER
