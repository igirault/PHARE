#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP


#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/numerics/constrained_transport/upwind_edge_b.hpp"
#include "core/utilities/index/index.hpp"

#include <cmath>

namespace PHARE::core
{
using UpwindConstrainedTransportInfo = OhmInfo;

template<typename GridLayout, template<typename> typename Reconstruction>
class UpwindConstrainedTransport : UpwindConstrainedTransportInfo
{
    using Super                     = UpwindConstrainedTransportInfo;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    using EdgeB_t                   = UpwindEdgeB<GridLayout, Reconstruction>;
    constexpr static auto dimension = GridLayout::dimension;
    using Super::hyper_mode;

public:
    using Info_t = Super;

    UpwindConstrainedTransport(UpwindConstrainedTransportInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
    {
    }

    void operator()(auto& ct_state, auto const& non_ideal_electric_state, auto& mhd_state) const
    {
        auto& E       = mhd_state.E;
        auto const& B = mhd_state.B;

        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);

        layout_.evalOnBox(Ex, [&](auto&... args) { ExEq_(ct_state, Ex, B, {args...}); });
        layout_.evalOnBox(Ey, [&](auto&... args) { EyEq_(ct_state, Ey, B, {args...}); });
        layout_.evalOnBox(Ez, [&](auto&... args) { EzEq_(ct_state, Ez, B, {args...}); });

        if (non_ideal_electric_state.active())
        {
            auto const& Enonideal = non_ideal_electric_state.E();
            for (auto c : {Component::X, Component::Y, Component::Z})
                layout_.evalOnBox(E(c),
                                  [&](auto&... args) { E(c)(args...) += Enonideal(c)(args...); });
        }
    }

private:
    void ExEq_(auto& ct_state, auto& Ex, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto const [aL, aR, dL, dR, BzL, BzR]
                = EdgeB_t::template states<Component::X, Component::Z>(ct_state, B, idx);

            auto FL = BzL * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);

            Ex(idx) = -(aL * FL + aR * FR) + (dR * BzR - dL * BzL);
        }
        else if constexpr (dimension == 3)
        {
            auto const [aS, aN, dS, dN, BzS, BzN]
                = EdgeB_t::template states<Component::X, Component::Z>(ct_state, B, idx);
            auto const [aB, aT, dB, dT, ByB, ByT]
                = EdgeB_t::template states<Component::X, Component::Y>(ct_state, B, idx);

            auto [vyS, vyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                ct_state.vt_z(Component::Y), idx);
            auto [vzB, vzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                ct_state.vt_y(Component::Z), idx);

            Ex(idx) = (aB * vzB * ByB + aT * vzT * ByT) - (aS * vyS * BzS + aN * vyN * BzN)
                      - (dT * ByT - dB * ByB) + (dN * BzN - dS * BzS);
        }
    }

    void EyEq_(auto& ct_state, auto& Ey, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension <= 2)
        {
            auto const [aL, aR, dL, dR, BzL, BzR]
                = EdgeB_t::template states<Component::Y, Component::Z>(ct_state, B, idx);

            auto FL = BzL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);

            Ey(idx) = (aL * FL + aR * FR) - (dR * BzR - dL * BzL);
        }
        else if constexpr (dimension == 3)
        {
            auto const [aW, aE, dW, dE, BzW, BzE]
                = EdgeB_t::template states<Component::Y, Component::Z>(ct_state, B, idx);
            auto const [aB, aT, dB, dT, BxB, BxT]
                = EdgeB_t::template states<Component::Y, Component::X>(ct_state, B, idx);

            auto [vxW, vxE] = Reconstruction_t::template reconstruct<Direction::X>(
                ct_state.vt_z(Component::X), idx);
            auto [vzB, vzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                ct_state.vt_x(Component::Z), idx);

            Ey(idx) = (aW * vxW * BzW + aE * vxE * BzE) - (aB * vzB * BxB + aT * vzT * BxT)
                      - (dE * BzE - dW * BzW) + (dT * BxT - dB * BxB);
        }
    }

    void EzEq_(auto& ct_state, auto& Ez, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 1)
        {
            auto const [aL, aR, dL, dR, ByL, ByR]
                = EdgeB_t::template states<Component::Z, Component::Y>(ct_state, B, idx);

            auto FL = ByL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);
            auto FR = ByR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);

            Ez(idx) = -(aL * FL + aR * FR) + (dR * ByR - dL * ByL);
        }
        else if constexpr (dimension >= 2)
        {
            auto const [aW, aE, dW, dE, ByW, ByE]
                = EdgeB_t::template states<Component::Z, Component::Y>(ct_state, B, idx);
            auto const [aS, aN, dS, dN, BxS, BxN]
                = EdgeB_t::template states<Component::Z, Component::X>(ct_state, B, idx);

            auto [vyS, vyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                ct_state.vt_x(Component::Y), idx);
            auto [vxW, vxE] = Reconstruction_t::template reconstruct<Direction::X>(
                ct_state.vt_y(Component::X), idx);

            Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                      + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);
        }
    }

    GridLayout layout_;
};
} // namespace PHARE::core

#endif
