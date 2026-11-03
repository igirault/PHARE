#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP


#include "core/numerics/reconstructions/constant.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/ohm/ohm.hpp"
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
    constexpr static auto dimension = GridLayout::dimension;
    using Super::hyper_mode;

public:
    using Info_t = Super;

    UpwindConstrainedTransport(UpwindConstrainedTransportInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
    {
    }

    void operator()(auto& ct_state, auto const& non_ideal_electric_state, auto& mhd_state,
                    auto const& B0) const
    {
        auto& E        = mhd_state.E;
        auto const& B1 = mhd_state.B1;

        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);

        layout_.evalOnBox(Ex, [&](auto&... args) { ExEq_(ct_state, Ex, B1, B0, {args...}); });
        layout_.evalOnBox(Ey, [&](auto&... args) { EyEq_(ct_state, Ey, B1, B0, {args...}); });
        layout_.evalOnBox(Ez, [&](auto&... args) { EzEq_(ct_state, Ez, B1, B0, {args...}); });

        if (non_ideal_electric_state.active())
        {
            auto const& Enonideal = non_ideal_electric_state.E();
            for (auto c : {Component::X, Component::Y, Component::Z})
                layout_.evalOnBox(E(c),
                                  [&](auto&... args) { E(c)(args...) += Enonideal(c)(args...); });
        }
    }

    template<typename MeshData>
    void degrade_E_near_inner_boundary(auto& ct_state, auto& mhd_state, auto const& B0,
                                       MeshData const& meshData) const
    {
        using Constant = ConstantReconstruction<GridLayout>;

        auto& E        = mhd_state.E;
        auto const& B1 = mhd_state.B1;

        auto degrade = [&](auto& Ec, auto&& recompute) {
            for (auto const& elem : meshData.getDegradedElemsFromCentering(layout_.centering(Ec)))
                recompute(elem.index.template as<std::size_t>());
        };

        degrade(E(Component::X),
                [&](auto const& idx) { ExEq_<Constant>(ct_state, E(Component::X), B1, B0, idx); });
        degrade(E(Component::Y),
                [&](auto const& idx) { EyEq_<Constant>(ct_state, E(Component::Y), B1, B0, idx); });
        degrade(E(Component::Z),
                [&](auto const& idx) { EzEq_<Constant>(ct_state, E(Component::Z), B1, B0, idx); });
    }

private:
    template<auto direction, auto src, auto dst, typename Rec = Reconstruction_t>
    static auto reconstructTotal_(auto const& B1, auto const& B0, Component component,
                                  MeshIndex<dimension> const& idx)
    {
        auto const [B1L, B1R] = Rec::template reconstruct<direction>(B1(component), idx);
        auto const b0 = GridLayout::template project<src, dst>(B0(component), idx);
        return std::make_pair(B1L + b0, B1R + b0);
    }

    template<auto src, auto dst>
    static auto totalOnEdge_(auto const& B1, auto const& B0, Component component,
                             MeshIndex<dimension> const& idx)
    {
        return B1(component)(idx) + GridLayout::template project<src, dst>(B0(component), idx);
    }

    template<typename Rec = Reconstruction_t>
    void ExEq_(auto& ct_state, auto& Ex, auto const& B1, auto const& B0,
               MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto [BzL, BzR]
                = reconstructTotal_<Direction::Y, MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::Ex,
                                    Rec>(B1, B0, Component::Z, idx);
            auto const By = totalOnEdge_<MHDQuantity::Scalar::By, MHDQuantity::Scalar::Ex>(
                B1, B0, Component::Y, idx);

            auto FL
                = BzL * ct_state.vt_y(Component::Y)(idx) - By * ct_state.vt_y(Component::Z)(idx);
            auto FR
                = BzR * ct_state.vt_y(Component::Y)(idx) - By * ct_state.vt_y(Component::Z)(idx);

            Ex(idx) = -(ct_state.aL_y(idx) * FL + ct_state.aR_y(idx) * FR)
                      + (ct_state.dR_y(idx) * BzR - ct_state.dL_y(idx) * BzL);
        }
        else if constexpr (dimension == 3)
        {
            auto aS = 0.5
                      * (ct_state.aL_y(idx)
                         + ct_state.aL_y(layout_.template previous<Direction::Z>(idx)));
            auto aN = 0.5
                      * (ct_state.aR_y(idx)
                         + ct_state.aR_y(layout_.template previous<Direction::Z>(idx)));
            auto aB = 0.5
                      * (ct_state.aL_z(idx)
                         + ct_state.aL_z(layout_.template previous<Direction::Y>(idx)));
            auto aT = 0.5
                      * (ct_state.aR_z(idx)
                         + ct_state.aR_z(layout_.template previous<Direction::Y>(idx)));
            auto dS = 0.5
                      * (ct_state.dL_y(idx)
                         + ct_state.dL_y(layout_.template previous<Direction::Z>(idx)));
            auto dN = 0.5
                      * (ct_state.dR_y(idx)
                         + ct_state.dR_y(layout_.template previous<Direction::Z>(idx)));
            auto dB = 0.5
                      * (ct_state.dL_z(idx)
                         + ct_state.dL_z(layout_.template previous<Direction::Y>(idx)));
            auto dT = 0.5
                      * (ct_state.dR_z(idx)
                         + ct_state.dR_z(layout_.template previous<Direction::Y>(idx)));

            auto [vyS, vyN]
                = Rec::template reconstruct<Direction::Y>(ct_state.vt_z(Component::Y), idx);
            auto [vzB, vzT]
                = Rec::template reconstruct<Direction::Z>(ct_state.vt_y(Component::Z), idx);

            auto [BzS, BzN]
                = reconstructTotal_<Direction::Y, MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::Ex,
                                    Rec>(B1, B0, Component::Z, idx);
            auto [ByB, ByT]
                = reconstructTotal_<Direction::Z, MHDQuantity::Scalar::By, MHDQuantity::Scalar::Ex,
                                    Rec>(B1, B0, Component::Y, idx);

            Ex(idx) = (aB * vzB * ByB + aT * vzT * ByT) - (aS * vyS * BzS + aN * vyN * BzN)
                      - (dT * ByT - dB * ByB) + (dN * BzN - dS * BzS);
        }
    }

    template<typename Rec = Reconstruction_t>
    void EyEq_(auto& ct_state, auto& Ey, auto const& B1, auto const& B0,
               MeshIndex<dimension> idx) const
    {
        if constexpr (dimension <= 2)
        {
            auto [BzL, BzR]
                = reconstructTotal_<Direction::X, MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::Ey,
                                    Rec>(B1, B0, Component::Z, idx);
            auto const Bx = totalOnEdge_<MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::Ey>(
                B1, B0, Component::X, idx);

            auto FL
                = BzL * ct_state.vt_x(Component::X)(idx) - Bx * ct_state.vt_x(Component::Z)(idx);
            auto FR
                = BzR * ct_state.vt_x(Component::X)(idx) - Bx * ct_state.vt_x(Component::Z)(idx);

            Ey(idx) = (ct_state.aL_x(idx) * FL + ct_state.aR_x(idx) * FR)
                      - (ct_state.dR_x(idx) * BzR - ct_state.dL_x(idx) * BzL);
        }
        else if constexpr (dimension == 3)
        {
            auto aW = 0.5
                      * (ct_state.aL_x(idx)
                         + ct_state.aL_x(layout_.template previous<Direction::Z>(idx)));
            auto aE = 0.5
                      * (ct_state.aR_x(idx)
                         + ct_state.aR_x(layout_.template previous<Direction::Z>(idx)));
            auto aB = 0.5
                      * (ct_state.aL_z(idx)
                         + ct_state.aL_z(layout_.template previous<Direction::X>(idx)));
            auto aT = 0.5
                      * (ct_state.aR_z(idx)
                         + ct_state.aR_z(layout_.template previous<Direction::X>(idx)));
            auto dW = 0.5
                      * (ct_state.dL_x(idx)
                         + ct_state.dL_x(layout_.template previous<Direction::Z>(idx)));
            auto dE = 0.5
                      * (ct_state.dR_x(idx)
                         + ct_state.dR_x(layout_.template previous<Direction::Z>(idx)));
            auto dB = 0.5
                      * (ct_state.dL_z(idx)
                         + ct_state.dL_z(layout_.template previous<Direction::X>(idx)));
            auto dT = 0.5
                      * (ct_state.dR_z(idx)
                         + ct_state.dR_z(layout_.template previous<Direction::X>(idx)));

            auto [vxW, vxE]
                = Rec::template reconstruct<Direction::X>(ct_state.vt_z(Component::X), idx);
            auto [vzB, vzT]
                = Rec::template reconstruct<Direction::Z>(ct_state.vt_x(Component::Z), idx);
            auto [BzW, BzE]
                = reconstructTotal_<Direction::X, MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::Ey,
                                    Rec>(B1, B0, Component::Z, idx);
            auto [BxB, BxT]
                = reconstructTotal_<Direction::Z, MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::Ey,
                                    Rec>(B1, B0, Component::X, idx);

            Ey(idx) = (aW * vxW * BzW + aE * vxE * BzE) - (aB * vzB * BxB + aT * vzT * BxT)
                      - (dE * BzE - dW * BzW) + (dT * BxT - dB * BxB);
        }
    }

    template<typename Rec = Reconstruction_t>
    void EzEq_(auto& ct_state, auto& Ez, auto const& B1, auto const& B0,
               MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 1)
        {
            auto [ByL, ByR]
                = reconstructTotal_<Direction::X, MHDQuantity::Scalar::By, MHDQuantity::Scalar::Ez,
                                    Rec>(B1, B0, Component::Y, idx);
            auto const Bx = totalOnEdge_<MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::Ez>(
                B1, B0, Component::X, idx);

            auto FL
                = ByL * ct_state.vt_x(Component::X)(idx) - Bx * ct_state.vt_x(Component::Y)(idx);
            auto FR
                = ByR * ct_state.vt_x(Component::X)(idx) - Bx * ct_state.vt_x(Component::Y)(idx);

            Ez(idx) = -(ct_state.aL_x(idx) * FL + ct_state.aR_x(idx) * FR)
                      + (ct_state.dR_x(idx) * ByR - ct_state.dL_x(idx) * ByL);
        }
        else if constexpr (dimension >= 2)
        {
            auto aW = 0.5
                      * (ct_state.aL_x(idx)
                         + ct_state.aL_x(layout_.template previous<Direction::Y>(idx)));
            auto aE = 0.5
                      * (ct_state.aR_x(idx)
                         + ct_state.aR_x(layout_.template previous<Direction::Y>(idx)));
            auto aS = 0.5
                      * (ct_state.aL_y(idx)
                         + ct_state.aL_y(layout_.template previous<Direction::X>(idx)));
            auto aN = 0.5
                      * (ct_state.aR_y(idx)
                         + ct_state.aR_y(layout_.template previous<Direction::X>(idx)));
            auto dW = 0.5
                      * (ct_state.dL_x(idx)
                         + ct_state.dL_x(layout_.template previous<Direction::Y>(idx)));
            auto dE = 0.5
                      * (ct_state.dR_x(idx)
                         + ct_state.dR_x(layout_.template previous<Direction::Y>(idx)));
            auto dS = 0.5
                      * (ct_state.dL_y(idx)
                         + ct_state.dL_y(layout_.template previous<Direction::X>(idx)));
            auto dN = 0.5
                      * (ct_state.dR_y(idx)
                         + ct_state.dR_y(layout_.template previous<Direction::X>(idx)));

            auto [vyS, vyN]
                = Rec::template reconstruct<Direction::Y>(ct_state.vt_x(Component::Y), idx);
            auto [vxW, vxE]
                = Rec::template reconstruct<Direction::X>(ct_state.vt_y(Component::X), idx);

            auto [BxS, BxN]
                = reconstructTotal_<Direction::Y, MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::Ez,
                                    Rec>(B1, B0, Component::X, idx);
            auto [ByW, ByE]
                = reconstructTotal_<Direction::X, MHDQuantity::Scalar::By, MHDQuantity::Scalar::Ez,
                                    Rec>(B1, B0, Component::Y, idx);

            Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                      + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);
        }
    }

    GridLayout layout_;
};
} // namespace PHARE::core

#endif
