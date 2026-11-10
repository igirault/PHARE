#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_CORRECTION_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_CORRECTION_HPP

#include "core/utilities/types.hpp"
#include "core/utilities/index/index.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/constrained_transport/upwind_edge_b.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/def.hpp"

namespace PHARE::core
{
template<typename GridLayout, template<typename> typename Reconstruction>
class PoyntingCorrection
{
    constexpr static auto dimension = GridLayout::dimension;
    using EdgeB_t                   = UpwindEdgeB<GridLayout, Reconstruction>;

public:
    explicit PoyntingCorrection(GridLayout const& layout)
        : layout_{layout}
    {
    }

    void operator()(auto const& ct_state, auto const& state, auto& fluxes) const
    {
        for_N<dimension>([&](auto i) {
            constexpr auto direction = static_cast<Direction>(i());

            layout_.evalOnBox(fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                auto& F_Etot = fluxes.template get_dir<direction>({indices...}).Etot();
                poynting_energy_flux_<direction>(ct_state, state.E, state.B,
                                                 MeshIndex<dimension>{indices...}, F_Etot);
            });
        });
    }

private:
    template<typename Integrand>
    struct EdgeIntegrand
    {
        using type                      = double;
        constexpr static auto dimension = GridLayout::dimension;

        NO_DISCARD double operator()(MeshIndex<dimension> const& i) const { return integrand(i); }

        Integrand integrand;
    };

    template<auto edgeE, auto face, Component edge, Component component>
    static double edge_product_on_face_(auto const& ct_state, auto const& Ec, auto const& B,
                                        MeshIndex<dimension> const& index)
    {
        auto const product = [&](MeshIndex<dimension> const& i) {
            return Ec(i) * EdgeB_t::template at<edge, component>(ct_state, B, i);
        };
        return core::project<GridLayout::centering(edgeE), GridLayout::centering(face)>(
            EdgeIntegrand<decltype(product)>{product}, index);
    }

    template<auto direction>
    void poynting_energy_flux_(auto const& ct_state, auto const& E, auto const& B,
                               MeshIndex<dimension> const& index, auto& F_Etot) const
    {
        using Scalar = MHDQuantity::Scalar;

        auto const& Ex = E(Component::X);
        auto const& Ey = E(Component::Y);
        auto const& Ez = E(Component::Z);

        if constexpr (direction == Direction::X)
            F_Etot += edge_product_on_face_<Scalar::Ey, Scalar::Bx, Component::Y, Component::Z>(
                          ct_state, Ey, B, index)
                      - edge_product_on_face_<Scalar::Ez, Scalar::Bx, Component::Z, Component::Y>(
                          ct_state, Ez, B, index);
        else if constexpr (direction == Direction::Y)
            F_Etot += edge_product_on_face_<Scalar::Ez, Scalar::By, Component::Z, Component::X>(
                          ct_state, Ez, B, index)
                      - edge_product_on_face_<Scalar::Ex, Scalar::By, Component::X, Component::Z>(
                          ct_state, Ex, B, index);
        else
            F_Etot += edge_product_on_face_<Scalar::Ex, Scalar::Bz, Component::X, Component::Y>(
                          ct_state, Ex, B, index)
                      - edge_product_on_face_<Scalar::Ey, Scalar::Bz, Component::Y, Component::X>(
                          ct_state, Ey, B, index);
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
