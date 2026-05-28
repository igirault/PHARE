#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/point/point.hpp"
#include "initializer/data_provider.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace PHARE::core
{
template<size_t dim>
constexpr auto getDirections()
{
    if constexpr (dim == 1)
    {
        return std::make_tuple(Direction::X);
    }
    else if constexpr (dim == 2)
    {
        return std::make_tuple(Direction::X, Direction::Y);
    }
    else if constexpr (dim == 3)
    {
        return std::make_tuple(Direction::X, Direction::Y, Direction::Z);
    }
}

template<auto direction, size_t dim>
auto getGrow(int const nghosts)
{
    Point<std::uint32_t, dim> p{};

    auto dir = static_cast<size_t>(direction);

    for (size_t i = 0; i < dim; ++i)
    {
        if (i != dir)
            p[i] = nghosts;
    }

    // always allocate the extra layer for the flux laplacian (hyper-resistivity)
    p[dir] += 1;

    return p;
}

template<typename GridLayout, typename MHDModel, template<typename> typename Reconstruction,
         typename RiemannSolver, typename Equations>
class Godunov : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

    using Reconstruction_t = Reconstruction<GridLayout>;
    using Reconstructor_t  = Reconstructor<Reconstruction_t>;
    using RiemannSolver_t  = RiemannSolver;

public:
    template<typename T>
    using Rec = Reconstruction<T>;

    constexpr static auto Hall = Equations::hall;

    Godunov(PHARE::initializer::PHAREDict const& dict)
        : gamma_{dict["heat_capacity_ratio"].template to<double>()}
        , eta_{dict["resistivity"].template to<double>()}
        , nu_{dict["hyper_resistivity"].template to<double>()}
        , hyper_mode_{cppdict::get_value(dict, "hyper_mode", std::string{"constant"}) == "constant"
                          ? HyperMode::constant
                          : HyperMode::spatial}
        , resistivity_{eta_ != 0.0}
        , hyper_resistivity_{nu_ != 0.0}
        , equations_{gamma_, eta_, nu_}
        , riemann_{gamma_}
    {
    }

    template<typename State, typename Fluxes>
    void operator()(auto& ct, State& state, Fluxes& fluxes)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - GodunovFluxes - GridLayout not set");

        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_->evalOnBiggerBox(
                fluxes.template expose_centering<direction>(),
                getGrow<direction, dimension>(Reconstruction_t::nghosts), [&](auto&... indices) {
                    if constexpr (Hall)
                    {
                        auto&& [uL, uR]
                            = Reconstructor_t::template reconstruct<direction>(state, {indices...});

                        auto const& [jL, jR] = Reconstructor_t::template center_reconstruct<
                            direction, GridLayout::edgeXToCellCenter, GridLayout::edgeYToCellCenter,
                            GridLayout::edgeZToCellCenter>(state.J, {indices...});

                        auto&& u      = std::forward_as_tuple(uL, uR);
                        auto const& j = std::forward_as_tuple(jL, jR);

                        // if constexpr (HyperResistivity)
                        // {
                        //     auto const& [laplJL, laplJR]
                        //         = Reconstructor_t::template reconstructed_laplacian<direction>(
                        //             layout_->inverseMeshSize(), state.J, {indices...});
                        //
                        //     auto const& LaplJ = std::forward_as_tuple(laplJL, laplJR);
                        //
                        //     auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i)
                        //     {
                        //         return equations_.template compute<direction>(
                        //             std::get<i>(u), std::get<i>(j), std::get<i>(LaplJ));
                        //     });
                        //
                        //     fluxes.template get_dir<direction>({indices...})
                        //         = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);
                        //
                        //     ct.template save<direction>(riemann_.vt, riemann_.jt,
                        //                                 riemann_.rhot, riemann_.uct_coefs,
                        //                                 {indices...});
                        // }
                        // else
                        // {
                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u),
                                                                          std::get<i>(j));
                        });

                        // if constexpr (Hall)
                        // {
                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                        ct.template save<direction>(riemann_.vt, riemann_.jt, riemann_.rhot,
                                                    riemann_.uct_coefs, {indices...});
                    }
                    else // Ideal
                    {
                        auto&& [uL, uR]
                            = Reconstructor_t::template reconstruct<direction>(state, {indices...});

                        auto&& u = std::forward_as_tuple(uL, uR);

                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR);

                        ct.template save<direction>(riemann_.vt, riemann_.uct_coefs, {indices...});
                    }
                });

            // Note: resistive contributions to F_B and F_Etot are now handled via CT:
            // - B is updated by Faraday using E (which includes ηJ from CT)
            // - Etot gets the resistive Poynting flux via E×B where E = E_ideal + ηJ
            // The old resistive_contributions code was dead (F_B not used) and would
            // double-count η(J×B) in F_Etot when combined with Poynting correction.
        });
    }

    void registerResources(MHDModel& model) {}

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const {}

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::forward_as_tuple(); }

    template<typename CT, typename State, typename Fluxes>
    void apply_poynting_correction(CT const& ct, State const& state, Fluxes& fluxes)
    {
        // Apply Poynting flux correction to perturbation energy Etot1:
        //   ∂Etot1/∂t -= ∇·(E × B1)
        // Must be called AFTER CT has computed both E and edge-B1 fields.
        // B-split: E from Ohm's law uses total B (B0+B1); the energy variable Etot1
        // (perturbation) has Poynting flux E × B1, so we use edge-centered B1 here.
        if (!this->hasLayout())
            throw std::runtime_error(
                "Error - GodunovFluxes::apply_poynting_correction - GridLayout not set");

        constexpr auto directions     = getDirections<dimension>();
        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_->evalOnBox(
                fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                    auto& F_Etot = fluxes.template get_dir<direction>({indices...}).Etot();
                    poynting_energy_flux_<direction>(ct, state.E, MeshIndex<dimension>{indices...},
                                                     F_Etot);
                });
        });
    }

    bool resistivity() const { return resistivity_; }
    bool hyper_resistivity() const { return hyper_resistivity_; }

private:
    template<auto direction>
    void poynting_energy_flux_(auto const& ct, auto const& E, MeshIndex<dimension> const& index,
                               auto& F_Etot) const
    {
        // Compute perturbation magnetic energy flux via Poynting vector: S·n̂ = (E × B1)·n̂
        // E components live on edges (from CT, computed via Ohm's law with total B)
        // B1 components are edge-centered (from CT, temporally consistent with E)
        //
        // B-split: Poynting flux for Etot1 uses perturbation B1, not total B,
        // since the static background B0 must not be transported.

        auto const& Ex = E(Component::X);
        auto const& Ey = E(Component::Y);
        auto const& Ez = E(Component::Z);

        if constexpr (direction == Direction::X && dimension >= 2)
        {
            // X-flux face: Sx = Ey*B1z - Ez*B1y
            auto const& B1y_at_Ez = ct.getB1y_at_Ez();
            auto const& B1z_at_Ey = ct.getB1z_at_Ey();

            double EzBy = 0.5
                          * (Ez(index) * B1y_at_Ez(index)
                             + Ez(layout_->template next<Direction::Y>(index))
                                   * B1y_at_Ez(layout_->template next<Direction::Y>(index)));

            double EyBz;
            if constexpr (dimension == 2)
            {
                EyBz = Ey(index) * B1z_at_Ey(index);
            }
            else
            {
                EyBz = 0.5
                       * (Ey(index) * B1z_at_Ey(index)
                          + Ey(layout_->template next<Direction::Z>(index))
                                * B1z_at_Ey(layout_->template next<Direction::Z>(index)));
            }

            F_Etot += EyBz - EzBy;
        }
        else if constexpr (direction == Direction::Y && dimension >= 2)
        {
            // Y-flux face: Sy = Ez*B1x - Ex*B1z
            auto const& B1x_at_Ez = ct.getB1x_at_Ez();
            auto const& B1z_at_Ex = ct.getB1z_at_Ex();

            double EzBx = 0.5
                          * (Ez(index) * B1x_at_Ez(index)
                             + Ez(layout_->template next<Direction::X>(index))
                                   * B1x_at_Ez(layout_->template next<Direction::X>(index)));

            double ExBz;
            if constexpr (dimension == 2)
            {
                ExBz = Ex(index) * B1z_at_Ex(index);
            }
            else
            {
                ExBz = 0.5
                       * (Ex(index) * B1z_at_Ex(index)
                          + Ex(layout_->template next<Direction::Z>(index))
                                * B1z_at_Ex(layout_->template next<Direction::Z>(index)));
            }

            F_Etot += EzBx - ExBz;
        }
        else if constexpr (direction == Direction::Z && dimension == 3)
        {
            // Z-flux face: Sz = Ex*B1y - Ey*B1x
            auto const& B1y_at_Ex = ct.getB1y_at_Ex();
            auto const& B1x_at_Ey = ct.getB1x_at_Ey();

            double ExBy = 0.5
                          * (Ex(index) * B1y_at_Ex(index)
                             + Ex(layout_->template next<Direction::Y>(index))
                                   * B1y_at_Ex(layout_->template next<Direction::Y>(index)));

            double EyBx = 0.5
                          * (Ey(index) * B1x_at_Ey(index)
                             + Ey(layout_->template next<Direction::X>(index))
                                   * B1x_at_Ey(layout_->template next<Direction::X>(index)));

            F_Etot += ExBy - EyBx;
        }
        else if constexpr (direction == Direction::X && dimension == 1)
        {
            // In 1D, Ey/Ez (y/z-edges) and B1_*_at_E* (CT-upwinded to x-face) all live
            // at the x-flux face — no averaging needed, same pattern as the 2D case above.
            // Etot1 is the perturbation energy, so its Poynting flux uses B1 (not total B).
            auto const& B1y_at_Ez = ct.getB1y_at_Ez();
            auto const& B1z_at_Ey = ct.getB1z_at_Ey();
            F_Etot += Ey(index) * B1z_at_Ey(index) - Ez(index) * B1y_at_Ez(index);
        }
    }


    double const gamma_;
    double const eta_;
    double const nu_;
    HyperMode const hyper_mode_;
    bool const resistivity_;
    bool const hyper_resistivity_;

    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
