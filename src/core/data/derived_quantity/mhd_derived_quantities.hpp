#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP

#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/utilities/index/index.hpp"

#include <memory>
#include <string>

namespace PHARE::core
{
template<typename State, typename GridLayout>
class MhdVelocity : public DerivedQuantity<State, GridLayout, 1>
{
    using Super = DerivedQuantity<State, GridLayout, 1>;

public:
    using typename Super::out_t;

    std::string name() const override { return "V"; }
    VectorCentering centering() const override { return VectorCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        auto& Vx = out(Component::X);
        auto& Vy = out(Component::Y);
        auto& Vz = out(Component::Z);

        layout.evalOnGhostBox(Vx,
                              [&](auto const&... args) { point_(state, Vx, Vy, Vz, {args...}); });
    }

private:
    static void point_(State const& state, auto& Vx, auto& Vy, auto& Vz,
                       MeshIndex<GridLayout::dimension> const index)
    {
        auto&& [vx, vy, vz]
            = rhoVToV(state.rho(index), state.rhoV(Component::X)(index),
                      state.rhoV(Component::Y)(index), state.rhoV(Component::Z)(index));
        Vx(index) = vx;
        Vy(index) = vy;
        Vz(index) = vz;
    }
};


template<typename State, typename GridLayout>
class MhdPressure : public DerivedQuantity<State, GridLayout, 0>
{
    using Super = DerivedQuantity<State, GridLayout, 0>;

public:
    using typename Super::out_t;

    explicit MhdPressure(double const gamma)
        : gamma_{gamma}
    {
    }

    std::string name() const override { return "P"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        layout.evalOnGhostBox(out,
                              [&](auto const&... args) { point_(gamma_, state, out, {args...}); });
    }

private:
    static void point_(double const gamma, State const& state, out_t& P,
                       MeshIndex<GridLayout::dimension> const index)
    {
        auto const& rho = state.rho;
        auto const vx   = state.rhoV(Component::X)(index) / rho(index);
        auto const vy   = state.rhoV(Component::Y)(index) / rho(index);
        auto const vz   = state.rhoV(Component::Z)(index) / rho(index);

        auto const bx = GridLayout::template project<GridLayout::faceXToCellCenter>(
            state.B(Component::X), index);
        auto const by = GridLayout::template project<GridLayout::faceYToCellCenter>(
            state.B(Component::Y), index);
        auto const bz = GridLayout::template project<GridLayout::faceZToCellCenter>(
            state.B(Component::Z), index);

        auto const etot = state.Etot(index);
        P(index)        = eosEtotToP(gamma, rho(index), vx, vy, vz, bx, by, bz, etot);
    }

    double gamma_;
};


template<typename State, typename GridLayout>
class MhdDivB : public DerivedQuantity<State, GridLayout, 0>
{
    using Super = DerivedQuantity<State, GridLayout, 0>;

public:
    using typename Super::out_t;

    std::string name() const override { return "divB"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        auto const& Bx = state.B(Component::X);
        auto const& By = state.B(Component::Y);
        auto const& Bz = state.B(Component::Z);

        layout.evalOnGhostBox(
            out, [&](auto const&... args) { point_(layout, Bx, By, Bz, out, {args...}); });
    }

private:
    static void point_(GridLayout const& layout, auto const& Bx, auto const& By, auto const& Bz,
                       out_t& out, MeshIndex<GridLayout::dimension> const index)
    {
        out(index) = layout.template deriv<Direction::X>(Bx, index);
        if constexpr (GridLayout::dimension >= 2)
            out(index) += layout.template deriv<Direction::Y>(By, index);
        if constexpr (GridLayout::dimension == 3)
            out(index) += layout.template deriv<Direction::Z>(Bz, index);
    }
};


template<typename State, typename GridLayout>
class MhdCurrentDensity : public DerivedQuantity<State, GridLayout, 1>
{
    using Super = DerivedQuantity<State, GridLayout, 1>;

public:
    using typename Super::out_t;

    std::string name() const override { return "J"; }
    VectorCentering centering() const override { return VectorCentering::Elike; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        Ampere<GridLayout>{layout}(state.B, out);
    }
};


template<typename State, typename GridLayout>
DerivedQuantityRegistry<State, GridLayout> makeMhdDerivedQuantities(double const gamma)
{
    DerivedQuantityRegistry<State, GridLayout> registry;
    registry.template add<1>(std::make_unique<MhdVelocity<State, GridLayout>>());
    registry.template add<0>(std::make_unique<MhdPressure<State, GridLayout>>(gamma));
    registry.template add<0>(std::make_unique<MhdDivB<State, GridLayout>>());
    registry.template add<1>(std::make_unique<MhdCurrentDensity<State, GridLayout>>());
    return registry;
}

} // namespace PHARE::core

#endif
