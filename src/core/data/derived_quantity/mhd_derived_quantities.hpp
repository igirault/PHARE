#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_MHD_DERIVED_QUANTITIES_HPP

#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

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
class MhdElectricField : public DerivedQuantity<State, GridLayout, 1>
{
    using Super                     = DerivedQuantity<State, GridLayout, 1>;
    constexpr static auto dimension = GridLayout::dimension;

public:
    using typename Super::out_t;

    MhdElectricField(double const eta, double const nu, HyperMode const hyper_mode)
        : eta_{eta}
        , nu_{nu}
        , hyper_mode_{hyper_mode}
    {
    }

    std::string name() const override { return "E"; }
    VectorCentering centering() const override { return VectorCentering::Elike; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        // A real, heap-backed local J (Field/VecField are non-owning views, so we
        // need a Grid to actually own the storage; mirrors the pattern used by
        // UsableTensorFieldMHD test fixtures, but here in production code).
        using Grid_t = Grid<NdArrayVector<dimension>, MHDQuantity::Scalar>;

        out_t J{"J_tmp_for_E", MHDQuantity::Vector::J};
        auto const qts = MHDQuantity::componentsQuantities(MHDQuantity::Vector::J);
        std::array<Grid_t, 3> storage{
            Grid_t{J.getComponentName(Component::X), qts[0], layout.allocSize(qts[0])},
            Grid_t{J.getComponentName(Component::Y), qts[1], layout.allocSize(qts[1])},
            Grid_t{J.getComponentName(Component::Z), qts[2], layout.allocSize(qts[2])}};
        for (std::size_t i = 0; i < 3; ++i)
            J[i].setBuffer(&storage[i]);

        Ampere<GridLayout>{layout}(state.B, J);

        auto& Ex = out(Component::X);
        auto& Ey = out(Component::Y);
        auto& Ez = out(Component::Z);

        // Shrink by 2: Ampere itself only fills J up to 1 cell from the edge of
        // B's ghost box (see Ampere::operator()'s own shrink-by-1), and the
        // hyper-resistive term below takes a laplacian of J (touching J's
        // neighbors), so E can only be safely evaluated 1 cell further in than J.
        Point<std::uint32_t, dimension> shrink;
        for (std::size_t i = 0; i < dimension; ++i)
            shrink[i] = 2;

        layout.evalOnShrinkedGhostBox(Ex, shrink, [&](auto const&... args) {
            point_<Component::X>(state, layout, J, MeshIndex<dimension>{args...}, Ex);
        });
        layout.evalOnShrinkedGhostBox(Ey, shrink, [&](auto const&... args) {
            point_<Component::Y>(state, layout, J, MeshIndex<dimension>{args...}, Ey);
        });
        layout.evalOnShrinkedGhostBox(Ez, shrink, [&](auto const&... args) {
            point_<Component::Z>(state, layout, J, MeshIndex<dimension>{args...}, Ez);
        });
    }

private:
    template<auto component, typename Field>
    void point_(State const& state, GridLayout const& layout, out_t const& J,
                MeshIndex<dimension> const index, Field& E) const
    {
        auto&& [vx, vy, vz]
            = rhoVToV(state.rho(index), state.rhoV(Component::X)(index),
                      state.rhoV(Component::Y)(index), state.rhoV(Component::Z)(index));

        E(index) = ideal_<component>(state, layout, vx, vy, vz, index)
                   + hall_<component>(state, layout, J, index) + eta_ * J(component)(index)
                   + hyperresistive_<component>(state, layout, J(component), index);
    }

    template<auto component>
    auto ideal_(State const& state, GridLayout const& layout, double const vx, double const vy,
                double const vz, MeshIndex<dimension> const index) const
    {
        auto const& B = state.B;
        if constexpr (component == Component::X)
        {
            auto const by
                = GridLayout::template project<GridLayout::ByToEx>(B(Component::Y), index);
            auto const bz
                = GridLayout::template project<GridLayout::BzToEx>(B(Component::Z), index);
            return -vy * bz + vz * by;
        }
        if constexpr (component == Component::Y)
        {
            auto const bx
                = GridLayout::template project<GridLayout::BxToEy>(B(Component::X), index);
            auto const bz
                = GridLayout::template project<GridLayout::BzToEy>(B(Component::Z), index);
            return -vz * bx + vx * bz;
        }
        if constexpr (component == Component::Z)
        {
            auto const bx
                = GridLayout::template project<GridLayout::BxToEz>(B(Component::X), index);
            auto const by
                = GridLayout::template project<GridLayout::ByToEz>(B(Component::Y), index);
            return -vx * by + vy * bx;
        }
    }

    template<auto component>
    auto hall_(State const& state, GridLayout const& layout, out_t const& J,
               MeshIndex<dimension> const index) const
    {
        auto const& B = state.B;
        auto const rhoE
            = GridLayout::template project<GridLayout::cellCenterToEdgeX>(state.rho, index);
        if constexpr (component == Component::X)
        {
            auto const by
                = GridLayout::template project<GridLayout::ByToEx>(B(Component::Y), index);
            auto const bz
                = GridLayout::template project<GridLayout::BzToEx>(B(Component::Z), index);
            return (J(Component::Y)(index) * bz - J(Component::Z)(index) * by) / rhoE;
        }
        if constexpr (component == Component::Y)
        {
            auto const bx
                = GridLayout::template project<GridLayout::BxToEy>(B(Component::X), index);
            auto const bz
                = GridLayout::template project<GridLayout::BzToEy>(B(Component::Z), index);
            return (J(Component::Z)(index) * bx - J(Component::X)(index) * bz) / rhoE;
        }
        if constexpr (component == Component::Z)
        {
            auto const bx
                = GridLayout::template project<GridLayout::BxToEz>(B(Component::X), index);
            auto const by
                = GridLayout::template project<GridLayout::ByToEz>(B(Component::Y), index);
            return (J(Component::X)(index) * by - J(Component::Y)(index) * bx) / rhoE;
        }
    }

    template<auto component, typename Field>
    auto hyperresistive_(State const& state, GridLayout const& layout, Field const& Jc,
                         MeshIndex<dimension> const index) const
    {
        if (hyper_mode_ == HyperMode::constant)
            return -nu_ * layout.laplacian(Jc, index);

        auto const& B = state.B;
        auto const bx = GridLayout::template project<GridLayout::BxToEx>(B(Component::X), index);
        auto const by = GridLayout::template project<GridLayout::ByToEx>(B(Component::Y), index);
        auto const bz = GridLayout::template project<GridLayout::BzToEx>(B(Component::Z), index);
        auto const rho
            = GridLayout::template project<GridLayout::cellCenterToEdgeX>(state.rho, index);
        auto const b = std::sqrt(bx * bx + by * by + bz * bz);
        return -nu_ * (b / rho + 1.0) * layout.laplacian(Jc, index);
    }

    double const eta_;
    double const nu_;
    HyperMode const hyper_mode_;
};


template<typename State, typename GridLayout>
DerivedQuantityRegistry<State, GridLayout>
makeMhdDerivedQuantities(double const gamma, double const eta, double const nu,
                         HyperMode const hyper_mode)
{
    DerivedQuantityRegistry<State, GridLayout> registry;
    registry.template add<1>(std::make_unique<MhdVelocity<State, GridLayout>>());
    registry.template add<0>(std::make_unique<MhdPressure<State, GridLayout>>(gamma));
    registry.template add<0>(std::make_unique<MhdDivB<State, GridLayout>>());
    registry.template add<1>(std::make_unique<MhdCurrentDensity<State, GridLayout>>());
    registry.template add<1>(
        std::make_unique<MhdElectricField<State, GridLayout>>(eta, nu, hyper_mode));
    return registry;
}

} // namespace PHARE::core

#endif
