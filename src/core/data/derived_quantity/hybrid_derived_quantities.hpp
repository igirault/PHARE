#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_HYBRID_DERIVED_QUANTITIES_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_HYBRID_DERIVED_QUANTITIES_HPP

#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/utilities/index/index.hpp"

#include <memory>
#include <string>

namespace PHARE::core
{
template<typename State, typename GridLayout>
class HybridCurrentDensity : public DerivedQuantity<State, GridLayout, 1>
{
    using Super = DerivedQuantity<State, GridLayout, 1>;

public:
    using typename Super::out_t;

    std::string name() const override { return "J"; }
    VectorCentering centering() const override { return VectorCentering::Elike; }
    DerivedCategory category() const override { return DerivedCategory::electromag; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        Ampere<GridLayout>{layout}(state.electromag.B, out);
    }
};


template<typename State, typename GridLayout>
class HybridDivB : public DerivedQuantity<State, GridLayout, 0>
{
    using Super = DerivedQuantity<State, GridLayout, 0>;

public:
    using typename Super::out_t;

    std::string name() const override { return "divB"; }
    ScalarCentering centering() const override { return ScalarCentering::cell; }
    DerivedCategory category() const override { return DerivedCategory::electromag; }

    void compute(State const& state, GridLayout const& layout, out_t& out,
                 double /*time*/) const override
    {
        auto const& Bx = state.electromag.B(Component::X);
        auto const& By = state.electromag.B(Component::Y);
        auto const& Bz = state.electromag.B(Component::Z);

        layout.evalOnGhostBox(out, [&](auto const&... args) {
            MeshIndex<GridLayout::dimension> const index{args...};
            out(index) = layout.template deriv<Direction::X>(Bx, index);
            if constexpr (GridLayout::dimension >= 2)
                out(index) += layout.template deriv<Direction::Y>(By, index);
            if constexpr (GridLayout::dimension == 3)
                out(index) += layout.template deriv<Direction::Z>(Bz, index);
        });
    }
};


template<typename State, typename GridLayout>
DerivedQuantityRegistry<State, GridLayout> makeHybridDerivedQuantities()
{
    DerivedQuantityRegistry<State, GridLayout> registry;
    registry.template add<1>(std::make_unique<HybridCurrentDensity<State, GridLayout>>());
    registry.template add<0>(std::make_unique<HybridDivB<State, GridLayout>>());
    return registry;
}

} // namespace PHARE::core

#endif
