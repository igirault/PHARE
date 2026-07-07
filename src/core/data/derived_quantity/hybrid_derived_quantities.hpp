#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_HYBRID_DERIVED_QUANTITIES_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_HYBRID_DERIVED_QUANTITIES_HPP

#include "core/data/derived_quantity/derived_quantity.hpp"
#include "core/numerics/ampere/ampere.hpp"

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
DerivedQuantityRegistry<State, GridLayout> makeHybridDerivedQuantities()
{
    DerivedQuantityRegistry<State, GridLayout> registry;
    registry.template add<1>(std::make_unique<HybridCurrentDensity<State, GridLayout>>());
    return registry;
}

} // namespace PHARE::core

#endif
