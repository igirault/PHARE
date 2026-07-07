#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_QUANTITY_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_DERIVED_QUANTITY_HPP

#include "core/data/derived_quantity/centering.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace PHARE::core
{
template<typename State, typename GridLayout, std::size_t rank>
struct derived_traits;

template<typename State, typename GridLayout>
struct derived_traits<State, GridLayout, 0>
{
    using out_t       = typename State::field_type;
    using centering_t = ScalarCentering;
};

template<typename State, typename GridLayout>
struct derived_traits<State, GridLayout, 1>
{
    using out_t       = typename State::vecfield_type;
    using centering_t = VectorCentering;
};


/** Diagnostic family a derived quantity is published under: "fluid" quantities
 *  live in the model tree (e.g. /mhd/), "electromag" ones under the EM_ names. */
enum class DerivedCategory { fluid, electromag };


/** Interface for computing a derived (post-processed) quantity from the primary
 *  variables of a State into a caller-provided buffer, over the ghost box. */
template<typename State, typename GridLayout, std::size_t rank>
class DerivedQuantity
{
    using traits = derived_traits<State, GridLayout, rank>;

public:
    using out_t       = typename traits::out_t;
    using centering_t = typename traits::centering_t;

    virtual ~DerivedQuantity() = default;

    virtual std::string name() const         = 0;
    virtual centering_t centering() const    = 0;
    virtual DerivedCategory category() const = 0;
    virtual void compute(State const& state, GridLayout const& layout, out_t& out,
                         double time) const
        = 0;
};


template<typename State, typename GridLayout>
class DerivedQuantityRegistry
{
    template<std::size_t rank>
    using DQ = DerivedQuantity<State, GridLayout, rank>;

public:
    template<std::size_t rank>
    void add(std::unique_ptr<DQ<rank>> dq)
    {
        std::get<rank>(quantities_).push_back(std::move(dq));
    }

    template<std::size_t rank>
    DQ<rank> const* find(std::string const& name) const
    {
        for (auto const& dq : std::get<rank>(quantities_))
            if (dq->name() == name)
                return dq.get();
        return nullptr;
    }

    template<std::size_t rank>
    auto const& quantities() const
    {
        return std::get<rank>(quantities_);
    }

private:
    std::tuple<std::vector<std::unique_ptr<DQ<0>>>, std::vector<std::unique_ptr<DQ<1>>>>
        quantities_;
};

} // namespace PHARE::core

#endif
