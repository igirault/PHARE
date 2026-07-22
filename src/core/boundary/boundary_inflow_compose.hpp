#ifndef PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP
#define PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP

#include "initializer/data_provider.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace PHARE::core::inflow_compose
{
// Helpers to build space-time functions for the inflow boundary: a constant lifted to a
// function, and the element-wise product of two functions (used to form the momentum
// rho*v from prescribed density and velocity). Both preserve the batch-evaluation
// invariant: each input function is invoked exactly once over the whole coordinate batch,
// then combined element-wise.

/** @brief A constant lifted into a space-time function: returns c at every node. Sized to
 * the first spatial coordinate span (the node count). Time is ignored. */
template<std::size_t dim>
initializer::SpaceTimeFunction<dim> constFunction(double const c)
{
    return [c](auto const&... args) -> std::shared_ptr<Span<double>> {
        auto const& first = std::get<0>(std::forward_as_tuple(args...));
        std::vector<double> out(first.size(), c);
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}

/** @brief Element-wise product f*g of two space-time functions. */
template<std::size_t dim>
initializer::SpaceTimeFunction<dim> mulFunction(initializer::SpaceTimeFunction<dim> f,
                                                initializer::SpaceTimeFunction<dim> g)
{
    return [f = std::move(f), g = std::move(g)](
               auto const&... args) -> std::shared_ptr<Span<double>> {
        auto sf = f(args...);
        auto sg = g(args...);
        std::vector<double> out(sf->size());
        for (std::size_t k = 0; k < out.size(); ++k)
            out[k] = (*sf)[k] * (*sg)[k];
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}

} // namespace PHARE::core::inflow_compose

#endif // PHARE_CORE_BOUNDARY_BOUNDARY_INFLOW_COMPOSE_HPP
