#ifndef PHARE_CORE_NUMERICS_HYPER_RESISTIVITY_HPP
#define PHARE_CORE_NUMERICS_HYPER_RESISTIVITY_HPP

#include "core/def.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/differential_operators.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/types.hpp"

#include <array>
#include <cstddef>

namespace PHARE::core
{
namespace detail
{
    template<auto NodeScalar, std::size_t c, std::size_t d>
    consteval auto hyperKappaLocation()
    {
        using Scalar = decltype(NodeScalar);
        if constexpr (c == d)
            return NodeScalar;
        else if constexpr (c + d == 1)
            return Scalar::Bz;
        else if constexpr (c + d == 2)
            return Scalar::By;
        else
            return Scalar::Bx;
    }

    template<auto Scalar0, std::size_t c>
    consteval auto currentComponent()
    {
        using Scalar = decltype(Scalar0);
        if constexpr (c == 0)
            return Scalar::Jx;
        else if constexpr (c == 1)
            return Scalar::Jy;
        else
            return Scalar::Jz;
    }

    template<std::size_t d, std::size_t dim>
    consteval auto flippedIn(std::array<QtyCentering, dim> centering)
    {
        centering[d] = flipped(centering[d]);
        return centering;
    }

    template<std::size_t d, std::size_t dim>
    MeshIndex<dim> shiftedIndex(MeshIndex<dim> idx, int const shift)
    {
        idx[d] += shift;
        return idx;
    }
} // namespace detail


template<auto component, auto NodeScalar, typename GridLayout>
NO_DISCARD double hyperResistiveDivergence(GridLayout const& layout, auto const& Jc,
                                           auto const& kappa,
                                           MeshIndex<GridLayout::dimension> const idx)
{
    constexpr auto c         = detail::vector_component_index<component>();
    constexpr auto centering = GridLayout::centering(detail::currentComponent<NodeScalar, c>());

    return -sum(for_N_make_array<GridLayout::dimension>([&](auto id) {
        constexpr auto d        = id();
        constexpr auto location = detail::hyperKappaLocation<NodeScalar, c, d>();
        static_assert(GridLayout::centering(location) == detail::flippedIn<d>(centering));

        auto const inv      = layout.inverseMeshSize(static_cast<Direction>(d));
        constexpr int upper = centering[d] == QtyCentering::primal ? 0 : 1;

        auto const kUpper
            = kappa.template operator()<location>(detail::shiftedIndex<d>(idx, upper));
        auto const kLower
            = kappa.template operator()<location>(detail::shiftedIndex<d>(idx, upper - 1));
        auto const j = detail::shifted<d>(Jc, idx, 0);

        return inv * inv
               * (kUpper * (detail::shifted<d>(Jc, idx, 1) - j)
                  - kLower * (j - detail::shifted<d>(Jc, idx, -1)));
    }));
}

} // namespace PHARE::core

#endif
