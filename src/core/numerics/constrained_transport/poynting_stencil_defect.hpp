#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_STENCIL_DEFECT_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_STENCIL_DEFECT_HPP

#include "core/def.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace PHARE::core
{
template<typename GridLayout>
class PoyntingStencilDefect
{
    static constexpr auto dimension = GridLayout::dimension;
    using Scalar                    = MHDQuantity::Scalar;
    using FaceBoxes                 = std::array<std::vector<Box<int, dimension>>, dimension>;

    static constexpr std::array<Scalar, 3> Eq{Scalar::Ex, Scalar::Ey, Scalar::Ez};
    static constexpr std::array<Scalar, 3> Bq{Scalar::Bx, Scalar::By, Scalar::Bz};

public:
    explicit PoyntingStencilDefect(GridLayout const& layout)
        : layout_{layout}
    {
    }

    void operator()(auto& fluxes, auto const& E, auto const& Bold, auto const& Bnew,
                    FaceBoxes const& faces, double const coef) const
    {
        for_N<dimension>([&](auto id) {
            constexpr auto d = static_cast<std::size_t>(id());
            for_N<3>([&](auto ic) {
                constexpr auto c = static_cast<std::size_t>(ic());
                if constexpr (c != d)
                {
                    constexpr auto b = 3 - c - d;
                    if constexpr (b < dimension)
                        for (auto const& box : faces[d])
                            correct_<d, c, b>(fluxes, E, Bold, Bnew, box, coef);
                }
            });
        });
    }

private:
    template<typename Fn>
    struct ExprField
    {
        using type                      = double;
        constexpr static auto dimension = GridLayout::dimension;

        NO_DISCARD double operator()(MeshIndex<dimension> const& i) const { return fn(i); }

        Fn fn;
    };

    template<std::size_t d, std::size_t c, std::size_t b>
    void correct_(auto& fluxes, auto const& E, auto const& Bold, auto const& Bnew,
                  Box<int, dimension> const& box, double const coef) const
    {
        constexpr auto direction = static_cast<Direction>(d);
        constexpr double sign    = ((d + 1) % 3 == c) ? 1.0 : -1.0;

        auto const& Ec  = E(static_cast<Component>(c));
        auto const& Bo  = Bold(static_cast<Component>(b));
        auto const& Bn  = Bnew(static_cast<Component>(b));
        auto const bmid = [&](MeshIndex<dimension> const& j) { return 0.5 * (Bo(j) + Bn(j)); };

        for_each_point(box, [&](auto const& amr) {
            auto const face     = toIndex_(layout_.AMRToLocal(amr));
            auto const lowIsOdd = amr[b] % 2 != 0;
            auto odd            = face;
            auto even           = face;
            (lowIsOdd ? even : odd)[b] += 1;

            auto const bEdge = project<GridLayout::centering(Bq[b]), GridLayout::centering(Eq[c])>(
                ExprField<decltype(bmid)>{bmid}, odd);
            fluxes.template get_dir<direction>(face).Etot()
                += coef * sign * 0.5 * (Ec(even) - Ec(odd)) * bEdge;
        });
    }

    static MeshIndex<dimension> toIndex_(auto const& point)
    {
        MeshIndex<dimension> index;
        for (std::size_t k = 0; k < dimension; ++k)
            index[k] = point[k];
        return index;
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
