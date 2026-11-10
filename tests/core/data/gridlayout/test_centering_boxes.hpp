#ifndef TESTS_CORE_DATA_GRIDLAYOUT_TEST_CENTERING_BOXES_HPP
#define TESTS_CORE_DATA_GRIDLAYOUT_TEST_CENTERING_BOXES_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cstddef>

namespace PHARE::core
{
template<typename Layout>
using Centering_t = std::array<QtyCentering, Layout::dimension>;

template<typename Layout>
using Box_t = Box<std::size_t, Layout::dimension>;

template<typename Layout>
Box_t<Layout> ghostBox(Layout const& layout, Centering_t<Layout> const& centering)
{
    Box_t<Layout> box;
    for (std::size_t d = 0; d < Layout::dimension; ++d)
    {
        box.lower[d] = layout.ghostStartIndex(centering[d], static_cast<Direction>(d));
        box.upper[d] = layout.ghostEndIndex(centering[d], static_cast<Direction>(d));
    }
    return box;
}

template<typename Layout>
Box_t<Layout> physicalBox(Layout const& layout, Centering_t<Layout> const& centering)
{
    Box_t<Layout> box;
    for (std::size_t d = 0; d < Layout::dimension; ++d)
    {
        box.lower[d] = layout.physicalStartIndex(centering[d], static_cast<Direction>(d));
        box.upper[d] = layout.physicalEndIndex(centering[d], static_cast<Direction>(d));
    }
    return box;
}

template<typename Layout>
std::array<double, 3> coordinates(Layout const& layout, Centering_t<Layout> const& centering,
                                  MeshIndex<Layout::dimension> const& idx)
{
    auto const& dl = layout.meshSize();
    std::array<double, 3> x{0., 0., 0.};
    for (std::size_t d = 0; d < Layout::dimension; ++d)
    {
        auto const start = layout.physicalStartIndex(centering[d], static_cast<Direction>(d));
        auto const shift = centering[d] == QtyCentering::dual ? dl[d] / 2 : 0.;
        x[d] = (static_cast<double>(idx[d]) - static_cast<double>(start)) * dl[d] + shift;
    }
    return x;
}

template<typename Layout, typename Field, typename Fn>
void fill(Layout const& layout, Field& field, Fn const& fn)
{
    auto const centering = layout.centering(field);
    for_each_point(ghostBox(layout, centering),
                   [&](auto const& idx) { field(idx) = fn(coordinates(layout, centering, idx)); });
}

} // namespace PHARE::core

#endif
