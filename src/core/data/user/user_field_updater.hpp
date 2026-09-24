#ifndef PHARE_CORE_DATA_USER_USER_FIELD_UPDATER_HPP
#define PHARE_CORE_DATA_USER_USER_FIELD_UPDATER_HPP

#include "core/utilities/span.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/space_time_function.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace PHARE::core
{
/**
 * @brief Fills fields from user functions of space and time, evaluated on their whole ghost box.
 *
 * The ghost box is walked in the order of the field buffer (last index fastest, like the C-ordered
 * storage), so node k of the coordinate arrays handed to the user function, value k it returns
 * and element k of the field buffer all refer to the same node: the result is copied in one go,
 * without any index.
 */
class UserFieldUpdater
{
public:
    /**
     * @brief fills a field, computing the coordinates of its nodes on the fly
     */
    template<typename Field, typename GridLayout>
    void static update(Field& field, GridLayout const& layout,
                       SpaceTimeFunction<GridLayout::dimension> const& f, double time)
    {
        auto constexpr dim = GridLayout::dimension;

        std::array<std::vector<double>, dim> xyz;
        for (auto& x : xyz)
            x.reserve(field.size());

        layout.evalOnGhostBox(field, [&](auto... ijk) {
            auto const x = layout.fieldNodeCoordinates(field, layout.localToAMR(Point{ijk...}));
            for (std::size_t d = 0; d < dim; ++d)
                xyz[d].push_back(x[d]);
        });

        std::array<CoordinateSpan, dim> coords;
        for (std::size_t d = 0; d < dim; ++d)
            coords[d] = CoordinateSpan{xyz[d].data(), xyz[d].size()};

        evaluate_(field, f, time, coords);
    }

    /**
     * @brief fills a vecfield component-wise, computing the coordinates on the fly
     */
    template<typename VecField, typename GridLayout>
    void static update(
        VecField& vecfield, GridLayout const& layout,
        std::array<SpaceTimeFunction<GridLayout::dimension>, VecField::size()> const& funcs,
        double time)
    {
        for (std::size_t i = 0; i < VecField::size(); ++i)
            update(vecfield[i], layout, funcs[i], time);
    }

    /**
     * @brief fills a vecfield component-wise, reading the coordinates from a cache filled by
     * fillCoordinates
     *
     * @param coordinates one vecfield per dimension, with the centering of vecfield: component c
     * of coordinates[d] holds coordinate d at the nodes of component c
     */
    template<typename VecField>
    void static update(
        VecField& vecfield,
        std::array<SpaceTimeFunction<VecField::dimension>, VecField::size()> const& funcs,
        double time, std::span<std::type_identity_t<VecField> const> coordinates)
    {
        auto constexpr dim = VecField::dimension;

        if (coordinates.size() != dim)
            throw std::runtime_error(
                "UserFieldUpdater::update: expected one coordinates vecfield per dimension");

        for (std::size_t c = 0; c < VecField::size(); ++c)
        {
            std::array<CoordinateSpan, dim> coords;
            for (std::size_t d = 0; d < dim; ++d)
                coords[d] = CoordinateSpan{coordinates[d][c].data(), coordinates[d][c].size()};

            evaluate_(vecfield[c], funcs[c], time, coords);
        }
    }

    /**
     * @brief fills the node coordinates cache of vecfields, over the whole ghost box
     *
     * Component c of coords[d] gets coordinate d at the nodes of component c: coords holds one
     * vecfield per dimension, all with the centering of the vecfield whose nodes they describe.
     */
    template<typename GridLayout, typename VecField>
    void static fillCoordinates(GridLayout const& layout, std::span<VecField> coords)
    {
        if (coords.size() != GridLayout::dimension)
            throw std::runtime_error(
                "UserFieldUpdater::fillCoordinates: expected one vecfield per dimension");

        for (std::size_t c = 0; c < VecField::size(); ++c)
        {
            auto const& reference = coords[0][c];
            layout.evalOnGhostBox(reference, [&](auto... ijk) {
                auto const x
                    = layout.fieldNodeCoordinates(reference, layout.localToAMR(Point{ijk...}));
                for (std::size_t d = 0; d < GridLayout::dimension; ++d)
                    coords[d][c](ijk...) = x[d];
            });
        }
    }

private:
    /**
     * @brief evaluates f on the given node coordinates and copies the result into field
     */
    template<typename Field, std::size_t dim>
    void static evaluate_(Field& field, SpaceTimeFunction<dim> const& f, double time,
                          std::array<CoordinateSpan, dim> const& coords)
    {
        for (auto const& x : coords)
            if (x.size != field.size())
                throw std::runtime_error(
                    "UserFieldUpdater: coordinate arrays do not match the field size");

        std::shared_ptr<Span<double>> const gridPtr // keep grid data alive
            = std::apply([&](auto const&... x) { return f(x..., time); }, coords);
        Span<double> const& grid = *gridPtr;

        // a user function returning the wrong number of values would be read out of bounds
        if (grid.size() != field.size())
            throw std::runtime_error("UserFieldUpdater: user function returned "
                                     + std::to_string(grid.size()) + " values for "
                                     + std::to_string(field.size()) + " nodes");

        std::copy(grid.data(), grid.data() + grid.size(), field.data());
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_DATA_USER_USER_FIELD_UPDATER_HPP
