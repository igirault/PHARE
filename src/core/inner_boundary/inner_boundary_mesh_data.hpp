#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_DATA_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_DATA_HPP

#include "core/def.hpp"
#include "core/data/field/field.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/inner_boundary/ghost_elem_pack.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"
#include "core/utilities/point/point.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace PHARE::core
{
template<std::size_t dim, typename PhysicalQuantityT>
struct InnerBoundaryMeshData
{
    using field_type           = Field<dim, typename PhysicalQuantityT::Scalar, double>;
    using vecfield_type        = VecField<field_type, PhysicalQuantityT>;
    using ghost_elem_data_type = GhostElemData<dim>;

    static constexpr std::size_t num_elem_types = (std::size_t{1} << dim);

    static constexpr std::size_t centeringToIdx(std::array<QtyCentering, dim> const& c) noexcept
    {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < dim; ++i)
            if (c[i] == QtyCentering::dual)
                idx |= (std::size_t{1} << i);
        return idx;
    }

    static constexpr auto idxToCentering(std::size_t idx) noexcept
    {
        std::array<QtyCentering, dim> c{};
        for (std::size_t i = 0; i < dim; ++i)
            c[i] = ((idx >> i) & 1u) ? QtyCentering::dual : QtyCentering::primal;
        return c;
    }

    static auto scalarFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        auto const n_primal = static_cast<std::size_t>(
            std::count(centering.begin(), centering.end(), QtyCentering::primal));

        if (n_primal == 0)
            return PhysicalQuantityT::Scalar::CellCentered;

        if (n_primal == dim)
            return PhysicalQuantityT::Scalar::NodeCentered;

        if (n_primal == 1)
        {
            auto const dir = static_cast<std::size_t>(
                std::distance(centering.begin(),
                              std::find(centering.begin(), centering.end(), QtyCentering::primal)));
            return PhysicalQuantityT::FaceCentered()[dir];
        }

        auto const dir = static_cast<std::size_t>(std::distance(
            centering.begin(), std::find(centering.begin(), centering.end(), QtyCentering::dual)));
        return PhysicalQuantityT::EdgeCentered()[dir];
    }

    field_type signedDistanceAtNodes;

    std::array<field_type, num_elem_types> elemStatus;

    GhostElemPack<dim> ghostElemsData;

    GhostElemPack<dim> degradedElemsData;

    explicit InnerBoundaryMeshData(std::string boundaryName = "")
        : signedDistanceAtNodes{boundaryName + "_signed_distance",
                                PhysicalQuantityT::Scalar::NodeCentered}
        , elemStatus{makeElemStatus_(boundaryName, std::make_index_sequence<num_elem_types>{})}
        , ghostElemsData{boundaryName + "_ghost_elems"}
        , degradedElemsData{boundaryName + "_degraded_elems"}
    {
    }

    field_type& getStatusFieldFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        return elemStatus[centeringToIdx(centering)];
    }

    field_type const&
    getStatusFieldFromCentering(std::array<QtyCentering, dim> const& centering) const
    {
        return elemStatus[centeringToIdx(centering)];
    }

    field_type& cellStatusField() { return elemStatus.back(); }
    field_type const& cellStatusField() const { return elemStatus.back(); }

    auto
    getStatusFieldsFromCenterings(std::array<std::array<QtyCentering, dim>, 3> const& centerings)
        -> std::tuple<field_type&, field_type&, field_type&>
    {
        return std::forward_as_tuple(getStatusFieldFromCentering(centerings[0]),
                                     getStatusFieldFromCentering(centerings[1]),
                                     getStatusFieldFromCentering(centerings[2]));
    }

    auto getStatusFieldsFromCenterings(
        std::array<std::array<QtyCentering, dim>, 3> const& centerings) const
        -> std::tuple<field_type const&, field_type const&, field_type const&>
    {
        return std::forward_as_tuple(getStatusFieldFromCentering(centerings[0]),
                                     getStatusFieldFromCentering(centerings[1]),
                                     getStatusFieldFromCentering(centerings[2]));
    }

    std::vector<ghost_elem_data_type>&
    getGhostDataFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        return ghostElemsData[centeringToIdx(centering)];
    }

    std::vector<ghost_elem_data_type> const&
    getGhostDataFromCentering(std::array<QtyCentering, dim> const& centering) const
    {
        return ghostElemsData[centeringToIdx(centering)];
    }

    std::vector<ghost_elem_data_type>&
    getDegradedElemsFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        return degradedElemsData[centeringToIdx(centering)];
    }

    std::vector<ghost_elem_data_type> const&
    getDegradedElemsFromCentering(std::array<QtyCentering, dim> const& centering) const
    {
        return degradedElemsData[centeringToIdx(centering)];
    }

    auto getGhostDataFromCenterings(std::array<std::array<QtyCentering, dim>, 3> const& centerings)
        -> std::tuple<std::vector<ghost_elem_data_type>&, std::vector<ghost_elem_data_type>&,
                      std::vector<ghost_elem_data_type>&>
    {
        return std::forward_as_tuple(getGhostDataFromCentering(centerings[0]),
                                     getGhostDataFromCentering(centerings[1]),
                                     getGhostDataFromCentering(centerings[2]));
    }

    auto
    getGhostDataFromCenterings(std::array<std::array<QtyCentering, dim>, 3> const& centerings) const
        -> std::tuple<std::vector<ghost_elem_data_type> const&,
                      std::vector<ghost_elem_data_type> const&,
                      std::vector<ghost_elem_data_type> const&>
    {
        return std::forward_as_tuple(getGhostDataFromCentering(centerings[0]),
                                     getGhostDataFromCentering(centerings[1]),
                                     getGhostDataFromCentering(centerings[2]));
    }

    NO_DISCARD bool isUsable() const
    {
        return std::apply(
            [this](auto const&... fields) {
                return isUsable(signedDistanceAtNodes, fields..., ghostElemsData,
                                degradedElemsData);
            },
            elemStatus);
    }

    NO_DISCARD bool isSettable() const
    {
        return std::apply(
            [this](auto const&... fields) {
                return isSettable(signedDistanceAtNodes, fields..., ghostElemsData,
                                  degradedElemsData);
            },
            elemStatus);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::apply(
            [this](auto&... fields) {
                return std::forward_as_tuple(signedDistanceAtNodes, fields..., ghostElemsData,
                                             degradedElemsData);
            },
            elemStatus);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::apply(
            [this](auto const&... fields) {
                return std::forward_as_tuple(signedDistanceAtNodes, fields..., ghostElemsData,
                                             degradedElemsData);
            },
            elemStatus);
    }

private:
    static std::string centeringToString_(std::array<QtyCentering, dim> const& c)
    {
        std::string s;
        for (auto const& qc : c)
            s += (qc == QtyCentering::primal ? 'P' : 'D');
        return s;
    }

    template<std::size_t... Is>
    static auto makeElemStatus_(std::string const& name, std::index_sequence<Is...>)
    {
        return std::array<field_type, sizeof...(Is)>{[&](std::size_t i) -> field_type {
            auto const c = idxToCentering(i);
            return field_type{name + "_status_" + centeringToString_(c), scalarFromCentering(c)};
        }(Is)...};
    }
};

} // namespace PHARE::core

#endif
