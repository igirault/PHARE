#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_CLASSIFIER_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_CLASSIFIER_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/inner_boundary/inner_boundary_geometry.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/numerics/interpolator/field_at_point.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace PHARE::core
{

template<std::size_t dim, typename GridLayoutT, typename PhysicalQuantityT>
class InnerBoundaryMeshClassifier
{
public:
    using point_type              = InnerBoundaryGeometry<dim>::point_type;
    using local_index_type        = Point<std::uint32_t, dim>;
    using signed_local_index_type = Point<int, dim>;
    using mesh_data_type          = InnerBoundaryMeshData<dim, PhysicalQuantityT>;
    using field_type              = mesh_data_type::field_type;

    struct Params
    {
        std::size_t nghosts = 1;
        double cut_eps      = 0.0;
        double inactive_eps = 0.0;

        NearBodyScheme scheme = NearBodyScheme::None;

        std::size_t full_reach = 1;

        std::size_t band_radius = 1;
    };

    struct Overrides
    {
        std::optional<std::size_t> nghosts;
        std::optional<double> cut_eps;
        std::optional<double> inactive_eps;
    };

    static InnerBoundaryMeshClassifier withDefaults(InnerBoundaryGeometry<dim> const& boundary,
                                                    GridLayoutT const& layout,
                                                    Overrides const& overrides = {})
    {
        auto const& dx        = layout.meshSize();
        auto const dx_min     = *std::min_element(dx.begin(), dx.end());
        auto const full_reach = [&]() -> std::uint32_t {
            if constexpr (requires { GridLayoutT::options.field_options.reconstruction_nghosts; })
                return GridLayoutT::options.field_options.reconstruction_nghosts + 1;
            else
                return GridLayoutT::options.field_ghost_width;
        }();
        auto const nghosts = [&]() -> std::uint32_t {
            if constexpr (requires { GridLayoutT::options.field_options.reconstruction_nghosts; })
                return 2
                       * std::min<std::uint32_t>(
                           GridLayoutT::options.field_options.reconstruction_nghosts, 2);
            else
                return GridLayoutT::options.field_ghost_width;
        }();

        bool const underResolved
            = boundary.characteristicLength() < (static_cast<double>(nghosts) + 1.0) * dx_min;

        auto const scheme = [&]() {
            if (underResolved)
                return NearBodyScheme::FirstOrder;
            if constexpr (requires { GridLayoutT::options.field_options.reconstruction_nghosts; })
                if (GridLayoutT::options.field_options.reconstruction_nghosts > 2)
                    return NearBodyScheme::SecondOrder;
            return NearBodyScheme::None;
        }();

        Params p;
        p.full_reach   = full_reach;
        p.band_radius  = full_reach - 1;
        p.scheme       = scheme;
        p.nghosts      = overrides.nghosts.value_or(underResolved ? std::size_t{1} : nghosts);
        p.cut_eps      = overrides.cut_eps.value_or(1e-6 * dx_min);
        p.inactive_eps = overrides.inactive_eps.value_or(p.cut_eps);
        return InnerBoundaryMeshClassifier{boundary, p};
    }

    InnerBoundaryMeshClassifier(InnerBoundaryGeometry<dim> const& boundary, Params params)
        : boundary_{boundary}
        , params_{params}
    {
    }

    NO_DISCARD Params const& params() const { return params_; }

    void operator()(GridLayoutT const& layout, mesh_data_type& meshData) const
    {
        validateCenterings_(meshData.signedDistanceAtNodes, meshData);
        fillNodePhi_(layout, meshData.signedDistanceAtNodes);

        auto const padded = classifyCellsPadded_(layout, meshData.signedDistanceAtNodes,
                                                 meshData.cellStatusField());
        classifyNonCellElems_(layout, meshData.signedDistanceAtNodes, padded, meshData);
        populateGhostLists_(layout, meshData);
        populateDegradedLists_(layout, meshData, padded);
    }

private:
    InnerBoundaryGeometry<dim> const& boundary_;
    Params params_;

    struct PaddedCells
    {
        std::array<std::size_t, dim> shape{};
        int pad = 0;
        std::vector<double> status;

        std::size_t flat(std::array<int, dim> const& q) const
        {
            std::size_t f = 0;
            for (std::size_t d = 0; d < dim; ++d)
                f = f * shape[d] + static_cast<std::size_t>(q[d]);
            return f;
        }

        std::array<int, dim> unflat(std::size_t f) const
        {
            std::array<int, dim> q{};
            for (std::size_t d = dim; d-- > 0;)
            {
                q[d] = static_cast<int>(f % shape[d]);
                f /= shape[d];
            }
            return q;
        }

        std::size_t flatLocal(signed_local_index_type const& local) const
        {
            std::array<int, dim> q{};
            for (std::size_t d = 0; d < dim; ++d)
                q[d] = local[d] + pad;
            return flat(q);
        }

        bool inBounds(std::array<int, dim> const& q) const
        {
            for (std::size_t d = 0; d < dim; ++d)
                if (q[d] < 0 || q[d] >= static_cast<int>(shape[d]))
                    return false;
            return true;
        }
    };

    static bool isCut_(double phi_min, double phi_max, double cut_eps)
    {
        if (phi_min < -cut_eps && phi_max > cut_eps)
            return true;
        return phi_min <= cut_eps && phi_max >= -cut_eps;
    }

    template<typename FieldT>
    static signed_local_index_type fieldAMRIndex_(GridLayoutT const& layout, FieldT const& field,
                                                  local_index_type const& local_idx)
    {
        signed_local_index_type amr_idx{};
        for (std::size_t i = 0; i < dim; ++i)
        {
            auto const dir = static_cast<Direction>(i);
            amr_idx[i]     = static_cast<int>(local_idx[i])
                         + (layout.AMRBox().lower[i]
                            - static_cast<int>(layout.physicalStartIndex(field, dir)));
        }
        return amr_idx;
    }

    static auto nodeCentering_()
    {
        std::array<QtyCentering, dim> centering{};
        centering.fill(QtyCentering::primal);
        return centering;
    }

    static auto cellCentering_()
    {
        std::array<QtyCentering, dim> centering{};
        centering.fill(QtyCentering::dual);
        return centering;
    }

    template<std::size_t iDim, typename Fn>
    static void forEachSupportNode_(local_index_type const& elem,
                                    std::array<QtyCentering, dim> const& centering,
                                    local_index_type& node, Fn&& fn)
    {
        if constexpr (iDim == dim)
        {
            fn(node);
        }
        else if (centering[iDim] == QtyCentering::dual)
        {
            node[iDim] = elem[iDim];
            forEachSupportNode_<iDim + 1>(elem, centering, node, std::forward<Fn>(fn));
            node[iDim] = elem[iDim] + 1;
            forEachSupportNode_<iDim + 1>(elem, centering, node, std::forward<Fn>(fn));
        }
        else
        {
            node[iDim] = elem[iDim];
            forEachSupportNode_<iDim + 1>(elem, centering, node, std::forward<Fn>(fn));
        }
    }

    template<std::size_t iDim, typename Fn>
    static void forEachAdjacentCell_(local_index_type const& elem,
                                     std::array<QtyCentering, dim> const& centering,
                                     signed_local_index_type& cell, Fn&& fn)
    {
        if constexpr (iDim == dim)
        {
            fn(cell);
        }
        else if (centering[iDim] == QtyCentering::primal)
        {
            cell[iDim] = static_cast<int>(elem[iDim]) - 1;
            forEachAdjacentCell_<iDim + 1>(elem, centering, cell, std::forward<Fn>(fn));
            cell[iDim] = static_cast<int>(elem[iDim]);
            forEachAdjacentCell_<iDim + 1>(elem, centering, cell, std::forward<Fn>(fn));
        }
        else
        {
            cell[iDim] = static_cast<int>(elem[iDim]);
            forEachAdjacentCell_<iDim + 1>(elem, centering, cell, std::forward<Fn>(fn));
        }
    }

    bool elemSurroundsFluidOrCutCell_(local_index_type const& elem,
                                      std::array<QtyCentering, dim> const& centering,
                                      PaddedCells const& padded) const
    {
        bool found = false;
        signed_local_index_type cell{};
        forEachAdjacentCell_<0>(elem, centering, cell, [&](auto const& adjacent_cell) {
            if (found)
                return;
            auto const s = padded.status[padded.flatLocal(adjacent_cell)];
            if (s == toDouble(ElemStatus::Fluid) || s == toDouble(ElemStatus::Cut))
                found = true;
        });
        return found;
    }

    bool elemSurroundsGhostCell_(local_index_type const& elem,
                                 std::array<QtyCentering, dim> const& centering,
                                 PaddedCells const& padded) const
    {
        bool has_ghost = false;
        signed_local_index_type cell{};
        forEachAdjacentCell_<0>(elem, centering, cell, [&](auto const& adjacent_cell) {
            if (!has_ghost
                && padded.status[padded.flatLocal(adjacent_cell)] == toDouble(ElemStatus::Ghost))
                has_ghost = true;
        });
        return has_ghost;
    }

    void fillNodePhi_(GridLayoutT const& layout, field_type& signed_distance_at_nodes) const
    {
        layout.evalOnGhostBox(signed_distance_at_nodes, [&](auto... idx) {
            auto const local_point = local_index_type{static_cast<std::uint32_t>(idx)...};
            auto const amr_point   = fieldAMRIndex_(layout, signed_distance_at_nodes, local_point);
            signed_distance_at_nodes(local_point) = boundary_.signedDistance(
                layout.fieldNodeCoordinates(signed_distance_at_nodes, amr_point));
        });
    }

    PaddedCells classifyCellsPadded_(GridLayoutT const& layout,
                                     field_type const& signed_distance_at_nodes,
                                     field_type& cell_status) const
    {
        PaddedCells padded;
        padded.pad = static_cast<int>(params_.nghosts + params_.full_reach + 1);

        auto const cell_shape = cell_status.shape();
        std::array<std::size_t, dim> node_shape{};
        std::array<int, dim> cell_offset{}, node_offset{};
        std::size_t n_cells = 1, n_nodes = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const dir  = static_cast<Direction>(d);
            padded.shape[d] = cell_shape[d] + 2 * static_cast<std::size_t>(padded.pad);
            node_shape[d]   = padded.shape[d] + 1;
            cell_offset[d]  = layout.AMRBox().lower[d]
                             - static_cast<int>(layout.physicalStartIndex(cell_status, dir))
                             - padded.pad;
            node_offset[d]
                = layout.AMRBox().lower[d]
                  - static_cast<int>(layout.physicalStartIndex(signed_distance_at_nodes, dir))
                  - padded.pad;
            n_cells *= padded.shape[d];
            n_nodes *= node_shape[d];
        }

        auto node_flat = [&](std::array<int, dim> const& q) {
            std::size_t f = 0;
            for (std::size_t d = 0; d < dim; ++d)
                f = f * node_shape[d] + static_cast<std::size_t>(q[d]);
            return f;
        };

        std::vector<double> phi(n_nodes);
        for (std::size_t f = 0; f < n_nodes; ++f)
        {
            signed_local_index_type amr{};
            auto rest = f;
            for (std::size_t d = dim; d-- > 0;)
            {
                amr[d] = static_cast<int>(rest % node_shape[d]) + node_offset[d];
                rest /= node_shape[d];
            }
            phi[f] = boundary_.signedDistance(
                layout.fieldNodeCoordinates(signed_distance_at_nodes, amr));
        }

        padded.status.assign(n_cells, toDouble(ElemStatus::Fluid));
        std::vector<std::size_t> frontier;
        for (std::size_t f = 0; f < n_cells; ++f)
        {
            auto const q   = padded.unflat(f);
            double phi_min = std::numeric_limits<double>::max();
            double phi_max = std::numeric_limits<double>::lowest();
            for (std::size_t corner = 0; corner < (std::size_t{1} << dim); ++corner)
            {
                auto node = q;
                for (std::size_t d = 0; d < dim; ++d)
                    node[d] += static_cast<int>((corner >> d) & 1u);
                auto const v = phi[node_flat(node)];
                phi_min      = std::min(phi_min, v);
                phi_max      = std::max(phi_max, v);
            }

            if (isCut_(phi_min, phi_max, params_.cut_eps))
            {
                padded.status[f] = toDouble(ElemStatus::Cut);
                frontier.push_back(f);
                continue;
            }

            signed_local_index_type amr{};
            for (std::size_t d = 0; d < dim; ++d)
                amr[d] = q[d] + cell_offset[d];
            auto const phi_center = boundary_.signedDistance(layout.cellCenteredCoordinates(amr));
            padded.status[f] = (phi_center < -params_.inactive_eps) ? toDouble(ElemStatus::Inactive)
                                                                    : toDouble(ElemStatus::Fluid);
        }

        for (std::size_t layer = 0; layer < params_.nghosts && !frontier.empty(); ++layer)
        {
            std::vector<std::size_t> next_frontier;
            next_frontier.reserve(frontier.size() * 2 * dim);
            for (auto const f : frontier)
            {
                auto const q = padded.unflat(f);
                for (std::size_t d = 0; d < dim; ++d)
                    for (int sgn : {-1, 1})
                    {
                        auto neigh = q;
                        neigh[d] += sgn;
                        if (!padded.inBounds(neigh))
                            continue;
                        auto const g = padded.flat(neigh);
                        if (padded.status[g] != toDouble(ElemStatus::Inactive))
                            continue;
                        padded.status[g] = toDouble(ElemStatus::Ghost);
                        next_frontier.push_back(g);
                    }
            }
            frontier = std::move(next_frontier);
        }

        layout.evalOnGhostBox(cell_status, [&](auto... idx) {
            auto const local = local_index_type{static_cast<std::uint32_t>(idx)...};
            signed_local_index_type signed_local{};
            for (std::size_t d = 0; d < dim; ++d)
                signed_local[d] = static_cast<int>(local[d]);
            cell_status(local) = padded.status[padded.flatLocal(signed_local)];
        });

        return padded;
    }

    void classifyNonCellElems_(GridLayoutT const& layout,
                               field_type const& signed_distance_at_nodes,
                               PaddedCells const& padded, mesh_data_type& meshData) const
    {
        for (std::size_t idx = 0; idx < mesh_data_type::num_elem_types; ++idx)
        {
            auto const centering = mesh_data_type::idxToCentering(idx);
            auto const n_dual    = static_cast<std::size_t>(
                std::count(centering.begin(), centering.end(), QtyCentering::dual));

            if (n_dual == dim)
                continue;

            auto& elem_field = meshData.elemStatus[idx];
            layout.evalOnGhostBox(elem_field, [&](auto... local_idx_args) {
                auto const local_elem
                    = local_index_type{static_cast<std::uint32_t>(local_idx_args)...};
                auto const amr_elem = fieldAMRIndex_(layout, elem_field, local_elem);

                double phi_min = std::numeric_limits<double>::max();
                double phi_max = std::numeric_limits<double>::lowest();
                local_index_type node_idx{};
                forEachSupportNode_<0>(local_elem, centering, node_idx,
                                       [&](auto const& support_node) {
                                           auto const phi = signed_distance_at_nodes(support_node);
                                           phi_min        = std::min(phi_min, phi);
                                           phi_max        = std::max(phi_max, phi);
                                       });

                if (isCut_(phi_min, phi_max, params_.cut_eps))
                {
                    elem_field(local_elem) = toDouble(ElemStatus::Cut);
                    return;
                }

                if (elemSurroundsFluidOrCutCell_(local_elem, centering, padded))
                {
                    elem_field(local_elem) = toDouble(ElemStatus::Fluid);
                    return;
                }

                if (elemSurroundsGhostCell_(local_elem, centering, padded))
                {
                    elem_field(local_elem) = toDouble(ElemStatus::Ghost);
                    return;
                }

                auto const phi_elem
                    = boundary_.signedDistance(layout.fieldNodeCoordinates(elem_field, amr_elem));
                elem_field(local_elem) = (phi_elem < -params_.inactive_eps)
                                             ? toDouble(ElemStatus::Inactive)
                                             : toDouble(ElemStatus::Fluid);
            });
        }
    }

    void populateGhostLists_(GridLayoutT const& layout, mesh_data_type& meshData) const
    {
        for (auto& vec : meshData.ghostElemsData)
            vec.clear();

        for (std::size_t idx = 0; idx < mesh_data_type::num_elem_types; ++idx)
        {
            auto const& status_field = meshData.elemStatus[idx];
            auto& ghost_list         = meshData.ghostElemsData[idx];
            auto const centerings    = mesh_data_type::idxToCentering(idx);

            layout.evalOnGhostBox(status_field, [&](auto... local_idx_args) {
                auto const local = local_index_type{static_cast<std::uint32_t>(local_idx_args)...};
                if (status_field(local) != toDouble(ElemStatus::Ghost))
                    return;

                auto const amr    = fieldAMRIndex_(layout, status_field, local);
                auto const pos    = layout.fieldNodeCoordinates(status_field, amr);
                auto const mirror = boundary_.symmetric(pos);
                ghost_list.push_back(
                    {local, mirror, boundary_.normal(pos),
                     FieldAtPoint<dim, 1>::pointIsInterpolable(layout, mirror, centerings)});
            });
        }
    }

    void populateDegradedLists_(GridLayoutT const& layout, mesh_data_type& meshData,
                                PaddedCells const& padded) const
    {
        for (auto& vec : meshData.degradedElemsData)
            vec.clear();

        if (params_.scheme == NearBodyScheme::None)
            return;

        bool const secondOrder     = params_.scheme == NearBodyScheme::SecondOrder;
        std::uint32_t const margin = secondOrder ? 2u : 1u;

        std::vector<char> nearBody(padded.status.size(), 0);
        std::vector<std::size_t> frontier;
        for (std::size_t f = 0; f < padded.status.size(); ++f)
        {
            auto const s = padded.status[f];
            if (s == toDouble(ElemStatus::Cut) || s == toDouble(ElemStatus::Ghost))
            {
                nearBody[f] = 1;
                frontier.push_back(f);
            }
        }

        if (secondOrder)
            dilateChebyshev_(nearBody, padded.shape, params_.band_radius);

        for (std::size_t layer = 0; !secondOrder && layer < params_.full_reach && !frontier.empty();
             ++layer)
        {
            std::vector<std::size_t> next_frontier;
            for (auto const f : frontier)
            {
                auto const q = padded.unflat(f);
                for (std::size_t d = 0; d < dim; ++d)
                {
                    for (int sgn : {-1, 1})
                    {
                        auto neigh = q;
                        neigh[d] += sgn;
                        if (!padded.inBounds(neigh))
                            continue;
                        auto const g = padded.flat(neigh);
                        if (nearBody[g])
                            continue;
                        nearBody[g] = 1;
                        next_frontier.push_back(g);
                    }
                }
            }
            frontier = std::move(next_frontier);
        }

        for (std::size_t idx = 0; idx < mesh_data_type::num_elem_types; ++idx)
        {
            auto const centering     = mesh_data_type::idxToCentering(idx);
            auto const& status_field = meshData.elemStatus[idx];
            auto& degraded_list      = meshData.degradedElemsData[idx];

            auto const nPrimal
                = std::count(centering.begin(), centering.end(), QtyCentering::primal);
            auto const shape = status_field.shape();

            auto mark = [&](auto... local_idx_args) {
                auto const local_elem
                    = local_index_type{static_cast<std::uint32_t>(local_idx_args)...};

                bool degraded = false;
                signed_local_index_type cell{};
                forEachAdjacentCell_<0>(
                    local_elem, centering, cell, [&](auto const& adjacent_cell) {
                        if (!degraded && nearBody[padded.flatLocal(adjacent_cell)])
                            degraded = true;
                    });

                if (degraded)
                    degraded_list.push_back({local_elem, {}, {}, false});
            };

            if (nPrimal != 1)
            {
                layout.evalOnBox(status_field, mark);
                continue;
            }

            layout.evalOnGhostBox(status_field, [&](auto... local_idx_args) {
                auto const local_elem
                    = local_index_type{static_cast<std::uint32_t>(local_idx_args)...};
                for (std::size_t d = 0; d < dim; ++d)
                    if (centering[d] == QtyCentering::primal
                        and (local_elem[d] < margin or local_elem[d] + margin >= shape[d]))
                        return;
                mark(local_idx_args...);
            });
        }
    }

    static void dilateChebyshev_(std::vector<char>& mask, auto const& shape,
                                 std::size_t const radius)
    {
        auto const r = static_cast<int>(radius);
        for (std::size_t d = 0; d < dim; ++d)
        {
            std::size_t stride = 1;
            for (std::size_t e = d + 1; e < dim; ++e)
                stride *= shape[e];
            auto const extent = static_cast<int>(shape[d]);

            auto const previous = mask;
            for (std::size_t flat = 0; flat < previous.size(); ++flat)
            {
                if (!previous[flat])
                    continue;
                auto const coord = static_cast<int>((flat / stride) % shape[d]);
                for (int k = std::max(-r, -coord); k <= std::min(r, extent - 1 - coord); ++k)
                    mask[static_cast<std::size_t>(
                        static_cast<long>(flat) + static_cast<long>(k) * static_cast<long>(stride))]
                        = 1;
            }
        }
    }

    static void validateCenterings_(field_type const& signed_distance_at_nodes,
                                    mesh_data_type const& meshData)
    {
        if (GridLayoutT::centering(signed_distance_at_nodes) != nodeCentering_())
            throw std::runtime_error("signed_distance_at_nodes has invalid centering");

        for (std::size_t idx = 0; idx < mesh_data_type::num_elem_types; ++idx)
        {
            auto const expected = mesh_data_type::idxToCentering(idx);
            if (GridLayoutT::centering(meshData.elemStatus[idx]) != expected)
                throw std::runtime_error("elemStatus entry at index " + std::to_string(idx)
                                         + " has invalid centering");
        }
    }
};

} // namespace PHARE::core

#endif
