#include "gtest/gtest.h"

#include "core/data/grid/gridlayout.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"
#include "core/models/options/mhd_options.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "simulator/simulator_def.hpp"

#include "tests/core/utilities/inner_boundary/plane_inner_boundary.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <cmath>
#include <vector>

namespace
{
constexpr double eps = 1e-12;

inline constexpr PHARE::SimOpts wenozOptions{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::WENOZ,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::None,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<wenozOptions> wenozFieldOptions{};

using GridLayout = PHARE::core::GridLayout<PHARE::MHDOptions<wenozFieldOptions>{}>;
using Mapper   = PHARE::core::InnerBoundaryMeshClassifier<2, GridLayout, PHARE::core::MHDQuantity>;

inline constexpr PHARE::SimOpts linearOptions{
    .dimension           = 2,
    .interp_order        = 1,
    .reconstruction_type = PHARE::MHDOpts::ReconstructionType::Linear,
    .slope_limiter_type  = PHARE::MHDOpts::SlopeLimiterType::VanLeer,
    .riemann_solver_type = PHARE::MHDOpts::RiemannSolverType::Rusanov};

inline constexpr PHARE::MHDFieldOptions<linearOptions> linearFieldOptions{};

using LinearGridLayout = PHARE::core::GridLayout<PHARE::MHDOptions<linearFieldOptions>{}>;
using LinearMapper
    = PHARE::core::InnerBoundaryMeshClassifier<2, LinearGridLayout, PHARE::core::MHDQuantity>;
using MeshData = PHARE::core::InnerBoundaryMeshData<2, PHARE::core::MHDQuantity>;
using ScalarField = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;

constexpr auto reconstructionNghosts = wenozFieldOptions.reconstruction_nghosts;

template<typename FieldT>
PHARE::core::Point<std::uint32_t, 2> physicalLocalIndex(GridLayout const& layout,
                                                        FieldT const& field, std::uint32_t ix,
                                                        std::uint32_t iy)
{
    using PHARE::core::Direction;

    return {layout.physicalStartIndex(field, Direction::X) + ix,
            layout.physicalStartIndex(field, Direction::Y) + iy};
}

constexpr std::array<PHARE::core::QtyCentering, 2> kCellC
    = {PHARE::core::QtyCentering::dual, PHARE::core::QtyCentering::dual};
constexpr std::array<PHARE::core::QtyCentering, 2> kFaceXC
    = {PHARE::core::QtyCentering::primal, PHARE::core::QtyCentering::dual};
constexpr std::array<PHARE::core::QtyCentering, 2> kFaceYC
    = {PHARE::core::QtyCentering::dual, PHARE::core::QtyCentering::primal};
constexpr std::array<PHARE::core::QtyCentering, 2> kNodeC
    = {PHARE::core::QtyCentering::primal, PHARE::core::QtyCentering::primal};

template<typename Layout>
struct InnerBoundaryMeshClassifierBuffers
{
    static constexpr char const* BOUNDARY_NAME = "test";

    explicit InnerBoundaryMeshClassifierBuffers(Layout const& layout)
        : phi_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::NodeCentered)}
        , tags{BOUNDARY_NAME}
    {
        ScalarField phi_field{std::string(BOUNDARY_NAME) + "_signed_distance",
                              PHARE::core::MHDQuantity::Scalar::NodeCentered, phi_storage.data(),
                              phi_storage.shape()};
        tags.signedDistanceAtNodes.setBuffer(&phi_field);

        elem_storages.reserve(MeshData::num_elem_types);
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
        {
            auto const c   = MeshData::idxToCentering(i);
            auto const qty = MeshData::scalarFromCentering(c);
            elem_storages.emplace_back(layout.allocSize(qty));
            ScalarField tmp{tags.elemStatus[i].name(), qty, elem_storages[i].data(),
                            elem_storages[i].shape()};
            tags.elemStatus[i].setBuffer(&tmp);
        }
        tags.ghostElemsData._data    = &ghost_array;
        tags.degradedElemsData._data = &degraded_array;
    }

    PHARE::core::NdArrayVector<2, double> phi_storage;
    std::vector<PHARE::core::NdArrayVector<2, double>> elem_storages;
    PHARE::core::GhostElemPack<2>::ghost_elem_array_type ghost_array{};
    PHARE::core::GhostElemPack<2>::ghost_elem_array_type degraded_array{};
    MeshData tags;
};
} // namespace

TEST(InnerBoundaryMeshClassifier, computesReasonableDefaultCutEpsFromLayout)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 1.};
    GridLayout layout{{0.2, 0.1}, {4u, 4u}, {0., 0.}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto const origin_node = physicalLocalIndex(layout, buffers.tags.signedDistanceAtNodes, 0u, 0u);
    EXPECT_LT(buffers.tags.signedDistanceAtNodes(origin_node), 0.);
}

TEST(InnerBoundaryMeshClassifier, tagsCutInactiveAndGhostGeometry)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto classifier = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto& cellField = buffers.tags.getStatusFieldFromCentering(kCellC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              cellField(physicalLocalIndex(layout, cellField, 0u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              cellField(physicalLocalIndex(layout, cellField, 1u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              cellField(physicalLocalIndex(layout, cellField, 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Fluid),
              cellField(physicalLocalIndex(layout, cellField, 3u, 0u)));

    auto& faceXField = buffers.tags.getStatusFieldFromCentering(kFaceXC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              faceXField(physicalLocalIndex(layout, faceXField, 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Fluid),
              faceXField(physicalLocalIndex(layout, faceXField, 1u, 0u)));
    auto& faceYField = buffers.tags.getStatusFieldFromCentering(kFaceYC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              faceYField(physicalLocalIndex(layout, faceYField, 0u, 0u)));

    EXPECT_FALSE(buffers.tags.getDegradedElemsFromCentering(kCellC).empty());
}

TEST(InnerBoundaryMeshClassifier, ghostCellListIsNonEmpty)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto classifier = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kCellC).empty());
}

TEST(InnerBoundaryMeshClassifier, ghostCellHasCorrectMirrorPointAndNormal)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto classifier = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto const target_idx
        = physicalLocalIndex(layout, buffers.tags.getStatusFieldFromCentering(kCellC), 0u, 0u);
    auto const& ghost_cells = buffers.tags.getGhostDataFromCentering(kCellC);
    auto it                 = std::find_if(ghost_cells.begin(), ghost_cells.end(),
                                           [&](auto const& g) { return g.index == target_idx; });
    ASSERT_NE(it, ghost_cells.end()) << "Ghost cell for physical[0,0] not found in ghostCells";

    EXPECT_NEAR(it->mirrorPoint[0], 1.5, eps);
    EXPECT_NEAR(it->mirrorPoint[1], 0.5, eps);
    EXPECT_NEAR(it->normal[0], 1.0, eps);
    EXPECT_NEAR(it->normal[1], 0.0, eps);
    EXPECT_TRUE(it->mirrorIsInterpolable);
}

TEST(InnerBoundaryMeshClassifier, ghostFaceListIsNonEmpty)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto classifier = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kFaceXC).empty());
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kFaceYC).empty());
}

TEST(InnerBoundaryMeshClassifier, ghostCellMirrorsAreInPatch)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto classifier = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto const& ghost_cells = buffers.tags.getGhostDataFromCentering(kCellC);
    ASSERT_FALSE(ghost_cells.empty());

    auto const& dx        = layout.meshSize();
    auto const& amrBox    = layout.AMRBox();
    auto inPhysicalDomain = [&](auto const& point) {
        for (auto d = 0u; d < 2u; ++d)
        {
            int const iCell = static_cast<int>(std::floor(point[d] / dx[d]));
            if (iCell < amrBox.lower[d] || iCell > amrBox.upper[d])
                return false;
        }
        return true;
    };

    std::size_t interpolable_in_physical = 0;
    for (auto const& g : ghost_cells)
    {
        if (inPhysicalDomain(g.mirrorPoint))
        {
            EXPECT_TRUE(g.mirrorIsInterpolable)
                << "Ghost cell whose mirror lies in the physical domain must be interpolable";
            ++interpolable_in_physical;
        }
    }
    EXPECT_GT(interpolable_in_physical, 0u)
        << "At least one ghost mirror must land in the physical domain";
}

TEST(InnerBoundaryMeshClassifier, underResolvedSphereProducesDegradedElements)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 1.0};
    GridLayout layout{
        {1.0, 1.0}, {12u, 12u}, {-6., -6.}, PHARE::core::Box<int, 2>{{-6, -6}, {5, 5}}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    EXPECT_FALSE(buffers.tags.getDegradedElemsFromCentering(kCellC).empty())
        << "an under-resolved sphere must emit degraded cell elements";
}

TEST(InnerBoundaryMeshClassifier, resolvedSphereProducesSecondOrderBand)
{
    auto const ng = GridLayout::options.field_ghost_width;

    double const dx           = 1.0;
    double const radius       = (static_cast<double>(ng) + 2.0) * dx;
    auto const half           = static_cast<std::uint32_t>(std::ceil(radius / dx)) + ng + 2u;
    std::uint32_t const ncell = 2u * half;
    double const origin       = -static_cast<double>(half) * dx;

    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, radius};
    auto const lower = -static_cast<int>(half);
    GridLayout layout{{dx, dx},
                      {ncell, ncell},
                      {origin, origin},
                      PHARE::core::Box<int, 2>{{lower, lower}, {-lower - 1, -lower - 1}}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    EXPECT_FALSE(buffers.tags.getDegradedElemsFromCentering(kCellC).empty())
        << "a resolved sphere must emit a second-order band";
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kCellC).empty());
}

template<typename FieldT>
PHARE::core::Point<double, 2> elementCoordinates(GridLayout const& layout, FieldT const& field,
                                                 PHARE::core::Point<std::uint32_t, 2> const& idx)
{
    using PHARE::core::Direction;
    auto const centering = GridLayout::centering(field);
    auto const& dx       = layout.meshSize();
    PHARE::core::Point<double, 2> coords;
    for (auto d = 0u; d < 2u; ++d)
    {
        auto const start = layout.physicalStartIndex(field, d == 0 ? Direction::X : Direction::Y);
        auto const shift = centering[d] == PHARE::core::QtyCentering::dual ? 0.5 : 0.;
        coords[d]        = layout.origin()[d]
                    + (static_cast<double>(idx[d]) - static_cast<double>(start) + shift) * dx[d];
    }
    return coords;
}

TEST(InnerBoundaryMeshClassifier, degradedListsFilledWhenResolved)
{
    double const dx           = 1.0;
    double const radius       = 20. * dx;
    std::uint32_t const ncell = 60u;
    double const origin       = -30. * dx;

    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, radius};
    GridLayout layout{
        {dx, dx}, {ncell, ncell}, {origin, origin}, PHARE::core::Box<int, 2>{{-30, -30}, {29, 29}}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    double const reach
        = (4. + static_cast<double>(reconstructionNghosts) + 1.) * dx * std::sqrt(2.);
    for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
    {
        auto const centering = MeshData::idxToCentering(i);
        auto const& degraded = buffers.tags.getDegradedElemsFromCentering(centering);
        EXPECT_FALSE(degraded.empty()) << "element type " << i;
        auto const& status = buffers.tags.getStatusFieldFromCentering(centering);
        for (auto const& elem : degraded)
            EXPECT_LE(
                std::abs(sphere.signedDistance(elementCoordinates(layout, status, elem.index))),
                reach);
    }
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kCellC).empty());
}

TEST(InnerBoundaryMeshClassifier, degradedListsFilledWhenUnderResolved)
{
    double const dx           = 1.0;
    double const radius       = 3. * dx;
    std::uint32_t const ncell = 40u;
    double const origin       = -20. * dx;

    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, radius};
    GridLayout layout{
        {dx, dx}, {ncell, ncell}, {origin, origin}, PHARE::core::Box<int, 2>{{-20, -20}, {19, 19}}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    double const reach
        = (static_cast<double>(reconstructionNghosts) + 1. + 2.) * dx * std::sqrt(2.);

    for (auto const& centering : {kFaceXC, kFaceYC})
    {
        auto const& degraded = buffers.tags.getDegradedElemsFromCentering(centering);
        EXPECT_FALSE(degraded.empty());

        auto const& status = buffers.tags.getStatusFieldFromCentering(centering);
        for (auto const& elem : degraded)
        {
            auto const coords = elementCoordinates(layout, status, elem.index);
            EXPECT_LE(std::abs(sphere.signedDistance(coords)), reach);
        }
    }
}

TEST(InnerBoundaryMeshClassifier, underResolvedSphereCenteredOnANodeClassifies)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.0, 0.0}, 1.5};
    GridLayout layout{
        {1.0, 1.0}, {12u, 12u}, {-6., -6.}, PHARE::core::Box<int, 2>{{-6, -6}, {5, 5}}};
    auto classifier = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    EXPECT_NO_THROW(classifier(layout, buffers.tags));
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kCellC).empty());
}

template<typename Layout>
Layout squareLayout(int lower, std::uint32_t ncell)
{
    auto const upper = lower + static_cast<int>(ncell) - 1;
    return Layout{{1.0, 1.0},
                  {ncell, ncell},
                  {static_cast<double>(lower), static_cast<double>(lower)},
                  PHARE::core::Box<int, 2>{{lower, lower}, {upper, upper}}};
}

template<typename MapperT, typename Layout>
void expectBoxStencilOfEvolvedCellsNeverReachesInactive(double radius, int reach)
{
    using PHARE::core::Direction;
    using PHARE::core::ElemStatus;
    using PHARE::core::toDouble;

    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, radius};
    auto const layout = squareLayout<Layout>(-20, 40u);
    auto classifier   = MapperT::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto const& cells = buffers.tags.getStatusFieldFromCentering(kCellC);
    auto const x0     = layout.physicalStartIndex(cells, Direction::X);
    auto const x1     = layout.physicalEndIndex(cells, Direction::X);
    auto const y0     = layout.physicalStartIndex(cells, Direction::Y);
    auto const y1     = layout.physicalEndIndex(cells, Direction::Y);

    std::size_t reached = 0;
    for (auto i = x0; i <= x1; ++i)
        for (auto j = y0; j <= y1; ++j)
        {
            auto const s = cells(i, j);
            if (s != toDouble(ElemStatus::Fluid) and s != toDouble(ElemStatus::Cut))
                continue;
            for (int di = -reach; di <= reach; ++di)
                for (int dj = -reach; dj <= reach; ++dj)
                    if (cells(i + di, j + dj) == toDouble(ElemStatus::Inactive))
                        ++reached;
        }
    EXPECT_EQ(reached, 0u) << "radius " << radius;
}

TEST(InnerBoundaryMeshClassifier, schemeIsSecondOrderOnResolvedWenozLevel)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 20.0};
    auto const layout = squareLayout<GridLayout>(-30, 60u);
    auto const params = Mapper::withDefaults(sphere, layout).params();

    EXPECT_EQ(params.scheme, PHARE::core::NearBodyScheme::SecondOrder);
    EXPECT_EQ(params.nghosts, 4u);
}

TEST(InnerBoundaryMeshClassifier, schemeIsFirstOrderWhenUnderResolved)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 3.0};
    auto const layout = squareLayout<GridLayout>(-20, 40u);
    auto const params = Mapper::withDefaults(sphere, layout).params();

    EXPECT_EQ(params.scheme, PHARE::core::NearBodyScheme::FirstOrder);
    EXPECT_EQ(params.nghosts, 1u);
}

TEST(InnerBoundaryMeshClassifier, underResolvedThresholdIsFiveCells)
{
    auto const layout = squareLayout<GridLayout>(-20, 40u);

    PHARE::core::SphereInnerBoundary<2> below{"sphere", {0.05, 0.05}, 4.9};
    EXPECT_EQ(Mapper::withDefaults(below, layout).params().scheme,
              PHARE::core::NearBodyScheme::FirstOrder);

    PHARE::core::SphereInnerBoundary<2> above{"sphere", {0.05, 0.05}, 5.1};
    EXPECT_EQ(Mapper::withDefaults(above, layout).params().scheme,
              PHARE::core::NearBodyScheme::SecondOrder);
}

TEST(InnerBoundaryMeshClassifier, linearReconstructionHasNoBand)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 20.0};
    auto const layout = squareLayout<LinearGridLayout>(-30, 60u);
    auto const params = LinearMapper::withDefaults(sphere, layout).params();

    EXPECT_EQ(params.scheme, PHARE::core::NearBodyScheme::None);
    EXPECT_EQ(params.nghosts, 4u);
}

TEST(InnerBoundaryMeshClassifier, boxStencilOfEvolvedCellsNeverReachesInactive)
{
    for (double radius : {7.3, 9.7, 12.3})
    {
        expectBoxStencilOfEvolvedCellsNeverReachesInactive<Mapper, GridLayout>(radius, 2);
        expectBoxStencilOfEvolvedCellsNeverReachesInactive<LinearMapper, LinearGridLayout>(radius,
                                                                                           2);
    }
}

std::set<std::pair<std::uint32_t, std::uint32_t>> listedSet(auto const& degraded)
{
    std::set<std::pair<std::uint32_t, std::uint32_t>> listed;
    for (auto const& elem : degraded)
        listed.insert({elem.index[0], elem.index[1]});
    return listed;
}

TEST(InnerBoundaryMeshClassifier, secondOrderBandCoversWenozStencils)
{
    using PHARE::core::Direction;
    using PHARE::core::ElemStatus;
    using PHARE::core::toDouble;

    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 20.0};
    auto const layout = squareLayout<GridLayout>(-30, 60u);
    auto classifier   = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    auto const& cells = buffers.tags.getStatusFieldFromCentering(kCellC);
    auto nearBody     = [&](int i, int j) {
        auto const s = cells(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
        return s == toDouble(ElemStatus::Cut) or s == toDouble(ElemStatus::Ghost);
    };

    auto checkFaces = [&](auto const& centering, int axis) {
        auto const& status = buffers.tags.getStatusFieldFromCentering(centering);
        auto const listed  = listedSet(buffers.tags.getDegradedElemsFromCentering(centering));
        auto const x0      = static_cast<int>(layout.physicalStartIndex(status, Direction::X));
        auto const x1      = static_cast<int>(layout.physicalEndIndex(status, Direction::X));
        auto const y0      = static_cast<int>(layout.physicalStartIndex(status, Direction::Y));
        auto const y1      = static_cast<int>(layout.physicalEndIndex(status, Direction::Y));
        std::size_t missed = 0;
        for (int i = x0; i <= x1; ++i)
            for (int j = y0; j <= y1; ++j)
            {
                bool touches = false;
                for (int k = -3; k <= 2; ++k)
                    touches = touches or (axis == 0 ? nearBody(i + k, j) : nearBody(i, j + k));
                if (touches
                    and !listed.contains(
                        {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)}))
                    ++missed;
            }
        EXPECT_EQ(missed, 0u) << "axis " << axis;
    };
    checkFaces(kFaceXC, 0);
    checkFaces(kFaceYC, 1);

    auto const& nodes  = buffers.tags.getStatusFieldFromCentering(kNodeC);
    auto const listed  = listedSet(buffers.tags.getDegradedElemsFromCentering(kNodeC));
    auto const x0      = static_cast<int>(layout.physicalStartIndex(nodes, Direction::X));
    auto const x1      = static_cast<int>(layout.physicalEndIndex(nodes, Direction::X));
    auto const y0      = static_cast<int>(layout.physicalStartIndex(nodes, Direction::Y));
    auto const y1      = static_cast<int>(layout.physicalEndIndex(nodes, Direction::Y));
    std::size_t missed = 0;
    for (int i = x0; i <= x1; ++i)
        for (int j = y0; j <= y1; ++j)
        {
            bool touches = false;
            for (int di = -3; di <= 2; ++di)
                for (int dj = -3; dj <= 2; ++dj)
                    touches = touches or nearBody(i + di, j + dj);
            if (touches
                and !listed.contains(
                    {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)}))
                ++missed;
        }
    EXPECT_EQ(missed, 0u) << "nodes";
}

TEST(InnerBoundaryMeshClassifier, secondOrderBandFacesRespectStencilMargin)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {2.05, 2.05}, 6.0};
    auto const layout = squareLayout<GridLayout>(0, 12u);
    auto classifier   = Mapper::withDefaults(sphere, layout);
    ASSERT_EQ(classifier.params().scheme, PHARE::core::NearBodyScheme::SecondOrder);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    classifier(layout, buffers.tags);

    for (auto const& [centering, axis] : {std::pair{kFaceXC, 0u}, std::pair{kFaceYC, 1u}})
    {
        auto const& degraded = buffers.tags.getDegradedElemsFromCentering(centering);
        EXPECT_FALSE(degraded.empty());
        auto const shape   = buffers.tags.getStatusFieldFromCentering(centering).shape();
        bool inGhostRegion = false;
        for (auto const& elem : degraded)
        {
            EXPECT_GE(elem.index[axis], 2u);
            EXPECT_LE(elem.index[axis] + 3u, shape[axis]);
            inGhostRegion = inGhostRegion or elem.index[axis] < 4u;
        }
        EXPECT_TRUE(inGhostRegion) << "band must reach the ghost region for this test to bite";
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
