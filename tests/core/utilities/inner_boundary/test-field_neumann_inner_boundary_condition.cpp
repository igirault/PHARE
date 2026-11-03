#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/data/field/field.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "tests/core/utilities/inner_boundary/inner_boundary_test_fixtures.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "tests/core/utilities/inner_boundary/plane_inner_boundary.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/inner_boundary/field_neumann_inner_boundary_condition.hpp"
#include "core/utilities/box/box.hpp"

namespace
{
constexpr double eps = 1e-12;

using GridLayout = PHARE::core::IBTestGridLayout2D;
using Classifier
    = PHARE::core::InnerBoundaryMeshClassifier<2, GridLayout, PHARE::core::MHDQuantity>;
using MeshData    = PHARE::core::InnerBoundaryMeshData<2, PHARE::core::MHDQuantity>;
using ScalarField = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;

struct DummyState
{
};

constexpr std::array<PHARE::core::QtyCentering, 2> kCellC
    = {PHARE::core::QtyCentering::dual, PHARE::core::QtyCentering::dual};

struct MeshDataBuffers
{
    static constexpr char const* BOUNDARY_NAME = "test";

    explicit MeshDataBuffers(GridLayout const& layout)
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

struct NeumannBCFixture
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    MeshDataBuffers buffers{layout};

    NeumannBCFixture()
    {
        Classifier::Overrides ov;
        ov.cut_eps      = 1e-12;
        ov.inactive_eps = 0.0;
        auto classifier = Classifier::withDefaults(plane, layout, ov);
        classifier(layout, buffers.tags);
    }
};

} // namespace

TEST(FieldNeumannInnerBoundaryCondition, constantFieldIsUnchanged)
{
    NeumannBCFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.buffers.tags;

    constexpr double C = 7.0;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField field{"rho", PHARE::core::MHDQuantity::Scalar::CellCentered, storage.data(),
                      storage.shape()};

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            field(i, j) = C;

    PHARE::core::FieldNeumannInnerBoundaryCondition<ScalarField, GridLayout, DummyState> bc;
    DummyState state;
    bc.apply(field, layout, meshData, PHARE::core::InnerBCContext<DummyState>{state, state, 0.0});

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            EXPECT_NEAR(field(i, j), C, eps) << "cell (" << i << ", " << j << ") changed";
}

TEST(FieldNeumannInnerBoundaryCondition, ghostCellReceivesMirrorPointValue)
{
    NeumannBCFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.buffers.tags;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField field{"rho", PHARE::core::MHDQuantity::Scalar::CellCentered, storage.data(),
                      storage.shape()};

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
        {
            auto amr_pos = layout.localToAMR(PHARE::core::Point<std::uint32_t, 2>{i, j});
            auto amr_ij  = PHARE::core::Point<int, 2>{static_cast<int>(amr_pos[0]),
                                                      static_cast<int>(amr_pos[1])};
            auto pos     = layout.fieldNodeCoordinates(field, amr_ij);
            field(i, j)  = pos[0] + pos[1];
        }

    for (auto const& g : meshData.getGhostDataFromCentering(kCellC))
        field(g.index) = 0.0;

    PHARE::core::FieldNeumannInnerBoundaryCondition<ScalarField, GridLayout, DummyState> bc;
    DummyState state;
    bc.apply(field, layout, meshData, PHARE::core::InnerBCContext<DummyState>{state, state, 0.0});

    auto const& ghost_cells = meshData.getGhostDataFromCentering(kCellC);
    ASSERT_FALSE(ghost_cells.empty());

    bool found_in_patch = false;
    for (auto const& g : ghost_cells)
    {
        if (!g.mirrorIsInterpolable)
        {
            EXPECT_NEAR(field(g.index), 0.0, eps)
                << "out-of-patch ghost at (" << g.index[0] << ", " << g.index[1]
                << ") must not be touched by the BC";
            continue;
        }
        found_in_patch        = true;
        double const expected = g.mirrorPoint[0] + g.mirrorPoint[1];
        EXPECT_NEAR(field(g.index), expected, 1e-10)
            << "ghost at (" << g.index[0] << ", " << g.index[1] << ")";
    }
    EXPECT_TRUE(found_in_patch) << "at least one in-patch ghost must exist";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
