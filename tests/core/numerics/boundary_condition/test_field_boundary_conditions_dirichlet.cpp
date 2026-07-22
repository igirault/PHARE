#include "gtest/gtest.h"

#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "tests/core/numerics/boundary_condition/hybrid_bc_test_fixtures.hpp"

using namespace PHARE::core;


TEST_F(FieldBC1D, DirichletSetsLowerGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, makeCtx(acc, 0.0));

    // Interior is constant = interiorValue, so ghost = 2*value - interiorValue
    double expected = 2.0 * value - interiorValue;
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(g), expected);
}

TEST_F(FieldBC1D, DirichletSetsUpperGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, makeCtx(acc, 0.0));

    double expected       = 2.0 * value - interiorValue;
    std::uint32_t allocSz = grid.shape()[0];
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(allocSz - 1 - g), expected);
}


// A space- and time-varying Dirichlet value f(x, t) = x + t must be evaluated at each
// ghost node's own coordinate and at ctx.time, then fed into the same linear extrapolation
// as the constant case. Running at two different times proves ctx.time threads through.
TEST_F(FieldBC1D, DirichletTimeVaryingFunctionLowerGhost)
{
    PHARE::initializer::SpaceTimeFunction<1> fn = [](std::vector<double> const& x, double t) {
        std::vector<double> out(x.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            out[k] = x[k] + t;
        return std::shared_ptr<Span<double>>{std::make_shared<VectorSpan<double>>(std::move(out))};
    };

    auto const amrLower = layout.AMRGhostBoxFor(field).lower;

    auto checkAtTime = [&](double const t) {
        for (std::uint32_t i = 0; i < grid.shape()[0]; ++i)
            field(i) = ghostSentinel;
        for (std::uint32_t i = physStart; i <= physEnd; ++i)
            field(i) = interiorValue;

        FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{fn};
        bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, makeCtx(acc, t));

        // pure-ghost nodes mirror into the (constant) interior, so ghost = 2*f(x_g,t) - interior
        for (std::uint32_t g = 0; g < ghostWidth; ++g)
        {
            Point<int, 1> amr;
            amr[0]         = amrLower[0] + static_cast<int>(g);
            double const x = layout.fieldNodeCoordinates(field, amr)[0];
            EXPECT_DOUBLE_EQ(field(g), 2.0 * (x + t) - interiorValue);
        }
    };

    checkAtTime(0.0);
    checkAtTime(2.5);
}


// ─── 2D scalar ────────────────────────────────────────────────────────────────

TEST_F(FieldBC2D, DirichletAtXBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interiorValue;
    FieldDirichletBoundaryCondition<Field2D, GridLayout2D> bc{value};
    bc.apply(field, BoundaryLocation::XLower, xLowerGhostCellBox2D(), layout, makeCtx(acc, 0.0));
    bc.apply(field, BoundaryLocation::XUpper, xUpperGhostCellBox2D(), layout, makeCtx(acc, 0.0));

    std::uint32_t const allocX = grid.shape()[0];
    std::uint32_t sy = layout.physicalStartIndex(qty, Direction::Y);
    std::uint32_t ey = layout.physicalEndIndex(qty, Direction::Y);
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(field(g, iy), expected) << "lower g=" << g << " iy=" << iy;
            EXPECT_DOUBLE_EQ(field(allocX - 1 - g, iy), expected) << "upper g=" << g;
        }
}

TEST_F(FieldBC2D, DirichletAtYBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interiorValue;
    FieldDirichletBoundaryCondition<Field2D, GridLayout2D> bc{value};
    bc.apply(field, BoundaryLocation::YLower, yLowerGhostCellBox2D(), layout, makeCtx(acc, 0.0));
    bc.apply(field, BoundaryLocation::YUpper, yUpperGhostCellBox2D(), layout, makeCtx(acc, 0.0));

    std::uint32_t const allocY = grid.shape()[1];
    std::uint32_t sx = layout.physicalStartIndex(qty, Direction::X);
    std::uint32_t ex = layout.physicalEndIndex(qty, Direction::X);
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(field(ix, g), expected) << "lower g=" << g << " ix=" << ix;
            EXPECT_DOUBLE_EQ(field(ix, allocY - 1 - g), expected) << "upper g=" << g;
        }
}


// ─── 3D scalar ────────────────────────────────────────────────────────────────

TEST_F(FieldBC3D, DirichletAtZBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interiorValue;
    FieldDirichletBoundaryCondition<Field3D, GridLayout3D> bc{value};
    bc.apply(field, BoundaryLocation::ZLower, zLowerGhostCellBox3D(), layout, makeCtx(acc, 0.0));
    bc.apply(field, BoundaryLocation::ZUpper, zUpperGhostCellBox3D(), layout, makeCtx(acc, 0.0));

    std::uint32_t const allocZ = grid.shape()[2];
    std::uint32_t sx = layout.physicalStartIndex(qty, Direction::X);
    std::uint32_t ex = layout.physicalEndIndex(qty, Direction::X);
    std::uint32_t sy = layout.physicalStartIndex(qty, Direction::Y);
    std::uint32_t ey = layout.physicalEndIndex(qty, Direction::Y);
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(field(ix, iy, g), expected) << "lower g=" << g;
                EXPECT_DOUBLE_EQ(field(ix, iy, allocZ - 1 - g), expected) << "upper g=" << g;
            }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
