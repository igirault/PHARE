#include "gtest/gtest.h"

#include "core/data/field/field.hpp"
#include "core/data/tensorfield/tensorfield.hpp"
#include "core/numerics/boundary_condition/field_b1_from_btot_boundary_condition.hpp"
#include "tests/core/numerics/boundary_condition/mhd_bc_test_fixtures.hpp"

using namespace PHARE::core;

// ════════════════════════════════════════════════════════════════════════════
// FieldB1FromBtotBoundaryCondition
//
// Prescribes the *total* field B = B0 + B1 to a constant Btot at the boundary,
// writing the ghost values of the perturbation B1 (= B - B0) with a spatially
// varying background B0 read from the accessor. Transverse components follow a
// Dirichlet on the total field; the normal component is set so that div B1 = 0.
// ════════════════════════════════════════════════════════════════════════════

namespace
{
constexpr std::array<double, 3> Btot{2.0, 0.5, -0.3};
constexpr double interiorB1 = 7.0;
} // namespace


// ─── 1D: transverse components follow a Dirichlet on the total field ─────────

struct B1FromBtotBC1D : testing::Test
{
    GridLayoutMHD1D layout{{0.1}, {nCellsMHD}, {0.0}};

    GridMHD1D rhoGrid{"rho", MHDQuantity::Scalar::rho, layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD1D PGrid{"P", MHDQuantity::Scalar::P, layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD1D EtotGrid{"Etot", MHDQuantity::Scalar::Etot1,
                       layout.allocSize(MHDQuantity::Scalar::Etot1)};

    UsableVecFieldMHD<1> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<1> Bvec{"B", layout, MHDQuantity::Vector::B1};
    UsableVecFieldMHD<1> B0vec{"B0", layout, MHDQuantity::Vector::B0};

    MHDPatchFieldAccessorTest<1> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec, &B0vec};

    B1FromBtotBC1D()
    {
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            auto& b1 = Bvec[comp];
            auto& b0 = B0vec[comp];
            for (std::uint32_t i = 0; i < b1.shape()[0]; ++i)
            {
                b1(i) = interiorB1;
                // distinct, position-dependent background so B0(ghost) != B0(mirror)
                b0(i) = 0.05 * static_cast<double>(comp + 1) + 0.01 * static_cast<double>(i);
            }
        }
    }
};

TEST_F(B1FromBtotBC1D, TransverseTotalFieldEqualsPrescribedAtXBoundaries)
{
    auto B = Bvec.super();
    FieldB1FromBtotBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D> bc{Btot};
    bc.apply(B, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout, makeCtx(acc));
    bc.apply(B, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout, makeCtx(acc));

    auto check = [&](BoundaryLocation loc, Box<std::uint32_t, 1> const& ghostBox) {
        Side const side = getSide(loc);
        for (std::size_t comp : {1u, 2u}) // transverse components for an x-boundary
        {
            auto qty       = MHDQuantity::componentsQuantities(MHDQuantity::Vector::B1)[comp];
            auto centering = layout.centering(qty)[static_cast<std::size_t>(Direction::X)];
            auto& b1       = Bvec[comp];
            auto& b0       = B0vec[comp];
            for (auto const& index : layout.toFieldBox(ghostBox, qty))
            {
                auto mirror        = layout.boundaryMirrored(Direction::X, side, centering, index);
                double total_index = b1(index[0]) + b0(index[0]);
                if (mirror[0] == index[0])
                    EXPECT_NEAR(total_index, Btot[comp], 1e-12) << "boundary node comp=" << comp;
                else
                {
                    double total_mirror   = b1(mirror[0]) + b0(mirror[0]);
                    double expected_total = 2.0 * Btot[comp] - total_mirror;
                    EXPECT_NEAR(total_index, expected_total, 1e-12)
                        << "ghost comp=" << comp << " index=" << index[0];
                }
            }
        }
    };

    check(BoundaryLocation::XLower, mhdLowerGhostCellBox());
    check(BoundaryLocation::XUpper, mhdUpperGhostCellBox());
}


// ─── 2D: normal component keeps the perturbation divergence zero ─────────────

struct B1FromBtotBC2D : testing::Test
{
    GridLayoutMHD2D layout{{0.1, 0.1}, {nCellsMHDX2D, nCellsMHDY2D}, {0.0, 0.0}};

    GridMHD2D rhoGrid{"rho", MHDQuantity::Scalar::rho, layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD2D PGrid{"P", MHDQuantity::Scalar::P, layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD2D EtotGrid{"Etot", MHDQuantity::Scalar::Etot1,
                       layout.allocSize(MHDQuantity::Scalar::Etot1)};

    UsableVecFieldMHD<2> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<2> Bvec{"B", layout, MHDQuantity::Vector::B1};
    UsableVecFieldMHD<2> B0vec{"B0", layout, MHDQuantity::Vector::B0};

    MHDPatchFieldAccessorTest<2> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec, &B0vec};

    B1FromBtotBC2D()
    {
        auto fill = [](auto& f, auto func) {
            auto [nx, ny] = f.shape();
            for (std::uint32_t ix = 0; ix < nx; ++ix)
                for (std::uint32_t iy = 0; iy < ny; ++iy)
                    f(ix, iy) = func(ix, iy);
        };
        // non-uniform By along y so the transverse divergence in the x-ghost loop is non-trivial
        fill(Bvec[0], [](auto, auto) { return interiorB1; });
        fill(Bvec[1], [](auto, auto iy) { return interiorB1 + 0.1 * static_cast<double>(iy); });
        fill(Bvec[2], [](auto, auto) { return interiorB1; });

        for (std::size_t comp = 0; comp < 3; ++comp)
            fill(B0vec[comp], [comp](auto ix, auto iy) {
                return 0.05 * static_cast<double>(comp + 1) + 0.01 * static_cast<double>(ix)
                       + 0.02 * static_cast<double>(iy);
            });
    }
};

TEST_F(B1FromBtotBC2D, NormalComponentKeepsXGhostPerturbationDivergenceZero)
{
    auto B = Bvec.super();
    FieldB1FromBtotBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D> bc{Btot};
    bc.apply(B, BoundaryLocation::XLower, mhd2DXLowerGhostBox(), layout, makeCtx(acc));
    bc.apply(B, BoundaryLocation::XUpper, mhd2DXUpperGhostBox(), layout, makeCtx(acc));

    auto& Bx        = Bvec[0];
    auto& By        = Bvec[1];
    auto divergence = [&](std::uint32_t ix, std::uint32_t iy) {
        return Bx(ix + 1, iy) - Bx(ix, iy) + By(ix, iy + 1) - By(ix, iy);
    };
    for (auto const& index : mhd2DXLowerGhostBox())
        EXPECT_NEAR(divergence(index[0], index[1]), 0.0, 1e-12)
            << "lower divergence at (" << index[0] << ", " << index[1] << ")";
    for (auto const& index : mhd2DXUpperGhostBox())
        EXPECT_NEAR(divergence(index[0], index[1]), 0.0, 1e-12)
            << "upper divergence at (" << index[0] << ", " << index[1] << ")";
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
