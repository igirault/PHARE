#include "gtest/gtest.h"

#include "core/boundary/boundary_defs.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_total_energy_from_pressure_boundary_condition.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "tests/core/numerics/boundary_condition/mhd_bc_test_fixtures.hpp"

using namespace PHARE::core;


namespace
{
constexpr std::array<double, 3> Bprescribed{0.4, -0.2, 0.7};
}


struct DirichletBEnergySubBC1D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = -0.5;
    static constexpr double vz_val   = 0.25;
    static constexpr double B_int    = 7.0;
    static constexpr double P_val    = 1.0;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (B_int * B_int + B_int * B_int + B_int * B_int);

    GridLayoutMHD1D layout{{0.1}, {nCellsMHD}, {0.0}};

    GridMHD1D rhoGrid{"rho", MHDQuantity::Scalar::rho, layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD1D PGrid{"P", MHDQuantity::Scalar::P, layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD1D EtotGrid{"Etot", MHDQuantity::Scalar::Etot,
                       layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<1> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<1> Bvec{"B", layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<1> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<1>& rhoField{*(&rhoGrid)};
    FieldMHD<1>& PField{*(&PGrid)};
    FieldMHD<1>& EtotField{*(&EtotGrid)};

    DirichletBEnergySubBC1D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            for (std::uint32_t i = 0; i < f.shape()[0]; ++i)
                f(i) = sentinel;
            std::uint32_t ps = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe = layout.physicalEndIndex(qty, Direction::X);
            for (std::uint32_t i = ps; i <= pe; ++i)
                f(i) = interior_val;
        };

        fill_scalar(rhoField, MHDQuantity::Scalar::rho, rho_val);
        fill_scalar(PField, MHDQuantity::Scalar::P, P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, B_int);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, B_int);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, B_int);
    }

    void applyAndCheck()
    {
        double const time = 0.0;
        auto rho_bc
            = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
        auto P_bc = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
        auto rhoV_bc
            = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
        auto B_bc
            = std::make_shared<FieldDirichletBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>(
                Bprescribed);

        FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D> bc{
            rho_bc, rhoV_bc, B_bc, P_bc, gamma};

        bc.apply(EtotField, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout,
                 makeCtx(acc, time));
        bc.apply(EtotField, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout,
                 makeCtx(acc, time));

        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            double const bd       = Bprescribed[comp];
            double const expected = 2.0 * bd - B_int;
            auto& bField          = Bvec[comp];
            auto const qty        = MHDQuantity::componentsQuantities(MHDQuantity::Vector::B)[comp];
            auto const centering  = layout.centering(qty)[0];

            for (auto const& index : layout.toFieldBox(mhdLowerGhostCellBox(), qty))
            {
                auto const mirror
                    = layout.boundaryMirrored(Direction::X, Side::Lower, centering, index);
                if (mirror[0] == index[0])
                    EXPECT_NEAR(bField(index[0]), bd, 1e-12) << "comp=" << comp << " t=" << time;
                else
                    EXPECT_NEAR(bField(index[0]), expected, 1e-12)
                        << "comp=" << comp << " t=" << time;
            }
        }

        auto const etotQty = MHDQuantity::Scalar::Etot;
        double const kinetic
            = 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val);
        auto expectedEtotAt = [&](Point<std::uint32_t, 1> const& index) {
            double const bx
                = GridLayoutMHD1D::template project<GridLayoutMHD1D::implT::faceXToCellCenter>(
                    Bvec[0], index);
            double const by
                = GridLayoutMHD1D::template project<GridLayoutMHD1D::implT::faceYToCellCenter>(
                    Bvec[1], index);
            double const bz
                = GridLayoutMHD1D::template project<GridLayoutMHD1D::implT::faceZToCellCenter>(
                    Bvec[2], index);
            double const magnetic = 0.5 * (bx * bx + by * by + bz * bz);
            return P_val / (gamma - 1.0) + kinetic + magnetic;
        };

        for (auto const& box : {mhdLowerGhostCellBox(), mhdUpperGhostCellBox()})
            for (auto const& index : layout.toFieldBox(box, etotQty))
                EXPECT_NEAR(EtotField(index[0]), expectedEtotAt(index), 1e-12)
                    << "index=" << index[0] << " t=" << time;
    }
};

TEST_F(DirichletBEnergySubBC1D, GhostEtotUsesPrescribedB)
{
    applyAndCheck();
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
