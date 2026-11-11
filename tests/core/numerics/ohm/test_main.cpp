
#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "core/numerics/ohm/ohm.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

#include "tests/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tests/core/data/gridlayout/test_centering_boxes.hpp"


#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <fstream>
#include <variant>

using namespace PHARE::core;


template<int dim, int interp>
class NDlayout
{
    NDlayout() {}

    using nDL = PHARE::core::PHARE_Types<PHARE::SimOpts{dim, interp}>::Hybrid::GridLayout_t;

public:
    static nDL create()
    {
        if constexpr (dim == 1)
        {
            return {{{0.1}}, {{50}}, {0.}};
        }
        else if constexpr (dim == 2)
        {
            return {{{0.1, 0.2}}, {{50, 40}}, {0., 0.}};
        }
        else if constexpr (dim == 3)
        {
            return {{{0.1, 0.2, 0.3}}, {{50, 40, 30}}, {0., 0., 0.}};
        }
    }
};




PHARE::initializer::PHAREDict createDict()
{
    PHARE::initializer::PHAREDict dict;

    dict["resistivity"]       = 1.0;
    dict["hyper_resistivity"] = 0.01;

    return dict;
}



template<typename TypeInfo /*= std::pair<DimConst<1>, InterpConst<1>>*/>
struct OhmTest : public ::testing::Test
{
    static constexpr auto dim    = typename TypeInfo::first_type{}();
    static constexpr auto interp = typename TypeInfo::second_type{}();

    using GridYee = PHARE::core::PHARE_Types<PHARE::SimOpts{dim, interp}>::Hybrid::GridLayout_t;
    using UsableVecFieldND = UsableVecField<dim>;
    using Grid_t           = Grid<NdArrayVector<dim>, HybridQuantity::Scalar>;
    using Ohm_t            = Ohm<GridYee>;

    GridYee layout = NDlayout<dim, interp>::create();

    Grid_t n;
    Grid_t P;
    UsableVecFieldND V, B, J, Enew;

    OhmInfo ohm_info;

    OhmTest()
        : n{"n", HybridQuantity::Scalar::rho, layout.allocSize(HybridQuantity::Scalar::rho)}
        , P{"P", HybridQuantity::Scalar::P, layout.allocSize(HybridQuantity::Scalar::P)}
        , V{"V", layout, HybridQuantity::Vector::V}
        , B{"B", layout, HybridQuantity::Vector::B}
        , J{"J", layout, HybridQuantity::Vector::J}
        , Enew{"Enew", layout, HybridQuantity::Vector::E}
        , ohm_info{OhmInfo::FROM(createDict())}
    {
        auto const& [Bx, By, Bz]          = B();
        auto const& [Jx, Jy, Jz]          = J();
        auto const& [Vx, Vy, Vz]          = V();
        auto const& [Exnew, Eynew, Eznew] = Enew();

        if constexpr (dim == 1)
        {
            auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
            auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);
            auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
            auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);

            for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
            {
                auto point = this->layout.fieldNodeCoordinates(
                    n, this->layout.localToAMR(Point{ix}.as_signed()));

                n(ix)  = std::cosh(0.5 * point[0]);
                Vx(ix) = std::sinh(0.2 * point[0]);
                Vy(ix) = std::sinh(0.3 * point[0]);
                Vz(ix) = std::sinh(0.4 * point[0]);
                P(ix)  = std::cosh(0.5 * point[0]);
                Bx(ix) = std::cosh(0.2 * point[0]);
                Jy(ix) = std::tanh(0.3 * point[0]);
                Jz(ix) = std::tanh(0.4 * point[0]);
            }

            for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
            {
                auto point = this->layout.fieldNodeCoordinates(
                    Bz, this->layout.localToAMR(Point{ix}.as_signed()));

                By(ix) = std::cosh(0.3 * point[0]);
                Bz(ix) = std::cosh(0.4 * point[0]);
                Jx(ix) = std::tanh(0.2 * point[0]);
            }
        }

        if constexpr (dim == 2)
        {
            auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
            auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);
            auto gsi_p_Y = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Y);
            auto gei_p_Y = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Y);
            auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
            auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);
            auto gsi_d_Y = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Y);
            auto gei_d_Y = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Y);

            for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
            {
                for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
                {
                    auto point = this->layout.fieldNodeCoordinates(
                        n, this->layout.localToAMR(Point{ix, iy}.as_signed()));

                    n(ix, iy)  = std::cosh(0.5 * point[0]) * std::cosh(0.5 * point[1]);
                    Vx(ix, iy) = std::sinh(0.2 * point[0]) * std::sinh(0.2 * point[1]);
                    Vy(ix, iy) = std::sinh(0.3 * point[0]) * std::sinh(0.3 * point[1]);
                    Vz(ix, iy) = std::sinh(0.4 * point[0]) * std::sinh(0.4 * point[1]);
                    P(ix, iy)  = std::cosh(0.5 * point[0]) * std::cosh(0.5 * point[1]);
                    Jz(ix, iy) = std::tanh(0.4 * point[0]) * std::tanh(0.4 * point[1]);
                }
                for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
                {
                    auto point = this->layout.fieldNodeCoordinates(
                        Bx, this->layout.localToAMR(Point{ix, iy}.as_signed()));

                    Bx(ix, iy) = std::cosh(0.2 * point[0]) * std::cosh(0.2 * point[1]);
                    Jy(ix, iy) = std::tanh(0.3 * point[0]) * std::tanh(0.3 * point[1]);
                }
            }
            for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
            {
                for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
                {
                    auto point = this->layout.fieldNodeCoordinates(
                        Jx, this->layout.localToAMR(Point{ix, iy}.as_signed()));

                    By(ix, iy) = std::cosh(0.3 * point[0]) * std::cosh(0.3 * point[1]);
                    Jx(ix, iy) = std::tanh(0.2 * point[0]) * std::tanh(0.2 * point[1]);
                }
                for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
                {
                    auto point = this->layout.fieldNodeCoordinates(
                        Bz, this->layout.localToAMR(Point{ix, iy}.as_signed()));

                    Bz(ix, iy) = std::cosh(0.4 * point[0]) * std::cosh(0.4 * point[1]);
                }
            }
        }

        if constexpr (dim == 3)
        {
            auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
            auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);
            auto gsi_p_Y = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Y);
            auto gei_p_Y = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Y);
            auto gsi_p_Z = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Z);
            auto gei_p_Z = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Z);
            auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
            auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);
            auto gsi_d_Y = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Y);
            auto gei_d_Y = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Y);
            auto gsi_d_Z = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Z);
            auto gei_d_Z = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Z);

            for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
            {
                for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
                {
                    for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            n, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        n(ix, iy, iz) = std::cosh(0.5 * point[0]) * std::cosh(0.5 * point[1])
                                        * std::cosh(0.5 * point[2]);
                        Vx(ix, iy, iz) = std::sinh(0.2 * point[0]) * std::sinh(0.2 * point[1])
                                         * std::sinh(0.2 * point[2]);
                        Vy(ix, iy, iz) = std::sinh(0.3 * point[0]) * std::sinh(0.3 * point[1])
                                         * std::sinh(0.3 * point[2]);
                        Vz(ix, iy, iz) = std::sinh(0.4 * point[0]) * std::sinh(0.4 * point[1])
                                         * std::sinh(0.4 * point[2]);
                        P(ix, iy, iz) = std::cosh(0.5 * point[0]) * std::cosh(0.5 * point[1])
                                        * std::cosh(0.5 * point[2]);
                    }
                    for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            Jz, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        Jz(ix, iy, iz) = std::tanh(0.4 * point[0]) * std::tanh(0.4 * point[1])
                                         * std::tanh(0.4 * point[2]);
                    }
                }
                for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
                {
                    for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            Jy, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        Jy(ix, iy, iz) = std::tanh(0.3 * point[0]) * std::tanh(0.3 * point[1])
                                         * std::tanh(0.3 * point[2]);
                    }
                    for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            Bx, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        Bx(ix, iy, iz) = std::cosh(0.2 * point[0]) * std::cosh(0.2 * point[1])
                                         * std::cosh(0.2 * point[2]);
                    }
                }
            }
            for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
            {
                for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
                {
                    for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            Jx, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        Jx(ix, iy, iz) = std::tanh(0.2 * point[0]) * std::tanh(0.2 * point[1])
                                         * std::tanh(0.2 * point[2]);
                    }
                    for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            By, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        By(ix, iy, iz) = std::cosh(0.3 * point[0]) * std::cosh(0.3 * point[1])
                                         * std::cosh(0.3 * point[2]);
                    }
                }
                for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
                {
                    for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
                    {
                        auto point = this->layout.fieldNodeCoordinates(
                            Bz, this->layout.localToAMR(Point{ix, iy, iz}.as_signed()));

                        Bz(ix, iy, iz) = std::cosh(0.4 * point[0]) * std::cosh(0.4 * point[1])
                                         * std::cosh(0.4 * point[2]);
                    }
                }
            }
        }
    }

    ~OhmTest() {}
};



using OhmTupleInfos
    = testing::Types<std::pair<DimConst<1>, InterpConst<1>>, std::pair<DimConst<2>, InterpConst<1>>,
                     std::pair<DimConst<3>, InterpConst<1>>>;

TYPED_TEST_SUITE(OhmTest, OhmTupleInfos);




TYPED_TEST(OhmTest, ThatOhmInfoCanBeBuiltFromDict)
{
    auto info = OhmInfo::FROM(createDict());
}


// asEnumConstant enumerates [0, Enum::count), which is what replaced the EnumTraits value table
static_assert(std::is_same_v<EnumConstantVariant_t<HyperMode>,
                             std::variant<std::integral_constant<HyperMode, HyperMode::constant>,
                                          std::integral_constant<HyperMode, HyperMode::spatial>>>);

TYPED_TEST(OhmTest, ThatAsEnumConstantLiftsEveryHyperMode)
{
    for (auto const mode : {HyperMode::constant, HyperMode::spatial})
        std::visit([&](auto const tag) { EXPECT_EQ(mode, decltype(tag)::value); },
                   asEnumConstant(mode));

    EXPECT_THROW(asEnumConstant(HyperMode::count), std::runtime_error);
}


// hyper_mode is written to the dict as the int of the HyperMode member exposed to python by
// cpp_etc (see pharein.initialize.add_enum_int), and read back as the enum
TYPED_TEST(OhmTest, ThatHyperModeIsReadFromDictAsAnEnum)
{
    EXPECT_EQ(HyperMode::constant, OhmInfo::FROM(createDict()).hyper_mode); // absent, defaulted

    auto dict          = createDict();
    dict["hyper_mode"] = static_cast<int>(HyperMode::spatial);
    EXPECT_EQ(HyperMode::spatial, OhmInfo::FROM(dict).hyper_mode);

    dict["hyper_mode"] = static_cast<int>(HyperMode::constant);
    EXPECT_EQ(HyperMode::constant, OhmInfo::FROM(dict).hyper_mode);
}



std::vector<double> read(std::string filename)
{
    std::ifstream readFile(filename);
    assert(readFile.is_open());
    std::vector<double> x;

    std::copy(std::istream_iterator<double>(readFile), std::istream_iterator<double>(),
              std::back_inserter(x));
    return x;
}



TYPED_TEST(OhmTest, ThatElectricFieldIsOkFromOhmsLaw)
{
    using Ohm_t = TestFixture::Ohm_t;

    TypeParam pair;
    auto constexpr dim    = pair.first();
    auto constexpr interp = pair.second();

    std::string prefix{"ohm"};

    auto filenameX
        = prefix + "x_yee_" + std::to_string(dim) + "D_order" + std::to_string(interp) + ".txt";
    auto filenameY
        = prefix + "y_yee_" + std::to_string(dim) + "D_order" + std::to_string(interp) + ".txt";
    auto filenameZ
        = prefix + "z_yee_" + std::to_string(dim) + "D_order" + std::to_string(interp) + ".txt";
    auto expected_ohmX = read(filenameX);
    auto expected_ohmY = read(filenameY);
    auto expected_ohmZ = read(filenameZ);

    auto const layout = NDlayout<dim, interp>::create();

    Ohm_t{this->ohm_info, layout}(this->n, this->V, this->P, this->B, this->J, this->Enew);


    auto const& [Exnew, Eynew, Eznew] = this->Enew();

    if constexpr (dim == 1)
    {
        auto psi_X = this->layout.physicalStartIndex(Exnew, Direction::X);
        auto pei_X = this->layout.physicalEndIndex(Exnew, Direction::X);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            EXPECT_THAT(Exnew(ix), ::testing::DoubleNear((expected_ohmX[ix]), 1e-12));
        }

        psi_X = this->layout.physicalStartIndex(Eynew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eynew, Direction::X);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            EXPECT_THAT(Eynew(ix), ::testing::DoubleNear((expected_ohmY[ix]), 1e-12));
        }

        psi_X = this->layout.physicalStartIndex(Eznew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eznew, Direction::X);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            EXPECT_THAT(Eznew(ix), ::testing::DoubleNear((expected_ohmZ[ix]), 1e-12));
        }
    }

    if constexpr (dim == 2)
    {
        auto psi_X = this->layout.physicalStartIndex(Exnew, Direction::X);
        auto pei_X = this->layout.physicalEndIndex(Exnew, Direction::X);
        auto psi_Y = this->layout.physicalStartIndex(Exnew, Direction::Y);
        auto pei_Y = this->layout.physicalEndIndex(Exnew, Direction::Y);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ex);
                auto index_ = ix * nPts_[1] + iy;
                EXPECT_THAT(Exnew(ix, iy), ::testing::DoubleNear((expected_ohmX[index_]), 1e-12));
            }
        }

        psi_X = this->layout.physicalStartIndex(Eynew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eynew, Direction::X);
        psi_Y = this->layout.physicalStartIndex(Eynew, Direction::Y);
        pei_Y = this->layout.physicalEndIndex(Eynew, Direction::Y);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ey);
                auto index_ = ix * nPts_[1] + iy;
                EXPECT_THAT(Eynew(ix, iy), ::testing::DoubleNear((expected_ohmY[index_]), 1e-12));
            }
        }

        psi_X = this->layout.physicalStartIndex(Eznew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eznew, Direction::X);
        psi_Y = this->layout.physicalStartIndex(Eznew, Direction::Y);
        pei_Y = this->layout.physicalEndIndex(Eznew, Direction::Y);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ez);
                auto index_ = ix * nPts_[1] + iy;
                EXPECT_THAT(Eznew(ix, iy), ::testing::DoubleNear((expected_ohmZ[index_]), 1e-12));
            }
        }
    }

    if constexpr (dim == 3)
    {
        auto psi_X = this->layout.physicalStartIndex(Exnew, Direction::X);
        auto pei_X = this->layout.physicalEndIndex(Exnew, Direction::X);
        auto psi_Y = this->layout.physicalStartIndex(Exnew, Direction::Y);
        auto pei_Y = this->layout.physicalEndIndex(Exnew, Direction::Y);
        auto psi_Z = this->layout.physicalStartIndex(Exnew, Direction::Z);
        auto pei_Z = this->layout.physicalEndIndex(Exnew, Direction::Z);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                for (auto iz = psi_Z; iz <= pei_Z; ++iz)
                {
                    auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ex);
                    auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                    EXPECT_THAT(Exnew(ix, iy, iz),
                                ::testing::DoubleNear((expected_ohmX[index_]), 1e-10));
                }
            }
        }

        psi_X = this->layout.physicalStartIndex(Eynew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eynew, Direction::X);
        psi_Y = this->layout.physicalStartIndex(Eynew, Direction::Y);
        pei_Y = this->layout.physicalEndIndex(Eynew, Direction::Y);
        psi_Z = this->layout.physicalStartIndex(Eynew, Direction::Z);
        pei_Z = this->layout.physicalEndIndex(Eynew, Direction::Z);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                for (auto iz = psi_Z; iz <= pei_Z; ++iz)
                {
                    auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ey);
                    auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                    EXPECT_THAT(Eynew(ix, iy, iz),
                                ::testing::DoubleNear((expected_ohmY[index_]), 1e-10));
                }
            }
        }

        psi_X = this->layout.physicalStartIndex(Eznew, Direction::X);
        pei_X = this->layout.physicalEndIndex(Eznew, Direction::X);
        psi_Y = this->layout.physicalStartIndex(Eznew, Direction::Y);
        pei_Y = this->layout.physicalEndIndex(Eznew, Direction::Y);
        psi_Z = this->layout.physicalStartIndex(Eznew, Direction::Z);
        pei_Z = this->layout.physicalEndIndex(Eznew, Direction::Z);

        for (auto ix = psi_X; ix <= pei_X; ++ix)
        {
            for (auto iy = psi_Y; iy <= pei_Y; ++iy)
            {
                for (auto iz = psi_Z; iz <= pei_Z; ++iz)
                {
                    auto nPts_  = this->layout.allocSize(HybridQuantity::Scalar::Ez);
                    auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                    EXPECT_THAT(Eznew(ix, iy, iz),
                                ::testing::DoubleNear((expected_ohmZ[index_]), 1e-10));
                }
            }
        }
    }
}


template<std::size_t c, std::size_t d>
constexpr auto hybridKappaLocation()
{
    using Scalar = HybridQuantity::Scalar;
    if constexpr (c == d)
        return Scalar::rho;
    else if constexpr (c + d == 1)
        return Scalar::Bz;
    else if constexpr (c + d == 2)
        return Scalar::By;
    else
        return Scalar::Bx;
}

TYPED_TEST(OhmTest, spatialHyperResistivityDissipates)
{
    using Layout              = TestFixture::GridYee;
    using Ohm_t               = TestFixture::Ohm_t;
    using Scalar              = HybridQuantity::Scalar;
    static constexpr auto dim = TestFixture::dim;
    auto const& layout        = this->layout;
    auto const& dl            = layout.meshSize();
    double const nu           = 0.02;
    std::array<double, 3> const direction{0.6, 0.3, 0.74};

    std::array<double, 3> x0{0., 0., 0.};
    for (std::size_t d = 0; d < dim; ++d)
        x0[d] = 0.5 * layout.nbrCells()[d] * dl[d];

    auto r2 = [&](auto const& x) {
        double s = 0.;
        for (std::size_t d = 0; d < 3; ++d)
            s += (x[d] - x0[d]) * (x[d] - x0[d]);
        return s;
    };
    auto bump = [&](auto const& x) {
        double w = 1.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            double const margin = 3. * dl[d];
            double const length = layout.nbrCells()[d] * dl[d] - 2. * margin;
            double const u      = (x[d] - margin) / length;
            w *= (u <= 0. or u >= 1.) ? 0. : std::pow(std::sin(M_PI * u), 2);
        }
        return w;
    };

    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(layout, this->J[c], [&](auto const& x) { return bump(x) * std::exp(-r2(x) / 0.25); });
        fill(layout, this->B[c], [&](auto const& x) {
            return direction[c] * 30. * (1. - std::exp(-std::pow(std::sqrt(r2(x)) / 0.375, 8)));
        });
        fill(layout, this->V[c], [](auto const&) { return 0.; });
    }
    fill(layout, this->n,
         [](auto const& x) { return 1.5 + 0.5 * std::sin(0.6 * x[0] + 1.1 * x[1] - 0.7 * x[2]); });
    fill(layout, this->P, [](auto const&) { return 1.; });

    auto electricField = [&](double const hyper) {
        Ohm_t{OhmInfo{0., hyper, HyperMode::spatial}, layout}(this->n, this->V, this->P, this->B,
                                                              this->J, this->Enew);
        std::array<std::vector<double>, 3> E;
        for (std::size_t c = 0; c < 3; ++c)
            for_each_point(physicalBox(layout, layout.centering(this->Enew[c])),
                           [&](auto const& idx) { E[c].push_back(this->Enew[c](idx)); });
        return E;
    };

    auto const withHyper    = electricField(nu);
    auto const withoutHyper = electricField(0.);

    double work = 0.;
    for (std::size_t c = 0; c < 3; ++c)
    {
        std::size_t i = 0;
        for_each_point(physicalBox(layout, layout.centering(this->Enew[c])), [&](auto const& idx) {
            work += this->J[c](idx) * (withHyper[c][i] - withoutHyper[c][i]);
            ++i;
        });
    }

    double expected = 0.;
    for_N<3>([&](auto ic) {
        constexpr auto c = ic();
        auto const& Jc   = this->J[c];
        for_N<dim>([&](auto id) {
            constexpr auto d   = id();
            constexpr auto dst = hybridKappaLocation<c, d>();
            auto const dJ      = partial<static_cast<Direction>(d)>(layout, Jc);
            EXPECT_EQ(Layout::centering(dst), dJ.centering());
            for_each_point(physicalBox(layout, dJ.centering()), [&](auto const& p) {
                auto const bx    = Layout::template project<Scalar::Bx, dst>(this->B[0], p);
                auto const by    = Layout::template project<Scalar::By, dst>(this->B[1], p);
                auto const bz    = Layout::template project<Scalar::Bz, dst>(this->B[2], p);
                auto const n     = Layout::template project<Scalar::rho, dst>(this->n, p);
                auto const kappa = nu * (std::sqrt(bx * bx + by * by + bz * bz) / (n + 0.1) + 1);
                expected += kappa * dJ(p) * dJ(p);
            });
        });
    });

    EXPECT_GT(work, 0.);
    EXPECT_NEAR(work, expected, 1e-12 * std::abs(expected));
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
