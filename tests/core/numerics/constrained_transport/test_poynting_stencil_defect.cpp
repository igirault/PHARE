#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"
#include "core/utilities/box/box.hpp"
#include "core/numerics/constrained_transport/poynting_stencil_defect.hpp"

#include "tests/core/data/gridlayout/test_centering_boxes.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>

namespace PHARE::core
{

template<std::size_t dim>
struct MHDParam
{
    static constexpr SimOpts opts{dim,
                                  0,
                                  0,
                                  MHDOpts::ReconstructionType::Constant,
                                  MHDOpts::SlopeLimiterType::None,
                                  MHDOpts::RiemannSolverType::Rusanov};
    using Model = PHARE_Types<opts>::MHD;
};

constexpr std::array<double, 3> meshSizes{0.1, 0.2, 0.3};
constexpr std::array<std::uint32_t, 3> cellCounts{20, 14, 10};

template<std::size_t dim, typename T>
std::array<T, dim> leading(std::array<T, 3> const& values)
{
    std::array<T, dim> out;
    std::copy_n(values.begin(), dim, out.begin());
    return out;
}

template<typename Model>
struct OwnedField
{
    template<typename Layout, typename Qty>
    OwnedField(Layout const& layout, Qty qty, double value)
        : grid{"f", layout, qty, value}
        , field{"f", qty}
    {
        field.setBuffer(&grid);
    }

    OwnedField(OwnedField const&)            = delete;
    OwnedField& operator=(OwnedField const&) = delete;

    Model::Grid_t grid;
    Model::Field_t field;
};

template<typename Model>
struct OwnedVecField
{
    template<typename Layout, typename Qty>
    OwnedVecField(Layout const& layout, Qty qty)
        : view{"v", qty}
        , grids{makeGrid(layout, 0), makeGrid(layout, 1), makeGrid(layout, 2)}
    {
        view.setBuffer(&grids);
    }

    OwnedVecField(OwnedVecField const&)            = delete;
    OwnedVecField& operator=(OwnedVecField const&) = delete;

    Model::VecField_t view;
    std::array<typename Model::Grid_t, 3> grids;

private:
    template<typename Layout>
    auto makeGrid(Layout const& layout, std::size_t i) const
    {
        return typename Model::Grid_t{view[i].name(), layout, view[i].physicalQuantity(),
                                      std::numeric_limits<double>::quiet_NaN()};
    }
};

template<typename Field, std::size_t dim>
struct EtotFluxes
{
    std::array<Field*, dim> etot;

    struct Ref
    {
        double& value;
        double& Etot() { return value; }
    };

    template<auto direction>
    Ref get_dir(MeshIndex<dim> const& index)
    {
        return Ref{(*etot[static_cast<std::size_t>(direction)])(index)};
    }
};


template<typename Param>
struct PoyntingStencilDefectTest : public ::testing::Test
{
    using Model                      = Param::Model;
    using Layout                     = Model::GridLayout_t;
    using Field                      = Model::Field_t;
    static constexpr std::size_t dim = Layout::dimension;

    Layout layout{leading<dim>(meshSizes), leading<dim>(cellCounts), Point<double, dim>{}};

    OwnedVecField<Model> E{layout, MHDQuantity::Vector::E};
    OwnedVecField<Model> Bold{layout, MHDQuantity::Vector::B};
    OwnedVecField<Model> Bnew{layout, MHDQuantity::Vector::B};
    OwnedField<Model> Fx{layout, MHDQuantity::Scalar::ScalarFlux_x, 0.};
    OwnedField<Model> Fy{layout, MHDQuantity::Scalar::ScalarFlux_y, 0.};
    OwnedField<Model> Fz{layout, MHDQuantity::Scalar::ScalarFlux_z, 0.};

    std::array<double, 3> const bOld{0.4, 0.7, -0.3};
    std::array<double, 3> const bNew{0.6, 0.5, -0.1};

    PoyntingStencilDefectTest()
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            fill(layout, E.view[c], [](auto const&) { return 0.; });
            fill(layout, Bold.view[c], [&](auto const&) { return bOld[c]; });
            fill(layout, Bnew.view[c], [&](auto const&) { return bNew[c]; });
        }
    }

    double bMid(std::size_t c) const { return 0.5 * (bOld[c] + bNew[c]); }

    auto& flux(std::size_t d)
    {
        std::array<Field*, 3> all{&Fx.field, &Fy.field, &Fz.field};
        return *all[d];
    }

    void apply(std::array<std::vector<Box<int, dim>>, dim> const& faces, double coef)
    {
        EtotFluxes<Field, dim> fluxes;
        for (std::size_t d = 0; d < dim; ++d)
            fluxes.etot[d] = &flux(d);
        PoyntingStencilDefect<Layout>{layout}(fluxes, E.view, Bold.view, Bnew.view, faces, coef);
    }

    static Box<int, dim> faceBox(std::size_t normal, int face, std::array<int, 3> lo,
                                 std::array<int, 3> hi)
    {
        Box<int, dim> box;
        for (std::size_t k = 0; k < dim; ++k)
        {
            box.lower[k] = k == normal ? face : lo[k];
            box.upper[k] = k == normal ? face : hi[k];
        }
        return box;
    }

    Point<int, dim> amrOf(MeshIndex<dim> const& local) const
    {
        Point<std::uint32_t, dim> p;
        for (std::size_t k = 0; k < dim; ++k)
            p[k] = local[k];
        return layout.localToAMR(p);
    }

    bool inBox(Box<int, dim> const& box, MeshIndex<dim> const& local) const
    {
        auto const amr = amrOf(local);
        for (std::size_t k = 0; k < dim; ++k)
            if (amr[k] < box.lower[k] or amr[k] > box.upper[k])
                return false;
        return true;
    }

    void expectCoarseFaceAverages(std::size_t d, Box<int, dim> const& box, double expected)
    {
        auto& F = flux(d);
        for_each_point(physicalBox(layout, layout.centering(F)), [&](auto const& i) {
            if (!inBox(box, i))
            {
                EXPECT_EQ(F(i), 0.);
                return;
            }
            auto const amr = amrOf(i);
            for (std::size_t k = 0; k < dim; ++k)
                if (k != d and amr[k] % 2 != 0)
                    return;
            double sum        = 0.;
            std::size_t count = 0;
            for (std::size_t corner = 0; corner < (1u << dim); ++corner)
            {
                if (corner & (1u << d))
                    continue;
                auto j = i;
                for (std::size_t k = 0; k < dim; ++k)
                    j[k] += (corner >> k) & 1u;
                sum += F(j);
                ++count;
            }
            EXPECT_NEAR(sum / count, expected, 1e-12);
        });
    }
};

using Params = ::testing::Types<MHDParam<1>, MHDParam<2>, MHDParam<3>>;

TYPED_TEST_SUITE(PoyntingStencilDefectTest, Params);


TYPED_TEST(PoyntingStencilDefectTest, quadraticEAlongTheFaceGivesTheAnalyticCorrection)
{
    static constexpr auto dim = TestFixture::dim;
    if constexpr (dim == 1)
        GTEST_SKIP() << "no edge between coarse edge lines along a 1D face";
    else
    {
        double const a    = 2.5;
        double const coef = 0.25;
        auto const& dl    = this->layout.meshSize();

        fill(this->layout, this->E.view[2], [&](auto const& x) { return a * x[1] * x[1] + x[0]; });

        auto const box = TestFixture::faceBox(0, 6, {0, 2, 2}, {0, 9, 7});
        std::array<std::vector<Box<int, dim>>, dim> faces;
        faces[0].push_back(box);
        this->apply(faces, coef);

        this->expectCoarseFaceAverages(0, box, -0.5 * coef * a * dl[1] * dl[1] * this->bMid(1));
    }
}


TYPED_TEST(PoyntingStencilDefectTest, yNormalFacesTakeTheOppositeSign)
{
    static constexpr auto dim = TestFixture::dim;
    if constexpr (dim == 1)
        GTEST_SKIP() << "no y faces in 1D";
    else
    {
        double const a    = -1.5;
        double const coef = 0.5;
        auto const& dl    = this->layout.meshSize();

        fill(this->layout, this->E.view[2], [&](auto const& x) { return a * x[0] * x[0] - x[1]; });

        auto const box = TestFixture::faceBox(1, 4, {4, 0, 0}, {13, 0, 5});
        std::array<std::vector<Box<int, dim>>, dim> faces;
        faces[1].push_back(box);
        this->apply(faces, coef);

        this->expectCoarseFaceAverages(1, box, 0.5 * coef * a * dl[0] * dl[0] * this->bMid(0));
    }
}


TYPED_TEST(PoyntingStencilDefectTest, linearEAlongTheFaceLeavesTheFluxUnchanged)
{
    static constexpr auto dim = TestFixture::dim;
    for (std::size_t c = 0; c < 3; ++c)
        fill(this->layout, this->E.view[c],
             [&](auto const& x) { return 0.3 + c + 1.7 * x[0] - 0.4 * x[1] + 0.9 * x[2]; });

    std::array<std::vector<Box<int, dim>>, dim> faces;
    for (std::size_t d = 0; d < dim; ++d)
        faces[d].push_back(TestFixture::faceBox(d, 4, {2, 2, 2}, {9, 9, 7}));
    this->apply(faces, 1.);

    for (std::size_t d = 0; d < dim; ++d)
        this->expectCoarseFaceAverages(d, faces[d][0], 0.);
}


TYPED_TEST(PoyntingStencilDefectTest, edgesOnCoarseLinesAreNotCorrected)
{
    static constexpr auto dim = TestFixture::dim;
    if constexpr (dim == 1)
        GTEST_SKIP() << "no edge along a 1D face";
    else
    {
        auto const onCoarseLine = [](int j) { return static_cast<double>((j * j + 3) % 7); };
        fill(this->layout, this->E.view[2], [&](auto const& x) {
            auto const j = static_cast<int>(std::lround(x[1] / this->layout.meshSize()[1]));
            return j % 2 == 0 ? onCoarseLine(j) : 0.5 * (onCoarseLine(j - 1) + onCoarseLine(j + 1));
        });

        std::array<std::vector<Box<int, dim>>, dim> faces;
        faces[0].push_back(TestFixture::faceBox(0, 6, {0, 2, 2}, {0, 9, 7}));
        this->apply(faces, 1.);

        this->expectCoarseFaceAverages(0, faces[0][0], 0.);
    }
}


TYPED_TEST(PoyntingStencilDefectTest, splittingAFaceAtAnOddIndexGivesTheSameCorrection)
{
    static constexpr auto dim = TestFixture::dim;
    if constexpr (dim == 1)
        GTEST_SKIP() << "no split along a 1D face";
    else
    {
        fill(this->layout, this->E.view[2],
             [&](auto const& x) { return std::sin(3. * x[1]) + x[1] * x[1] * x[0]; });

        std::array<std::vector<Box<int, dim>>, dim> whole, split;
        whole[0].push_back(TestFixture::faceBox(0, 6, {0, 2, 2}, {0, 9, 7}));
        split[0].push_back(TestFixture::faceBox(0, 6, {0, 2, 2}, {0, 4, 7}));
        split[0].push_back(TestFixture::faceBox(0, 6, {0, 5, 2}, {0, 9, 7}));

        this->apply(whole, 0.75);
        auto& Fx = this->flux(0);
        std::vector<double> reference;
        for_each_point(physicalBox(this->layout, this->layout.centering(Fx)),
                       [&](auto const& i) { reference.push_back(Fx(i)); });

        for_each_point(physicalBox(this->layout, this->layout.centering(Fx)),
                       [&](auto const& i) { Fx(i) = 0.; });
        this->apply(split, 0.75);

        std::size_t k = 0;
        for_each_point(physicalBox(this->layout, this->layout.centering(Fx)),
                       [&](auto const& i) { EXPECT_EQ(Fx(i), reference[k++]); });
    }
}

} // namespace PHARE::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
