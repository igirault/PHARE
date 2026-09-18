#include "core/models/external_field_updater_dipole.hpp"
#include "core/utilities/point/point.hpp"

#include "phare_core.hpp"

#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include "tests/core/models/test_external_field_fixtures.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <numbers>
#include <numeric>

using namespace PHARE;
using namespace PHARE::core;


namespace
{
//! an MHD-enabled option set, the axes other than the dimension are irrelevant here
template<std::size_t dim>
constexpr SimOpts mhd_opts{.dimension            = dim,
                           .interp_order         = 1,
                           .time_integrator_type = MHDOpts::TimeIntegratorType::Euler,
                           .reconstruction_type  = MHDOpts::ReconstructionType::Constant,
                           .slope_limiter_type   = MHDOpts::SlopeLimiterType::None,
                           .riemann_solver_type  = MHDOpts::RiemannSolverType::Rusanov};

template<std::size_t dim>
using MHDTypes = typename PHARE_Types<mhd_opts<dim>>::MHD;


/**
 * @brief analytical magnetic field of a dipole, i.e. the exact curl of the potential the
 * updater implements
 *
 * The moment has one component per dimension.
 *
 * In 3D @f$\mathbf{B} = \frac{1}{4\pi r^{3}}
 *                       \left[3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}-\mathbf{m}\right]@f$
 * and in 2D @f$\mathbf{B} = \frac{1}{2\pi r^{2}}
 *                       \left[2(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}-\mathbf{m}\right]@f$
 * with @f$\mathbf{m}@f$ in the plane, and @f$B_z = 0@f$.
 */
template<std::size_t dim>
Point<double, 3> expectedB(Point<double, dim> const& x, Point<double, dim> const& x0,
                           Point<double, dim> const& m)
{
    auto const r          = x - x0;
    double const rSquared = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
    double const mDotR    = std::inner_product(r.begin(), r.end(), m.begin(), 0.0);

    if constexpr (dim == 2)
    {
        double const factor = 1. / (2. * std::numbers::pi * rSquared);
        return {factor * (2. * mDotR * r[0] / rSquared - m[0]),
                factor * (2. * mDotR * r[1] / rSquared - m[1]), 0.};
    }
    else
    {
        double const factor = 1. / (4. * std::numbers::pi * rSquared * std::sqrt(rSquared));
        return {factor * (3. * mDotR * r[0] / rSquared - m[0]),
                factor * (3. * mDotR * r[1] / rSquared - m[1]),
                factor * (3. * mDotR * r[2] / rSquared - m[2])};
    }
}


template<std::size_t dim_, std::uint32_t cells_>
struct DipoleSetup
{
    auto static constexpr dim   = dim_;
    auto static constexpr cells = cells_;

    //! measured error of the discrete curl at 64 cells: 9.2e-5 in 2D, 8.0e-4 in 3D, for a field
    //! of order 0.5 over the domain. Taken here with ~50% margin.
    auto static constexpr tolerance = dim == 2 ? 1.5e-4 : 1.2e-3;

    using GridLayout_t = typename MHDTypes<dim>::GridLayout_t;
    using VecField_t   = typename MHDTypes<dim>::VecField_t;
    using Updater_t    = ExternalFieldUpdaterDipole<VecField_t, GridLayout_t>;
    using Position_t   = Updater_t::point_type;
    using Moment_t     = Updater_t::vector_type;

    //! the moment, one component per dimension
    Moment_t static moment()
    {
        if constexpr (dim == 2)
            return {0.3, -0.2};
        else
            return {0.3, -0.2, 0.5};
    }

    //! kept outside the [0, 1]^dim domain so that the field stays smooth on the whole mesh
    Position_t static position()
    {
        if constexpr (dim == 2)
            return {-0.5, 0.5};
        else
            return {-0.5, 0.5, 0.5};
    }

    TestGridLayout<GridLayout_t> layout{cells};

    UsableExternalField<dim> externalField{"external", layout};
    UsableVecFieldMHD<dim> a0{"a0", layout, MHDQuantity::Vector::E};

    Updater_t updater{position(), moment()};

    void update(double time = 0.) { updater(externalField, a0, layout, time); }

    //! largest |B0 - B_analytical| over the physical domain, all components
    double maxErrorOnDomain()
    {
        double maxError = 0.;
        for_N<3>([&](auto i) {
            constexpr auto component = static_cast<Component>(decltype(i)::value);
            auto& field              = externalField.B0(component);
            layout.evalOnGhostBox(field, [&](auto... ijk) {
                auto const x = layout.fieldNodeCoordinates(field, layout.localToAMR(Point{ijk...}));
                auto const expected = expectedB<dim>(x, position(), moment());
                maxError            = std::max(maxError, std::abs(field(ijk...) - expected[i]));
            });
        });
        return maxError;
    }

    //! largest |dB0/dt| over the whole ghost box, all components
    double maxAbsTimeDerivative()
    {
        double maxAbs = 0.;
        for_N<3>([&](auto i) {
            auto& field = externalField.dB0dt(static_cast<Component>(decltype(i)::value));
            for (auto const& v : field)
                maxAbs = std::max(maxAbs, std::abs(v));
        });
        return maxAbs;
    }
};

} // namespace

//! NOTE: the parameter cannot be named Setup: ::testing::Test declares a private member of
//! that name (its guard against SetUp being misspelled), and class scope wins over the
//! template parameter scope during lookup.
template<typename SetupT>
struct DipoleTest : public ::testing::Test
{
    void SetUp() override { setup.update(); }

    SetupT setup;
};

using Setups = ::testing::Types<DipoleSetup<2, 64>, DipoleSetup<3, 64>>;
TYPED_TEST_SUITE(DipoleTest, Setups);


TYPED_TEST(DipoleTest, isNotTimeDependent)
{
    EXPECT_FALSE(this->setup.updater.isTimeDependent());
}


TYPED_TEST(DipoleTest, retrievesTheAnalyticalDipoleField)
{
    EXPECT_LT(this->setup.maxErrorOnDomain(), TypeParam::tolerance);
}

TYPED_TEST(DipoleTest, retrievesAZeroTimeDerivative)
{
    EXPECT_DOUBLE_EQ(this->setup.maxAbsTimeDerivative(), 0.);
}


/**
 * @brief the tolerance above is only meaningful if the error is indeed that of a 2nd order curl
 */
TYPED_TEST(DipoleTest, convergesAtSecondOrder)
{
    using Coarse = DipoleSetup<TypeParam::dim, TypeParam::cells>;
    using Fine   = DipoleSetup<TypeParam::dim, 2 * TypeParam::cells>;

    Coarse coarse;
    Fine fine;
    coarse.update();
    fine.update();

    auto const ratio = coarse.maxErrorOnDomain() / fine.maxErrorOnDomain();
    EXPECT_GT(ratio, 3.5);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
