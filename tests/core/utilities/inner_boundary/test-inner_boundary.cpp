#include "gtest/gtest.h"

#include "core/inner_boundary/inner_boundary_factory.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"

#include "initializer/data_provider.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr double eps = 1e-12;
}

using namespace PHARE;

TEST(InnerBoundarySphere, basicGeometry)
{
    core::SphereInnerBoundary<2> sphere{"sphere", {0., 0.}, 1.};

    auto const p = core::Point<double, 2>{2., 0.};
    EXPECT_NEAR(sphere.signedDistance(p), 1., eps);

    auto const n = sphere.normal(p);
    EXPECT_NEAR(n[0], 1., eps);
    EXPECT_NEAR(n[1], 0., eps);

    auto const proj = sphere.project(p);
    EXPECT_NEAR(proj[0], 1., eps);
    EXPECT_NEAR(proj[1], 0., eps);

    auto const sym = sphere.symmetric(p);
    EXPECT_NEAR(sym[0], 0., eps);
    EXPECT_NEAR(sym[1], 0., eps);

    EXPECT_NEAR(sphere.radius(), 1., eps);
    EXPECT_NEAR(sphere.center()[0], 0., eps);
}

TEST(InnerBoundaryFactory, createsSphere)
{
    initializer::PHAREDict dict;
    dict["inner_boundary"]["name"]   = std::string{"my_sphere"};
    dict["inner_boundary"]["shape"]  = std::string{"sphere"};
    dict["inner_boundary"]["center"] = std::vector<double>{1., 2.};
    dict["inner_boundary"]["radius"] = 2.;

    auto boundary = core::InnerBoundaryFactory<2>::create(dict);
    ASSERT_NE(boundary, nullptr);
    EXPECT_NEAR(boundary->signedDistance(core::Point<double, 2>{1., 6.}), 2., eps);
}

TEST(InnerBoundaryFactory, rejectsPlane)
{
    initializer::PHAREDict dict;
    dict["inner_boundary"]["name"]   = std::string{"my_plane"};
    dict["inner_boundary"]["shape"]  = std::string{"plane"};
    dict["inner_boundary"]["point"]  = std::vector<double>{0., 0.};
    dict["inner_boundary"]["normal"] = std::vector<double>{0., 1.};

    EXPECT_THROW(core::InnerBoundaryFactory<2>::create(dict), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
