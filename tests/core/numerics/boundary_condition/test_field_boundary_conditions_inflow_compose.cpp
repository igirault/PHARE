#include "gtest/gtest.h"

#include "core/boundary/boundary_inflow_compose.hpp"
#include "initializer/data_provider.hpp"

#include <memory>
#include <vector>

using namespace PHARE::core::inflow_compose;
using PHARE::core::Span;
using PHARE::core::VectorSpan;
using PHARE::initializer::SpaceTimeFunction;

namespace
{
// A 1D space-time function f(x, t) built from a plain lambda over the batch.
SpaceTimeFunction<1> make1D(std::function<double(double, double)> f)
{
    return [f](std::vector<double> const& x, double t) -> std::shared_ptr<Span<double>> {
        std::vector<double> out(x.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            out[k] = f(x[k], t);
        return std::make_shared<VectorSpan<double>>(std::move(out));
    };
}
} // namespace

TEST(InflowCompose, ConstFunctionBroadcastsToNodeCount)
{
    auto c = constFunction<1>(2.5);
    std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    auto s = c(x, 7.0); // time is ignored
    ASSERT_EQ(s->size(), x.size());
    for (std::size_t k = 0; k < x.size(); ++k)
        EXPECT_DOUBLE_EQ((*s)[k], 2.5);
}

TEST(InflowCompose, MulFunctionMultipliesElementwise)
{
    auto f = make1D([](double x, double) { return x; });
    auto g = make1D([](double, double t) { return t; });
    auto p = mulFunction<1>(f, g); // x * t
    std::vector<double> x{1.0, 2.0, 4.0};
    auto s = p(x, 3.0);
    ASSERT_EQ(s->size(), x.size());
    EXPECT_DOUBLE_EQ((*s)[0], 3.0);
    EXPECT_DOUBLE_EQ((*s)[1], 6.0);
    EXPECT_DOUBLE_EQ((*s)[2], 12.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
