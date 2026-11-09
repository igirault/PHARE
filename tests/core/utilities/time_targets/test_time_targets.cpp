#include "core/utilities/time_targets.hpp"
#include "amr/load_balancing/load_balancer_details.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace PHARE::core;


TEST(TimeAtStep, freshRunIsStepTimesDt)
{
    EXPECT_EQ(time_at_step({0.0, 0}, 3, 0.1), 3.0 * 0.1);
}

TEST(TimeAtStep, reanchoredRunCountsFromAnchor)
{
    EXPECT_EQ(time_at_step({3.0 * 0.1, 3}, 5, 0.05), 3.0 * 0.1 + 2.0 * 0.05);
}


TEST(AdaptiveTarget, plainStep)
{
    EXPECT_EQ(adaptive_target(0.5, 0.1, 1.0), 0.5 + 0.1);
}

TEST(AdaptiveTarget, clampsToFinal)
{
    EXPECT_EQ(adaptive_target(0.95, 0.1, 1.0), 1.0);
}

TEST(AdaptiveTarget, snapsSliverToFinal)
{
    EXPECT_EQ(adaptive_target(0.9, 0.1 - 1e-9, 1.0), 1.0);
}


TEST(IncrementLandingOn, fromZero)
{
    EXPECT_EQ(0.0 + increment_landing_on(0.0, 0.1), 0.1);
}

TEST(IncrementLandingOn, nearbyTimes)
{
    double const t = 3.0 * 0.1;
    EXPECT_EQ(t + increment_landing_on(t, 0.4), 0.4);
}

TEST(IncrementLandingOn, farApartTimes)
{
    double const t = 1e-20;
    EXPECT_EQ(t + increment_landing_on(t, 0.1), 0.1);
}

TEST(IncrementLandingOn, randomPairsLandExactlyWithinAFactorTwo)
{
    std::mt19937_64 gen{42};
    std::uniform_real_distribution<double> u{0.0, 1e3};
    for (int i = 0; i < 100000; ++i)
    {
        auto const target = u(gen);
        auto const t      = std::uniform_real_distribution<double>{target / 2, target}(gen);
        if (!(target > t))
            continue;
        EXPECT_EQ(t + increment_landing_on(t, target), target);
    }
}

TEST(IncrementLandingOn, landsExactlyOnARoundedSum)
{
    std::mt19937_64 gen{7};
    std::uniform_real_distribution<double> u{0.0, 1e3};
    for (int i = 0; i < 100000; ++i)
    {
        auto const t      = u(gen) * 1e-3;
        auto const target = t + u(gen);
        if (!(target > t))
            continue;
        EXPECT_EQ(t + increment_landing_on(t, target), target);
    }
}

TEST(IncrementLandingOn, unreachableTargetOvershootsByOneUlp)
{
    double const t      = 19.30026167262983;
    double const target = 297.62996909823011;
    auto const landed   = t + increment_landing_on(t, target);
    EXPECT_EQ(landed, std::nextafter(target, std::numeric_limits<double>::infinity()));
}

TEST(IncrementLandingOn, randomPairsNeverLandBelowTarget)
{
    std::mt19937_64 gen{42};
    std::uniform_real_distribution<double> u{0.0, 1e3};
    auto constexpr inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 100000; ++i)
    {
        auto const a = u(gen);
        auto const b = u(gen);
        if (a == b)
            continue;
        auto const t      = std::min(a, b);
        auto const target = std::max(a, b);
        auto const landed = t + increment_landing_on(t, target);
        EXPECT_GE(landed, target);
        EXPECT_LE(landed, std::nextafter(target, inf));
    }
}

TEST(IncrementLandingOn, throwsOnNonIncreasingTarget)
{
    EXPECT_THROW(increment_landing_on(2.0, 2.0), std::runtime_error);
}


TEST(ResolveAnchor, freshStart)
{
    auto const a = resolve_anchor(TimeStepType::constant, 0.1, std::nullopt, 0.0, 0.0, 0);
    EXPECT_EQ(a.time, 0.0);
    EXPECT_EQ(a.step, 0);
}

TEST(ResolveAnchor, sameDtKeepsStoredAnchor)
{
    auto const a
        = resolve_anchor(TimeStepType::constant, 0.05,
                         TimeRecord{TimeStepType::constant, 0.05, TimeAnchor{0.3, 3}}, 0.0, 0.5, 7);
    EXPECT_EQ(a.time, 0.3);
    EXPECT_EQ(a.step, 3);
}

TEST(ResolveAnchor, sameDtWithoutStoredAnchorReanchorsAtRestoredTime)
{
    auto const a
        = resolve_anchor(TimeStepType::constant, 0.1,
                         TimeRecord{TimeStepType::constant, 0.1, std::nullopt}, 0.0, 0.5, 7);
    EXPECT_EQ(a.time, 0.5);
    EXPECT_EQ(a.step, 7);
}

TEST(ResolveAnchor, sameDtWithoutStoredAnchorOnTheGridAnchorsAtStart)
{
    double const drifted = std::nextafter(std::nextafter(3.0 * 0.1, 0.0), 0.0);
    auto const a
        = resolve_anchor(TimeStepType::constant, 0.1,
                         TimeRecord{TimeStepType::constant, 0.1, std::nullopt}, 0.0, drifted, 3);
    EXPECT_EQ(a.time, 0.0);
    EXPECT_EQ(a.step, 0);
}

TEST(ResolveAnchor, adaptivePreviousRunWithoutStoredAnchorReanchors)
{
    auto const a
        = resolve_anchor(TimeStepType::constant, 0.1,
                         TimeRecord{TimeStepType::adaptive, 0.0, std::nullopt}, 0.0, 0.3, 3);
    EXPECT_EQ(a.time, 0.3);
    EXPECT_EQ(a.step, 3);
}

TEST(ResolveAnchor, changedDtReanchorsAtRestoredTime)
{
    auto const a
        = resolve_anchor(TimeStepType::constant, 0.05,
                         TimeRecord{TimeStepType::constant, 0.1, TimeAnchor{0.0, 0}}, 0.0, 0.3, 3);
    EXPECT_EQ(a.time, 0.3);
    EXPECT_EQ(a.step, 3);
}

TEST(ResolveAnchor, oneUlpDtDifferenceReanchors)
{
    auto const dt = std::nextafter(0.1, 1.0);
    auto const a
        = resolve_anchor(TimeStepType::constant, dt,
                         TimeRecord{TimeStepType::constant, 0.1, TimeAnchor{0.0, 0}}, 0.0, 0.3, 3);
    EXPECT_EQ(a.step, 3);
}

TEST(ResolveAnchor, adaptivePreviousRunReanchors)
{
    auto const a = resolve_anchor(TimeStepType::constant, 0.1,
                                  TimeRecord{TimeStepType::adaptive, 0.0, TimeAnchor{0.0, 0}}, 0.0,
                                  0.41, 17);
    EXPECT_EQ(a.time, 0.41);
    EXPECT_EQ(a.step, 17);
}

TEST(ResolveAnchor, adaptiveRunUsesRestoredTime)
{
    auto const a
        = resolve_anchor(TimeStepType::adaptive, 0.0,
                         TimeRecord{TimeStepType::constant, 0.1, TimeAnchor{0.0, 0}}, 0.0, 0.3, 3);
    EXPECT_EQ(a.time, 0.3);
    EXPECT_EQ(a.step, 3);
}


TEST(LandedOnTarget, acceptsTargetAndOneUlpAbove)
{
    auto constexpr inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(landed_on_target(0.4, 0.4));
    EXPECT_TRUE(landed_on_target(std::nextafter(0.4, inf), 0.4));
}

TEST(LandedOnTarget, rejectsBelowOrFurtherAbove)
{
    auto constexpr inf = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(landed_on_target(std::nextafter(0.4, 0.0), 0.4));
    EXPECT_FALSE(landed_on_target(std::nextafter(std::nextafter(0.4, inf), inf), 0.4));
}


TEST(RebalanceOnCadence, onInitOnlyOnFirstAdvance)
{
    EXPECT_TRUE(PHARE::amr::rebalance_on_cadence(true, 0, true, 40));
    EXPECT_FALSE(PHARE::amr::rebalance_on_cadence(true, 0, false, 41));
}

TEST(RebalanceOnCadence, everyCountsCumulativeSteps)
{
    EXPECT_TRUE(PHARE::amr::rebalance_on_cadence(false, 10, true, 40));
    EXPECT_FALSE(PHARE::amr::rebalance_on_cadence(false, 10, false, 41));
    EXPECT_FALSE(PHARE::amr::rebalance_on_cadence(false, 10, true, 0));
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
