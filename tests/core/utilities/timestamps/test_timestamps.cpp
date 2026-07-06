#include "core/utilities/timestamps.hpp"

#include "gtest/gtest.h"

using namespace PHARE::core;

TEST(VariableTimeStamper, accumulatesConstantDt)
{
    VariableTimeStamper stamper{0.1};

    EXPECT_DOUBLE_EQ(stamper += 0.1, 0.1);
    EXPECT_DOUBLE_EQ(stamper += 0.1, 0.2);
    EXPECT_DOUBLE_EQ(stamper += 0.1, 0.3);
}

TEST(VariableTimeStamper, staysContinuousWhenDtChanges)
{
    VariableTimeStamper stamper{0.1};

    EXPECT_DOUBLE_EQ(stamper += 0.1, 0.1);
    EXPECT_DOUBLE_EQ(stamper += 0.1, 0.2);

    // dt grows: the accumulated time must keep advancing from where it left off, not reset
    EXPECT_DOUBLE_EQ(stamper += 0.2, 0.4);
    EXPECT_DOUBLE_EQ(stamper += 0.2, 0.6);

    // dt shrinks again
    EXPECT_DOUBLE_EQ(stamper += 0.05, 0.65);
    EXPECT_DOUBLE_EQ(stamper += 0.05, 0.7);
}

TEST(VariableTimeStamper, seedsFromANonZeroInitTime)
{
    // this is the restart case: TimeStamperFactory always seeds init_time at 0 and relies on the
    // first call resetting dt_, but the class itself supports a nonzero seed
    VariableTimeStamper stamper{0.1, /*init_time=*/5.0};

    EXPECT_DOUBLE_EQ(stamper += 0.1, 5.1);
    EXPECT_DOUBLE_EQ(stamper += 0.1, 5.2);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
