// AI-generated tests (Claude), reviewed by hand before committing.

#include "RunningAverage.hpp"

#include <gtest/gtest.h>

TEST(RunningAverageTest, EmptyAverageIsZero)
{
    RunningAverage average(5);
    EXPECT_DOUBLE_EQ(average.mean(), 0.0);
}

TEST(RunningAverageTest, AveragesAddedValues)
{
    RunningAverage average(5);
    average.add(2.0);
    average.add(4.0);
    average.add(6.0);
    EXPECT_DOUBLE_EQ(average.mean(), 4.0);
}

TEST(RunningAverageTest, DropsOldestBeyondWindow)
{
    RunningAverage average(3);
    average.add(1.0);
    average.add(2.0);
    average.add(3.0);
    average.add(4.0); // evicts 1.0, window is now {2,3,4}
    EXPECT_DOUBLE_EQ(average.mean(), 3.0);
}

TEST(RunningAverageTest, KeepsOnlyWindowSizeSamples)
{
    RunningAverage average(2);
    for (int i = 0; i < 100; ++i)
        average.add(static_cast<double>(i));
    // Only the last two values (98, 99) survive.
    EXPECT_DOUBLE_EQ(average.mean(), 98.5);
}

TEST(RunningAverageTest, ResetClearsHistory)
{
    RunningAverage average(5);
    average.add(10.0);
    average.add(20.0);
    average.reset();
    EXPECT_DOUBLE_EQ(average.mean(), 0.0);
}
