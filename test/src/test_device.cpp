// AI-generated tests (Claude), reviewed by hand before committing.

#include "device/Array.hpp"
#include "device/Vector.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

// These exercise the host<->device transfer plumbing and therefore require a
// working CUDA device.

TEST(DeviceArrayTest, ResetFillsHostWithInitialValue)
{
    DeviceArray<int> array(4, 7);
    array.reset();

    for (size_t i = 0; i < array.size(); ++i)
        EXPECT_EQ(array.hostPtr()[i], 7);
}

TEST(DeviceArrayTest, HostDeviceRoundTripRestoresData)
{
    DeviceArray<int> array(3, 5);
    array.reset();
    array.ensureDeviceAllocation();

    // Clobber the host copy, then pull it back from the device.
    array.hostPtr()[0] = 999;
    array.hostPtr()[1] = -1;
    array.updateHostData();

    EXPECT_EQ(array.hostPtr()[0], 5);
    EXPECT_EQ(array.hostPtr()[1], 5);
    EXPECT_EQ(array.hostPtr()[2], 5);
}

TEST(DeviceArrayTest, MoveTransfersOwnership)
{
    DeviceArray<int> source(8, 1);
    DeviceArray<int> moved = std::move(source);

    EXPECT_EQ(moved.size(), 8u);
    EXPECT_EQ(source.size(), 0u);
    EXPECT_EQ(source.hostPtr(), nullptr);
}

TEST(DeviceVectorTest, TracksHostSize)
{
    DeviceVector<int> vector(std::vector<int>{1, 2, 3});
    EXPECT_EQ(vector.size(), 3u);

    vector.push_back(4);
    EXPECT_EQ(vector.size(), 4u);
}

TEST(DeviceVectorTest, LazyAllocationProvidesDevicePointer)
{
    DeviceVector<float> vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    vector.ensureDeviceAllocation();

    EXPECT_NE(vector.devicePtr(), nullptr);
    EXPECT_EQ(vector.deviceSpan().size(), 4u);
}

TEST(DeviceVectorTest, MoveKeepsContents)
{
    DeviceVector<int> source(std::vector<int>{10, 20, 30});
    source.ensureDeviceAllocation();

    DeviceVector<int> moved = std::move(source);
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved.hostPtr()[0], 10);
    EXPECT_NE(moved.devicePtr(), nullptr);
}
