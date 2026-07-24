// AI-generated tests (Claude), reviewed by hand before committing.
//
// Host-only behavior of the device containers: these never allocate or copy
// device memory, so they don't need a CUDA device and run per-case. The cases
// that do touch the GPU live in test_cuda.cu, sharing one CUDA context.

#include "device/Array.hpp"
#include "device/Vector.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

TEST(DeviceArrayTest, ResetFillsHostWithInitialValue)
{
    DeviceArray<int> array(4, 7);
    array.reset();

    for (size_t i = 0; i < array.size(); ++i)
        EXPECT_EQ(array.hostPtr()[i], 7);
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
