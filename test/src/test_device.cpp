// AI-generated tests (Claude), reviewed by hand before committing.
//
// Host-only behavior of the device containers: these never allocate or copy
// device memory, so they don't need a CUDA device and run per-case. The cases
// that do touch the GPU live in test_cuda.cu, sharing one CUDA context.

#include "Buffers.hpp"
#include "Utils.hpp"
#include "device/Array.hpp"
#include "device/Vector.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
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

TEST(DeviceArrayTest, UninitializedHasTheRequestedSizeAndNoDeviceAllocation)
{
    auto array = DeviceArray<int>::uninitialized(5);

    EXPECT_EQ(array.size(), 5u);
    EXPECT_NE(array.hostPtr(), nullptr);
    EXPECT_EQ(array.devicePtr(), nullptr);

    for (int i = 0; i < 5; ++i)
        array.hostSpan()[i] = i * i;

    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(array.hostSpan()[i], i * i);
}

TEST(DeviceArrayTest, ConstHostSpanViewsTheHostData)
{
    const DeviceArray<int> array(3, 4);

    const std::span<const int> span = array.hostSpan();

    EXPECT_EQ(span.size(), 3u);
    EXPECT_EQ(span.data(), array.hostPtr());
    EXPECT_EQ(span[2], 4);
}

TEST(DeviceVectorTest, TracksHostSize)
{
    DeviceVector<int> vector(std::vector<int>{1, 2, 3});
    EXPECT_EQ(vector.size(), 3u);

    vector.push_back(4);
    EXPECT_EQ(vector.size(), 4u);
}

TEST(BuffersTest, TotalCastsSumsEveryPixelAndResetClearsThem)
{
    Buffers buffers(Size2i{4, 2});
    ASSERT_EQ(buffers.casts.size(), 8u);
    EXPECT_EQ(buffers.totalCasts(), 0u);

    buffers.casts.hostPtr()[0] = 2;
    buffers.casts.hostPtr()[7] = 5;
    buffers.color.hostPtr()[3] = Vec4{1, 1, 1, 1};
    EXPECT_EQ(buffers.totalCasts(), 7u);

    buffers.reset();

    EXPECT_EQ(buffers.totalCasts(), 0u);
    EXPECT_EQ(buffers.color.hostPtr()[3].w, 0.0f);
}

TEST(BuffersTest, TotalCastsDoesNotOverflowThirtyTwoBits)
{
    Buffers buffers(Size2i{2, 1});
    constexpr uint32_t nearly_full = 0xF0000000u;
    buffers.casts.hostPtr()[0] = nearly_full;
    buffers.casts.hostPtr()[1] = nearly_full;

    EXPECT_EQ(buffers.totalCasts(), 2 * static_cast<uint64_t>(nearly_full));
}
