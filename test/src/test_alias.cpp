#include "Image.hpp"
#include "TracingTestHelpers.hpp"
#include "Utils.hpp"
#include "tracing/DistributionSamplers.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <random>
#include <ratio>
#include <vector>

/////////////////////////////////////////
///             DISCLAIMER            ///
///                                   ///
/// These test were ai generated with ///
/// Google's Gemini 3, then reviewed  ///
/// And corrected here and there by   ///
/// hand. Some tests were later added ///
/// or rewritten by Claude, reviewed  ///
/// by hand before committing.        ///
/////////////////////////////////////////

namespace {
    void VerifyTableFidelity(const std::vector<float>& input, const AliasTable& table, float tolerance = 1e-4f)
    {
        const size_t N = input.size();
        ASSERT_EQ(table.entries.size(), N) << "Table size must match input size";

        double totalWeight = 0.0;
        for (float w : input) totalWeight += w;

        // 1. Calculate Expected Probabilities
        std::vector<double> expected_probs(N);
        for (size_t i = 0; i < N; ++i) {
            expected_probs[i] = (totalWeight > 0) ? (input[i] / totalWeight) : 0.0;
        }

        // 2. Calculate Actual Probabilities derived from the Table
        std::vector<double> actual_probs(N, 0.0);
        const AliasEntry* entries = table.entries.hostPtr();

        for (size_t j = 0; j < N; ++j) {
            const auto& entry = entries[j];

            // Probability of picking this slot is 1/N
            double slot_prob = 1.0 / static_cast<double>(N);

            // Contribution from A
            if (entry.A >= 0 && entry.A < N) {
                actual_probs[entry.A] += slot_prob * entry.p_A;
            }

            // Contribution from B (only if probability is not 1.0)
            if (entry.p_A < 1.0f && entry.B >= 0 && entry.B < N) {
                actual_probs[entry.B] += slot_prob * (1.0f - entry.p_A);
            }
        }

        // 3. Compare
        for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(actual_probs[i], expected_probs[i], tolerance)
                << "Mismatch for index " << i << " with input weight " << input[i];
        }
    }
}

TEST(AliasTableGen, EmptyInput) {
    std::vector<float> input = {};
    AliasTable table = generateAliasTable(input);
    EXPECT_EQ(table.entries.size(), 0);
}

TEST(AliasTableGen, SingleElement) {
    // If there is only one element, it should always be picked.
    std::vector<float> input = {42.0f};
    AliasTable table = generateAliasTable(input);

    ASSERT_EQ(table.entries.size(), 1);

    // With one element, p_A must be 1.0 (always pick A)
    // B is irrelevant, but p_A should ensure we never check B
    EXPECT_FLOAT_EQ(table.entries.hostPtr()[0].p_A, 1.0f);
    EXPECT_EQ(table.entries.hostPtr()[0].A, 0);
}

TEST(AliasTableGen, UniformDistribution) {
    // 3 elements with equal weight
    std::vector<float> input = {10.f, 10.f, 10.f};
    AliasTable table = generateAliasTable(input);

    ASSERT_EQ(table.entries.size(), 3);

    // In a perfect uniform distribution, every slot handles itself perfectly.
    // p_A should be 1.0 for all, B is irrelevant.
    for(int i=0; i<3; ++i) {
        EXPECT_FLOAT_EQ(table.entries.hostPtr()[i].p_A, 1.0f);
        EXPECT_EQ(table.entries.hostPtr()[i].A, i);
    }

    VerifyTableFidelity(input, table);
}

TEST(AliasTableGen, SimpleBiased) {
    // Simple case: Item 0 has weight 1, Item 1 has weight 3.
    // Total = 4. Avg = 2.
    // Item 0 (val 1) is "Small" (1 < 2). P = 1/2 = 0.5.
    // Item 1 (val 3) is "Large".
    std::vector<float> input = {1.0f, 3.0f};
    AliasTable table = generateAliasTable(input);

    ASSERT_EQ(table.entries.size(), 2);
    VerifyTableFidelity(input, table);
}

TEST(AliasTableGen, LargeVariance) {
    // One huge weight, one tiny weight
    std::vector<float> input = {0.1f, 1000.0f};
    AliasTable table = generateAliasTable(input);

    VerifyTableFidelity(input, table, 1e-3f);
}

TEST(AliasTableGen, ZeroImportanceSum_ReturnsUniform) {
    // Input is all zeros
    std::vector<float> input = {0.0f, 0.0f, 0.0f, 0.0f};
    AliasTable table = generateAliasTable(input);

    ASSERT_EQ(table.entries.size(), 4);

    const auto* ptr = table.entries.hostPtr();
    for(int i = 0; i < 4; ++i) {
        // Uniform means every index maps to itself with probability 1.0
        EXPECT_FLOAT_EQ(ptr[i].p_A, 1.0f) << "Index " << i << " should be pure A";
        EXPECT_EQ(ptr[i].A, i) << "Index " << i << " should point to itself";
        // B is irrelevant when p_A is 1.0, but usually -1
    }
}

TEST(AliasTableGen, NegativeImportanceIsTreatedAsZero) {
    const std::vector<float> input = {-5.0f, 10.0f};
    const std::vector<float> importancesAfterClamping = {0.0f, 10.0f};

    const AliasTable table = generateAliasTable(input);
    VerifyTableFidelity(importancesAfterClamping, table);

    const AliasTableView view = table.hostView();
    test::DeterministicRng rng;
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(sample(view, rng).index, 1);
    }
}

TEST(AliasTableGen, ManyItems) {
    // Stress test with slightly more items to check vector resizing/logic
    std::vector<float> input(100);
    for(int i=0; i<100; ++i) {
        input[i] = static_cast<float>(i + 1); // 1, 2, ..., 100
    }

    AliasTable table = generateAliasTable(input);
    VerifyTableFidelity(input, table, 1e-3f);
}

TEST(AliasTableGen, FloatingPointPrecisionStub) {
    // Test a case that typically causes "overfull" bucket leakage
    // if accumulation uses integers or precision is handled poorly.
    // 1/3 cannot be represented perfectly in float.
    std::vector<float> input = {1.0f, 1.0f, 1.0f};
    // Perturb slightly
    input[0] += 0.00001f;
    input[1] -= 0.00001f;

    AliasTable table = generateAliasTable(input);

    // Ensure the last residuals were forced to 1.0f to prevent
    // accessing uninitialized 'B' indices.
    const auto* ptr = table.entries.hostPtr();
    for(int i=0; i<3; ++i) {
        // If p_A is very close to 1, it should effectively be 1,
        // or B must be a valid index.
        if (ptr[i].p_A < 0.999f) {
            EXPECT_GE(ptr[i].B, 0);
            EXPECT_LT(ptr[i].B, 3);
        }
    }
    VerifyTableFidelity(input, table);
}

TEST(AliasTableGen, AlternatingHighLow) {
    std::vector<float> input = {0.1f, 10.0f, 0.1f, 10.0f, 0.1f, 10.0f};
    AliasTable table = generateAliasTable(input);
    VerifyTableFidelity(input, table);
}

TEST(AliasTableGen, StressTest) {
    // 1. Setup
    constexpr size_t entry_count = 200000;
    std::vector<float> input(entry_count);

    // Use a fixed seed for reproducibility
    std::mt19937 rng(42);

    // Lognormal(0, 1) produces a heavy right skew.
    // Most values will be small (< 1.0), but some will be very large.
    // This effectively tests the algorithm's ability to pair many small items
    // with a few large "wealthy" items.
    std::lognormal_distribution<float> dist(0.0f, 1.0f);

    for(size_t i = 0; i < entry_count; ++i) {
        input[i] = dist(rng);
    }

    // 2. Execution (with timing)
    auto start = std::chrono::high_resolution_clock::now();

    AliasTable table = generateAliasTable(input);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "[          ] " << entry_count << " lognormal entries generated in " << ms.count() << " ms" << std::endl;

    // 3. Validation
    ASSERT_EQ(table.entries.size(), entry_count);

    // Verify fidelity.
    // Note: Lognormal distributions can have extreme dynamic range.
    // We trust that VerifyTableFidelity uses double precision for accumulation
    // to handle the summation of tiny and large weights correctly.
    VerifyTableFidelity(input, table, 1e-3f);
}

TEST(AliasTableViewTest, HostViewPointsAtHostEntries)
{
    const AliasTable table = generateAliasTable(std::vector<float>{1.0f, 2.0f, 3.0f});

    const AliasTableView view = table.hostView();

    EXPECT_EQ(view.entries, table.entries.hostPtr());
    EXPECT_EQ(view.entry_count, 3);
}

TEST(AliasTableViewTest, HostViewOfEmptyTableHasNoEntries)
{
    const AliasTable table = generateAliasTable(std::vector<float>{});

    EXPECT_EQ(table.hostView().entry_count, 0);
}

TEST(AliasTableViewTest, HostViewSamplesOnlyNonZeroImportances)
{
    const AliasTable table = generateAliasTable(std::vector<float>{0.0f, 1.0f, 0.0f});
    const AliasTableView view = table.hostView();

    test::DeterministicRng rng;
    for (int i = 0; i < 200; ++i)
    {
        const auto drawn = sample(view, rng);

        EXPECT_EQ(drawn.index, 1);
        EXPECT_FLOAT_EQ(drawn.pdf, 1.0f);
    }
}

TEST(AliasImageTableViewTest, HostViewPointsAtHostEntriesAndKeepsImageSize)
{
    auto luminance = Image1f::create(Size2i{4, 3}, 1.0f);
    const AliasImageTable table = generateAliasTable(luminance.hostView());

    const AliasImageTableView view = table.hostView();

    EXPECT_EQ(view.pixel_entries, table.pixel_entries.hostPtr());
    EXPECT_EQ(view.row_entries, table.row_entries.hostPtr());
    EXPECT_EQ(view.image_size.width, 4);
    EXPECT_EQ(view.image_size.height, 3);
}

TEST(AliasImageTableViewTest, HostViewSamplesTheOnlyBrightPixel)
{
    auto luminance = Image1f::create(Size2i{4, 3}, 0.0f);
    luminance[Vec2i{.x = 2, .y = 1}] = 5.0f;
    const AliasImageTable table = generateAliasTable(luminance.hostView());
    const AliasImageTableView view = table.hostView();

    test::DeterministicRng rng;
    for (int i = 0; i < 200; ++i)
    {
        const auto drawn = sample(view, rng);

        EXPECT_EQ(drawn.pixel_coordinate.x, 2);
        EXPECT_EQ(drawn.pixel_coordinate.y, 1);
    }
}

namespace
{
    template<typename WeightAt>
    Image1f imageOfWeights(Size2i size, WeightAt weight_at)
    {
        auto weights = Image1f::create_uninitialized(size);

        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                weights[Vec2i{x, y}] = weight_at(x, y);

        return weights;
    }

    void VerifyImageTableMatchesWeights(const Image1f &weights, double tolerance = 1e-5)
    {
        const Size2i size = weights.size;
        const AliasImageTable table = generateAliasTable(weights.hostView());

        double total = 0.0;
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                total += weights[Vec2i{x, y}];

        std::vector<double> expected(size.area());
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                expected[x + y * size.width] = total > 0.0 ? weights[Vec2i{x, y}] / total : 1.0 / size.area();

        struct Outcome
        {
            int index;
            double probability;
            float pdf;
        };

        auto outcomesOf = [](const AliasEntry &entry)
        {
            return std::array{
                Outcome{entry.A, entry.p_A, entry.pdf_A},
                Outcome{entry.B, 1.0 - entry.p_A, entry.pdf_B},
            };
        };

        const AliasEntry *rows = table.row_entries.hostPtr();
        const AliasEntry *pixels = table.pixel_entries.hostPtr();

        std::vector<double> decoded(size.area(), 0.0);

        for (int row_bucket = 0; row_bucket < size.height; ++row_bucket)
        {
            for (const Outcome &row : outcomesOf(rows[row_bucket]))
            {
                if (row.index < 0 || row.probability <= 0.0) continue;

                for (int column_bucket = 0; column_bucket < size.width; ++column_bucket)
                {
                    for (const Outcome &column : outcomesOf(pixels[row.index * size.width + column_bucket]))
                    {
                        if (column.index < 0 || column.probability <= 0.0) continue;

                        const int pixel = row.index * size.width + column.index;

                        decoded[pixel] += row.probability * column.probability / size.height / size.width;

                        EXPECT_NEAR(static_cast<double>(row.pdf) * column.pdf, expected[pixel], tolerance)
                            << "the pdf reported when drawing pixel " << pixel << " must be its probability";
                    }
                }
            }
        }

        for (int pixel = 0; pixel < size.area(); ++pixel)
            EXPECT_NEAR(decoded[pixel], expected[pixel], tolerance) << "probability of drawing pixel " << pixel;
    }
}

TEST(AliasImageTableDistributionTest, DrawsEveryPixelWithItsShareOfTheTotalWeight)
{
    constexpr int row_of_zeros = 2;

    const auto weights = imageOfWeights(Size2i{7, 5}, [](int x, int y)
    {
        return y == row_of_zeros ? 0.0f : static_cast<float>((x * 3 + y * 5) % 7);
    });

    VerifyImageTableMatchesWeights(weights);
}

TEST(AliasImageTableDistributionTest, WorksForDegenerateImageShapes)
{
    for (const Size2i size : {Size2i{1, 1}, Size2i{1, 6}, Size2i{6, 1}, Size2i{2, 2}})
    {
        SCOPED_TRACE(::testing::Message() << size.width << "x" << size.height);

        const auto weights = imageOfWeights(size, [](int x, int y) { return 1.0f + x + 2.0f * y; });
        VerifyImageTableMatchesWeights(weights);
    }
}

TEST(AliasImageTableDistributionTest, SinglePixelImageIsDrawnWithProbabilityOne)
{
    const auto weights = Image1f::create(Size2i{1, 1}, 3.0f);
    const AliasImageTable table = generateAliasTable(weights.hostView());

    test::DeterministicRng rng;
    for (int i = 0; i < 20; ++i)
    {
        const auto drawn = sample(table.hostView(), rng);

        EXPECT_EQ(drawn.pixel_coordinate.x, 0);
        EXPECT_EQ(drawn.pixel_coordinate.y, 0);
        EXPECT_FLOAT_EQ(drawn.pdf, 1.0f);
    }
}

TEST(AliasImageTableDistributionTest, AllZeroImageFallsBackToUniform)
{
    const auto weights = Image1f::create(Size2i{3, 2}, 0.0f);

    VerifyImageTableMatchesWeights(weights);
}

TEST(AliasImageTableDistributionTest, HandlesOneBrightPixelAmongVeryDimOnes)
{
    constexpr float dim = 1e-3f;
    constexpr float sun = 1e5f;

    auto weights = Image1f::create(Size2i{16, 8}, dim);
    weights[Vec2i{.x = 5, .y = 3}] = sun;

    VerifyImageTableMatchesWeights(weights);
}
