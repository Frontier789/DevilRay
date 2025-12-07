#include "tracing/DistributionSamplers.hpp"

#include <gtest/gtest.h>

/////////////////////////////////////////
///             DISCLAIMER            ///
///                                   ///
/// These test were ai generated with ///
/// Google's Gemini 3, then reviewed  ///
/// And corrected here and there by   ///
/// hand.                             ///
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

// =========================================================================
// 3. Google Tests
// =========================================================================

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

TEST(AliasTableGen, NegativeValues) {
    // The Alias method mathematically assumes positive weights.
    // However, we should ensure the code doesn't crash or hang.
    // If the implementation doesn't filter negatives, it produces undefined probabilities.
    // This test ensures stability (no infinite loops/crashes).
    std::vector<float> input = {-5.0f, 10.0f};
    
    // Just checking it doesn't crash
    EXPECT_NO_FATAL_FAILURE(generateAliasTable(input));
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

// Corner Case: The "Robin Hood" necessity
// Some inputs cause a cascade where a priority queue performs better than a stack,
// but the stack (std::vector) implementation is standard. 
// We verify that the output is still mathematically valid regardless of internal pairing.
TEST(AliasTableGen, AlternatingHighLow) {
    std::vector<float> input = {0.1f, 10.0f, 0.1f, 10.0f, 0.1f, 10.0f};
    AliasTable table = generateAliasTable(input);
    VerifyTableFidelity(input, table);
}

TEST(AliasTableGen, StressTest) {
    // 1. Setup
    constexpr size_t N = 1000000;
    std::vector<float> input(N);

    // Use a fixed seed for reproducibility
    std::mt19937 rng(42); 
    
    // Lognormal(0, 1) produces a heavy right skew.
    // Most values will be small (< 1.0), but some will be very large.
    // This effectively tests the algorithm's ability to pair many small items
    // with a few large "wealthy" items.
    std::lognormal_distribution<float> dist(0.0f, 1.0f);

    for(size_t i = 0; i < N; ++i) {
        input[i] = dist(rng);
    }

    // 2. Execution (with timing)
    auto start = std::chrono::high_resolution_clock::now();
    
    AliasTable table = generateAliasTable(input);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "[          ] 1M Lognormal entries generated in " << ms.count() << " ms" << std::endl;

    // 3. Validation
    ASSERT_EQ(table.entries.size(), N);

    // Verify fidelity. 
    // Note: Lognormal distributions can have extreme dynamic range.
    // We trust that VerifyTableFidelity uses double precision for accumulation 
    // to handle the summation of tiny and large weights correctly.
    VerifyTableFidelity(input, table, 1e-3f);
}
