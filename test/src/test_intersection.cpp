
#include "tracing/Intersection.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <cmath>

/////////////////////////////////////////
///             DISCLAIMER            ///
///                                   ///
/// These test were ai generated with ///
/// Google's Gemini 3, then reviewed  ///
/// And corrected here and there by   ///
/// hand.                             ///
/////////////////////////////////////////

void ExpectValidNumbers(const TriangleIntersection& intersect) {
    EXPECT_TRUE(std::isfinite(intersect.t)) << "Intersection 't' is NaN or Inf";
    EXPECT_TRUE(std::isfinite(intersect.bari.x)) << "Barycentric x is NaN or Inf";
    EXPECT_TRUE(std::isfinite(intersect.bari.y)) << "Barycentric y is NaN or Inf";
    EXPECT_TRUE(std::isfinite(intersect.bari.z)) << "Barycentric z is NaN or Inf";
}

// --- Test Suite ---
class IntersectionTest : public ::testing::Test {
protected:
    // A standard CCW triangle on the XY plane (Z=0)
    // Vertices: A(0,0,0), B(1,0,0), C(0,1,0)
    TriangleVertices unitTri = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
};

// 1. HAPPY PATH CASES
TEST_F(IntersectionTest, HitsCenterOfTriangle) {
    // Ray starts at (0.2, 0.2, 5) and shoots straight down the Z-axis
    Ray ray{{0.2f, 0.2f, 5.0f}, {0.0f, 0.0f, -1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);

    ASSERT_TRUE(result.has_value()) << "Ray should have hit the center of the triangle";
    ExpectValidNumbers(*result);
    EXPECT_NEAR(result->t, 5.0f, 1e-5f);
    
    // Check that barycentric coordinates are within the valid [0, 1] range
    EXPECT_GE(result->bari.x, 0.0f);
    EXPECT_GE(result->bari.y, 0.0f);
    EXPECT_GE(result->bari.z, 0.0f);
    
    float barySum = result->bari.x + result->bari.y + result->bari.z;
    EXPECT_NEAR(barySum, 1.0f, 1e-5f);
}

TEST_F(IntersectionTest, HitsExactlyOnVertex) {
    // Ray pointing directly at vertex A (0,0,0) from Z=10
    Ray ray{{0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, -1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);

    ASSERT_TRUE(result.has_value());
    ExpectValidNumbers(*result);
    EXPECT_NEAR(result->t, 10.0f, 1e-5f);
}

// 2. MISS CASES (GEOMETRY)
TEST_F(IntersectionTest, MissesJustOutsideEdge) {
    // Ray at (-0.01, 0.5, 1) - slightly to the left of the triangle
    Ray ray{{-0.01f, 0.5f, 1.0f}, {0.0f, 0.0f, -1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IntersectionTest, MissesBehindRayOrigin) {
    // Ray starts at (0.2, 0.2, 1) but points UP (+Z), away from triangle at Z=0
    Ray ray{{0.2f, 0.2f, 1.0f}, {0.0f, 0.0f, 1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);
    EXPECT_FALSE(result.has_value());
}

// 3. EDGE CASES (MATHEMATICAL)
TEST_F(IntersectionTest, ParallelRayDoesNotIntersect) {
    // Ray is parallel to the triangle plane (sliding along X-axis at Z=1)
    Ray ray{{-1.0f, 0.2f, 1.0f}, {1.0f, 0.0f, 0.0f}};
    auto result = testTriangleIntersection(ray, unitTri);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IntersectionTest, DegenerateTriangleFails) {
    // Triangle where all points are identical (a single point, no area)
    TriangleVertices flatTri = {{0,0,0}, {0,0,0}, {0,0,0}};
    Ray ray{{0,0,1}, {0,0,-1}};
    auto result = testTriangleIntersection(ray, flatTri);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IntersectionTest, NoBackfaceCulling) {
    // Ray hits from "underneath" (Z = -1, pointing UP to Z=0)
    // Most renderers/engines treat this as a miss if culling is enabled
    Ray ray{{0.2f, 0.2f, -1.0f}, {0.0f, 0.0f, 1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);
    
    // Adjust this expectation based on whether your function allows backfaces
    ASSERT_TRUE(result.has_value()); 
    ExpectValidNumbers(*result);
}

TEST_F(IntersectionTest, VeryDistantIntersection) {
    Ray ray{{0.2f, 0.2f, 10000.0f}, {0.0f, 0.0f, -1.0f}};
    auto result = testTriangleIntersection(ray, unitTri);
    ASSERT_TRUE(result.has_value());
    ExpectValidNumbers(*result);

    EXPECT_NEAR(result->t, 10000.0f, 1e-2f);
}

// --- Degenerate Case Tests ---

TEST_F(IntersectionTest, DegenerateTriangleCollinear) {
    // Vertices form a straight line (zero area)
    TriangleVertices lineTri = {{0,0,0}, {1,0,0}, {2,0,0}};
    Ray ray{{0.5f, 0, 1}, {0,0,-1}};
    
    auto result = testTriangleIntersection(ray, lineTri);
    EXPECT_FALSE(result.has_value()) << "Collinear vertices should not result in a hit";
}

TEST_F(IntersectionTest, DegenerateRayZeroDirection) {
    // Ray has no direction (v = 0,0,0)
    Ray ray{{0.2f, 0.2f, 1.0f}, {0.0f, 0.0f, 0.0f}};
    
    auto result = testTriangleIntersection(ray, unitTri);
    EXPECT_FALSE(result.has_value()) << "Zero direction should not result in a hit";
}

TEST_F(IntersectionTest, DegenerateRayNaNInDirection) {
    // Ray contains a NaN in direction, common in buggy physics/cam logic
    Ray ray{{0,0,0}, {NAN, NAN, NAN}};
    
    auto result = testTriangleIntersection(ray, unitTri);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IntersectionTest, ExtremelyThinTriangle) {
    // Triangle is valid but extremely thin (sliver)
    // Testing for precision/robustness near epsilon limits
    TriangleVertices sliverTri = {{0,0,0}, {100.0f, 0, 0}, {100.0f, 2e-8f, 0}};
    Ray ray{{50.0f, 1e-9f, 1.0f}, {0,0,-1}};
    
    auto result = testTriangleIntersection(ray, sliverTri);
    ASSERT_TRUE(result.has_value());
    ExpectValidNumbers(*result);
}

TEST_F(IntersectionTest, MultipleAngleSanityCheck) {
    // A point we know is inside the triangle
    constexpr Vec3 targetInside{0.2f, 0.2f, 0.0f};
    constexpr float expectedU = 0.2f;
    constexpr float expectedV = 0.2f;
    constexpr float expectedW = 0.6f;

    // A point we know is outside the triangle
    constexpr Vec3 targetOutside{5.0f, 5.0f, 0.0f};

    // Test ray origins from various positions in the positive Z hemisphere
    const std::vector<Vec3> origins = {
        {0.2f, 0.2f, 10.0f},  // Top-down
        {10.0f, 0.2f, 10.0f}, // 45-degree slant
        {-5.0f, -5.0f, 2.0f}, // Low grazing angle
        {0.0f, 0.0f, 1.0f}    // Originating near a vertex
    };

    for (const auto& origin : origins) {
        // 1. TEST VALID HITS
        // Calculate direction: (target - origin)
        const Vec3 dirHit = targetInside - origin;
        const Ray rayHit{origin, dirHit};

        const auto resultHit = testTriangleIntersection(rayHit, unitTri);
        
        EXPECT_TRUE(resultHit.has_value()) 
            << "Ray from (" << origin.x << "," << origin.y << "," << origin.z << ") should hit.";
        
        if (resultHit.has_value()) {
            ExpectValidNumbers(*resultHit);
            // Since direction is (Target - Origin), t should be approximately 1.0
            EXPECT_NEAR(resultHit->t, 1.0f, 1e-4f);

            EXPECT_NEAR(resultHit->bari.x + resultHit->bari.y + resultHit->bari.z, 1.0f, 1e-5f);
            EXPECT_NEAR(resultHit->bari.x, expectedW, 1e-4f); 
            EXPECT_NEAR(resultHit->bari.y, expectedU, 1e-4f);
            EXPECT_NEAR(resultHit->bari.z, expectedV, 1e-4f);
        }

        // 2. TEST VALID MISSES
        // Calculate direction to the outside point
        const Vec3 dirMiss = targetOutside - origin;
        const Ray rayMiss{origin, dirMiss};

        const auto resultMiss = testTriangleIntersection(rayMiss, unitTri);
        
        EXPECT_FALSE(resultMiss.has_value()) 
            << "Ray from (" << origin.x << "," << origin.y << "," << origin.z << ") should miss.";
    }
}