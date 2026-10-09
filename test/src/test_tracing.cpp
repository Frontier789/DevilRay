// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/Intersection.hpp"
#include "tracing/Benchmark.hpp"
#include "models/MeshUtils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>
#include <span>

// Defined in IntersectionImpl.hpp and compiled into the library, but not exposed
// through a public header, so we declare the signatures we exercise here.
struct BoxInterval
{
    float enter_time;
    float exit_time;
};
BoxInterval findBoxInterval(float min, float max, float p, float v);
std::optional<float> testBoxIntersection(const AABB &box, const Ray &ray);

using test::expectVec3Near;
using test::HostObject;
using test::makeHostObject;
using test::unitTriangleMesh;

namespace
{
    const AABB unitBox{.min = {-1, -1, -1}, .max = {1, 1, 1}};

    float bruteForceClosestT(const Mesh &mesh, const Ray &ray)
    {
        float best = std::numeric_limits<float>::infinity();
        for (const auto &tri : mesh.triangles)
        {
            const TriangleVertices verts{
                .a = mesh.points[tri.a.pi],
                .b = mesh.points[tri.b.pi],
                .c = mesh.points[tri.c.pi],
            };
            const auto hit = intersectTriangle(ray, verts);
            if (hit.valid())
                best = std::min(best, hit.t);
        }
        return best;
    }

    Mesh scatteredTriangleMesh(test::DeterministicRng &rng, int count)
    {
        Mesh mesh;
        mesh.name = "scattered";
        mesh.normals = {Vec3{0, 0, 1}};
        for (int i = 0; i < count; ++i)
        {
            const float cx = (rng.rnd() - 0.5f) * 4.0f;
            const float cy = (rng.rnd() - 0.5f) * 4.0f;
            const float cz = rng.rnd() * -5.0f;
            const uint32_t base = static_cast<uint32_t>(mesh.points.size());
            mesh.points.push_back(Vec3{cx, cy, cz});
            mesh.points.push_back(Vec3{cx + 0.6f, cy, cz});
            mesh.points.push_back(Vec3{cx, cy + 0.6f, cz});
            mesh.triangles.push_back(Triangle{.a = {base, 0}, .b = {base + 1, 0}, .c = {base + 2, 0}});
        }
        return mesh;
    }
}

// --- findBoxInterval ---

TEST(BoxIntervalTest, OrdersEnterBeforeExit)
{
    const auto positive = findBoxInterval(-1, 1, -5, 1);
    EXPECT_FLOAT_EQ(positive.enter_time, 4);
    EXPECT_FLOAT_EQ(positive.exit_time, 6);
}

TEST(BoxIntervalTest, NegativeDirectionStillOrdersInterval)
{
    const auto negative = findBoxInterval(-1, 1, 5, -1);
    EXPECT_FLOAT_EQ(negative.enter_time, 4);
    EXPECT_FLOAT_EQ(negative.exit_time, 6);
}

// --- testBoxIntersection ---

TEST(BoxIntersectionTest, HitsFromOutside)
{
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {0, 0, -5}, .v = {0, 0, 1}});
    ASSERT_TRUE(hit.has_value());
    EXPECT_FLOAT_EQ(*hit, 4.0f);
}

TEST(BoxIntersectionTest, OriginInsideClampsToZero)
{
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {0, 0, 0}, .v = {0, 0, 1}});
    ASSERT_TRUE(hit.has_value());
    EXPECT_FLOAT_EQ(*hit, 0.0f);
}

TEST(BoxIntersectionTest, BoxBehindOriginMisses)
{
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {0, 0, 5}, .v = {0, 0, 1}});
    EXPECT_FALSE(hit.has_value());
}

TEST(BoxIntersectionTest, AxisAlignedRayInsideSlabHits)
{
    // v.x = v.y = 0, but the origin lies within the X and Y slabs.
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {0.5f, -0.5f, -5}, .v = {0, 0, 1}});
    ASSERT_TRUE(hit.has_value());
    EXPECT_FLOAT_EQ(*hit, 4.0f);
}

TEST(BoxIntersectionTest, ParallelRayOutsideSlabMisses)
{
    // Travels along Z but sits outside the X slab the whole time.
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {5, 0, -5}, .v = {0, 0, 1}});
    EXPECT_FALSE(hit.has_value());
}

TEST(BoxIntersectionTest, CornerGrazeHits)
{
    const auto hit = testBoxIntersection(unitBox, Ray{.p = {-5, -5, 0}, .v = {1, 1, 0}});
    ASSERT_TRUE(hit.has_value());
    EXPECT_FLOAT_EQ(*hit, 4.0f);
}

// --- triangleFaceNormal ---

TEST(TriangleNormalTest, MatchesCrossProductDirection)
{
    const TriangleVertices tri{.a = {0, 0, 0}, .b = {1, 0, 0}, .c = {0, 1, 0}};
    expectVec3Near(triangleFaceNormal(tri).normalized(), {0, 0, 1});
}

// triangleFaceNormal is deliberately left unnormalized: barycentricCoordinates
// relies on its squared length being (2*area)^2, so the magnitude is part of the
// contract, not an accident.
TEST(TriangleNormalTest, MagnitudeIsTwiceTriangleArea)
{
    const TriangleVertices tri{.a = {0, 0, 0}, .b = {2, 0, 0}, .c = {0, 0, 3}};
    const float area = triangleArea(tri.a, tri.b, tri.c);
    EXPECT_NEAR(area, 3.0f, 1e-6f);
    EXPECT_NEAR(triangleFaceNormal(tri).length(), 2.0f * area, 1e-5f);
}

// --- intersectMesh (BBH traversal) ---
// The wavefront intersectMesh returns {t, triangleID}; world position is
// recovered as ray.p + ray.v * t and the shading normal via surfaceNormal.

TEST(MeshIntersectionTest, HitsSingleTriangle)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMeshView mesh = object.hostView();

    const Ray ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    const auto hit = intersectMesh(ray, mesh);
    ASSERT_TRUE(hit.valid());
    EXPECT_NEAR(hit.t, 5.0f, 1e-5f);
    expectVec3Near(ray.p + ray.v * hit.t, {0.2f, 0.2f, 0});
}

TEST(MeshIntersectionTest, MissReturnsInvalid)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMeshView mesh = object.hostView();

    const auto hit = intersectMesh(Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}, mesh);
    EXPECT_FALSE(hit.valid());
}

TEST(MeshIntersectionTest, AppliesModelToWorldTranslation)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    object.transform = Transform{.s = {1, 1, 1}, .p = {10, 0, 0}};
    const TriangleMeshView mesh = object.hostView();

    const Ray ray{.p = {10.2f, 0.2f, 5}, .v = {0, 0, -1}};
    const auto hit = intersectMesh(ray, mesh);
    ASSERT_TRUE(hit.valid());
    expectVec3Near(ray.p + ray.v * hit.t, {10.2f, 0.2f, 0});
}

TEST(MeshIntersectionTest, NormalFlipsTowardIncomingRay)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMeshView mesh = object.hostView();

    const Ray above{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    const auto fromAbove = intersectMesh(above, mesh);
    ASSERT_TRUE(fromAbove.valid());
    const auto nAbove = surfaceNormal(mesh, fromAbove.triangleID, above.v, above.p + above.v * fromAbove.t);
    expectVec3Near(nAbove, {0, 0, 1});
    EXPECT_LE(nAbove.dot(above.v), 0.0f);

    const Ray below{.p = {0.2f, 0.2f, -5}, .v = {0, 0, 1}};
    const auto fromBelow = intersectMesh(below, mesh);
    ASSERT_TRUE(fromBelow.valid());
    expectVec3Near(
        surfaceNormal(mesh, fromBelow.triangleID, below.v, below.p + below.v * fromBelow.t),
        {0, 0, -1});
}

TEST(MeshIntersectionTest, ReturnsClosestOfStackedTriangles)
{
    Mesh mesh;
    mesh.name = "stack";
    mesh.normals = {Vec3{0, 0, 1}};
    for (float z : {-2.0f, 0.0f, -4.0f})
    {
        const uint32_t base = static_cast<uint32_t>(mesh.points.size());
        mesh.points.push_back(Vec3{0, 0, z});
        mesh.points.push_back(Vec3{1, 0, z});
        mesh.points.push_back(Vec3{0, 1, z});
        mesh.triangles.push_back(Triangle{.a = {base, 0}, .b = {base + 1, 0}, .c = {base + 2, 0}});
    }

    HostObject object = makeHostObject(std::move(mesh));
    const Ray ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    const auto hit = intersectMesh(ray, object.hostView());
    ASSERT_TRUE(hit.valid());
    EXPECT_NEAR(hit.t, 5.0f, 1e-5f);
    EXPECT_NEAR((ray.p + ray.v * hit.t).z, 0.0f, 1e-5f);
}

TEST(MeshIntersectionTest, TraversalMatchesBruteForce)
{
    test::DeterministicRng rng;
    Mesh mesh = scatteredTriangleMesh(rng, 40);
    const Mesh meshCopy = mesh; // makeHostObject reorders triangles; keep originals for brute force

    HostObject object = makeHostObject(std::move(mesh));
    const TriangleMeshView view = object.hostView();

    test::DeterministicRng rayRng{7};
    for (int i = 0; i < 25; ++i)
    {
        const Ray ray{
            .p = {(rayRng.rnd() - 0.5f) * 4.0f, (rayRng.rnd() - 0.5f) * 4.0f, 5.0f},
            .v = {0, 0, -1},
        };

        const auto hit = intersectMesh(ray, view);
        const float bruteT = bruteForceClosestT(meshCopy, ray);

        if (std::isinf(bruteT))
        {
            EXPECT_FALSE(hit.valid()) << "ray " << i;
        }
        else
        {
            ASSERT_TRUE(hit.valid()) << "ray " << i;
            EXPECT_NEAR(hit.t, bruteT, 1e-4f) << "ray " << i;
        }
    }
}

// --- intersectScene across multiple objects ---

TEST(SceneIntersectionTest, ReturnsNearestObject)
{
    HostObject near = makeHostObject(unitTriangleMesh());
    near.material = 11;
    HostObject far = makeHostObject(unitTriangleMesh());
    far.material = 22;
    far.transform = Transform{.s = {1, 1, 1}, .p = {0, 0, -3}};

    const std::array<TriangleMeshView, 2> objects{near.hostView(), far.hostView()};

    const auto hit = intersectScene(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, objects);
    ASSERT_TRUE(hit.valid());
    EXPECT_EQ(objects[hit.meshID].material, 11);
    EXPECT_NEAR(hit.t, 5.0f, 1e-5f);
}

TEST(SceneIntersectionTest, EmptySceneMisses)
{
    const auto hit = intersectScene(Ray{.p = {0, 0, 0}, .v = {0, 0, -1}}, std::span<const TriangleMeshView>{});
    EXPECT_FALSE(hit.valid());
}

// --- benchmark counters ---

TEST(BenchmarkCountsTest, SingleTriangleHitCountsOneOfEach)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const std::array<TriangleMeshView, 1> objects{object.hostView()};

    benchmark::HitTests counts{};
    const auto hit = intersectSceneBenchmark(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, objects, counts);

    ASSERT_TRUE(hit.valid());
    EXPECT_EQ(counts.bbox_tests, 1);
    EXPECT_EQ(counts.triangle_tests, 1);
}

TEST(BenchmarkCountsTest, MissingBoxSkipsTriangleTests)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const std::array<TriangleMeshView, 1> objects{object.hostView()};

    benchmark::HitTests counts{};
    const auto hit = intersectSceneBenchmark(Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}, objects, counts);

    EXPECT_FALSE(hit.valid());
    EXPECT_EQ(counts.bbox_tests, 1);
    EXPECT_EQ(counts.triangle_tests, 0);
}

TEST(BenchmarkCountsTest, HierarchyPrunesTriangleTests)
{
    test::DeterministicRng rng;
    HostObject object = makeHostObject(scatteredTriangleMesh(rng, 64));
    const TriangleMeshView mesh = object.hostView();
    const std::array<TriangleMeshView, 1> objects{mesh};

    benchmark::HitTests counts{};
    intersectSceneBenchmark(Ray{.p = {0, 0, 5}, .v = {0, 0, -1}}, objects, counts);

    EXPECT_GT(counts.bbox_tests, 0);
    EXPECT_LT(counts.triangle_tests, mesh.triangle_count)
        << "BBH should test fewer triangles than a brute force scan";
}

TEST(BenchmarkCountsTest, SkipAndCountingTraversalAgree)
{
    test::DeterministicRng rng;
    HostObject object = makeHostObject(scatteredTriangleMesh(rng, 32));
    const TriangleMeshView mesh = object.hostView();
    const std::array<TriangleMeshView, 1> objects{mesh};

    const Ray ray{.p = {0.3f, -0.2f, 5}, .v = {0, 0, -1}};
    const auto plain = intersectScene(ray, objects);

    benchmark::HitTests counts{};
    const auto counted = intersectSceneBenchmark(ray, objects, counts);

    EXPECT_EQ(plain.valid(), counted.valid());
    if (plain.valid() && counted.valid())
        EXPECT_FLOAT_EQ(plain.t, counted.t);
}
