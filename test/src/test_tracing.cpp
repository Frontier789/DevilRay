// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/Intersection.hpp"
#include "tracing/Scene.hpp"
#include "tracing/Benchmark.hpp"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <span>

// Defined in IntersectionTestsImpl.hpp and compiled into the library; not exposed
// through a public header, so we declare the signatures we exercise here.
struct BoxInterval
{
    float enter_time;
    float exit_time;
};
BoxInterval findBoxInterval(float min, float max, float p, float v);
std::optional<float> testBoxIntersection(const AABB &box, const Ray &ray);
Vec3 triangleNormal(const TriangleVertices &triangle);
std::optional<Intersection> cast(const Ray &ray, std::span<const TriangleMesh> objects, const ObjectsInfo &info);

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
            const auto hit = testTriangleIntersection(ray, verts);
            if (hit.has_value())
                best = std::min(best, hit->t);
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

// --- triangleNormal ---

TEST(TriangleNormalTest, MatchesCrossProductDirection)
{
    const TriangleVertices tri{.a = {0, 0, 0}, .b = {1, 0, 0}, .c = {0, 1, 0}};
    expectVec3Near(triangleNormal(tri), {0, 0, 1});
}

TEST(TriangleNormalTest, IsUnitLength)
{
    const TriangleVertices tri{.a = {0, 0, 0}, .b = {2, 0, 0}, .c = {0, 0, 3}};
    EXPECT_NEAR(triangleNormal(tri).length(), 1.0f, 1e-6f);
}

// --- getIntersection on a TriangleMesh (BBH traversal) ---

TEST(MeshIntersectionTest, HitsSingleTriangle)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    const auto hit = getIntersection(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->t, 5.0f, 1e-5f);
    expectVec3Near(hit->p, {0.2f, 0.2f, 0});
    EXPECT_FLOAT_EQ(hit->triangle.area, 0.5f);
}

TEST(MeshIntersectionTest, MissReturnsNullopt)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    const auto hit = getIntersection(Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}, mesh);
    EXPECT_FALSE(hit.has_value());
}

TEST(MeshIntersectionTest, AppliesModelToWorldTranslation)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    object.transform = Transform{.s = {1, 1, 1}, .p = {10, 0, 0}};
    const TriangleMesh mesh = object.view();

    const auto hit = getIntersection(Ray{.p = {10.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh);
    ASSERT_TRUE(hit.has_value());
    expectVec3Near(hit->p, {10.2f, 0.2f, 0});
}

TEST(MeshIntersectionTest, WorldSpaceAreaScalesWithTransform)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    object.transform = Transform{.s = {2, 3, 1}, .p = {0, 0, 0}};
    const TriangleMesh mesh = object.view();

    const auto hit = getIntersection(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->triangle.area, 0.5f * 2.0f * 3.0f, 1e-5f);
}

TEST(MeshIntersectionTest, NormalFlipsTowardIncomingRay)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    const auto fromAbove = getIntersection(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh);
    ASSERT_TRUE(fromAbove.has_value());
    expectVec3Near(fromAbove->n, {0, 0, 1});
    EXPECT_LE(fromAbove->n.dot(Vec3{0, 0, -1}), 0.0f);

    const auto fromBelow = getIntersection(Ray{.p = {0.2f, 0.2f, -5}, .v = {0, 0, 1}}, mesh);
    ASSERT_TRUE(fromBelow.has_value());
    expectVec3Near(fromBelow->n, {0, 0, -1});
}

TEST(MeshIntersectionTest, WindingFlagFollowsRayDirection)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    const auto fromAbove = getIntersection(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh);
    ASSERT_TRUE(fromAbove.has_value());
    EXPECT_FALSE(fromAbove->triangle.ccw);

    const auto fromBelow = getIntersection(Ray{.p = {0.2f, 0.2f, -5}, .v = {0, 0, 1}}, mesh);
    ASSERT_TRUE(fromBelow.has_value());
    EXPECT_TRUE(fromBelow->triangle.ccw);
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
    const auto hit = getIntersection(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, object.view());
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->t, 5.0f, 1e-5f);
    EXPECT_NEAR(hit->p.z, 0.0f, 1e-5f);
}

TEST(MeshIntersectionTest, TraversalMatchesBruteForce)
{
    test::DeterministicRng rng;
    Mesh mesh = scatteredTriangleMesh(rng, 40);
    const Mesh meshCopy = mesh; // makeHostObject reorders triangles; keep originals for brute force

    HostObject object = makeHostObject(std::move(mesh));
    const TriangleMesh view = object.view();

    test::DeterministicRng rayRng{7};
    for (int i = 0; i < 25; ++i)
    {
        const Ray ray{
            .p = {(rayRng.rnd() - 0.5f) * 4.0f, (rayRng.rnd() - 0.5f) * 4.0f, 5.0f},
            .v = {0, 0, -1},
        };

        const auto hit = getIntersection(ray, view);
        const float bruteT = bruteForceClosestT(meshCopy, ray);

        if (std::isinf(bruteT))
        {
            EXPECT_FALSE(hit.has_value()) << "ray " << i;
        }
        else
        {
            ASSERT_TRUE(hit.has_value()) << "ray " << i;
            EXPECT_NEAR(hit->t, bruteT, 1e-4f) << "ray " << i;
        }
    }
}

// --- cast across multiple objects ---

TEST(CastTest, ReturnsNearestObject)
{
    HostObject near = makeHostObject(unitTriangleMesh());
    near.material = 11;
    HostObject far = makeHostObject(unitTriangleMesh());
    far.material = 22;
    far.transform = Transform{.s = {1, 1, 1}, .p = {0, 0, -3}};

    const std::array<TriangleMesh, 2> objects{near.view(), far.view()};
    const ObjectsInfo info{.total_radiant_power = 1.0f};

    const auto hit = cast(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, objects, info);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->mat, 11);
    EXPECT_NEAR(hit->t, 5.0f, 1e-5f);
}

TEST(CastTest, EmptySceneMisses)
{
    const ObjectsInfo info{.total_radiant_power = 1.0f};
    const auto hit = cast(Ray{.p = {0, 0, 0}, .v = {0, 0, -1}}, std::span<const TriangleMesh>{}, info);
    EXPECT_FALSE(hit.has_value());
}

// --- benchmark counters ---

TEST(BenchmarkCountsTest, SingleTriangleHitCountsOneOfEach)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    benchmark::HitTests counts;
    const auto hit = getIntersectionBenchmark(Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}, mesh, counts);

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(counts.bbox_tests, 1);
    EXPECT_EQ(counts.triangle_tests, 1);
}

TEST(BenchmarkCountsTest, MissingBoxSkipsTriangleTests)
{
    HostObject object = makeHostObject(unitTriangleMesh());
    const TriangleMesh mesh = object.view();

    benchmark::HitTests counts;
    const auto hit = getIntersectionBenchmark(Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}, mesh, counts);

    EXPECT_FALSE(hit.has_value());
    EXPECT_EQ(counts.bbox_tests, 1);
    EXPECT_EQ(counts.triangle_tests, 0);
}

TEST(BenchmarkCountsTest, HierarchyPrunesTriangleTests)
{
    test::DeterministicRng rng;
    HostObject object = makeHostObject(scatteredTriangleMesh(rng, 64));
    const TriangleMesh mesh = object.view();

    benchmark::HitTests counts;
    getIntersectionBenchmark(Ray{.p = {0, 0, 5}, .v = {0, 0, -1}}, mesh, counts);

    EXPECT_GT(counts.bbox_tests, 0);
    EXPECT_LT(counts.triangle_tests, mesh.tris_count)
        << "BBH should test fewer triangles than a brute force scan";
}

TEST(BenchmarkCountsTest, SkipAndCountingTraversalAgree)
{
    test::DeterministicRng rng;
    HostObject object = makeHostObject(scatteredTriangleMesh(rng, 32));
    const TriangleMesh mesh = object.view();

    const Ray ray{.p = {0.3f, -0.2f, 5}, .v = {0, 0, -1}};
    const auto plain = getIntersection(ray, mesh);

    benchmark::HitTests counts;
    const auto counted = getIntersectionBenchmark(ray, mesh, counts);

    EXPECT_EQ(plain.has_value(), counted.has_value());
    if (plain.has_value() && counted.has_value())
        EXPECT_FLOAT_EQ(plain->t, counted->t);
}
