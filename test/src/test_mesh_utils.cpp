// AI-generated tests (Claude), reviewed by hand before committing.
//
// Covers include/models/MeshUtils.hpp: barycentricCoordinates and the
// vertex-normal interpolation in surfaceNormal. The interpolation path was
// previously only exercised through meshes whose three vertex normals were
// identical, which cannot tell blending apart from returning the face normal.

#include "TracingTestHelpers.hpp"

#include "models/MeshUtils.hpp"
#include "tracing/Intersection.hpp"

#include <gtest/gtest.h>

#include <cmath>

using test::expectVec3Near;
using test::HostObject;
using test::makeHostObject;

namespace
{
    // Unit triangle on the XY plane: A(0,0,0) B(1,0,0) C(0,1,0).
    const TriangleVertices unitTri{.a = {0, 0, 0}, .b = {1, 0, 0}, .c = {0, 1, 0}};

    Vec3 baryOf(const TriangleVertices &tri, const Vec3 &hit)
    {
        return barycentricCoordinates(tri, triangleFaceNormal(tri), hit);
    }

    // Point inside `tri` with the given barycentric weights.
    Vec3 pointFromWeights(const TriangleVertices &tri, const Vec3 &w)
    {
        return tri.a * w.x + tri.b * w.y + tri.c * w.z;
    }

    // Single triangle on the XY plane carrying three distinct vertex normals,
    // so barycentric blending is observable.
    Mesh smoothTriangleMesh(const Vec3 &na, const Vec3 &nb, const Vec3 &nc)
    {
        Mesh mesh;
        mesh.name = "smooth-triangle";
        mesh.points = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};
        mesh.normals = {na, nb, nc};
        mesh.triangles = {Triangle{.a = {0, 0}, .b = {1, 1}, .c = {2, 2}}};
        return mesh;
    }

    const Vec3 kNormalA{1, 0, 0};
    const Vec3 kNormalB{0, 1, 0};
    const Vec3 kNormalC{0, 0, 1};

    // Looks down -Z at the XY-plane triangle, so the face normal (0,0,+1) already
    // points against the ray and surfaceNormal performs no flip.
    const Vec3 kRayFromAbove{0, 0, -1};
}

// --- barycentricCoordinates ---

// Pins the (w_A, w_B, w_C) ordering of the returned weights.
TEST(BarycentricCoordinatesTest, OneAtOwnVertex)
{
    expectVec3Near(baryOf(unitTri, unitTri.a), {1, 0, 0});
    expectVec3Near(baryOf(unitTri, unitTri.b), {0, 1, 0});
    expectVec3Near(baryOf(unitTri, unitTri.c), {0, 0, 1});
}

TEST(BarycentricCoordinatesTest, CentroidIsEvenlyWeighted)
{
    const Vec3 centroid = (unitTri.a + unitTri.b + unitTri.c) / 3.0f;
    expectVec3Near(baryOf(unitTri, centroid), {1 / 3.0f, 1 / 3.0f, 1 / 3.0f});
}

TEST(BarycentricCoordinatesTest, EdgeMidpointsSplitBetweenEndpoints)
{
    expectVec3Near(baryOf(unitTri, (unitTri.a + unitTri.b) * 0.5f), {0.5f, 0.5f, 0});
    expectVec3Near(baryOf(unitTri, (unitTri.b + unitTri.c) * 0.5f), {0, 0.5f, 0.5f});
    expectVec3Near(baryOf(unitTri, (unitTri.c + unitTri.a) * 0.5f), {0.5f, 0, 0.5f});
}

// Round-trips arbitrary weights through a triangle that is neither axis aligned
// nor unit sized.
TEST(BarycentricCoordinatesTest, RecoversWeightsOnGeneralTriangle)
{
    const TriangleVertices tri{.a = {1, 0, 0}, .b = {0, 2, 0}, .c = {0, 0, 3}};
    const Vec3 weights{0.2f, 0.5f, 0.3f};

    expectVec3Near(baryOf(tri, pointFromWeights(tri, weights)), weights, 1e-5f);
}

// Barycentric weights are affine invariant, so translating and (non-uniformly)
// scaling both the triangle and the query point must not move them.
TEST(BarycentricCoordinatesTest, IsInvariantUnderAffineTransform)
{
    const TriangleVertices tri{.a = {1, 0, 0}, .b = {0, 2, 0}, .c = {0, 0, 3}};
    const Vec3 weights{0.25f, 0.6f, 0.15f};
    const Transform xform{.s = {2, 3, 4}, .p = {-5, 7, 1}};

    const TriangleVertices moved{
        .a = xform.applyToPoint(tri.a),
        .b = xform.applyToPoint(tri.b),
        .c = xform.applyToPoint(tri.c),
    };

    expectVec3Near(baryOf(moved, pointFromWeights(moved, weights)), weights, 1e-5f);
}

TEST(BarycentricCoordinatesTest, WeightsSumToOneOverInterior)
{
    for (float u = 0.0f; u <= 1.0f; u += 0.1f)
    {
        for (float v = 0.0f; u + v <= 1.0f; v += 0.1f)
        {
            const Vec3 bary = baryOf(unitTri, Vec3{u, v, 0});

            EXPECT_NEAR(bary.x + bary.y + bary.z, 1.0f, 1e-5f) << "at u=" << u << " v=" << v;
            EXPECT_GE(bary.x, -1e-5f);
            EXPECT_GE(bary.y, -1e-5f);
            EXPECT_GE(bary.z, -1e-5f);
        }
    }
}

// Documents the extrapolation behaviour: outside the triangle at least one
// weight goes negative while the three still sum to one.
TEST(BarycentricCoordinatesTest, GoesNegativeOutsideTriangle)
{
    const Vec3 bary = baryOf(unitTri, Vec3{1, 1, 0});

    expectVec3Near(bary, {-1, 1, 1});
    EXPECT_NEAR(bary.x + bary.y + bary.z, 1.0f, 1e-5f);
}

// --- surfaceNormal: vertex-normal interpolation ---

TEST(SurfaceNormalTest, PicksVertexNormalAtEachVertex)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    const TriangleMesh mesh = object.view();

    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{0, 0, 0}), kNormalA);
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{1, 0, 0}), kNormalB);
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{0, 1, 0}), kNormalC);
}

// Equal weights blend to (1,1,1)/sqrt(3); a flat-normal implementation would
// return (0,0,1) here instead.
TEST(SurfaceNormalTest, BlendsVertexNormalsAtCentroid)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    const TriangleMesh mesh = object.view();

    const float k = 1.0f / std::sqrt(3.0f);
    expectVec3Near(
        surfaceNormal(mesh, 0, kRayFromAbove, Vec3{1 / 3.0f, 1 / 3.0f, 0}),
        {k, k, k});
}

TEST(SurfaceNormalTest, IsAlwaysNormalized)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    const TriangleMesh mesh = object.view();

    for (float u = 0.0f; u <= 1.0f; u += 0.125f)
    {
        for (float v = 0.0f; u + v <= 1.0f; v += 0.125f)
        {
            const Vec3 n = surfaceNormal(mesh, 0, kRayFromAbove, Vec3{u, v, 0});
            EXPECT_NEAR(n.length(), 1.0f, 1e-5f) << "at u=" << u << " v=" << v;
        }
    }
}

// The flip is driven by the face normal, so it must apply to the interpolated
// normal as a whole rather than to each vertex normal individually.
TEST(SurfaceNormalTest, FlipsInterpolatedNormalTowardIncomingRay)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    const TriangleMesh mesh = object.view();

    const Vec3 rayFromBelow{0, 0, 1};
    const Vec3 hit{1 / 3.0f, 1 / 3.0f, 0};

    const Vec3 above = surfaceNormal(mesh, 0, kRayFromAbove, hit);
    const Vec3 below = surfaceNormal(mesh, 0, rayFromBelow, hit);

    expectVec3Near(below, above * -1.0f);
}

TEST(SurfaceNormalTest, ConstantVertexNormalsReproduceFaceNormal)
{
    const Vec3 flat{0, 0, 1};
    HostObject object = makeHostObject(smoothTriangleMesh(flat, flat, flat));
    const TriangleMesh mesh = object.view();

    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{0.25f, 0.25f, 0}), flat);
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{0, 1, 0}), flat);
}

// Normals transform by the inverse scale, not the scale: stretching the surface
// along x must tilt a 45-degree normal *away* from x.
TEST(SurfaceNormalTest, AppliesInverseScaleToInterpolatedNormal)
{
    const Vec3 tilted = Vec3{1, 0, 1}.normalized();
    HostObject object = makeHostObject(smoothTriangleMesh(tilted, tilted, tilted));
    object.transform = Transform{.s = {2, 1, 1}, .p = {0, 0, 0}};
    const TriangleMesh mesh = object.view();

    // Centroid of the scaled triangle A(0,0,0) B(2,0,0) C(0,1,0).
    const Vec3 hit{2 / 3.0f, 1 / 3.0f, 0};

    const Vec3 expected = Vec3{0.5f, 0, 1}.normalized();
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, hit), expected);
}

// surfaceNormal takes the hit point in world space; feeding it a model-space
// point under a translated transform would pick the wrong vertex weights.
TEST(SurfaceNormalTest, UsesWorldSpaceHitUnderTranslation)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    object.transform = Transform{.s = {1, 1, 1}, .p = {10, -3, 0}};
    const TriangleMesh mesh = object.view();

    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{10, -3, 0}), kNormalA);
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{11, -3, 0}), kNormalB);
    expectVec3Near(surfaceNormal(mesh, 0, kRayFromAbove, Vec3{10, -2, 0}), kNormalC);
}

// End-to-end over the contract the integrator uses: intersectMesh returns
// {t, triangleID} and the shading normal comes from ray.p + ray.v * t.
TEST(SurfaceNormalTest, MatchesInterpolationAtMeshHitPoint)
{
    HostObject object = makeHostObject(smoothTriangleMesh(kNormalA, kNormalB, kNormalC));
    const TriangleMesh mesh = object.view();

    const Ray ray{.p = {0.5f, 0.25f, 4}, .v = {0, 0, -1}};
    const auto hit = intersectMesh(ray, mesh);
    ASSERT_TRUE(hit.valid());

    const Vec3 point = ray.p + ray.v * hit.t;
    expectVec3Near(point, {0.5f, 0.25f, 0});

    // Weights at (0.5, 0.25) are (0.25, 0.5, 0.25) for the unit triangle.
    const Vec3 expected = (kNormalA * 0.25f + kNormalB * 0.5f + kNormalC * 0.25f).normalized();
    expectVec3Near(surfaceNormal(mesh, hit.triangleID, ray.v, point), expected);
}
