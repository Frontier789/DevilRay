// AI-generated tests (Claude), reviewed by hand before committing.

#include "tracing/GpuTris.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
    // GpuTris is move-only and has no default constructor, so a small unit cube
    // mesh is produced fresh for each test that needs one.
    Mesh tetrahedronMesh()
    {
        Mesh mesh;
        mesh.name = "tetra";
        mesh.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        mesh.triangles = {
            Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}},
            Triangle{.a = {0, 0}, .b = {1, 0}, .c = {3, 0}},
            Triangle{.a = {0, 0}, .b = {2, 0}, .c = {3, 0}},
            Triangle{.a = {1, 0}, .b = {2, 0}, .c = {3, 0}},
        };
        return mesh;
    }

    float meshSurfaceArea(const GpuTris &tris)
    {
        float total = 0;
        const auto *points = tris.points.hostPtr();
        const auto *triangles = tris.triangles.hostPtr();
        for (size_t i = 0; i < tris.triangles.size(); ++i)
            total += triangleArea(points[triangles[i].a.pi], points[triangles[i].b.pi], points[triangles[i].c.pi]);
        return total;
    }
}

TEST(ConvertMeshToTrisTest, CopiesGeometryAndBuildsStructures)
{
    Mesh mesh = tetrahedronMesh();
    const size_t pointCount = mesh.points.size();
    const size_t triangleCount = mesh.triangles.size();

    GpuTris tris = convertMeshToTris(mesh);

    EXPECT_EQ(tris.points.size(), pointCount);
    EXPECT_EQ(tris.triangles.size(), triangleCount);
    EXPECT_GT(tris.bbh.nodes.size(), 0u);
    EXPECT_EQ(tris.triangleSampler.entries.size(), triangleCount);
}

TEST(ConvertMeshToTrisTest, SkipsSamplerWhenDisabled)
{
    Mesh mesh = tetrahedronMesh();
    GpuTris tris = convertMeshToTris(mesh, /*generateTriangleSampler=*/false);
    EXPECT_EQ(tris.triangleSampler.entries.size(), 0u);
}

TEST(CreateQuadMeshTest, ProducesAxisAlignedSquare)
{
    GpuTris quad = createQuadMesh(Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 2.0f);

    ASSERT_EQ(quad.points.size(), 4u);
    ASSERT_EQ(quad.triangles.size(), 2u);

    const auto *points = quad.points.hostPtr();
    for (size_t i = 0; i < quad.points.size(); ++i)
    {
        EXPECT_NEAR(std::abs(points[i].x), 1.0f, 1e-5f);
        EXPECT_NEAR(std::abs(points[i].y), 1.0f, 1e-5f);
        EXPECT_NEAR(points[i].z, 0.0f, 1e-5f);
    }

    // A size-2 quad has total area 4.
    EXPECT_NEAR(meshSurfaceArea(quad), 4.0f, 1e-5f);
}

TEST(ViewGpuTrisTest, WiresUpDevicePointersAndSurfaceArea)
{
    GpuTris quad = createQuadMesh(Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 2.0f);
    const TriangleMesh view = viewGpuTris(quad);

    EXPECT_EQ(view.tris_count, 2);
    EXPECT_NE(view.points, nullptr);
    EXPECT_NE(view.normals, nullptr);
    EXPECT_NE(view.triangles, nullptr);
    EXPECT_NE(view.tris_sampler, nullptr);
    EXPECT_FALSE(view.bbh.isEmpty());

    EXPECT_NEAR(view.base_surface_area, 4.0f, 1e-5f);
    EXPECT_FLOAT_EQ(view.surface_area, view.base_surface_area);
}
