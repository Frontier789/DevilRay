// AI-generated tests (Claude), reviewed by hand before committing.

#include "Utils.hpp"
#include "VectorAssertions.hpp"
#include "models/Mesh.hpp"
#include "tracing/TriangleMesh.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace
{
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

    float meshSurfaceArea(const TriangleMesh &tris)
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

    TriangleMesh tris = convertMeshToTris(mesh);

    EXPECT_EQ(tris.points.size(), pointCount);
    EXPECT_EQ(tris.triangles.size(), triangleCount);
    EXPECT_GT(tris.bbh.nodes.size(), 0u);
    EXPECT_EQ(tris.triangle_sampler.entries.size(), triangleCount);
}

TEST(ConvertMeshToTrisTest, SkipsSamplerWhenDisabled)
{
    Mesh mesh = tetrahedronMesh();
    TriangleMesh tris = convertMeshToTris(mesh, /*generateTriangleSampler=*/false);
    EXPECT_EQ(tris.triangle_sampler.entries.size(), 0u);
}

TEST(TriangleMeshViewTest, UniformScaleScalesSurfaceAreaQuadratically)
{
    constexpr float base_area = 4.0f;
    constexpr float uniform_scale = 3.0f;
    TriangleMeshView view{};
    view.surface_area = base_area;
    view.base_surface_area = base_area;

    view.setScale(Vec3{uniform_scale, uniform_scale, uniform_scale});

    EXPECT_NEAR(view.surface_area, base_area * uniform_scale * uniform_scale, 1e-4f);
    EXPECT_FLOAT_EQ(view.base_surface_area, base_area);
    test::expectVec3Near(view.model_to_world.s, {uniform_scale, uniform_scale, uniform_scale});
}

TEST(TriangleMeshViewTest, PositionAndScaleDoNotDisturbEachOther)
{
    TriangleMeshView positioned_then_scaled{};
    positioned_then_scaled.setPosition(Vec3{1, 2, 3});
    positioned_then_scaled.setScale(Vec3{2, 2, 2});

    TriangleMeshView scaled_then_positioned{};
    scaled_then_positioned.setScale(Vec3{2, 2, 2});
    scaled_then_positioned.setPosition(Vec3{1, 2, 3});

    for (const auto &view : {positioned_then_scaled, scaled_then_positioned})
    {
        test::expectVec3Near(view.model_to_world.p, {1, 2, 3});
        test::expectVec3Near(view.model_to_world.s, {2, 2, 2});
    }
}

TEST(CreateQuadMeshTest, ProducesAxisAlignedSquare)
{
    TriangleMesh quad = createQuadMesh(Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 2.0f);

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
