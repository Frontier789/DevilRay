// AI-generated test helpers (Claude), reviewed by hand before committing.

#pragma once

#include "models/Mesh.hpp"
#include "models/BBH.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/DistributionSamplers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <span>
#include <vector>

namespace test
{
    inline void expectVec3Near(const Vec3 &actual, const Vec3 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
    }

    inline void expectVec4Near(const Vec4 &actual, const Vec4 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
        EXPECT_NEAR(actual.w, expected.w, tol);
    }

    // Reproducible RNG exposing the `.rnd()` interface every sampler expects.
    struct DeterministicRng
    {
        std::mt19937 generator;
        std::uniform_real_distribution<float> distribution{0.0f, 1.0f};

        explicit DeterministicRng(uint32_t seed = 0xC0FFEEu) : generator(seed) {}

        float rnd() { return distribution(generator); }
    };

    // RNG that replays preset values, then keeps returning the final one.
    struct ScriptedRng
    {
        std::vector<float> values;
        size_t cursor = 0;

        float rnd()
        {
            const float value = values[std::min(cursor, values.size() - 1)];
            ++cursor;
            return value;
        }
    };

    // Owns a mesh and its acceleration structure, and exposes a TriangleMesh whose
    // pointers reference host memory so the intersection code runs on the CPU.
    struct HostObject
    {
        Mesh mesh;
        BBH bbh;
        AliasTable triangle_sampler;
        Transform transform{};
        int material = 0;

        TriangleMesh view()
        {
            TriangleMesh object{};
            object.points = mesh.points.data();
            object.normals = mesh.normals.data();
            object.triangles = mesh.triangles.data();
            object.triangle_count = static_cast<int>(mesh.triangles.size());
            object.model_to_world = transform;
            object.material = material;
            object.triangle_sampler = triangle_sampler.entries.hostPtr();
            object.surface_area = 0;
            object.base_surface_area = 0;
            object.bbh = BBHGpuView{std::span<const BBHNode>{bbh.nodes.hostPtr(), bbh.nodes.size()}};
            return object;
        }
    };

    inline std::vector<float> triangleAreasOf(const Mesh &mesh)
    {
        std::vector<float> areas;
        for (const auto &tri : mesh.triangles)
        {
            areas.push_back(triangleArea(
                mesh.points[tri.a.pi], mesh.points[tri.b.pi], mesh.points[tri.c.pi]));
        }
        return areas;
    }

    // generateSimpleBBH reorders mesh.triangles, so the area-weighted sampler is
    // built afterwards to stay consistent with the post-sort triangle layout.
    inline HostObject makeHostObject(Mesh mesh, bool withTriangleSampler = false)
    {
        BBH bbh = generateSimpleBBH(mesh);

        AliasTable sampler{};
        if (withTriangleSampler)
            sampler = generateAliasTable(triangleAreasOf(mesh));

        return HostObject{
            .mesh = std::move(mesh),
            .bbh = std::move(bbh),
            .triangle_sampler = std::move(sampler),
        };
    }

    // Single triangle on the XY plane: A(0,0,0) B(1,0,0) C(0,1,0).
    inline Mesh unitTriangleMesh()
    {
        Mesh mesh;
        mesh.name = "unit-triangle";
        mesh.points = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};
        mesh.normals = {Vec3{0, 0, 1}};
        mesh.triangles = {Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}}};
        return mesh;
    }

    // A flat triangle parallel to the XY plane at the given z, large enough to
    // cover the origin column. Useful as an occluder or stacked hit target.
    inline Mesh flatTriangleAtZ(float z)
    {
        Mesh mesh;
        mesh.name = "flat-triangle";
        mesh.points = {Vec3{-1, -1, z}, Vec3{3, -1, z}, Vec3{-1, 3, z}};
        mesh.normals = {Vec3{0, 0, 1}};
        mesh.triangles = {Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}}};
        return mesh;
    }

    // A square in the z=0 plane spanning [-half, half]^2, split into two triangles.
    inline Mesh squareMeshXY(float half)
    {
        Mesh mesh;
        mesh.name = "square";
        mesh.points = {
            Vec3{-half, -half, 0},
            Vec3{half, -half, 0},
            Vec3{half, half, 0},
            Vec3{-half, half, 0},
        };
        mesh.normals = {Vec3{0, 0, 1}};
        mesh.triangles = {
            Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}},
            Triangle{.a = {0, 0}, .b = {2, 0}, .c = {3, 0}},
        };
        return mesh;
    }
}
