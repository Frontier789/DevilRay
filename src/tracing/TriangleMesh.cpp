#include "tracing/TriangleMesh.hpp"

#include <iostream>

void TriangleMeshView::setPosition(const Vec3 &pos)
{
    model_to_world.p = pos;
}

void TriangleMeshView::setScale(const Vec3 &scale)
{
    model_to_world.s = scale;

    const float vol = scale.x * scale.y * scale.z;
    const float area_factor = std::cbrt(vol * vol);
    this->surface_area = this->base_surface_area * area_factor;
}

TriangleMesh convertMeshToTris(Mesh &mesh, bool generateTriangleSampler)
{
    std::vector<float> triangleAreas;
    for (const auto &[a,b,c] : mesh.triangles)
    {
        const auto A = mesh.points[a.pi];
        const auto B = mesh.points[b.pi];
        const auto C = mesh.points[c.pi];

        triangleAreas.push_back(triangleArea(A, B, C));
    }

    AliasTable triangle_sampler;

    if (generateTriangleSampler) {
        triangle_sampler = generateAliasTable(triangleAreas);

        std::cout << "Generated alias table for '" << mesh.name << "'" << std::endl;
        // for (const auto e : std::span{triangle_sampler.entries.hostPtr(), triangle_sampler.entries.size()})
        // {
        //     std::cout << "  A=" << e.A << ", B=" << e.B << " p_A=" << e.p_A << " pdf_A=" << e.pdf_A << " pdf_B=" << e.pdf_B << std::endl;
        // }
    }

    BBH bbh = generateSimpleBBH(mesh);

    return TriangleMesh{
        .points = DeviceVector(mesh.points),
        .normals = DeviceVector(mesh.normals),
        .triangles = DeviceVector(mesh.triangles),
        .triangle_sampler = std::move(triangle_sampler),
        .bbh = std::move(bbh),
    };
}

namespace
{
    float totalSurfaceArea(const TriangleMesh &mesh)
    {
        double total = 0;

        for (int i=0;i<mesh.triangles.size();++i)
        {
            const auto &indices = mesh.triangles.hostPtr()[i];

            const auto A = mesh.points.hostPtr()[indices.a.pi];
            const auto B = mesh.points.hostPtr()[indices.b.pi];
            const auto C = mesh.points.hostPtr()[indices.c.pi];

            total += triangleArea(A, B, C);
        }

        return static_cast<float>(total);
    }
}

TriangleMeshView TriangleMesh::view()
{
    points.ensureDeviceAllocation();
    normals.ensureDeviceAllocation();
    triangles.ensureDeviceAllocation();

    const float surface_area = totalSurfaceArea(*this);

    return TriangleMeshView{
        .points = points.devicePtr(),
        .normals = normals.devicePtr(),
        .triangles = triangles.devicePtr(),
        .triangle_count = static_cast<int>(triangles.size()),
        .model_to_world = Transform{},
        .material = 0,
        .triangle_sampler = triangle_sampler.view(),
        .surface_area = surface_area,
        .base_surface_area = surface_area,
        .bbh = bbh.view(),
    };
}

TriangleMesh createQuadMesh(Vec3 center, Vec3 normal, Vec3 right, float size)
{
    const auto up = right.cross(normal);
    const auto half = size / 2.0f;

    Mesh mesh;
    mesh.name = "quad";
    mesh.points = {
        center + right * (-half) + up * (-half),
        center + right * ( half) + up * (-half),
        center + right * ( half) + up * ( half),
        center + right * (-half) + up * ( half),
    };
    mesh.normals = { normal };
    mesh.triangles = {
        Triangle{.a = {0, 0}, .b = {2, 0}, .c = {1, 0}},
        Triangle{.a = {0, 0}, .b = {3, 0}, .c = {2, 0}},
    };

    return convertMeshToTris(mesh);
}
