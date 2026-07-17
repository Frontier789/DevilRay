#include "tracing/GpuTris.hpp"

#include <iostream>

GpuTris convertMeshToTris(Mesh &mesh, bool generateTriangleSampler)
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

    return GpuTris{
        .points = DeviceVector(mesh.points),
        .normals = DeviceVector(mesh.normals),
        .triangles = DeviceVector(mesh.triangles),
        .triangle_sampler = std::move(triangle_sampler),
        .bbh = std::move(bbh),
    };
}

namespace
{
    float totalSurfaceArea(GpuTris &tris)
    {
        double total = 0;

        for (int i=0;i<tris.triangles.size();++i)
        {
            const auto &indices = tris.triangles.hostPtr()[i];

            const auto A = tris.points.hostPtr()[indices.a.pi];
            const auto B = tris.points.hostPtr()[indices.b.pi];
            const auto C = tris.points.hostPtr()[indices.c.pi];

            total += triangleArea(A, B, C);
        }

        return static_cast<float>(total);
    }
}

TriangleMesh viewGpuTris(GpuTris &tris)
{
    TriangleMesh obj;

        std::cout << "L" << __LINE__ << std::endl;
    tris.points.ensureDeviceAllocation();
        std::cout << "L" << __LINE__ << std::endl;
    obj.points = tris.points.devicePtr();
        std::cout << "L" << __LINE__ << std::endl;

    tris.normals.ensureDeviceAllocation();
        std::cout << "L" << __LINE__ << std::endl;
    obj.normals = tris.normals.devicePtr();
        std::cout << "L" << __LINE__ << std::endl;

    tris.triangles.ensureDeviceAllocation();
        std::cout << "L" << __LINE__ << std::endl;
    obj.triangles = tris.triangles.devicePtr();
        std::cout << "L" << __LINE__ << std::endl;

    tris.triangle_sampler.entries.ensureDeviceAllocation();
        std::cout << "L" << __LINE__ << std::endl;
    obj.triangle_sampler = tris.triangle_sampler.entries.devicePtr();
        std::cout << "L" << __LINE__ << std::endl;

    tris.bbh.nodes.ensureDeviceAllocation();
        std::cout << "L" << __LINE__ << std::endl;
    obj.bbh = createBBHGpuView(tris.bbh);
        std::cout << "L" << __LINE__ << std::endl;

    obj.triangle_count = tris.triangles.size();
        std::cout << "L" << __LINE__ << std::endl;
    obj.base_surface_area = totalSurfaceArea(tris);
        std::cout << "L" << __LINE__ << std::endl;
    obj.surface_area = obj.base_surface_area;
        std::cout << "L" << __LINE__ << std::endl;

    return obj;
}

GpuTris createQuadMesh(Vec3 center, Vec3 normal, Vec3 right, float size)
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
