#include "tracing/GpuTris.hpp"

#include <iostream>

GpuTris convertMeshToTris(const Mesh &mesh)
{
    std::vector<float> triangleAreas;
    for (const auto &[a,b,c] : mesh.triangles)
    {
        const auto A = mesh.points[a.pi];
        const auto B = mesh.points[b.pi];
        const auto C = mesh.points[c.pi];

        triangleAreas.push_back(triangleArea(A, B, C));
    }

    auto triangleSampler = generateAliasTable(triangleAreas);
    
    std::cout << "Generated alias table for '" << mesh.name << "'" << std::endl;
    for (const auto e : std::span{triangleSampler.entries.hostPtr(), triangleSampler.entries.size()})
    {
        std::cout << "  A=" << e.A << ", B=" << e.B << " p_A=" << e.p_A << " pdf_A=" << e.pdf_A << " pdf_B=" << e.pdf_B << std::endl;
    }

    return GpuTris{
        .points = DeviceVector(mesh.points),
        .normals = DeviceVector(mesh.normals),
        .triangles = DeviceVector(mesh.triangles),
        .triangleSampler = std::move(triangleSampler),
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

TrisCollection viewGpuTris(GpuTris &tris)
{
    TrisCollection obj;

    tris.points.ensureDeviceAllocation();
    obj.points = tris.points.devicePtr();
    
    tris.normals.ensureDeviceAllocation();
    obj.normals = tris.normals.devicePtr();
    
    tris.triangles.ensureDeviceAllocation();
    obj.triangles = tris.triangles.devicePtr();

    tris.triangleSampler.entries.ensureDeviceAllocation();
    obj.tris_sampler = tris.triangleSampler.entries.devicePtr();

    obj.p = Vec3{};
    obj.s = Vec3{1,1,1};
    obj.tris_count = tris.triangles.size();
    obj.surface_area = totalSurfaceArea(tris);

    return obj;
}
