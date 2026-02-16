#include "tracing/GpuTris.hpp"

GpuTris convertMeshToTris(const Mesh &mesh)
{
    return GpuTris{
        .points = DeviceVector(mesh.points),
        .normals = DeviceVector(mesh.normals),
        .triangles = DeviceVector(mesh.triangles),
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

            const auto a = tris.points.hostPtr()[indices.a.pi];
            const auto b = tris.points.hostPtr()[indices.b.pi];
            const auto c = tris.points.hostPtr()[indices.c.pi];

            const auto u = a - b;
            const auto v = a - c;

            const auto area = std::abs(u.cross(v).length()) / 2;

            total += area;
        }

        return static_cast<float>(total);
    }
}

TrisCollection viewGpuTris(GpuTris &tris)
{
    TrisCollection obj;

    tris.points.ensureDeviceAllocation();
    tris.points.updateDeviceData();
    obj.points = tris.points.devicePtr();
    
    tris.normals.ensureDeviceAllocation();
    tris.normals.updateDeviceData();
    obj.normals = tris.normals.devicePtr();
    
    tris.triangles.ensureDeviceAllocation();
    tris.triangles.updateDeviceData();
    obj.triangles = tris.triangles.devicePtr();

    obj.p = Vec3{};
    obj.tris_count = tris.triangles.size();
    obj.surface_area = totalSurfaceArea(tris);

    return obj;
}
