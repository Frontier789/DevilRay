#pragma once

#include "Utils.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Scene.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/CameraRay.hpp"
#include "tracing/LightSampling.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "tracing/Wavefront.hpp"
#include "tracing/Path.hpp"
#include "tracing/RenderBuffers.hpp"
#include "tracing/ShadingUtils.hpp"
#include "models/MeshUtils.hpp"
#include "device/CudaRng.hpp"
#include "device/Random.hpp"

#include "Buffers.hpp"
#include "DebugOptions.hpp"

#include <optional>
#include <span>

constexpr int VACUUM_MAT = -1;
constexpr float VACUUM_IOR = 1.0f;

HD Vec4 checkerPattern(const Vec2f &uv, int checker_count, Vec4 dark, Vec4 bright);

__global__ void initPaths(
    PathVertexDataView vertex, int path_count
)
{
    const int idx = KERNEL_IDX(path_count);

    vertex.throughput[idx] = Vec4{1, 1, 1, 0};
    vertex.prevSpecular[idx] = true;
    vertex.bsdfPdfPrev[idx] = 0;
    vertex.alive[idx] = true;
}

__global__ void initCameraRays(
    WavefrontDataView wavefront,
    PixelSampling pixel_sampling,
    Camera camera,
    curandState *rand_states
)
{
    int idx = KERNEL_IDX(camera.resolution.area());

    const auto width = camera.resolution.width;
    const auto pixel_pos = Vec2f{static_cast<float>(idx % width), static_cast<float>(idx / width)};

    auto rng = CudaRng{.state = rand_states + idx};

    const Ray ray = cameraRay(camera, pixel_pos, pixel_sampling, 0, rng);

    wavefront.rays[idx] = ray;
    wavefront.current_mat[idx] = VACUUM_MAT;
    wavefront.sort_index[idx] = idx;
}

__global__ void extendPaths(
    WavefrontDataView wavefront,
    PathVertexDataView vertex,
    std::span<const TriangleMeshView> objects,
    int *alive_count,
    uint32_t *casts,
    int depth
)
{
    const int thread_idx = KERNEL_IDX(*alive_count);
    const int idx = wavefront.sort_index[thread_idx];
    if (!vertex.alive[idx]) return;
    
    casts[idx] += 1;

    const auto ray = wavefront.rays[idx];
    const auto intersection = intersectScene(ray, objects);

    if (intersection.valid()) {
        vertex.t[idx] = intersection.t;
        vertex.ids[idx] = TriangleIdentifier{
            .meshID = intersection.meshID,
            .triangleID = intersection.triangleID,
        };
    } else {
        vertex.alive[idx] = false;
    }
}

inline HD Vec3 reflect(const Vec3 &i, const Vec3 &n, const float dotp)
{
    return i - n * 2*dotp;
}

inline HD float schlick(const float dotp, const float n1, const float n2)
{
    const float r = (n1 - n2) / (n1 + n2);
    const float R0 = r*r;

    const float cosT1 = 1 - dotp;
    const float cosT1_2 = cosT1*cosT1;
    const float cosT1_4 = cosT1_2*cosT1_2;
    const float cosT1_5 = cosT1_4*cosT1;

    return R0 + (1.0 - R0) * cosT1_5;
}

template<typename Rng>
inline HD Vec3 reflectOrRefractRay(
    const Vec3 &dir,
    const TransparentMaterial &hit_material,
    int &inside_mat, int hit_mat,
    bool enter,
    const Vec3 &normal,
    Rng &rng
){
    float n1;
    float n2;

    if (!enter) {
        n1 = hit_material.inside_medium.ior;
        n2 = VACUUM_IOR;
    } else {
        n1 = VACUUM_IOR;
        n2 = hit_material.inside_medium.ior;
    }

    // Reflect or refract
    const auto eta = n1/n2;
    const auto d = dir.dot(normal);
    const auto k = 1 - eta*eta * (1 - d*d);

    Vec3 v;

    // Total internal reflection
    if (n1 > n2 && k < 0)
    {
        v = reflect(dir, normal, d);
    }
    else
    {
        const auto F = schlick(-d, n1, n2);

        // Fresnel reflection
        if (rng.rnd() < F)
        {
            v = reflect(dir, normal, d);
        }
        else
        {
            if (enter) inside_mat = hit_mat;
            else inside_mat = VACUUM_MAT;

            const float dotp = -d;
            v = normal * -1 * (std::sqrt(k) - dotp * eta) + dir * eta;
        }
    }

    return v;
}

__global__ void sampleBsdfDirection(
    PathVertexDataView vertex,
    WavefrontDataView wavefront,
    curandState *rand_states,
    std::span<const TriangleMeshView> objects,
    std::span<const Material> materials,
    int *alive_count
)
{
    const int thread_idx = KERNEL_IDX(*alive_count);
    const int idx = wavefront.sort_index[thread_idx];
    if (!vertex.alive[idx]) return;

    const auto ids = vertex.ids[idx];
    const auto &object = objects[ids.meshID];
    auto rng = CudaRng{rand_states + idx};

    const auto ray = wavefront.rays[idx];
    const auto pos = ray.p + ray.v * vertex.t[idx];
    const auto n = surfaceNormal(object, ids.triangleID, ray.v, pos);

    if (const auto material = std::get_if<DiffuseMaterial>(&materials[object.material]))
    {
        const auto w_out = cosineWeightedHemisphereSample(n, rng);
        const float bsdf_pdf = cosineWeightedHemisphereDirPdf(w_out, n);
    
        wavefront.rays[idx] = Ray{
            .p = pos + n * 1e-5f,
            .v = w_out,
        };
        wavefront.current_mat[idx] = VACUUM_MAT;

        vertex.bsdfPdfPrev[idx] = bsdf_pdf;
        vertex.prevSpecular[idx] = false;
        vertex.throughput[idx] *= material->diffuse_reflectance;
    }
    else if (const auto material = std::get_if<TransparentMaterial>(&materials[object.material]))
    {
        int &current_mat = wavefront.current_mat[idx];
        const bool enter = current_mat == VACUUM_MAT; // Assume no nesting of objects

        const auto newDir = reflectOrRefractRay(ray.v, *material, current_mat, object.material, enter, n, rng);
        const auto normalTowardsV = newDir.dot(n) > 0 ? n : -n;

        wavefront.rays[idx] = Ray{
            .p = pos + normalTowardsV * 1e-5f,
            .v = newDir,
        };

        vertex.bsdfPdfPrev[idx] = 1;
        vertex.prevSpecular[idx] = true;
    }
}

// // // // // // SHADE // // // // // // 

inline HD void shadeDiffuseMaterial(
    PathVertexDataView vertex,
    WavefrontDataView wavefront,
    const TriangleIdentifier &ids,
    const Material &material,
    int idx,
    const TriangleMeshView &object,
    CudaRng &rng,
    std::span<const TriangleMeshView> objects,
    std::span<const Material> materials,
    AliasTableView light_table,
    ObjectsInfo info,
    RenderBuffersView output
)
{
    const auto &ray = wavefront.rays[idx];
    const auto prevPos = ray.p;
    const auto pos = ray.p + ray.v * vertex.t[idx];
    const auto n = surfaceNormal(object, ids.triangleID, ray.v, pos);

    const auto diffuse_material = std::get_if<DiffuseMaterial>(&material);

    const auto nee_pdf = computeNeePdf(prevPos, pos, n, material, info);
    Vec4 color = misWeightedEmission(
        diffuse_material->emission, vertex.throughput[idx],
        vertex.prevSpecular[idx], MisPdfs{
            .bsdf_pdf = vertex.bsdfPdfPrev[idx],
            .nee_pdf = nee_pdf,
        }
    );

    LightSample light_sample = samplePointOnLights(objects, light_table, rng);
    const auto *light_material = std::get_if<DiffuseMaterial>(&materials[light_sample.mat]);

    const auto nee_pdf_bsdf = cosineWeightedHemispherePdf(pos, light_sample.p, n);
    const auto nee_pdf_nee = light_sample.pdf * areaToSolidAngle(pos, light_sample.p, light_sample.n);

    const auto Ld_nee = evaluateDirectLighting(
        pos, n, diffuse_material->diffuse_reflectance,
        light_sample, light_material->emission, objects);

    if (nee_pdf_nee + nee_pdf_bsdf > 0)
        color = color + Ld_nee * vertex.throughput[idx] * powerHeuristic(nee_pdf_nee, nee_pdf_bsdf);
    
    output.colors[idx] += color;
}

__global__ void shade(
    PathVertexDataView vertex,
    WavefrontDataView wavefront,
    curandState *rand_states,
    std::span<const TriangleMeshView> objects,
    std::span<const Material> materials,
    AliasTableView light_table,
    int *alive_count,
    ObjectsInfo info,
    RenderBuffersView output
)
{
    const int thread_idx = KERNEL_IDX(*alive_count);
    const int idx = wavefront.sort_index[thread_idx];
    if (!vertex.alive[idx]) return;
    
    const auto ids = vertex.ids[idx];
    const auto &object = objects[ids.meshID];
    const auto &material = materials[object.material];
    auto rng = CudaRng{rand_states + idx};

    if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material))
    {
        shadeDiffuseMaterial(
            vertex, wavefront,
            ids, material,
            idx, object, rng,
            objects, materials,
            light_table, info, output
        );
    }
    else if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {
        // PASS
    }
}

__global__ void debugShade(
    PathVertexDataView vertex,
    WavefrontDataView wavefront,
    std::span<const TriangleMeshView> objects,
    std::span<const Material> materials,
    DebugOptions debug,
    int *alive_count,
    RenderBuffersView output
)
{
    int idx = KERNEL_IDX(*alive_count);

    Vec4 color{0, 0, 0, 0};

    if (vertex.alive[idx])
    {
        const auto ids = vertex.ids[idx];
        const auto &object = objects[ids.meshID];
        const auto &material = materials[object.material];

        const auto world_ray = wavefront.rays[wavefront.sort_index[idx]];
        const auto model_ray = object.model_to_world.applyInverse(world_ray);

        const auto &tri = object.triangles[ids.triangleID];
        const auto triangle = TriangleVertices{
            .a = object.points[tri.a.pi],
            .b = object.points[tri.b.pi],
            .c = object.points[tri.c.pi],
        };
        const auto hit = world_ray.p + world_ray.v * vertex.t[idx];

        color = getDebugColor(material);

        switch (debug)
        {
            case DebugOptions::BariCoords:
            {
                const auto model_hit = model_ray.p + model_ray.v * vertex.t[idx];
                const auto normal = triangleFaceNormal(triangle);

                const auto bari = barycentricCoordinates(triangle, normal, model_hit);
                color = Vec4{bari.x, bari.y, bari.z, 0};
                break;
            }
            case DebugOptions::WindingOrder:
            {
                constexpr auto clockWiseColor = Vec4{0.53, 0.82, 1.0, 0.0};
                constexpr auto counterClockWiseColor = Vec4{1.0, 0.73, 0.47, 0.0};

                const auto model_normal = triangleFaceNormal(triangle);
                const bool ccw = model_normal.dot(model_ray.v) > 0;

                color = ccw ? counterClockWiseColor : clockWiseColor;
                break;
            }
            case DebugOptions::Normal:
            {
                const auto n = surfaceNormal(object, ids.triangleID, world_ray.v, hit);

                color = Vec4::from((n + Vec3{1,1,1})/2, 0);
                break;
            }
            case DebugOptions::UVChecker:
                color = checkerPattern(Vec2f{0, 0}, 7,
                    Vec4{0.5, 0.5, 0.5, 0}, Vec4{0.8, 0.8, 0.8, 0}) * getDebugColor(material);
                break;
            case DebugOptions::Off:
                break;
        }
    }

    output.colors[idx] += color;
}
