#include "benchmark.hpp"

#include "tracing/IntersectionTestsImpl.hpp"
#include "tracing/LightSampling.hpp"
#include "device/DevUtils.hpp"
#include <iostream>

struct CudaRandom
{
    curandState *state;

    __device__ float rnd()
    {
        return curand_uniform(state);
    }
};

HD Ray generateRay(Vec3 center, float radius, CudaRandom &rng)
{
    const auto dir0 = uniformSphereSample(rng);
    const auto dir1 = uniformSphereSample(rng);

    const auto r = radius * 1.1f;
    const auto p0 = dir0 * r + center;
    const auto p1 = (dir1 - dir0) * r + center;

    const auto ray = Ray{.p = p0, .v = (p1 - p0).normalized()};

    return ray;
}

__global__ void runRaycasts(
    curandState *rand_states, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh tris,
    Vec3 center, float radius
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRandom{rand_states + idx};

    const auto ray = generateRay(center, radius, rng);

    const auto intersection = getIntersectionBenchmark(ray, tris, stats[idx]);

    if (intersection.has_value())
    {
        stats[idx].registerTriangleHit();
    }
}

void benchmarkRayCast(
    CudaRandomStates &rand_states, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
)
{
    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(rand_states.devicePtr(), stats, ray_count, tris, center, radius);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct IntersectionTime
{
    float t;

    HD static constexpr IntersectionTime missed() { return IntersectionTime{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};

struct IntersectionTriangle
{
    float t;
    int triangleID;

    HD static constexpr IntersectionTriangle missed() { return IntersectionTriangle{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};

struct IntersectionMesh
{
    float t;
    int triangleID;
    int meshID;

    HD static constexpr IntersectionMesh missed() { return IntersectionMesh{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};


HD IntersectionTime testTriangleIntersection_dev(const Ray &ray, const TriangleVertices &triangle)
{
    if (ray.p.anyNan() || ray.v.anyNan()) return IntersectionTime::missed();

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);

    auto dp = (A - ray.p).dot(n_f);
    const auto d = ray.v.dot(n_f);

    if (d < 1e-7f) return IntersectionTime::missed();

    const float t = dp / d;

    return IntersectionTime{
        .t = t
    };
}





template<Benchmark B>
HD void getIntersectionTris_dev(
    const Ray &ray, const TriangleMesh &tris,
    int tris_begin, int tris_end,
    IntersectionTriangle &best,
    B &benchmark
){
    for (int i=tris_begin;i<tris_end;++i)
    {
        const auto &indices = tris.triangles[i];

        const auto triangle = TriangleVertices{
            .a = tris.points[indices.a.pi],
            .b = tris.points[indices.b.pi],
            .c = tris.points[indices.c.pi],
        };

        benchmark.registerTriangleTest();
        const auto intersection = testTriangleIntersection_dev(ray, triangle);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = IntersectionTriangle{
            .t = intersection.t,
            .triangleID = i,
        };
    }
}








template<Benchmark B>
HD IntersectionTriangle getIntersectionImpl_dev(
    const Ray &ray_in_world,
    const TriangleMesh &tris,
    B &benchmark
){
    auto best = IntersectionTriangle::missed();
    
    const auto &bbh = tris.bbh;
    const auto ray = tris.model_to_world.applyInverse(ray_in_world);

    // getIntersectionTris(ray, tris, 0, tris.triangle_count, best, benchmark);
    // return best;

    uint32_t bit_trail = 0;
    uint32_t depth = 0;
    int current_index = 0;

    while (current_index >= 0 && current_index < bbh.nodes.size())
    {
        const auto &node = bbh.nodes[current_index];
        benchmark.registerBBoxTest();
        const auto bboxHit = testBoxIntersection(node.box, ray);

        if (bboxHit.has_value())
        {
            const auto t = *bboxHit;

            if (!best.valid() || t < best.t)
            {
                if (node.isLeaf())
                {
                    getIntersectionTris_dev(ray, tris, node.tris_begin, node.tris_end, best, benchmark);
                }
                else
                {
                    const auto near_child = nearChild(ray, depth, current_index, node);
                    
                    bit_trail &= ~(1u << depth);

                    current_index = near_child;
                    depth++;

                    continue;
                }
            }
        }

        while (true)
        {
            if (current_index == 0) {
                current_index = -1;
                break;
            }

            const auto parent_index = bbh.nodes[current_index].parent_index;
            const auto &parent = bbh.nodes[parent_index];

            depth--;
            const auto far_child_visited = bit_trail & (1u << depth);

            if (!far_child_visited)
            {
                const auto far_child = farChild(ray, depth, parent_index, parent);

                bit_trail |= (1u << depth);

                current_index = far_child;
                depth++;
                break;
            }
            else
            {
                current_index = parent_index;
            }
        }
    }

    return best;
}


HD auto getIntersection_dev(const Ray &ray, const TriangleMesh &tris)
{
    benchmark::Skip skip_benchmarks;
    return getIntersectionImpl_dev(ray, tris, skip_benchmarks);
}



HD IntersectionMesh cast_dev(const Ray &ray, const std::span<const TriangleMesh> objects, const ObjectsInfo &info)
{
    auto best = IntersectionMesh::missed();

    for (int i=0; i<objects.size(); ++i)
    {
        auto intersection = getIntersection_dev(ray, objects[i]);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = IntersectionMesh {
            .t = intersection.t,
            .triangleID = intersection.triangleID,
            .meshID = i,
        };
    }

    return best;
}





struct TriangleIdentifier
{
    int meshID;
    int triangleID;

    constexpr bool valid() const
    {
        return meshID >= 0;
    }

    static constexpr TriangleIdentifier invalid()
    {
        return TriangleIdentifier{
            .meshID = -1,
            .triangleID = -1,
        };
    }
};

struct WavefrontData
{
    DeviceBuffer<Ray> rays;
};

struct WavefrontDataDevice
{
    std::span<Ray> rays;
};

struct PathVertexData
{
    DeviceBuffer<float> t;
    DeviceBuffer<float> bsdfPdfPrev;
    DeviceBuffer<Vec4> throughput;
    DeviceBuffer<int> prevSpecular;
    DeviceBuffer<TriangleIdentifier> ids;
};

struct PathVertexDataDevice
{
    std::span<float> t;
    std::span<float> bsdfPdfPrev;
    std::span<Vec4> throughput;
    std::span<int> prevSpecular;
    std::span<TriangleIdentifier> ids;
};

struct OutputData
{
    DeviceBuffer<Vec4> colors;
};

struct OutputDataDevice
{
    std::span<Vec4> colors;
};

__global__ void initWavefront(
    WavefrontDataDevice wavefront,
    PathVertexDataDevice vertex,
    curandState *rand_states,
    Vec3 center, float radius, int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRandom{rand_states + idx};

    const auto ray = generateRay(center, radius, rng);

    wavefront.rays[idx] = ray;

    vertex.throughput[idx] = Vec4{1, 1, 1, 0};
    vertex.prevSpecular[idx] = 1;
    vertex.bsdfPdfPrev[idx] = 0;
}

__global__ void findNextVertex(
    WavefrontDataDevice wavefront,
    PathVertexDataDevice nextVertex,
    std::span<const TriangleMesh> objects,
    int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ray = wavefront.rays[idx];
    const auto intersection = cast_dev(ray, objects, ObjectsInfo{.total_radiant_power = 1});

    if (intersection.valid()) {
        nextVertex.t[idx] = intersection.t;
        nextVertex.ids[idx] = TriangleIdentifier{
            .meshID = intersection.meshID,
            .triangleID = intersection.triangleID,
        };
    } else {
        nextVertex.ids[idx] = TriangleIdentifier::invalid();
    }
}

inline HD Vec3 surfaceNormal(const TriangleMesh &object, int triangleID, const Vec3 &ray_dir)
{
    const auto &triangle = object.triangles[triangleID];

    const auto triangle_vertices = TriangleVertices{
        .a = object.points[triangle.a.pi],
        .b = object.points[triangle.b.pi],
        .c = object.points[triangle.c.pi],
    };

    auto n = triangleNormal(triangle_vertices);
    const auto &inv_s = object.model_to_world.s.inv();
    n = (n * inv_s).normalized();

    if (n.dot(ray_dir) > 0) n = n * -1;

    return n;
}

__global__ void sampleBrdfDirection(
    PathVertexDataDevice vertex,
    WavefrontDataDevice wavefront,
    curandState *randStates,
    std::span<const TriangleMesh> objects,
    const std::span<const Material> materials,
    int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ids = vertex.ids[idx];
    if (!ids.valid()) return;

    const auto &object = objects[ids.meshID];
    auto rng = CudaRandom{randStates + idx};

    const auto ray = wavefront.rays[idx];
    const auto pos = ray.p + ray.v * vertex.t[idx];
    const auto n = surfaceNormal(object, ids.triangleID, ray.v);

    const auto w_out = cosineWeightedHemisphereSample(n, rng);
    const float bsdf_pdf = cosineWeightedHemisphereDirPdf(w_out, n);

    wavefront.rays[idx] = Ray{
        .p = pos + n * 1e-5f,
        .v = w_out,
    };

    const auto &material = *std::get_if<DiffuseMaterial>(&materials[object.material]);

    vertex.bsdfPdfPrev[idx] = bsdf_pdf;
    vertex.prevSpecular[idx] = 0;
    vertex.throughput[idx] *= material.diffuse_reflectance;
}

struct MisPdfs {
    float bsdf_pdf;
    float nee_pdf;
};

inline HD Vec4 misWeightedEmission(
    const Vec4 &emission,
    const Vec4 &throughput,
    bool prev_bounce_specular,
    MisPdfs path_pdfs)
{
    if (prev_bounce_specular) {
        return emission * throughput;
    } else if (path_pdfs.bsdf_pdf + path_pdfs.nee_pdf > 0) {
        return emission * throughput * powerHeuristic(path_pdfs.bsdf_pdf, path_pdfs.nee_pdf);
    }
    return Vec4{0,0,0,0};
}

struct LightSample
{
    Vec3 p;
    Vec3 n;
    int mat;
    float pdf;
};

#pragma nv_exec_check_disable
template<typename Rng>
HD LightSample samplePointOnLights(
    std::span<const TriangleMesh> objects,
    std::span<const AliasEntry> light_table,
    Rng &rng)
{
    const auto [index, object_pdf] = sample(light_table, rng);
    const auto &object = objects[index];
    const auto mat = object.material;

    // printf("Rolled %d\n", index);

    const auto tris_table = std::span{object.triangle_sampler, static_cast<size_t>(object.triangle_count)};
    const auto [i, triangle_pdf] = sample(tris_table, rng);

    const auto triangle = object.triangles[i];
    const auto A = object.points[triangle.a.pi];
    const auto B = object.points[triangle.b.pi];
    const auto C = object.points[triangle.c.pi];

    const auto p = uniformTriangleSample(A, B, C, rng);

    const auto Aw = A * object.model_to_world.s;
    const auto Bw = B * object.model_to_world.s;
    const auto Cw = C * object.model_to_world.s;

    const auto perp = (Aw - Bw).cross(Aw - Cw);
    const auto perp_length = perp.length();
    const auto n = perp / perp_length;
    const auto world_triangle_area = perp_length / 2.0f;

    return LightSample{
        .p = object.model_to_world.applyToPoint(p),
        .n = n,
        .mat = mat,
        .pdf = object_pdf * triangle_pdf / world_triangle_area,
    };
}

inline HD float computeNeePdf(
    const Vec3 &prevPos,
    const Vec3 &pos,
    const Vec3 &normal,
    const Material &material,
    const ObjectsInfo &info)
{
    const auto next_radiant_exitance = luminance(radiantExitance(material));
    const auto pdf_nee_area = next_radiant_exitance / info.total_radiant_power;
    const float nee_pdf = pdf_nee_area * areaToSolidAngle(prevPos, pos, normal);

    return nee_pdf;
}

inline HD bool visible(Vec3 p0, Vec3 p1, std::span<const TriangleMesh> objects, const ObjectsInfo &info)
{
    const auto distance = (p1 - p0).length();
    const auto v = (p1 - p0) / distance;

    Ray ray{.p = p0 + v * 1e-5, .v = v};

    const auto hit = cast_dev(ray, objects, info);
    if (!hit.valid()) return true;

    return hit.t > distance - 1e-5 * 2;
}


inline HD Vec4 evaluateDirectLighting(
    const Vec3 &surface_pos,
    const Vec3 &surface_normal,
    const Vec4 &diffuse_reflectance,
    const LightSample &light_sample,
    const Vec4 &light_emission,
    std::span<const TriangleMesh> objects,
    const ObjectsInfo &info)
{
    if (!visible(surface_pos, light_sample.p, objects, info))
        return Vec4{0,0,0,0};

    const auto distance = (light_sample.p - surface_pos).length();
    const auto w_light = (light_sample.p - surface_pos) / distance;

    const auto brdf = diffuse_reflectance / pi;

    const auto cos_at_surface = w_light.dot(surface_normal);
    const auto cos_at_light   = w_light.dot(light_sample.n);

    if (cos_at_surface <= 0)
        return Vec4{0,0,0,0};

    const auto geometric_term = cos_at_surface * std::abs(cos_at_light) / (distance * distance);
    return light_emission * brdf * geometric_term / light_sample.pdf;
}

__global__ void shade(
    PathVertexDataDevice vertex,
    WavefrontDataDevice wavefront,
    std::span<const Material> materials,
    std::span<const TriangleMesh> objects,
    curandState *rand_states,
    std::span<const AliasEntry> light_table,
    int ray_count,
    ObjectsInfo info,
    OutputDataDevice output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ids = vertex.ids[idx];
    if (!ids.valid()) return;

    const auto &object = objects[ids.meshID];
    const auto &material = materials[object.material];
    auto rng = CudaRandom{rand_states + idx};

    Vec4 color{0,0,0,0};

    if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material))
    {
        const auto &ray = wavefront.rays[idx];
        const auto prevPos = ray.p;
        const auto pos = ray.p + ray.v * vertex.t[idx];
        const auto n = surfaceNormal(object, ids.triangleID, ray.v);

        const auto nee_pdf = computeNeePdf(prevPos, pos, n, material, info);
        color = color + misWeightedEmission(
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
            light_sample, light_material->emission, objects, info);

        if (nee_pdf_nee + nee_pdf_bsdf > 0)
            color = color + Ld_nee * vertex.throughput[idx] * powerHeuristic(nee_pdf_nee, nee_pdf_bsdf);
    }
    else if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {
        // PASS
    }

    output.colors[idx] = output.colors[idx] + color;
}

void testWavefront(Mesh &mesh)
{
    constexpr int RAY_COUNT = 1'000'000;

    auto tris = GpuTris{convertMeshToTris(mesh, true)};
    auto gpuTris = viewGpuTris(tris);
    const auto bounds = calculateMeshBounds(mesh);

    gpuTris.material = 0;

    DiffuseMaterial light_material{};
    light_material.emission = Vec4{1, 1, 1, 0};
    light_material.diffuse_reflectance = Vec4{0.7, 0.7, 0.7, 0};
    DeviceVector<Material> materials{std::vector<Material>{Material{light_material}}};
    materials.ensureDeviceAllocation();

    const auto total_radiant_power =
        gpuTris.surface_area * luminance(radiantExitance(materials.hostPtr()[0]));
    const auto info = ObjectsInfo{.total_radiant_power = total_radiant_power};

    auto light_sampler = generateAliasTable(std::vector<float>{1.0f});
    light_sampler.entries.ensureDeviceAllocation();
    const std::span<const AliasEntry> light_table = light_sampler.entries.deviceSpan();

    // Upload the single object into device memory
    auto objects = DeviceBuffer<TriangleMesh>::allocate(1);
    objects.deviceData.copyFromHost(&gpuTris, sizeof(TriangleMesh));
    const std::span<const TriangleMesh> objectsView = objects.deviceSpan();

    CudaRandomStates rng(Size2i{.width = RAY_COUNT, .height = 1});
    WavefrontData wavefront{.rays = DeviceBuffer<Ray>::allocate(RAY_COUNT)};
    PathVertexData pathVertexData{
        .t = DeviceBuffer<float>::allocate(RAY_COUNT),
        .bsdfPdfPrev = DeviceBuffer<float>::allocate(RAY_COUNT),
        .throughput = DeviceBuffer<Vec4>::allocate(RAY_COUNT),
        .prevSpecular = DeviceBuffer<int>::allocate(RAY_COUNT),
        .ids = DeviceBuffer<TriangleIdentifier>::allocate(RAY_COUNT),
    };
    OutputData output{.colors = DeviceBuffer<Vec4>::allocate(RAY_COUNT)};

    WavefrontDataDevice wavefrontView{.rays = wavefront.rays.deviceSpan()};
    PathVertexDataDevice pathVertexDataView{
        .t = pathVertexData.t.deviceSpan(),
        .bsdfPdfPrev = pathVertexData.bsdfPdfPrev.deviceSpan(),
        .throughput = pathVertexData.throughput.deviceSpan(),
        .prevSpecular = pathVertexData.prevSpecular.deviceSpan(),
        .ids = pathVertexData.ids.deviceSpan(),
    };
    OutputDataDevice outputView{.colors = output.colors.deviceSpan()};

    dim3 dimBlock(128, 1);
    dim3 dimGrid((RAY_COUNT + dimBlock.x - 1) / dimBlock.x, 1);

    const auto runCudaCalls = [&]{
        initWavefront<<<dimGrid, dimBlock>>>(wavefrontView, pathVertexDataView, rng.devicePtr(), bounds.center, bounds.extent * 0.5f, RAY_COUNT);
        for (int depth=0;depth<8;++depth) {
            findNextVertex<<<dimGrid, dimBlock>>>(wavefrontView, pathVertexDataView, objectsView, RAY_COUNT);
            shade<<<dimGrid, dimBlock>>>(pathVertexDataView, wavefrontView, materials.deviceSpan(), objectsView, rng.devicePtr(), light_table, RAY_COUNT, info, outputView);
            sampleBrdfDirection<<<dimGrid, dimBlock>>>(pathVertexDataView, wavefrontView, rng.devicePtr(), objectsView, materials.deviceSpan(), RAY_COUNT);
        }
        cudaDeviceSynchronize();
        CUDA_ERROR_CHECK();
    };

    runCudaCalls();

    Timer t;
    runCudaCalls();

    std::cout << "Wavefront init: " << t.elapsedSeconds()*1000 << "ms" << std::endl;
}
