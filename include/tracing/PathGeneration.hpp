#pragma once

#include "Utils.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Scene.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/CameraRay.hpp"
#include "tracing/LightSampling.hpp"
#include "tracing/DistributionSamplers.hpp"

#include "Buffers.hpp"
#include "DebugOptions.hpp"

#include <optional>
#include <span>

struct ColorSample
{
    Vec4 color;
    int casts;
};

HD std::optional<Intersection> cast(
    const Ray &ray,
    const std::span<const TriangleMesh> objects,
    const ObjectsInfo &info
);

HD Vec4 checkerPattern(
    const Vec2f &uv,
    const int checker_count,
    const Vec4 dark,
    const Vec4 bright
);

inline HD Vec4 checkerPattern(
    const Vec2f &uv,
    const int checker_count
)
{
    return checkerPattern(uv, checker_count,
        Vec4{0.5,0.5,0.5,0},
        Vec4{0.8,0.8,0.8,0}
    );
}

template<typename T, int N>
struct Stack {
    T arr[N];
    int size;

    inline constexpr Stack() : size(0) {}

    inline constexpr bool empty() const {return size == 0;}
    inline constexpr bool isTop(T value) const {
        if (empty()) return false;
        return arr[size-1] == value;
    }

    inline constexpr void push(T value) {
        arr[size] = value;
        ++size;
    }

    inline constexpr void pop() {
        --size;
    }

    inline constexpr T top() {
        --size;
        return arr[size];
    }
};

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

struct SampleStats
{
    int ray_casts;
};

struct PathSampler
{
    Vec4 throughput{1,1,1,0};
    Stack<const TriangleMesh *, 3> obj_stack;
    Ray ray;
};

inline HD std::optional<Intersection> nextVertex(
    PathSampler &sampler,
    const std::span<const TriangleMesh> objects,
    const ObjectsInfo &info,
    SampleStats &stats
){
    auto &ray = sampler.ray;

    const auto intersection = cast(ray, objects, info);
    ++stats.ray_casts;

    return intersection;
}

template<typename Rng>
HD void generateNewRay(
    PathSampler &sampler,
    const Intersection &intersection,
    const std::span<const Material> materials,
    Rng &rng
){
    auto &[throughput, obj_stack, ray] = sampler;

    auto normal = intersection.n;

    const auto &material = materials[intersection.mat];

    if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {
        const auto w_out = cosineWeightedHemisphereSample(normal, rng);

        sampler.throughput = sampler.throughput * diffuse_material->diffuse_reflectance;

        sampler.ray = Ray{
            .p = intersection.p,
            .v = w_out,
        };
    }

    if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {

        // Find iors
        float n1;
        float n2;

        // leaving last object
        if (obj_stack.isTop(intersection.object)) {
            n1 = transparent_material->inside_medium.ior;

            obj_stack.pop();

            if (obj_stack.empty()) {
                n2 = 1;
            } else {
                const auto &topMat = materials[obj_stack.top()->material];
                n2 = std::get_if<TransparentMaterial>(&topMat)->inside_medium.ior;
            }

            obj_stack.push(intersection.object);
        } else {
            n2 = transparent_material->inside_medium.ior;

            if (obj_stack.empty()) {
                n1 = 1;
            } else {
                const auto &topMat = materials[obj_stack.top()->material];
                n1 = std::get_if<TransparentMaterial>(&topMat)->inside_medium.ior;
            }
        }

        // Reflect or refract
        const auto eta = n1/n2;
        const auto d = dot(ray.v, normal);
        const auto k = 1 - eta*eta * (1 - d*d);

        Vec3 v;

        // Total internal reflection
        if (n1 > n2 && k < 0)
        {
            v = reflect(ray.v, normal, d);
        }
        else
        {
            const auto F = schlick(-d, n1, n2);

            // Fresnel reflection
            if (rng.rnd() < F)
            {
                v = reflect(ray.v, normal, d);
            }
            else
            {
                if (obj_stack.isTop(intersection.object)) {
                    obj_stack.pop();
                } else {
                    obj_stack.push(intersection.object);
                }

                normal = normal * -1;
                const float dotp = -d;
                v = normal * (std::sqrt(k) - dotp * eta) + ray.v * eta;
            }
        }

        sampler.ray = Ray{
            .p = intersection.p,
            .v = v,
        };
    }

    sampler.ray.p = sampler.ray.p + normal * 1e-5;
}

inline HD bool visible(Vec3 p0, Vec3 p1, std::span<const TriangleMesh> objects, const ObjectsInfo &info)
{
    const auto distance = (p1 - p0).length();
    const auto v = (p1 - p0) / distance;

    Ray ray{.p = p0 + v * 1e-5, .v = v};

    const auto hit = cast(ray, objects, info);
    if (!hit.has_value()) return true;

    return hit->t > distance - 1e-5 * 2;
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

    const auto tris_table = std::span{object.tris_sampler, static_cast<size_t>(object.tris_count)};
    const auto [i, triangle_pdf] = sample(tris_table, rng);

    const auto triangle = object.triangles[i];
    const auto A = object.points[triangle.a.pi];
    const auto B = object.points[triangle.b.pi];
    const auto C = object.points[triangle.c.pi];

    const auto p = uniformTriangleSample(A, B, C, rng);

    const auto Aw = A * object.modelToWorld.s;
    const auto Bw = B * object.modelToWorld.s;
    const auto Cw = C * object.modelToWorld.s;

    const auto perp = (Aw - Bw).cross(Aw - Cw);
    const auto perp_length = perp.length();
    const auto n = perp / perp_length;
    const auto world_triangle_area = perp_length / 2.0f;

    return LightSample{
        .p = object.modelToWorld.applyToPoint(p),
        .n = n,
        .mat = mat,
        .pdf = object_pdf * triangle_pdf / world_triangle_area,
    };
}

template<typename Rng>
HD void sampleColorDebug(
    Vec2f sensorPos,
    Vec4 &pixel,
    SampleStats &stats,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const TriangleMesh> objects,
    const ObjectsInfo &info,
    std::span<const Material> materials,
    Rng &rng)
{
    PathSampler sampler;
    sampler.ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

    const auto intersection = nextVertex(sampler, objects, info, stats);
    if (!intersection.has_value()) return;

    const auto mat = intersection->mat;
    const auto uv = intersection->uv;

    pixel.w++;

    const auto &material = materials[mat];
    pixel = pixel + checkerPattern(uv, 7) * getDebugColor(material);
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

inline HD MisPdfs computeNextBounceMisPdfs(
    const PathEntry &vertex,
    const PathEntry &next_vertex,
    std::span<const Material> materials,
    const ObjectsInfo &info)
{
    const float bsdf_pdf = cosineWeightedHemispherePdf(vertex.p, next_vertex.p, vertex.n);

    const auto &next_material = materials[next_vertex.mat];
    const auto next_radiant_exitance = luminance(radiantExitance(next_material));
    const auto pdf_nee_area = next_radiant_exitance / info.total_radiant_power;
    const float nee_pdf = pdf_nee_area * areaToSolidAngle(vertex.p, next_vertex.p, next_vertex.n);

    return {bsdf_pdf, nee_pdf};
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

template<typename Rng>
HD void sampleColor(
    Vec2f sensorPos,
    Vec4 &pixel,
    SampleStats &stats,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const TriangleMesh> objects,
    const ObjectsInfo &info,
    std::span<const Material> materials,
    std::span<const AliasEntry> light_table,
    DebugOptions debug,
    Rng &rng)
{
    if (debug == DebugOptions::UVChecker)
        return sampleColorDebug(sensorPos, pixel, stats, std::move(camera), pixel_sampling, objects, info, materials, rng);

    constexpr int max_depth = 10;
    constexpr auto iterations = 3;

    std::array<PathEntry, max_depth> entries;
    PathEntry *path = entries.data();

    for (int i=0;i<iterations;++i) {
        auto path_end = path;

        {
            PathSampler sampler;
            sampler.ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

            if (debug != DebugOptions::Off)
            {
                const auto intersection = nextVertex(sampler, objects, info, stats);

                Vec4 color{0,0,0,0};
                if (intersection.has_value())
                {
                    const auto &material = materials[intersection->mat];
                    color = getDebugColor(material);

                    switch (debug) {
                        case DebugOptions::BariCoords:
                        {
                            const auto bari = intersection->triangle.bari;
                            color = Vec4{bari.x, bari.y, bari.z, 0};
                            break;
                        }
                        case DebugOptions::WindingOrder:
                        {
                            constexpr auto clockWiseColor = Vec4{0.53, 0.82, 1.0, 0.0};
                            constexpr auto counterClockWiseColor = Vec4{1.0, 0.73, 0.47, 0.0};
                            color = intersection->triangle.ccw ? counterClockWiseColor : clockWiseColor;
                            break;
                        }
                        case DebugOptions::UVChecker:
                            color = checkerPattern(intersection->uv, 7) * getDebugColor(material);
                            break;
                        case DebugOptions::Off:
                            break;
                    }
                }

                pixel.w++;
                pixel = pixel + color;

                continue;
            }

            for (int depth=0; depth<max_depth; ++depth)
            {
                const auto intersection = nextVertex(sampler, objects, info, stats);
                if (!intersection.has_value()) break;

                *path_end = PathEntry {
                    .p = intersection->p,
                    .uv = intersection->uv,
                    .n = intersection->n,
                    .mat = intersection->mat,
                    .total_throughput = sampler.throughput,
                    .triangle_area = intersection->triangle.area
                };
                ++path_end;

                generateNewRay(sampler, *intersection, materials, rng);
            }
        }

        // printf("Light sample: p=%f,%f,%f e=%f,%f,%f\n",
        //     light_sample.p.x, light_sample.p.y, light_sample.p.z,
        //     light_material->emission.x, light_material->emission.y, light_material->emission.z
        // );

        Vec4 color{0,0,0,0};
        bool prev_bounce_specular = true;

        auto path_pdfs = MisPdfs{
            .bsdf_pdf = 0,
            .nee_pdf = 0,
        };

        for (PathEntry *vertex = path; vertex!=path_end; ++vertex)
        {
            const auto next_vertex = vertex+1;

            const auto &material = materials[vertex->mat];

            // BSDF sampling alone
            // if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {
            //     color = color + diffuse_material->emission * vertex->total_throughput;
            // }
            // continue;

            // MIS
            if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {

                color = color + misWeightedEmission(
                    diffuse_material->emission, vertex->total_throughput,
                    prev_bounce_specular, path_pdfs
                );

                path_pdfs = MisPdfs{
                    .bsdf_pdf = 0,
                    .nee_pdf = 0,
                };

                if (next_vertex != path_end) {
                    path_pdfs = computeNextBounceMisPdfs(*vertex, *next_vertex, materials, info);
                }

                LightSample light_sample = samplePointOnLights(objects, light_table, rng);
                const auto *light_material = std::get_if<DiffuseMaterial>(&materials[light_sample.mat]);

                const auto nee_pdf_bsdf = cosineWeightedHemispherePdf(vertex->p, light_sample.p, vertex->n);
                const auto nee_pdf_nee = light_sample.pdf * areaToSolidAngle(vertex->p, light_sample.p, light_sample.n);

                const auto Ld_nee = evaluateDirectLighting(
                    vertex->p, vertex->n, diffuse_material->diffuse_reflectance,
                    light_sample, light_material->emission, objects, info);

                if (nee_pdf_nee + nee_pdf_bsdf > 0)
                    color = color + Ld_nee * vertex->total_throughput * powerHeuristic(nee_pdf_nee, nee_pdf_bsdf);

                prev_bounce_specular = false;
            }
            else if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {
                prev_bounce_specular = true;
            }
        }

        pixel.w++;
        pixel = pixel + color;
    }
}
