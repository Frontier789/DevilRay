#pragma once

#include "Utils.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/CameraRay.hpp"
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

HD std::optional<Intersection> cast(const Ray &ray, const std::span<const Object> objects);

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

HD inline int getMaterial(const Object &object)
{
    return std::visit([](const auto &o){return o.mat;}, object);
}

struct SampleStats
{
    int ray_casts;
};

struct PathSampler
{
    Vec4 transmission{1,1,1,0};
    Stack<const Object *, 3> obj_stack;
    Ray ray;
};

inline HD std::optional<Intersection> nextVertex(
    PathSampler &sampler,
    const std::span<const Object> objects,
    SampleStats &stats
){
    auto &[transmission, obj_stack, ray] = sampler;

    const auto intersection = cast(ray, objects);
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
    auto &[transmission, obj_stack, ray] = sampler;

    auto normal = intersection.n;

    const auto &material = materials[intersection.mat];

    if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {
        const auto new_v = cosineWeightedHemisphereSample(normal, rng);

        sampler.transmission = sampler.transmission * diffuse_material->diffuse_reflectance;

        sampler.ray = Ray{
            .p = intersection.p,
            .v = new_v,
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
                const auto &topMat = materials[getMaterial(*obj_stack.top())];
                n2 = std::get_if<TransparentMaterial>(&topMat)->inside_medium.ior;
            }

            obj_stack.push(intersection.object);
        } else {
            n2 = transparent_material->inside_medium.ior;

            if (obj_stack.empty()) {
                n1 = 1;
            } else {
                const auto &topMat = materials[getMaterial(*obj_stack.top())];
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

inline HD bool visible(Vec3 p0, Vec3 p1, std::span<const Object> objects)
{
    const auto distance = (p1 - p0).length();
    const auto v = (p1 - p0) / distance;

    Ray ray{.p = p0 + v * 1e-5, .v = v};

    const auto hit = cast(ray, objects);
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
    std::span<const Object> objects,
    std::span<const AliasEntry> light_table,
    Rng &rng)
{
    const auto index = sample(light_table, rng);
    const auto &object = objects[index];
    const auto mat = getMaterial(object);

    // printf("Rolled %d\n", index);

    return std::visit(Overloaded{
        [&](const Sphere &s){
            const auto n = uniformSphereSample(rng);
            const auto p = n * s.radius + s.center;

            return LightSample{
                .p = p,
                .n = n,
                .mat = mat,
                .pdf = 1.0f / (4.0f * pi * s.radius * s.radius),
            };
        },
        [&](const Square &s){
            const auto up = s.right.cross(s.n);

            const auto r = s.right * (rng.rnd() - 0.5f) * s.size;
            const auto u = up * (rng.rnd() - 0.5f) * s.size;

            const auto p = r + u + s.p;
            
            return LightSample{
                .p = p,
                .n = s.n,
                .mat = mat,
                .pdf = 1.0f / (s.size * s.size),
            };
        },
        [&](const TrisCollection &s){
            return LightSample{ // TODO
                .p = Vec3{},
                .n = Vec3{},
                .mat = 0,
                .pdf = 0,
            };
        }
    }, object);
}

template<typename Rng>
HD void sampleColorDebug(
    Vec2f sensorPos,
    Vec4 &pixel,
    SampleStats &stats,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const Object> objects,
    std::span<const Material> materials,
    Rng &rng)
{
    PathSampler sampler;
    sampler.ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

    const auto intersection = nextVertex(sampler, objects, stats);
    if (!intersection.has_value()) return;

    const auto mat = intersection->mat;
    const auto uv = intersection->uv;

    pixel.w++;
    
    const auto &material = materials[mat];
    pixel = pixel + checkerPattern(uv, 7) * getDebugColor(material);
}

template<typename Rng>
HD void sampleColor(
    Vec2f sensorPos,
    Vec4 &pixel,
    SampleStats &stats,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const Object> objects,
    std::span<const Material> materials,
    std::span<const AliasEntry> light_table,
    DebugOptions debug,
    Rng &rng)
{
    if (debug == DebugOptions::UVChecker) {
        return sampleColorDebug(sensorPos, pixel, stats, std::move(camera), pixel_sampling, objects, materials, rng);
    }

    constexpr int max_depth = 20;
    constexpr auto iterations = 1;

    std::array<PathEntry, max_depth> entries;
    PathEntry *path = entries.data();
    
    for (int i=0;i<iterations;++i) {
        auto pathEnd = path;
        
        {
            PathSampler sampler;
            sampler.ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

            if (debug != DebugOptions::Off)
            {
                const auto intersection = nextVertex(sampler, objects, stats);

                Vec4 color{0,0,0,0};
                if (intersection.has_value())
                {
                    const auto &material = materials[intersection->mat];
                    color = getDebugColor(material);

                    switch (debug) {
                        case DebugOptions::BariCoords:
                            if (intersection->triangle.has_value()) {
                                const auto bari = intersection->triangle->bari;
                                color = Vec4{bari.x, bari.y, bari.z, 0};
                            }
                            break;
                        case DebugOptions::WindingOrder:
                            if (intersection->triangle.has_value()) {
                                constexpr auto clockWiseColor = Vec4{0.53, 0.82, 1.0, 0.0};
                                constexpr auto counterClockWiseColor = Vec4{1.0, 0.73, 0.47, 0.0};
                                color = intersection->triangle->ccw ? counterClockWiseColor : clockWiseColor;
                            }
                            break;
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
                const auto intersection = nextVertex(sampler, objects, stats);
                if (!intersection.has_value()) break;

                *pathEnd = PathEntry {
                    .p = intersection->p,
                    .uv = intersection->uv,
                    .n = intersection->n,
                    .mat = intersection->mat,
                    .total_transmission = sampler.transmission,
                };
                ++pathEnd;

                // const auto l = luminance(sampler.transmission);
                // if (l > 10.0f) {
                //     printf("Extreme limunance path: \n");
                //     printf("\tLuminance: %f\n", l);
                //     printf("\tDepth: %d\n", depth);
                    
                //     int d = 0;
                //     for (PathEntry *qit = path; qit!=pathEnd; ++qit)
                //     {
                //         d+=1;
                //         printf("\tVertex %d: %f,%f,%f, t=%f\n", d, qit->p.x, qit->p.y, qit->p.z, luminance(qit->total_transmission));
                //     }
                // }

                generateNewRay(sampler, *intersection, materials, rng);
            }
        }
        
        LightSample light_sample = samplePointOnLights(objects, light_table, rng);
        const auto *light_material = std::get_if<DiffuseMaterial>(&materials[light_sample.mat]);

        // printf("Light sample: p=%f,%f,%f e=%f,%f,%f\n", 
        //     light_sample.p.x, light_sample.p.y, light_sample.p.z,
        //     light_material->emission.x, light_material->emission.y, light_material->emission.z
        // );

        Vec4 color{0,0,0,0};
        bool ray_constrained = true;

        for (PathEntry *pit = path; pit!=pathEnd; ++pit)
        {
            const auto &material = materials[pit->mat];

            if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {

                // color = color + diffuse_material->emission * pit->total_transmission;

                if (ray_constrained) {
                    color = color + diffuse_material->emission * pit->total_transmission;
                }
                
                {
                    if (visible(pit->p, light_sample.p, objects)) {
                        const auto distance = (light_sample.p - pit->p).length();
                        const auto v = (light_sample.p - pit->p) / distance;

                        const auto brdf = diffuse_material->diffuse_reflectance / pi;

                        const auto cos_phi_x_i = v.dot(pit->n);
                        const auto cos_phi_y   = v.dot(light_sample.n);
                        
                        if (cos_phi_y > 0) {
                            const auto geometric_term = std::abs(cos_phi_x_i * cos_phi_y) / distance / distance;
    
                            color = color + light_material->emission * brdf * geometric_term / light_sample.pdf * pit->total_transmission;
                        }
                    }
                }

                ray_constrained = false;
            }
            else if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {
                ray_constrained = true;
            }
        }

        pixel.w++;
        pixel = pixel + color;
    }
}
