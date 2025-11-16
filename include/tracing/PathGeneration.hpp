#pragma once

#include "Utils.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/CameraRay.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "Buffers.hpp"

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
        const auto new_v = uniformHemisphereSample(normal, rng);

        const auto weakening_factor = dot(normal, new_v);
        const auto path_sampling_probability = 1 / (2*pi);
        const auto new_v_radiance = 1 / pi;
        const auto beta = weakening_factor * new_v_radiance / path_sampling_probability;
        sampler.transmission = sampler.transmission * beta * diffuse_material->diffuse_reflectance;

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

template<typename Rng>
HD void sampleColor(
    Vec2f sensorPos,
    Vec4 &pixel,
    SampleStats &stats,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const Object> objects,
    std::span<const Material> materials,
    bool debug,
    Rng &rng)
{
    const int max_depth = debug ? 1 : Buffers::maxPathLength;
    const auto iterations = debug ? 1 : 10;

    std::array<PathEntry, Buffers::maxPathLength> entries;
    PathEntry *path = entries.data();
    
    for (int i=0;i<iterations;++i) {
        auto pathEnd = path;
        
        {
            PathSampler sampler;
            sampler.ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

            for (int depth=0; depth<max_depth; ++depth)
            {
                const auto intersection = nextVertex(sampler, objects, stats);
                if (!intersection.has_value()) break;

                generateNewRay(sampler, *intersection, materials, rng);
                
                *pathEnd = PathEntry {
                    .p = intersection->p,
                    .uv = intersection->uv,
                    .n = intersection->n,
                    .mat = intersection->mat,
                    .total_transmission = sampler.transmission,
                };
                ++pathEnd;
            }
        }

        Vec4 color{0,0,0,0};

        for (PathEntry *pit = path; pit!=pathEnd; ++pit)
        {
            const auto &material = materials[pit->mat];

            if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {
                color = color + diffuse_material->emission * pit->total_transmission;
            }
        }

        pixel.w++;
        pixel = pixel + color;
    }
}
