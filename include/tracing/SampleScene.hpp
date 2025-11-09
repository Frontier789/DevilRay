#pragma once

#include "Utils.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/PixelSampling.hpp"
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

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformHemisphereSample(const Vec3 &normal, Rng &r)
{
    const float theta0 = 2 * std::numbers::pi_v<float> * r.rnd();
    const float theta1 = std::acos(1 - 2 * r.rnd());

    const float x = std::sin(theta1) * std::sin(theta0);
    const float y = std::sin(theta1) * std::cos(theta0);
    const float z = std::cos(theta1);

    const auto v = Vec3{x,y,z};

    if (dot(v, normal) < 0) return v*-1;

    return v;
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

template<typename Rng>
HD Ray cameraRay(const Camera &cam, Vec2f pixelCoord, PixelSampling sampling, int index, Rng &rng)
{
    Vec2f subPixelCoord;

    if (sampling == PixelSampling::Center) {
        subPixelCoord = Vec2{0.5, 0.5};
    }
    if (sampling == PixelSampling::UniformRandom) {
        subPixelCoord = Vec2{rng.rnd(), rng.rnd()};
    }

    const auto pixelCenter = pixelCoord + subPixelCoord;
    const auto physicalPixelCenter = pixelCenter * cam.physical_pixel_size - cam.intrinsics.center;

    const auto dir = physicalPixelCenter / cam.intrinsics.focal_length;
    
    return Ray{
        .p = Vec3{0,0,0},
        .v = Vec3{dir.x, dir.y, 1},
    };
}

struct SampleStats
{
    int ray_casts;
};

template<typename Rng>
HD void samplePath(
    const Ray &initialRay,
    const int max_depth,
    const std::span<const Object> objects,
    const std::span<const Material> materials,
    PathEntry *&path,
    Rng &rng,
    SampleStats &stats
){
    Vec4 transmission{1,1,1,0};

    Ray ray = initialRay;

    Stack<const Object *, 3> obj_stack;

    for (int depth=0;depth<max_depth;++depth)
    {
        const auto intersection = cast(ray, objects);
        ++stats.ray_casts;

        if (!intersection.has_value()) break;

        *path = PathEntry {
            .p = intersection->p,
            .uv = intersection->uv,
            .n = intersection->n,
            .mat = intersection->mat,
            .total_transmission = transmission,
        };
        ++path;

        auto normal = intersection->n;

        if (dot(intersection->p - ray.p, intersection->p - ray.p) < 1e-12) {
            ray.p = ray.p + normal * 1e-6;
            continue;
        }

        const auto &material = materials[intersection->mat];

        if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material)) {
            const auto new_v = uniformHemisphereSample(normal, rng);

            const auto weakening_factor = dot(normal, new_v);
            transmission = transmission * weakening_factor * diffuse_material->diffuse_reflectance;

            if (transmission.max() < 0.001) break;

            ray = Ray{
                .p = intersection->p,
                .v = new_v,
            };
        }

        if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {

            // Find iors
            float n1;
            float n2;

            // leaving last object
            if (obj_stack.isTop(intersection->object)) {
                n1 = transparent_material->inside_medium.ior;
                
                obj_stack.pop();

                if (obj_stack.empty()) {
                    n2 = 1;
                } else {
                    const auto &topMat = materials[getMaterial(*obj_stack.top())];
                    n2 = std::get_if<TransparentMaterial>(&topMat)->inside_medium.ior;
                }

                obj_stack.push(intersection->object);
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
                    if (obj_stack.isTop(intersection->object)) {
                        obj_stack.pop();
                    } else {
                        obj_stack.push(intersection->object);
                    }

                    normal = normal * -1;
                    
                    const float dotp = -d;
                    v = normal * (std::sqrt(k) - dotp * eta) + ray.v * eta;
                }
            }

            ray = Ray{
                .p = intersection->p,
                .v = v,
            };
        }

        ray.p = ray.p + normal * 1e-5;
    }
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
    PathEntry *path,
    bool debug,
    Rng &rng)
{
    const int max_depth = debug ? 1 : Outputs::maxPathLength;
    const auto iterations = debug ? 1 : 10;
    
    for (int i=0;i<iterations;++i) {
        const auto ray = cameraRay(camera, sensorPos, pixel_sampling, pixel.w, rng);

        auto pathEnd = path;

        samplePath(ray, max_depth, objects, materials, pathEnd, rng, stats);

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
