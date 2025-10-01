#pragma once

#include "Utils.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Intersection.hpp"

#include <optional>
#include <span>

struct ColorSample
{
    Vec4 color;
    int casts;
};

HD std::optional<Intersection> cast(const Ray &ray, const std::span<const Object> objects);

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

template<typename Rng>
HD ColorSample sampleColor(Ray ray, const int max_depth, const std::span<const Object> objects, const bool debug, Rng &rng)
{
    Vec4 transmission{1,1,1,0};
    Vec4 color{0,0,0,0};
    int ray_casts = 0;

    for (int depth=0;depth<max_depth;++depth)
    {
        const auto intersection = cast(ray, objects);
        ++ray_casts;

        if (!intersection.has_value()) break;

        if (dot(intersection->n, ray.v) > 1e-5) {
            color.y += 10000;
        }

        if (dot(intersection->p - ray.p, intersection->p - ray.p) < 1e-12) {
            ray.p = ray.p + intersection->n * 1e-6;
            continue;
            // color.x += 10000;
        }

        const auto &material = *intersection->mat;

        if (debug)
        {
            color = color + material.debug_color * checkerPattern(intersection->uv, 8);
        }
        else
        {
            color = color + material.emission * transmission;
        }

        const auto new_v = uniformHemisphereSample(intersection->n, rng);

        const auto weakening_factor = dot(intersection->n, new_v);
        transmission = transmission * weakening_factor * material.diffuse_reflectance;

        ray = Ray{
            .p = intersection->p,
            .v = new_v
        };

        ray.p = ray.p + intersection->n * 1e-5;
    }

    return ColorSample{
        .color = color,
        .casts = ray_casts,
    };
}