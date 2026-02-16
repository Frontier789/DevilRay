#include <vector>
#include <iostream>

#include "device/DevUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/PathGeneration.hpp"
#include "tracing/LightSampling.hpp"

#include "IntersectionTestsImpl.hpp"
#include "RendererImpl.hpp"

HD std::optional<Intersection> cast(const Ray &ray, const std::span<const Object> objects)
{
    std::optional<Intersection> best = std::nullopt;

    for (const auto &obj : objects)
    {
        const auto intersection = testIntersection(ray, obj);

        if (!intersection.has_value()) continue;

        if (!best.has_value() || best->t > intersection->t)
        {
            best = intersection;
        }
    }

    return best;
}

HD std::optional<Intersection> testIntersection(const Ray &ray, const Object &object)
{
    return std::visit([&](auto&& o) {
        auto i = getIntersection(ray, o);
        if (i.has_value()) {
            i->object = &object;
        }
        return i;
    }, object);
}

HD std::optional<TriangleIntersection> testTriangleIntersection(const Ray &ray, const TriangleVertices &triangle)
{
    if (ray.p.anyNan() || ray.v.anyNan()) return std::nullopt;

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);
    const auto sgn_area2 = n_f.dot(n_f);

    if (sgn_area2 < 1e-14f) return std::nullopt;

    auto n = n_f.normalized();

    auto dp = dot(ray.p - A, n);

    if (dp < 0) {
        n = n * -1;
        dp = -dp;
    }

    // (ray.p - o) . n + ray.v . n * t = 0

    const auto d = -dot(ray.v, n);
    if (d < 1e-7f) return std::nullopt;

    const float t = dp / d;

    const Vec3 p = ray.p + ray.v * t;

    const auto n_1 = (p - A).cross(C - A);
    const auto n_2 = (B - A).cross(p - A);

    const auto w_B = n_f.dot(n_1) / sgn_area2;
    const auto w_C = n_f.dot(n_2) / sgn_area2;
    const auto w_A = 1 - w_B - w_C;

    if (w_A < 0 || w_B < 0 || w_C < 0) return std::nullopt;

    return TriangleIntersection{
        .t = t,
        .bari = Vec3{w_A, w_B, w_C},
    };
}

HD Vec4 checkerPattern(
    const Vec2f &uv, 
    const int checker_count, 
    const Vec4 dark, 
    const Vec4 bright
){
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;

    const float checker = checker_x ^ checker_y;

    return bright * checker + dark * (1-checker);
}


HD float surfaceAreaImpl(const Square &square)
{
    return square.size * square.size;
}

HD float surfaceAreaImpl(const Sphere &sphere)
{
    const auto r = sphere.radius;

    return 4*pi * r*r;
}

HD float surfaceAreaImpl(const TrisCollection &tris)
{
    return tris.surface_area;
}

HD float surfaceArea(const Object &object)
{
    return std::visit([](auto &&o){return surfaceAreaImpl(o);}, object);
}

HD Vec4 radiantExitanceImpl(const TransparentMaterial &mat)
{
    return Vec4{0,0,0,0};
}

HD Vec4 radiantExitanceImpl(const DiffuseMaterial &mat)
{
    return mat.emission * pi;
}

HD Vec4 radiantExitance(const Material &mat)
{
    return std::visit([](auto &&o){return radiantExitanceImpl(o);}, mat);
}

void TrisCollection::setPosition(const Vec3 &pos)
{
    this->p = pos;
}

void TrisCollection::setScale(const Vec3 &scale)
{
    this->s = scale;

    // TODO: scale surface area
}
