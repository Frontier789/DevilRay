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