#pragma once

#include "Utils.hpp"
#include "tracing/Material.hpp"
#include "tracing/Objects.hpp"

struct Intersection
{
    float t;
    Vec3 p;
    Vec2f uv;
    Vec3 n;
    int mat;
    const Object *object;
};

HD std::optional<Intersection> testIntersection(const Ray &ray, const Object &object);
