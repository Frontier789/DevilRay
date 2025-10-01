#pragma once

#include "Utils.hpp"
#include "Intersection.hpp"

#include <optional>
#include <variant>

struct Square
{
    Vec3 p;
    Vec3 n;
    Vec3 right;
    float size;
    Material *mat;
};

struct Sphere
{
    Vec3 center;
    float radius;
    Material *mat;
};

using Object = std::variant<Square, Sphere>;

HD std::optional<Intersection> testIntersection(const Ray &ray, const Object &object);
