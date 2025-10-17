#pragma once

#include "Utils.hpp"

#include <optional>
#include <variant>

struct Square
{
    Vec3 p;
    Vec3 n;
    Vec3 right;
    float size;
    int mat;
};

struct Sphere
{
    Vec3 center;
    float radius;
    int mat;
};

using Object = std::variant<Square, Sphere>;
