#pragma once

#include "Utils.hpp"

#include <optional>
#include <variant>

struct ObjectBase
{
    int mat;
};

struct Square : ObjectBase
{
    Vec3 p;
    Vec3 n;
    Vec3 right;
    float size;
};

struct Sphere : ObjectBase
{
    Vec3 center;
    float radius;
};

using Object = std::variant<Square, Sphere>;
