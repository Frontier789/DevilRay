#pragma once

#include "Utils.hpp"
#include "tracing/Material.hpp"

struct Intersection
{
    float t;
    Vec3 p;
    Vec2f uv;
    Vec3 n;
    int mat;
};