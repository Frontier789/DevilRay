#pragma once

#include "Utils.hpp"
#include "models/Mesh.hpp"

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

struct TrisCollection : ObjectBase
{
    Vec3 *points;
    Vec3 *normals;
    Triangle *triangles;

    Vec3 s;
    Vec3 p;
    int tris_count;
    float surface_area;

    void setPosition(const Vec3 &pos);
    void setScale(const Vec3 &scale);
};

using Object = std::variant<Square, Sphere, TrisCollection>;
