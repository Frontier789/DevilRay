#pragma once

#include "Utils.hpp"
#include "tracing/Material.hpp"

#include <optional>

struct TriangleMesh;

struct TriangleIntersection
{
    float t;
    Vec3 bari;
};

struct TriangleVertices
{
    Vec3 a;
    Vec3 b;
    Vec3 c;
};

HD std::optional<TriangleIntersection> testTriangleIntersection(const Ray &ray, const TriangleVertices &triangle);

struct TriangleHitData
{
    Vec3 bari;
    float area;
    bool ccw;
};

struct Intersection
{
    float t;
    Vec3 p;
    Vec2f uv;
    Vec3 n;
    int mat;
    const TriangleMesh *object;
    TriangleHitData triangle;
};

struct PathEntry
{
    Vec3 p;
    Vec2f uv;
    Vec3 n;
    int mat;
    Vec4 total_throughput;

    float triangle_area;
};
