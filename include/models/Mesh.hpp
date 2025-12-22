#pragma once

#include "Utils.hpp"

#include <vector>

struct Vertex
{
    uint32_t pi;
    uint32_t ni;
};

struct Triangle
{
    Vertex a;
    Vertex b;
    Vertex c;
};

struct Mesh
{
    std::vector<Vec3> points;
    std::vector<Vec3> normals;
    std::vector<Triangle> triangles;

    std::string name;
};

Mesh loadMesh(const std::string &fileName);
