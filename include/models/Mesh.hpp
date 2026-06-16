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

struct MeshBounds
{
    Vec3 center;
    float extent;
};

Mesh loadMesh(const std::string &fileName);
void generateCoarseNormals(Mesh &mesh);
MeshBounds calculateMeshBounds(const Mesh &mesh);
void normalizeMeshSize(Mesh &mesh);
