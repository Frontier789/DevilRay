#include "models/Mesh.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>

namespace
{
    Vertex parseVertex(std::string str)
    {
        for (auto &c : str) {
            if (c == '/') c = ' ';
        }

        std::stringstream ss(std::move(str));

        std::vector<int> indices;
        int i;
        while (ss >> i) indices.push_back(i-1);

        if (indices.size() != 3) {
            throw std::runtime_error("Unexpected number of indices in face vertex: " + std::to_string(indices.size()));
        }

        Vertex v;
        v.pi = indices[0];
        v.ni = indices[2];

        return v;
    }
} // namespace


Mesh loadMesh(const std::string &fileName)
{
    if (!std::filesystem::exists(fileName)) {
        throw std::runtime_error("File " + fileName + " does not exist");
    }

    std::ifstream in(fileName);
    std::string line;

    Mesh mesh;
    mesh.name = "Mesh";

    while (std::getline(in, line))
    {
        std::stringstream ss(std::move(line));

        std::string tag;
        ss >> tag;

        if (tag == "#") continue;
        if (tag == "mtllib") continue;
        if (tag == "s") continue;
        if (tag == "v") {
            Vec3 p;
            ss >> p.x >> p.y >> p.z;
            mesh.points.push_back(p);
            continue;
        }
        if (tag == "vn") {
            Vec3 n;
            ss >> n.x >> n.y >> n.z;
            mesh.normals.push_back(n);
            continue;
        }
        if (tag == "vt") continue;
        if (tag == "o") {
            ss >> mesh.name;
            continue;
        }
        if (tag == "f") {
            Vertex a,b,c;
            std::string indices;

            ss >> indices; a = parseVertex(indices);
            ss >> indices; b = parseVertex(indices);
            ss >> indices; c = parseVertex(indices);

            mesh.triangles.emplace_back(a,b,c);
            continue;
        }

        throw std::runtime_error("Unrecognized tag: " + tag);
    }

    return mesh;
}
