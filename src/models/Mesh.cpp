#include "models/Mesh.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>

namespace
{
    Vertex parseVertex(std::string str)
    {
        for (const auto c : str)
        {
            if (std::string("0123456789/").find(c) == std::string::npos)
            {
                throw std::runtime_error(std::string("Found unkown character in face descriptor: '") + c + "'");
            }
        }

        std::stringstream ss(str);
        std::string index_text;

        Vertex v{
            .pi = 1,
            .ni = 1,
        };

        int index_kind = 0;

        while (std::getline(ss, index_text, '/'))
        {
            if (!index_text.empty() && index_kind < 3)
            {
                const int index = std::stoi(index_text);

                if (index_kind == 0) v.pi = index;
                if (index_kind == 1);
                if (index_kind == 2) v.ni = index;
            }

            ++index_kind;
        }

        if (index_kind > 3)
        {
            throw std::runtime_error("Unexpected number of indices in face vertex: " + std::to_string(index_kind));
        }

        return v;
    }

    void trim(std::string &line)
    {
        if (line.empty()) return;

        size_t i = line.size();
        
        while (i > 0 && (line[i-1] == ' ' || line[i-1] == '\t' || line[i-1] == '\n' || line[i-1] == '\r'))
        {
            --i;
        }

        line = line.substr(0, i);
    }

    void fixVertexIndices(Mesh &m)
    {
        if (m.normals.empty())
        {
            m.normals.push_back(Vec3{0,0,0});
        }

        const int pointCount = m.points.size();
        const int normalCount = m.normals.size();
        
        for (int i=0; i<m.triangles.size(); ++i)
        {
            Triangle &t = m.triangles[i];

            for (Vertex *v : {&t.a, &t.b, &t.c})
            {
                if (v->pi < 0) v->pi = pointCount + v->pi;
                else v->pi--;
                
                if (v->ni < 0) v->ni = normalCount + v->ni;
                else v->ni--;

                if (v->pi < 0 || v->pi >= pointCount)
                {
                    throw std::runtime_error(std::format(
                        "Invalid position index on triangle {}: {}. Number of points is {}.",
                        i, v->pi, pointCount
                    ));
                }
                
                if (v->ni < 0 || v->ni >= normalCount)
                {
                    throw std::runtime_error(std::format(
                        "Invalid normal index on triangle {}: {}. Number of normals is {}.",
                        i, v->ni, normalCount
                    ));
                }
            }
        }
    }

    void checkStream(std::stringstream &ss)
    {
        if (ss.fail())
        {
            throw std::runtime_error("Failed to parse: " + ss.str());
        }
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
        trim(line);

        std::stringstream ss(std::move(line));

        std::string tag;
        ss >> tag;

        if (tag == "#") continue;
        if (tag == "mtllib") continue;
        if (tag == "s") continue;
        if (tag == "v") {
            Vec3 p;
            ss >> p.x >> p.y >> p.z;
            checkStream(ss);

            mesh.points.push_back(p);
            continue;
        }
        if (tag == "vn") {
            Vec3 n;
            ss >> n.x >> n.y >> n.z;
            checkStream(ss);
            
            mesh.normals.push_back(n);
            continue;
        }
        if (tag == "vt") continue;
        if (tag == "o") {
            ss >> mesh.name;
            continue;
        }
        if (tag == "f") {
            std::vector<Vertex> verts;
            std::string indices;

            while (ss >> indices)
                verts.push_back(parseVertex(std::move(indices)));
            
            if (verts.size() < 3) continue;

            if (verts.size() > 3)
            {
                throw std::runtime_error("Triangulation not supported. Found face with " + std::to_string(verts.size()) + " vertices");
            }

            mesh.triangles.emplace_back(verts[0], verts[1], verts[2]);
            continue;
        }

        throw std::runtime_error("Unrecognized tag: " + tag);
    }

    fixVertexIndices(mesh);

    return mesh;
}
