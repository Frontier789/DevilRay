#include "models/Mesh.hpp"
#include "tracing/Intersection.hpp"
#include "Utils.hpp"

#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv)
{
    const std::string meshPath = (argc > 1) ? argv[1] : "models/cube.obj";
    constexpr int rayCount = 1'000'000;

    std::cout << "Loading mesh: " << meshPath << std::endl;
    const Mesh mesh = loadMesh(meshPath);
    std::cout << "Triangles: " << mesh.triangles.size()
              << ", points: "  << mesh.points.size() << std::endl;

    // Pre-extract triangle vertices for fast iteration.
    std::vector<TriangleVertices> tris;
    tris.reserve(mesh.triangles.size());

    Vec3 bbMin{ std::numeric_limits<float>::infinity(),  std::numeric_limits<float>::infinity(),  std::numeric_limits<float>::infinity()};
    Vec3 bbMax{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

    for (const auto &t : mesh.triangles)
    {
        const TriangleVertices v{
            .a = mesh.points[t.a.pi],
            .b = mesh.points[t.b.pi],
            .c = mesh.points[t.c.pi],
        };
        tris.push_back(v);

        for (const Vec3 &p : {v.a, v.b, v.c})
        {
            bbMin = Vec3{std::min(bbMin.x, p.x), std::min(bbMin.y, p.y), std::min(bbMin.z, p.z)};
            bbMax = Vec3{std::max(bbMax.x, p.x), std::max(bbMax.y, p.y), std::max(bbMax.z, p.z)};
        }
    }

    const Vec3 center = (bbMin + bbMax) * 0.5f;
    const float radius = ((bbMax - bbMin).length()) * 1.5f + 1.0f;

    // Generate rays: origin on a sphere around the mesh, direction pointing
    // toward a random point inside the bounding box. This guarantees a healthy
    // mix of hits and misses.
    std::cout << "Generating " << rayCount << " random rays..." << std::endl;

    std::vector<Ray> rays;
    rays.reserve(rayCount);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (int i = 0; i < rayCount; ++i)
    {
        // Uniform direction on sphere
        const float z = 1.0f - 2.0f * u01(rng);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = 2.0f * pi * u01(rng);
        const Vec3 dir{r * std::cos(phi), r * std::sin(phi), z};

        const Vec3 origin = center + dir * radius;

        // Aim at a random point inside the (slightly enlarged) bbox
        const Vec3 target{
            bbMin.x + (bbMax.x - bbMin.x) * u01(rng),
            bbMin.y + (bbMax.y - bbMin.y) * u01(rng),
            bbMin.z + (bbMax.z - bbMin.z) * u01(rng),
        };

        const Vec3 v = (target - origin).normalized();
        rays.push_back(Ray{origin, v});
    }

    // Benchmark: for each ray, test against every triangle, keep nearest hit.
    std::cout << "Tracing..." << std::endl;

    const Timer timer;

    long long hitCount = 0;
    double accumT = 0.0;

    for (const Ray &ray : rays)
    {
        float bestT = std::numeric_limits<float>::infinity();
        bool hit = false;

        for (const TriangleVertices &tri : tris)
        {
            const auto isect = testTriangleIntersection(ray, tri);
            if (isect.has_value() && isect->t < bestT)
            {
                bestT = isect->t;
                hit = true;
            }
        }

        if (hit)
        {
            ++hitCount;
            accumT += bestT;
        }
    }

    const float seconds = timer.elapsed_seconds();
    const long long isectTests = static_cast<long long>(rays.size()) * static_cast<long long>(tris.size());

    std::cout << "----------------------------------------\n";
    std::cout << "Rays:                 " << rays.size()  << "\n";
    std::cout << "Triangles:            " << tris.size() << "\n";
    std::cout << "Intersection tests:   " << isectTests << "\n";
    std::cout << "Hits:                 " << hitCount
              << "  (" << (100.0 * hitCount / rays.size()) << "%)\n";
    std::cout << "Avg hit t:            "
              << (hitCount > 0 ? accumT / hitCount : 0.0) << "\n";
    std::cout << "Elapsed:              " << seconds << " s\n";
    std::cout << "Rays / second:        " << (rays.size() / seconds) << "\n";
    std::cout << "Tri tests / second:   " << (isectTests / seconds) << "\n";
    std::cout << "ns / ray:             " << (seconds * 1e9 / rays.size()) << "\n";
    std::cout << "ns / tri test:        " << (seconds * 1e9 / isectTests) << "\n";

    return 0;
}
