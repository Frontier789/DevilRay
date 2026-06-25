#include "benchmark.hpp"

#include "Utils.hpp"
#include "models/Mesh.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace
{
    constexpr const char *defaultMeshFile = "models/bunny.obj";
    constexpr int defaultRayCount = 10'000'000;

    void printUsage(const std::string &program)
    {
        std::cout << "Usage: " << program << " [mesh.obj] [ray_count]\n"
                  << "\n"
                  << "Casts random rays through a mesh's bounding-box hierarchy and reports\n"
                  << "triangle/bbox test counts.\n"
                  << "\n"
                  << "Arguments (both optional):\n"
                  << "  mesh.obj    Path to the mesh to benchmark (default: " << defaultMeshFile << ")\n"
                  << "  ray_count   Number of rays to cast (default: " << defaultRayCount << ")\n"
                  << "\n"
                  << "Options:\n"
                  << "  -h, --help  Show this help and exit\n";
    }
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (argc == 0) args.push_back("devil_ray_benchmark");

    for (const auto &arg : args)
    {
        if (arg == "-h" || arg == "--help")
        {
            printUsage(args[0]);
            return 0;
        }
    }

    if (args.size() > 3)
    {
        std::cerr << "Error: too many arguments.\n\n";
        printUsage(args[0]);
        return 1;
    }

    const std::string mesh_file = args.size() > 1 ? args[1] : defaultMeshFile;

    int ray_count = defaultRayCount;
    if (args.size() > 2)
    {
        try
        {
            ray_count = std::stoi(args[2]);
        }
        catch (const std::exception &)
        {
            std::cerr << "Error: ray_count must be an integer, got '" << args[2] << "'.\n\n";
            printUsage(args[0]);
            return 1;
        }

        if (ray_count <= 0)
        {
            std::cerr << "Error: ray_count must be positive, got " << ray_count << ".\n\n";
            printUsage(args[0]);
            return 1;
        }
    }

    try
    {
        Mesh mesh = loadMesh(mesh_file);
        generateCoarseNormals(mesh);

        std::cout << "Loaded mesh '" << mesh.name << "':\n"
                  << "\t" << mesh.points.size() << " points, \n"
                  << "\t" << mesh.normals.size() << " normals, \n"
                  << "\t" << mesh.triangles.size() << " tris" << std::endl;

        std::cout << std::endl;
        std::cout << "== Benchmark Started ==" << std::endl;
        Timer timer;

        auto bench = BenchmarkGenerator::create(ray_count, mesh);
        bench.step();

        const auto results = bench.aggregateResults();
        const auto rays = bench.ray_count;

        const auto elapsed = timer.elapsedSeconds();

        std::cout << "Number of rays: " << rays << std::endl;
        std::cout << "Triangle hits: " << results.triangle_hits << " (" << results.triangle_hits / static_cast<double>(rays) * 100 << "%)" << std::endl;
        std::cout << "Triangle tests: " << results.triangle_tests << " (" << results.triangle_tests / static_cast<double>(rays) << "/ray)" << std::endl;
        std::cout << "BBox tests: " << results.bbox_tests << " (" << results.bbox_tests / static_cast<double>(rays) << "/ray)" << std::endl;
        std::cout << "== Benchmark Finished in " << elapsed << "s ==" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
