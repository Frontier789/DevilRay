#include "models/Mesh.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

/////////////////////////////////////////
///             DISCLAIMER            ///
///                                   ///
/// These test were ai generated with ///
/// Google's Gemini 3, then reviewed  ///
/// And corrected at several places   ///
/// by hand. Some tests were later    ///
/// added by Claude, reviewed by hand ///
/// before committing.                ///
/////////////////////////////////////////

const auto output_folder = std::filesystem::path("test_output/test_mesh");

void createObjFile(const std::filesystem::path& file, const std::string& content)
{
    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder);
    }

    std::ofstream ofs(file.string());
    ofs << content;
}

// --- Test Suite ---

TEST(MeshLoaderTest, HandlesFullVertexDefinition) {
    // Tests the f v/vt/vn format.
    // Even though Mesh doesn't store vt, the parser must correctly skip it.
    const std::string filename = "full_vertex.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0.5 0.5\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/1/1 3/1/1\n";

    createObjFile(output_folder / filename, content);
    Mesh m = loadMesh(output_folder / filename);

    ASSERT_EQ(m.triangles.size(), 1);
    EXPECT_EQ(m.triangles[0].a.pi, 0);
    EXPECT_EQ(m.triangles[0].a.ni, 0);
    EXPECT_EQ(m.triangles[0].b.pi, 1);
    EXPECT_EQ(m.triangles[0].b.ni, 0);
    EXPECT_EQ(m.triangles[0].c.pi, 2);
    EXPECT_EQ(m.triangles[0].c.ni, 0);
}

TEST(MeshLoaderTest, QuadsThrowError) {
    const std::string filename = "quad.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "f 1 2 3 4\n";

    createObjFile(output_folder / filename, content);
    ASSERT_THROW(loadMesh(output_folder / filename), std::runtime_error);
}

TEST(MeshLoaderTest, NGonsThrow) {
    const std::string filename = "ngon.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 2 1 0\n"
        "v 1 2 0\n"
        "v 0 1 0\n"
        "f 1 2 3 4 5\n";

    createObjFile(output_folder / filename, content);
    ASSERT_THROW(loadMesh(output_folder / filename), std::runtime_error);
}

TEST(MeshLoaderTest, DiscardsDegenerateFaces) {
    const std::string filename = "degenerate.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 1 1\n"
        "f 1//1 2//1\n"   // Line
        "f 1//1\n";     // Point

    createObjFile(output_folder / filename, content);
    Mesh m = loadMesh(output_folder / filename);

    EXPECT_EQ(m.triangles.size(), 0);
}

TEST(MeshLoaderTest, HandlesAllFaceIndexFormats) {
    const std::string filename = "index_formats.obj";
    const std::string content =
        "v 0 0 0\n"   // Index 1
        "v 1 0 0\n"   // Index 2
        "v 0 1 0\n"   // Index 3
        "vt 0.5 0.5\n"
        "vn 0 0 1\n"   // Index 1
        "vn 0 1 0\n"   // Index 2
        "f 1 2 3\n"              // Format: v
        "f 1/1 2/1 3/1\n"        // Format: v/vt
        "f 1//2 2//2 3//2\n"     // Format: v//vn
        "f 1/1/2 2/1/2 3/1/2\n"; // Format: v/vt/vn

    createObjFile(output_folder / filename, content);
    Mesh m = loadMesh(output_folder / filename);

    ASSERT_EQ(m.triangles.size(), 4);

    const auto &position_only = m.triangles[0];
    const auto &position_and_texture = m.triangles[1];
    const auto &position_and_normal = m.triangles[2];
    const auto &position_texture_and_normal = m.triangles[3];

    constexpr uint32_t first_normal = 0;
    constexpr uint32_t second_normal = 1;

    EXPECT_EQ(position_only.a.pi, 0);
    EXPECT_EQ(position_only.a.ni, first_normal) << "A face without normals falls back to the first normal";

    EXPECT_EQ(position_and_texture.b.pi, 1);
    EXPECT_EQ(position_and_texture.b.ni, first_normal);

    EXPECT_EQ(position_and_normal.c.pi, 2);
    EXPECT_EQ(position_and_normal.c.ni, second_normal);

    EXPECT_EQ(position_texture_and_normal.a.pi, 0);
    EXPECT_EQ(position_texture_and_normal.a.ni, second_normal);
}

TEST(MeshLoaderTest, ThrowsOnInvalidContent) {
    const std::vector<std::pair<std::string, std::string>> invalidCases = {
        {"position_index_out_of_range.obj", "v 0 0 0\nf 1 2 3\n"},
        {"normal_index_out_of_range.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//5 2//5 3//5\n"},
        {"non_numeric_coordinate.obj", "v 0.0 apple 1.0\n"},
        {"too_many_slashes_in_face_vertex.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1///1 2///1 3///1\n"},
        {"zero_index.obj", "v 0 0 0\nf 0 0 0\n"},
        {"vertex_missing_z_coordinate.obj", "v 0.1 0.2\nf 1 1 1\n"},
        {"index_overflowing_int.obj", "v 0 0 0\nf 9999999999999999999999 1 1\n"},
    };

    for (const auto& [filename, content] : invalidCases) {
        createObjFile(output_folder / filename, content);

        EXPECT_THROW(loadMesh(output_folder / filename), std::exception)
            << "Failed to throw for case: " << filename;

        std::filesystem::remove(output_folder / filename);
    }
}

TEST(MeshLoaderTest, RelativeNegativeIndicesAreNotSupported) {
    const std::string filename = "relative_indices.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "f -3//-1 -2//-1 -1//-1\n";

    createObjFile(output_folder / filename, content);

    EXPECT_THROW(loadMesh(output_folder / filename), std::runtime_error);
}

TEST(MeshLoaderTest, MissingFileThrows) {
    EXPECT_THROW(loadMesh((output_folder / "does_not_exist.obj").string()), std::runtime_error);
}

TEST(MeshLoaderTest, UnrecognizedTagThrows) {
    const std::string filename = "unrecognized_tag.obj";
    createObjFile(output_folder / filename, "bogus 1 2 3\n");

    EXPECT_THROW(loadMesh(output_folder / filename), std::runtime_error);
}

TEST(MeshLoaderTest, ObjectTagSetsMeshName) {
    const std::string filename = "named_object.obj";
    createObjFile(output_folder / filename, "o teapot\n");

    EXPECT_EQ(loadMesh(output_folder / filename).name, "teapot");
}

TEST(MeshLoaderTest, SkipsCommentsAndAttributesItDoesNotUse) {
    const std::string filename = "skipped_lines.obj";
    const std::string content =
        "# a comment\n"
        "mtllib materials.mtl\n"
        "s off\n"
        "\n"
        "vt 0.5 0.5\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "f 1 2 3\n";

    createObjFile(output_folder / filename, content);
    const Mesh m = loadMesh(output_folder / filename);

    EXPECT_EQ(m.points.size(), 3);
    EXPECT_EQ(m.normals.size(), 1);
    EXPECT_EQ(m.triangles.size(), 1);
}

TEST(MeshLoaderTest, FacesWithoutNormalsStillReferenceAValidNormal) {
    const std::string filename = "no_normals.obj";
    const std::string content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n";

    createObjFile(output_folder / filename, content);
    const Mesh m = loadMesh(output_folder / filename);

    ASSERT_EQ(m.triangles.size(), 1);
    for (const Vertex &vertex : {m.triangles[0].a, m.triangles[0].b, m.triangles[0].c}) {
        EXPECT_LT(vertex.ni, m.normals.size());
    }
}

namespace
{
    bool allIndicesAreInRange(const Mesh &mesh)
    {
        for (const Triangle &triangle : mesh.triangles)
        {
            for (const Vertex &vertex : {triangle.a, triangle.b, triangle.c})
            {
                if (vertex.pi >= mesh.points.size()) return false;
                if (vertex.ni >= mesh.normals.size()) return false;
            }
        }

        return true;
    }
}

TEST(CheckedInModelsTest, EveryObjFileLoadsWithIndicesInRange) {
    const auto models_folder = std::filesystem::path("models");
    ASSERT_TRUE(std::filesystem::is_directory(models_folder));

    int obj_files_checked = 0;
    for (const auto &entry : std::filesystem::directory_iterator(models_folder))
    {
        if (entry.path().extension() != ".obj") continue;

        const Mesh mesh = loadMesh(entry.path().string());

        EXPECT_GT(mesh.triangles.size(), 0) << entry.path();
        EXPECT_TRUE(allIndicesAreInRange(mesh)) << entry.path();
        ++obj_files_checked;
    }

    EXPECT_GT(obj_files_checked, 0);
}

// --- Geometry utilities (no file IO) ---

namespace
{
    Mesh triangleMesh(std::vector<Vec3> points)
    {
        Mesh mesh;
        mesh.name = "geometry";
        mesh.points = std::move(points);
        mesh.triangles = {Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}}};
        return mesh;
    }
}

TEST(MeshGeometryTest, CalculateBoundsFindsCentroidAndExtent)
{
    const Mesh mesh = triangleMesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}});
    const auto bounds = calculateMeshBounds(mesh);

    EXPECT_NEAR(bounds.center.x, 2.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(bounds.center.y, 2.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(bounds.center.z, 0.0f, 1e-5f);

    const float farthest = (Vec3{2, 0, 0} - bounds.center).length();
    EXPECT_NEAR(bounds.extent, farthest, 1e-5f);
}

TEST(MeshGeometryTest, NormalizeRecentersAndRescales)
{
    Mesh mesh = triangleMesh({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}});
    normalizeMeshSize(mesh);

    const auto bounds = calculateMeshBounds(mesh);
    EXPECT_NEAR(bounds.center.x, 0.0f, 1e-5f);
    EXPECT_NEAR(bounds.center.y, 0.0f, 1e-5f);
    EXPECT_NEAR(bounds.extent, 1.0f, 1e-5f);
}

TEST(MeshGeometryTest, CoarseNormalsAreFaceNormals)
{
    Mesh mesh = triangleMesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    generateCoarseNormals(mesh);

    ASSERT_EQ(mesh.normals.size(), 1u);
    EXPECT_NEAR(mesh.normals[0].x, 0.0f, 1e-6f);
    EXPECT_NEAR(mesh.normals[0].y, 0.0f, 1e-6f);
    EXPECT_NEAR(mesh.normals[0].z, 1.0f, 1e-6f);

    EXPECT_EQ(mesh.triangles[0].a.ni, 0u);
    EXPECT_EQ(mesh.triangles[0].b.ni, 0u);
    EXPECT_EQ(mesh.triangles[0].c.ni, 0u);
}

TEST(MeshGeometryTest, CoarseNormalsAreGeneratedPerTriangle)
{
    Mesh mesh;
    mesh.name = "two-faces";
    mesh.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.triangles = {
        Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}},
        Triangle{.a = {0, 0}, .b = {1, 0}, .c = {3, 0}},
    };

    generateCoarseNormals(mesh);

    ASSERT_EQ(mesh.normals.size(), 2u);
    EXPECT_EQ(mesh.triangles[0].a.ni, 0u);
    EXPECT_EQ(mesh.triangles[1].a.ni, 1u);
}