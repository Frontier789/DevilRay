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
/// by hand.                          ///
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
    // A quad (4 vertices) should be split into 2 triangles.
    // f 1 2 3 4 -> (1, 2, 3) and (1, 3, 4)
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
    // A 5-sided polygon should result in 3 triangles.
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
    // A "face" with only 2 vertices is a line, not a triangle.
    // A "face" with 1 vertex is a point.
    // Your loader should either skip these or handle them without crashing.
    const std::string filename = "degenerate.obj";
    const std::string content = 
        "v 0 0 0\n"
        "v 1 1 1\n"
        "f 1//1 2//1\n"   // Line
        "f 1//1\n";     // Point

    createObjFile(output_folder / filename, content);
    Mesh m = loadMesh(output_folder / filename);

    // Since our struct is 'Triangle', these should not be loaded
    EXPECT_EQ(m.triangles.size(), 0);
}

TEST(MeshLoaderTest, HandlesAllFaceIndexFormats) {
    // Tests the four standard OBJ face formats:
    // 1. v                   (Only position)
    // 2. v/vt                (Position and Texture)
    // 3. v//vn               (Position and Normal, skipping Texture)
    // 4. v/vt/vn             (Position, Texture, and Normal)
    const std::string filename = "index_formats.obj";
    const std::string content = 
        "v 0 0 0\n"   // Index 1
        "v 1 0 0\n"   // Index 2
        "v 0 1 0\n"   // Index 3
        "vt 0.5 0.5\n"
        "vn 0 0 1\n"   // Index 1
        "f 1 2 3\n"              // Format: v
        "f 1/1 2/1 3/1\n"        // Format: v/vt
        "f 1//1 2//1 3//1\n"     // Format: v//vn
        "f 1/1/1 2/1/1 3/1/1\n"; // Format: v/vt/vn

    createObjFile(output_folder / filename, content);
    Mesh m = loadMesh(output_folder / filename);

    ASSERT_EQ(m.triangles.size(), 4);

    // 1. Test 'f v v v' (Indices should have position, but no normal)
    EXPECT_EQ(m.triangles[0].a.pi, 0);
    // Note: ni behavior depends on your loader's default (e.g., 0 or UINT32_MAX)

    // 2. Test 'f v/vt v/vt v/vt' (Should still map pi correctly)
    EXPECT_EQ(m.triangles[1].b.pi, 1);

    // 3. Test 'f v//vn v//vn v//vn' (Skips middle, maps pi and ni)
    EXPECT_EQ(m.triangles[2].c.pi, 2);
    EXPECT_EQ(m.triangles[2].c.ni, 0);

    // 4. Test 'f v/vt/vn v/vt/vn v/vt/vn' (Maps pi and ni, ignores vt)
    EXPECT_EQ(m.triangles[3].a.pi, 0);
    EXPECT_EQ(m.triangles[3].a.ni, 0);
}

TEST(MeshLoaderTest, ThrowsOnInvalidContent) {
    const std::vector<std::pair<std::string, std::string>> invalidCases = {
        // 1. Out of range position index (Forward reference)
        {"oob_pos.obj", "v 0 0 0\nf 1 2 3\n"},
        
        // 2. Out of range normal index
        {"oob_norm.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//5 2//5 3//5\n"},
        
        // 3. Invalid numeric data (Garbage string)
        {"bad_coords.obj", "v 0.0 apple 1.0\n"},
        
        // 4. Malformed face format (Too many slashes)
        {"bad_face_syntax.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1///1 2///1 3///1\n"},
        
        // 5. Zero index (Invalid in OBJ)
        {"zero_index.obj", "v 0 0 0\nf 0 0 0\n"},

        // 6. Negative index out of bounds (Relative index pointing before start of file)
        {"negative_oob.obj", "v 0 0 0\nf -2 -2 -2\n"},

        // 8. Incomplete vertex (missing Z coordinate)
        {"incomplete_vertex.obj", "v 0.1 0.2\nf 1 1 1\n"},

        // 11. Large index overflow (Testing against string-to-int limits)
        {"overflow_index.obj", "v 0 0 0\nf 9999999999999999999999 1 1\n"},
    };

    for (const auto& [filename, content] : invalidCases) {
        createObjFile(output_folder / filename, content);
        
        // Expecting the loader to validate bounds/syntax and throw
        EXPECT_THROW(loadMesh(output_folder / filename), std::exception)
            << "Failed to throw runtime_error for case: " << filename;

        std::filesystem::remove(output_folder / filename);
    }
}