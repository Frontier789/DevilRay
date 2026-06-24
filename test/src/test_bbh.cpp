// AI-generated tests (Claude), reviewed by hand before committing.

#include "models/BBH.hpp"
#include "models/Mesh.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <span>
#include <vector>

namespace
{
    // A row of small, well-separated triangles along the X axis, so the median
    // split (which cycles X, Y, Z by depth) has an unambiguous ordering.
    Mesh spreadTriangleMesh(int count)
    {
        Mesh mesh;
        mesh.name = "spread";
        mesh.normals = {Vec3{0, 0, 1}};

        for (int i = 0; i < count; ++i)
        {
            const float x = static_cast<float>(i) * 10.0f;
            const uint32_t base = static_cast<uint32_t>(mesh.points.size());
            mesh.points.push_back(Vec3{x, 0, 0});
            mesh.points.push_back(Vec3{x + 1, 0, 0});
            mesh.points.push_back(Vec3{x, 1, 0});
            mesh.triangles.push_back(Triangle{.a = {base, 0}, .b = {base + 1, 0}, .c = {base + 2, 0}});
        }

        return mesh;
    }

    std::span<const BBHNode> hostNodes(const BBH &bbh)
    {
        return std::span<const BBHNode>{bbh.nodes.hostPtr(), bbh.nodes.size()};
    }

    int recomputeDepth(std::span<const BBHNode> nodes, int index)
    {
        const auto &node = nodes[index];
        int depth = 1;
        if (node.left_child != -1)
            depth = std::max(depth, 1 + recomputeDepth(nodes, node.left_child));
        if (node.right_child != -1)
            depth = std::max(depth, 1 + recomputeDepth(nodes, node.right_child));
        return depth;
    }

    void collectSubtreeIndices(std::span<const BBHNode> nodes, int index, std::vector<int> &out)
    {
        out.push_back(index);
        const auto &node = nodes[index];
        if (node.left_child != -1)
            collectSubtreeIndices(nodes, node.left_child, out);
        if (node.right_child != -1)
            collectSubtreeIndices(nodes, node.right_child, out);
    }

    float maxXExtent(const Mesh &mesh, const Triangle &tri)
    {
        return std::max({mesh.points[tri.a.pi].x, mesh.points[tri.b.pi].x, mesh.points[tri.c.pi].x});
    }
}

TEST(BBHTest, RootCoversAllTriangles)
{
    Mesh mesh = spreadTriangleMesh(8);
    const int total = static_cast<int>(mesh.triangles.size());

    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    ASSERT_FALSE(nodes.empty());
    EXPECT_EQ(nodes[0].tris_begin, 0);
    EXPECT_EQ(nodes[0].tris_end, total);
}

TEST(BBHTest, LeafRangesPartitionAllTriangles)
{
    Mesh mesh = spreadTriangleMesh(13);
    const int total = static_cast<int>(mesh.triangles.size());

    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    std::vector<std::pair<int, int>> leafRanges;
    for (const auto &node : nodes)
    {
        if (node.isLeaf())
        {
            EXPECT_LT(node.tris_begin, node.tris_end) << "Leaf must contain at least one triangle";
            leafRanges.emplace_back(node.tris_begin, node.tris_end);
        }
    }

    std::sort(leafRanges.begin(), leafRanges.end());

    int expectedBegin = 0;
    for (const auto &[begin, end] : leafRanges)
    {
        EXPECT_EQ(begin, expectedBegin) << "Leaf ranges must tile [0, N) without gaps or overlap";
        expectedBegin = end;
    }
    EXPECT_EQ(expectedBegin, total);
}

TEST(BBHTest, DepthFieldMatchesStructuralDepth)
{
    for (int count : {1, 2, 4, 7, 16, 31})
    {
        Mesh mesh = spreadTriangleMesh(count);
        const BBH bbh = generateSimpleBBH(mesh);
        EXPECT_EQ(bbh.depth, recomputeDepth(hostNodes(bbh), 0)) << "count=" << count;
    }
}

TEST(BBHTest, RootSplitsAlongXAxisByMedian)
{
    Mesh mesh = spreadTriangleMesh(8);
    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    ASSERT_NE(nodes[0].left_child, -1);
    ASSERT_NE(nodes[0].right_child, -1);

    const auto &left = nodes[nodes[0].left_child];
    const auto &right = nodes[nodes[0].right_child];

    float maxLeft = -1e30f;
    for (int i = left.tris_begin; i < left.tris_end; ++i)
        maxLeft = std::max(maxLeft, maxXExtent(mesh, mesh.triangles[i]));

    float minRight = 1e30f;
    for (int i = right.tris_begin; i < right.tris_end; ++i)
        minRight = std::min(minRight, maxXExtent(mesh, mesh.triangles[i]));

    EXPECT_LE(maxLeft, minRight) << "Left subtree must sort before the right along the split axis";
}

TEST(BBHTest, SkipIndexAlwaysAdvances)
{
    Mesh mesh = spreadTriangleMesh(16);
    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
        EXPECT_GT(nodes[i].skip_index, i) << "node " << i;

    EXPECT_EQ(nodes[0].skip_index, static_cast<int>(nodes.size()))
        << "Root skip target must be the end of the array";
}

TEST(BBHTest, LeafSkipsToNextNode)
{
    Mesh mesh = spreadTriangleMesh(16);
    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
        if (nodes[i].isLeaf())
            EXPECT_EQ(nodes[i].skip_index, i + 1) << "Leaf " << i << " must skip to its immediate successor";
}

TEST(BBHTest, SkipIndexSpansContiguousSubtree)
{
    Mesh mesh = spreadTriangleMesh(16);
    const BBH bbh = generateSimpleBBH(mesh);
    const auto nodes = hostNodes(bbh);

    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
    {
        std::vector<int> subtree;
        collectSubtreeIndices(nodes, i, subtree);

        const int maxIndex = *std::max_element(subtree.begin(), subtree.end());
        const int minIndex = *std::min_element(subtree.begin(), subtree.end());

        EXPECT_EQ(minIndex, i) << "Subtree of " << i << " must start at the node itself";
        EXPECT_EQ(maxIndex + 1, nodes[i].skip_index) << "Skip must point just past the subtree of " << i;
        EXPECT_EQ(static_cast<int>(subtree.size()), nodes[i].skip_index - i)
            << "Subtree indices must be contiguous";
    }
}

TEST(BBHTest, IsLeafReflectsChildren)
{
    BBHNode leaf{};
    leaf.left_child = -1;
    leaf.right_child = -1;
    EXPECT_TRUE(leaf.isLeaf());

    BBHNode internal{};
    internal.left_child = 1;
    internal.right_child = -1;
    EXPECT_FALSE(internal.isLeaf());
}

TEST(BBHTest, GetBoxesOnDepthReturnsLevelCounts)
{
    Mesh mesh = spreadTriangleMesh(4);
    const BBH bbh = generateSimpleBBH(mesh);

    const auto level0 = getBoxesOnDepth(bbh, 0);
    ASSERT_EQ(level0.size(), 1u);
    EXPECT_EQ(level0[0].tris_begin, 0);
    EXPECT_EQ(level0[0].tris_end, 4);

    EXPECT_EQ(getBoxesOnDepth(bbh, 1).size(), 2u);
    EXPECT_EQ(getBoxesOnDepth(bbh, 2).size(), 4u);
}

TEST(BBHTest, GetBoxesOnDepthZeroIsRootBox)
{
    Mesh mesh = spreadTriangleMesh(8);
    const BBH bbh = generateSimpleBBH(mesh);

    const auto root = getBoxesOnDepth(bbh, 0);
    ASSERT_EQ(root.size(), 1u);

    const auto &rootBox = bbh.nodes.hostPtr()[0].box;
    EXPECT_FLOAT_EQ(root[0].box.min.x, rootBox.min.x);
    EXPECT_FLOAT_EQ(root[0].box.max.x, rootBox.max.x);
}
