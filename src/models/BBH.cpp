#include "models/BBH.hpp"

#include <span>
#include <iostream>

namespace
{
    AABB findBoundingBox(const std::span<const Triangle> triangles, const Mesh &mesh)
    {
        AABB bbox = AABB::empty();

        for (const auto &tri : triangles)
        {
            for (const Vertex &v : {tri.a, tri.b, tri.c})
                bbox = bbox.extend(mesh.points[v.pi]);
        }

        // const auto eps = std::numeric_limits<float>::epsilon();
        // bbox.min = bbox.min - Vec3(eps, eps, eps);
        // bbox.max = bbox.max + Vec3(eps, eps, eps);

        return bbox;
    }

    float maxExtentAlong(const Triangle &tri, const Vec3 &direction, const Mesh &mesh)
    {
        const auto A = mesh.points[tri.a.pi];
        const auto B = mesh.points[tri.b.pi];
        const auto C = mesh.points[tri.c.pi];

        const auto A_extent = A.dot(direction);
        const auto B_extent = B.dot(direction);
        const auto C_extent = C.dot(direction);

        return std::max(A_extent, std::max(B_extent, C_extent));
    }

    int generateBBHLayer(std::vector<BBHNode> &nodes,
                         std::vector<Triangle> &triangles,
                         int tris_begin, int tris_end,
                         int depth, int parent_index,
                         const Mesh &mesh)
    {
        if (tris_end - tris_begin < 1) return -1;

        const auto trianglesInBox = std::span{triangles}.subspan(tris_begin, tris_end - tris_begin);
        const auto node_bbox = findBoundingBox(trianglesInBox, mesh);
        const auto node_index = static_cast<int>(nodes.size());

        nodes.push_back(BBHNode{
            .box = node_bbox,
            .parent_index = parent_index,
            .tris_begin = tris_begin,
            .tris_end = tris_end,
        });

        if (tris_end - tris_begin > 1)
        {
            const auto direction = Vec3(depth%3 == 0, depth%3 == 1, depth%3 == 2);

            std::sort(triangles.begin() + tris_begin, triangles.begin() + tris_end, [direction, &mesh](const Triangle &tri1, const Triangle &tri2){
                return maxExtentAlong(tri1, direction, mesh) < maxExtentAlong(tri2, direction, mesh);
            });

            const auto tris_mid = (tris_begin + tris_end) / 2;

            if (tris_mid - tris_begin >= 1) {
                const auto left_child_index = generateBBHLayer(nodes, triangles, tris_begin, tris_mid, depth+1, node_index, mesh);
                nodes[node_index].left_child = left_child_index;
            }

            if (tris_end - tris_mid >= 1) {
                const auto right_child_index = generateBBHLayer(nodes, triangles, tris_mid, tris_end, depth+1, node_index, mesh);
                nodes[node_index].right_child = right_child_index;
            }
        }

        return node_index;
    }

    int findBBHDepth(const BBHNode &node, const std::vector<BBHNode> &all_nodes)
    {
        int depth = 1;

        if (node.left_child != -1) {
            depth = std::max(depth,
                1+findBBHDepth(all_nodes[node.left_child], all_nodes)
            );
        }

        if (node.right_child != -1) {
            depth = std::max(depth,
                1+findBBHDepth(all_nodes[node.right_child], all_nodes)
            );
        }

        return depth;
    }
}

BBH generateSimpleBBH(Mesh &mesh)
{
    std::cout << "TRACE: generateSimpleBBH for '" << mesh.name << "'" << std::endl;

    std::vector<BBHNode> nodes;

    auto &sorted_triangles = mesh.triangles;

    generateBBHLayer(nodes, sorted_triangles, 0, sorted_triangles.size(), 0, -1, mesh);

    return BBH{
        .depth = findBBHDepth(nodes[0], nodes),
        .nodes = DeviceVector{std::move(nodes)},
    };
}

BBHGpuView createBBHGpuView(BBH &bbh)
{
    bbh.nodes.ensureDeviceAllocation();

    return BBHGpuView{
        .nodes = bbh.nodes.deviceSpan()
    };
}

namespace
{
    void extractBoxes(std::vector<BBHNode> &nodes, const BBH &bbh, int current_index, int depth)
    {
        const auto &node = bbh.nodes.hostPtr()[current_index];

        if (depth == 0)
        {
            nodes.push_back(node);
            return;
        }

        if (node.left_child  != -1) extractBoxes(nodes, bbh, node.left_child,  depth-1);
        if (node.right_child != -1) extractBoxes(nodes, bbh, node.right_child, depth-1);
    }
}

std::vector<BBHNode> getBoxesOnDepth(const BBH &bbh, int depth)
{
    std::vector<BBHNode> nodes;

    extractBoxes(nodes, bbh, 0, depth);

    return nodes;
}
