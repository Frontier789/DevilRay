#pragma once

#include <device/Vector.hpp>

#include "Utils.hpp"
#include "Mesh.hpp"

struct BBHNode
{
    AABB box;
    int left_child = -1;
    int right_child = -1;
    int skip_index = -1;

    int tris_begin;
    int tris_end;

    constexpr bool isLeaf() const
    {
        return left_child == -1 && right_child == -1;
    }
};

struct BBH
{
    int depth;
    DeviceVector<BBHNode> nodes;
};

struct BBHGpuView
{
    std::span<const BBHNode> nodes;

    constexpr bool isEmpty() const { return nodes.size() == 0; }
};

BBH generateSimpleBBH(Mesh &mesh);
BBHGpuView createBBHGpuView(BBH &bbh);
std::vector<BBHNode> getBoxesOnDepth(const BBH &bbh, int depth);
