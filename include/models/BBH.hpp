#pragma once

#include "device/Vector.hpp"

#include "Utils.hpp"
#include "Mesh.hpp"

struct BBHNode
{
    AABB box;
    int left_child = -1;
    int right_child = -1;
    int parent_index = -1;

    int tris_begin;
    int tris_end;

    constexpr bool isLeaf() const
    {
        return left_child == -1 && right_child == -1;
    }
};

struct BBHView
{
    std::span<const BBHNode> nodes;

    constexpr bool isEmpty() const { return nodes.size() == 0; }
};

struct BBH
{
    int depth;
    DeviceVector<BBHNode> nodes;

    BBHView view();
    BBHView hostView() const;
};

BBH generateSimpleBBH(Mesh &mesh);
std::vector<BBHNode> getBoxesOnDepth(const BBH &bbh, int depth);
