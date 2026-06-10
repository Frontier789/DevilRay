#pragma once

#include "Utils.hpp"
#include "Mesh.hpp"

struct BBHNode
{
    AABB box;
    int left_child;
    int right_child;
};

struct BBH
{
    int depth;
    std::vector<BBHNode> nodes;
};

BBH generateSimpleBBH(const Mesh &mesh);
