#pragma once

#include <device/Vector.hpp>

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
    DeviceVector<BBHNode> nodes;
};

struct BBHGpuView
{
    std::span<const BBHNode> nodes;
};

BBH generateSimpleBBH(const Mesh &mesh);
BBHGpuView createBBHGpuView(BBH &bbh);
