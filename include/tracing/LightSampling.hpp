#pragma once

#include "Utils.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/Material.hpp"

inline HD Vec4 radiantExitanceImpl(const TransparentMaterial &mat)
{
    return Vec4{0,0,0,0};
}

inline HD Vec4 radiantExitanceImpl(const DiffuseMaterial &mat)
{
    return mat.emission * pi;
}

inline HD Vec4 radiantExitance(const Material &mat)
{
    return std::visit([](auto &&o){return radiantExitanceImpl(o);}, mat);
}
