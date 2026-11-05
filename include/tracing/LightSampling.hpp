#pragma once

#include "Utils.hpp"
#include "tracing/Material.hpp"

#include <variant>

inline HD Vec4 radiantExitance(const Material &mat)
{
    return std::visit(Overloaded{
        [](const TransparentMaterial &) { return Vec4{0,0,0,0}; },
        [](const DiffuseMaterial &mat) { return mat.emission * pi; },
    }, mat);
}
