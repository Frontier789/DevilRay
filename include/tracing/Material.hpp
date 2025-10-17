#pragma once

#include "Utils.hpp"

#include "tracing/Medium.hpp"

#include <variant>

struct BaseMaterial
{
    Vec4 debug_color;
};

struct TransparentMaterial : BaseMaterial
{
    Medium inside_medium;
};

struct DiffuseMaterial : BaseMaterial
{
    Vec4 emission;
    Vec4 diffuse_reflectance;
};

using Material = std::variant<TransparentMaterial, DiffuseMaterial>;

inline constexpr Vec4 getDebugColor(const Material &mat)
{
    return std::visit([](auto &&m){return m.debug_color;}, mat);
}
