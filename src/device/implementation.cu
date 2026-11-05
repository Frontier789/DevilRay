#include "RendererImpl.hpp"
#include "Utils.hpp"
#include "tracing/IntersectionImpl.hpp"
#include "tracing/PathGeneration.hpp"

HD Vec4 checkerPattern(
    const Vec2f &uv,
    const int checker_count,
    const Vec4 dark,
    const Vec4 bright
){
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;

    const float checker = checker_x ^ checker_y;

    return bright * checker + dark * (1-checker);
}
