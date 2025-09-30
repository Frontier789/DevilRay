#pragma once

#include "Utils.hpp"

struct Intrinsics
{
    float focal_length;
    Vec2f center;
};

struct Camera
{
    Intrinsics intrinsics;
    Size2i resolution;
    Size2f physical_pixel_size;
};

constexpr Ray cameraRay(const Camera &cam, Vec2f pixelCoord)
{
    const auto pixelCenter = pixelCoord + Vec2f{0.5, 0.5};
    const auto physicalPixelCenter = pixelCenter * cam.physical_pixel_size - cam.intrinsics.center;

    const auto dir = physicalPixelCenter / cam.intrinsics.focal_length;
    
    return Ray{
        .p = Vec3{0,0,0},
        .v = Vec3{dir.x, dir.y, 1},
    };
}
