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

HD Ray cameraRay(const Camera &cam, Vec2f pixelCoord);
