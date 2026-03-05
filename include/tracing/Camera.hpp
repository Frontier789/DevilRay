#pragma once

#include "Utils.hpp"
#include "models/Matrix.hpp"

struct Intrinsics
{
    float focal_length;
    Vec2f center;
};

struct Camera
{
    mutable Matrix4x4f transform;

    Intrinsics intrinsics;
    Size2i resolution;
    Size2f physical_pixel_size;
};

