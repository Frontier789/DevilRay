#pragma once

#include "Utils.hpp"
#include "tracing/PixelSampling.hpp"

template<typename Rng>
HD Ray cameraRay(
    const Camera &cam,
    Vec2f pixelCoord,
    PixelSampling sampling,
    int index,
    Rng &rng)
{
    Vec2f subPixelCoord;

    if (sampling == PixelSampling::Center) {
        subPixelCoord = Vec2{0.5, 0.5};
    }
    if (sampling == PixelSampling::UniformRandom) {
        subPixelCoord = Vec2{rng.rnd(), rng.rnd()};
    }

    const auto pixelCenter = pixelCoord + subPixelCoord;
    const auto physicalPixelCenter = pixelCenter * cam.physical_pixel_size.toVec() - cam.intrinsics.center;

    const auto dir = physicalPixelCenter / cam.intrinsics.focal_length;
    
    const auto v = Vec3{dir.x, dir.y, 1}.normalized();

    return Ray{
        .p = cam.transform.getOffset(),
        .v = cam.transform.applyToDirection(v),
    };
}
