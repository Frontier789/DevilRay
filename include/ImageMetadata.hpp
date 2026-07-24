#pragma once

#include <cstdint>
#include <string>

struct ImageMetadata
{
    uint64_t totalRayCasts = 0;
    double raysPerPixel = 0.0;
    int maxPathDepth = 0;
    std::string pixelSampling;
    std::string outputLinearity;
};

bool writeImageMetadata(const std::string &path, const ImageMetadata &meta);
