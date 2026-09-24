#pragma once

#include "Utils.hpp"

#include <vector>
#include <string>

void savePNG(
    const std::string &fileName,
    const std::vector<uint32_t> &pixelData,
    Size2i resolution
);

struct Image4f
{
    std::vector<Vec4> pixels;

    Size2i size;
};

struct Image1f
{
    std::vector<float> pixels;

    Size2i size;
};

struct ImageView1f
{
    float *pixels;

    Size2i size;
};

Image4f loadHDR(const std::string &fileName);
Image1f intensity(const Image4f &image);
