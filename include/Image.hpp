#pragma once

#include "Utils.hpp"

#include <vector>
#include <string>

void savePNG(
    const std::string &fileName,
    const std::vector<uint32_t> &pixelData,
    Size2i resolution
);

template<typename T>
struct ImageView
{
    T *pixels;

    Size2i size;
};

using ImageView1f = ImageView<float>;

template<typename T>
struct Image
{
    std::vector<T> pixels;

    Size2i size;

    T &operator[](Vec2i p) { return pixels[p.x + size.width * p.y]; }

    static Image<T> create(Size2i s, T def_val = T{})
    {
        return Image<T>{
            .pixels = std::vector<T>(s.area(), def_val),
            .size = s
        };
    }

    ImageView<T> view() {
        return ImageView<T>{
            .pixels = pixels.data(),
            .size = size,
        };
    }
};

using Image4f = Image<Vec4>;
using Image1f = Image<float>;


Image4f loadHDR(const std::string &fileName);
Image1f intensity(const Image4f &image);
