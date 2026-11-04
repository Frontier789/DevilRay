#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

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
using ImageView4f = ImageView<Vec4>;

template<typename T>
struct Image
{
    DeviceArray<T> pixels;

    Size2i size;

    T &operator[](Vec2i p) { return pixels.hostPtr()[p.x + size.width * p.y]; }
    const T &operator[](Vec2i p) const { return pixels.hostPtr()[p.x + size.width * p.y]; }

    static Image<T> create(Size2i s, T def_val = T{})
    {
        return Image<T>{
            .pixels = DeviceArray<T>(s.area(), def_val),
            .size = s
        };
    }

    static Image<T> create_uninitialized(Size2i s)
    {
        return Image<T>{
            .pixels = DeviceArray<T>::uninitialized(s.area()),
            .size = s,
        };
    }

    ImageView<T> hostView()
    {
        return ImageView<T>{
            .pixels = pixels.hostPtr(),
            .size = size,
        };
    }

    ImageView<T> view()
    {
        pixels.ensureDeviceAllocation();

        return ImageView<T>{
            .pixels = pixels.devicePtr(),
            .size = size,
        };
    }
};

using Image4f = Image<Vec4>;
using Image1f = Image<float>;


Image4f loadHDR(const std::string &fileName);
Image1f intensity(const Image4f &image);
