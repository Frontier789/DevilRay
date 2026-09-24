#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "stb_image.h"
#include "stb_image_write.h"

#include "Image.hpp"

#include <iostream>
#include <cstring>

void savePNG(
    const std::string &file_name,
    const std::vector<uint32_t> &pixel_data,
    Size2i resolution
){
    const auto result = stbi_write_png(file_name.c_str(), resolution.width, resolution.height, 4, pixel_data.data(), resolution.width * sizeof(pixel_data[0]));

    if (result == 0)
    {
        std::cout << "WARN: Failed to save " << file_name << ": " << stbi_failure_reason() << std::endl;
    }
}

Image4f loadHDR(const std::string &file_name)
{
    Size2i size;
    int components = 0;

    const auto data = stbi_loadf(file_name.c_str(), &size.width, &size.height, &components, STBI_rgb_alpha);

    if (!data)
    {
        const auto reason = std::string(stbi_failure_reason());
        throw std::runtime_error("Failed to load " + file_name + ". Reason: " + reason);
    }

    std::vector<Vec4> pixels(size.area());
    std::memcpy(pixels.data(), data, size.area() * sizeof(Vec4));
    stbi_image_free(data);

    return Image4f{
        .pixels = std::move(pixels),
        .size = size,
    };
}

Image1f intensity(const Image4f &image)
{
    std::vector<float> intens;

    std::transform(image.pixels.begin(), image.pixels.end(), std::back_inserter(intens), luminance);

    return Image1f{
        .pixels = std::move(intens),
        .size = image.size,
    };
}
