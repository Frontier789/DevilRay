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

    auto img = Image4f::create_uninitialized(size);
    std::memcpy(img.pixels.hostPtr(), data, size.area() * sizeof(Vec4));
    stbi_image_free(data);

    return img;
}

Image1f intensity(const Image4f &image)
{
    auto intens = Image1f::create_uninitialized(image.size);
    const auto data = image.pixels.hostPtr();
    const auto pixel_count = image.size.area();

    std::transform(data, data + pixel_count, intens.pixels.hostPtr(), luminance);

    return intens;
}
