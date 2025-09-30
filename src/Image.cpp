#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "stb_image.h"
#include "stb_image_write.h"

#include "Image.hpp"

#include <iostream>

void savePNG(
    const std::string &fileName,
    const std::vector<uint32_t> &pixelData,
    Size2i resolution
){
    const auto result = stbi_write_png(fileName.c_str(), resolution.width, resolution.height, 4, pixelData.data(), resolution.width * sizeof(pixelData[0]));

    if (result == 0)
    {
        std::cout << "WARN: Failed to save " << fileName << ": " << stbi_failure_reason() << std::endl;
    }
}