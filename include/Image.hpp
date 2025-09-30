#pragma once

#include "Utils.hpp"

#include <vector>
#include <string>

void savePNG(
    const std::string &fileName,
    const std::vector<uint32_t> &pixelData,
    Size2i resolution
);