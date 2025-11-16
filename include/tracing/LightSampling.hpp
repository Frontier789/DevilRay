#pragma once

#include "Utils.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Material.hpp"

HD float surfaceArea(const Object &object);

HD Vec4 radiantExitance(const Material &mat);
