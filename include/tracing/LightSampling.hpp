#pragma once

#include "Utils.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/Material.hpp"

HD float surfaceArea(const TriangleMesh &mesh);

HD Vec4 radiantExitance(const Material &mat);
