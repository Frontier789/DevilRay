#pragma once

#include "Utils.hpp"
#include "device/Buffer.hpp"

#include <span>

struct TriangleIdentifier
{
    int meshID;
    int triangleID;
};

struct PathVertexData
{
    DeviceBuffer<float> t;
    DeviceBuffer<float> bsdfPdfPrev;
    DeviceBuffer<Vec4> throughput;
    DeviceBuffer<int> prevSpecular;
    DeviceBuffer<TriangleIdentifier> ids;
    DeviceBuffer<int> alive;
};

struct PathVertexDataDevice
{
    float *t;
    float *bsdfPdfPrev;
    Vec4 *throughput;
    int *prevSpecular;
    TriangleIdentifier *ids;
    int *alive;
};
