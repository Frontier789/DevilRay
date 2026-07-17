#pragma once

#include "Utils.hpp"
#include "device/Buffer.hpp"

#include <span>

struct TriangleIdentifier
{
    int meshID;
    int triangleID;

    constexpr bool valid() const {return meshID >= 0;}

    static constexpr TriangleIdentifier invalid() {return TriangleIdentifier{ .meshID = -1, .triangleID = -1 };}
};

struct PathVertexData
{
    DeviceBuffer<float> t;
    DeviceBuffer<float> bsdfPdfPrev;
    DeviceBuffer<Vec4> throughput;
    DeviceBuffer<int> prevSpecular;
    DeviceBuffer<TriangleIdentifier> ids;
};

struct PathVertexDataDevice
{
    float *t;
    float *bsdfPdfPrev;
    Vec4 *throughput;
    int *prevSpecular;
    TriangleIdentifier *ids;
};
