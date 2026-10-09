#pragma once

#include "Utils.hpp"
#include "device/Buffer.hpp"

#include <span>

struct TriangleIdentifier
{
    int meshID;
    int triangleID;
};

struct PathVertexDataView
{
    float *t;
    float *bsdfPdfPrev;
    Vec4 *throughput;
    int *prevSpecular;
    TriangleIdentifier *ids;
    int *alive;
};

struct PathVertexData
{
    DeviceBuffer<float> t;
    DeviceBuffer<float> bsdfPdfPrev;
    DeviceBuffer<Vec4> throughput;
    DeviceBuffer<int> prevSpecular;
    DeviceBuffer<TriangleIdentifier> ids;
    DeviceBuffer<int> alive;

    PathVertexDataView view()
    {
        return PathVertexDataView{
            .t = t.devicePtr(),
            .bsdfPdfPrev = bsdfPdfPrev.devicePtr(),
            .throughput = throughput.devicePtr(),
            .prevSpecular = prevSpecular.devicePtr(),
            .ids = ids.devicePtr(),
            .alive = alive.devicePtr(),
        };
    }
};
