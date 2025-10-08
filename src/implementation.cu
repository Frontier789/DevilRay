#include <vector>
#include <iostream>

#include "DeviceUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/SampleScene.hpp"

#include "IntersectionTestsImpl.hpp"
#include "DeviceVectorImpl.hpp"
#include "RendererImpl.hpp"

std::optional<Intersection> cast(const Ray &ray, const std::span<const Object> objects)
{
    std::optional<Intersection> best = std::nullopt;

    for (const auto &obj : objects)
    {
        const auto intersection = testIntersection(ray, obj);

        if (!intersection.has_value()) continue;

        if (!best.has_value() || best->t > intersection->t)
        {
            best = intersection;
        }
    }

    return best;
}

Ray cameraRay(const Camera &cam, Vec2f pixelCoord)
{
    const auto pixelCenter = pixelCoord + Vec2f{0.5, 0.5};
    const auto physicalPixelCenter = pixelCenter * cam.physical_pixel_size - cam.intrinsics.center;

    const auto dir = physicalPixelCenter / cam.intrinsics.focal_length;
    
    return Ray{
        .p = Vec3{0,0,0},
        .v = Vec3{dir.x, dir.y, 1},
    };
}

std::optional<Intersection> testIntersection(const Ray &ray, const Object &object)
{
    return std::visit([&](auto&& o) {return getIntersection(ray, o);}, object);
}

Vec4 checkerPattern(
    const Vec2f &uv, 
    const int checker_count, 
    const Vec4 dark, 
    const Vec4 bright
){
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;

    const float checker = checker_x ^ checker_y;

    return bright * checker + dark * (1-checker);
}


__global__ void initRand(curandState *randStates, int width, int height, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &randStates[idx]);
}

void CudaRandomStates::init()
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid;
    dimGrid.x = (size.width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (size.height + dimBlock.y - 1) / dimBlock.y;

    initRand<<<dimGrid, dimBlock>>>(rand_states, size.width, size.height, 42);
    CUDA_ERROR_CHECK();
}

template struct DeviceVector<Vec4>;
