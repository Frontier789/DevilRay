#include "Image.hpp"
#include "Renderer.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <set>
#include <map>


struct CudaRandom
{
    curandState *state;

    __device__ float rnd()
    {
        return curand_uniform(state);
    }
};

__global__ void cuda_render(
    Size2i size,
    Vec4 *pixels,
    uint32_t *casts,
    Camera camera,
    PixelSampling pixel_sampling,
    std::span<const TriangleMesh> objects,
    const ObjectsInfo info,
    std::span<const Material> materials,
    std::span<const AliasEntry> light_table,
    DebugOptions debug,
    curandState *randStates)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.width || y >= size.height) return;

    int idx = y * size.width + x;

    auto random = CudaRandom{randStates + idx};

    SampleStats stats{.ray_casts = 0};
    sampleColor(Vec2{x, y}, pixels[idx], stats, camera, pixel_sampling, objects, info, materials, light_table, debug, random);

    casts[idx] += stats.ray_casts;
}

void Renderer::scheduleDeviceRender()
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid;
    dimGrid.x = (m_resolution.width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (m_resolution.height + dimBlock.y - 1) / dimBlock.y;

    static bool printed = false;
    if (!printed) {
        std::cout << "Running grid size " << dimGrid.x << "x" << dimGrid.y << std::endl;
        printed = true;

        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_render);
        std::cout << "suggested minGridSize = " << minGridSize << std::endl;
        std::cout << "suggested blockSize = " << blockSize << std::endl;
    }

    Camera localCamera;
    PixelSampling localPixelSampling;
    DebugOptions localDebug;

    {
        std::scoped_lock guard{m_renderMutex};
        localCamera = m_camera;
        localPixelSampling = m_pixel_sampling;
        localDebug = m_debug;
    }

    const auto objects = std::span{m_scene.objects.devicePtr(), m_scene.objects.size()};
    const auto materials = std::span{m_scene.materials.devicePtr(), m_scene.materials.size()};
    const auto light_table = std::span{m_light_sampler.entries.devicePtr(), m_light_sampler.entries.size()};

    cuda_render<<<dimGrid, dimBlock>>>(
        m_resolution,
        m_buffers.color.devicePtr(),
        m_buffers.casts.devicePtr(),
        localCamera,
        localPixelSampling,
        objects,
        m_scene.info,
        materials,
        light_table,
        localDebug,
        m_cuda_randoms.devicePtr()
    );

    CUDA_ERROR_CHECK();

    cudaDeviceSynchronize();

    CUDA_ERROR_CHECK();
}
