#include "Image.hpp"
#include "Renderer.hpp"
#include "device/CudaRng.hpp"

#include "tracing/PathGeneration.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <set>
#include <map>


// Bumps the sample counter (the .w channel of the progressive accumulation
// buffer) once per frame. Radiance is summed into .xyz by the shade kernel
// across every bounce; the sample count must advance exactly once per frame.
__global__ void accumulateSampleCount(Vec4 *colors, int path_count)
{
    int idx = KERNEL_IDX(path_count);
    colors[idx].w += 1;
}

void Renderer::scheduleDeviceRender()
{
    const int path_count = m_resolution.area();

    Camera localCamera;
    PixelSampling localPixelSampling;

    {
        std::scoped_lock guard{m_renderMutex};
        localCamera = m_camera;
        localPixelSampling = m_pixel_sampling;
    }

    const auto objects = std::span{m_scene.objects.devicePtr(), m_scene.objects.size()};
    const auto materials = std::span{m_scene.materials.devicePtr(), m_scene.materials.size()};
    const auto light_table = std::span{m_light_sampler.entries.devicePtr(), m_light_sampler.entries.size()};

    WavefrontDataDevice wavefront{
        .rays = m_wavefront.rays.devicePtr(),
    };
    PathVertexDataDevice vertex{
        .t = m_pathVertices.t.devicePtr(),
        .bsdfPdfPrev = m_pathVertices.bsdfPdfPrev.devicePtr(),
        .throughput = m_pathVertices.throughput.devicePtr(),
        .prevSpecular = m_pathVertices.prevSpecular.devicePtr(),
        .ids = m_pathVertices.ids.devicePtr(),
    };
    RenderBuffersDevice output{
        .colors = m_buffers.color.devicePtr(),
        .resolution = m_resolution,
    };

    constexpr int blockSize = 256;
    const int gridSize = (path_count + blockSize - 1) / blockSize;

    curandState *rand = m_cuda_randoms.devicePtr();

    initPaths<<<gridSize, blockSize>>>(vertex, path_count);
    initCameraRays<<<gridSize, blockSize>>>(wavefront, localPixelSampling, localCamera, rand);

    for (int depth = 0; depth < Buffers::maxPathLength; ++depth)
    {
        extendPaths<<<gridSize, blockSize>>>(wavefront, vertex, objects, path_count);
        shade<<<gridSize, blockSize>>>(vertex, wavefront, rand, objects, materials, light_table, path_count, m_scene.info, output);
        sampleBsdfDirection<<<gridSize, blockSize>>>(vertex, wavefront, rand, objects, materials, path_count);
    }

    accumulateSampleCount<<<gridSize, blockSize>>>(m_buffers.color.devicePtr(), path_count);

    CUDA_ERROR_CHECK();

    cudaDeviceSynchronize();

    CUDA_ERROR_CHECK();
}
