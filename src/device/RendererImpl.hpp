#include "Image.hpp"
#include "Renderer.hpp"
#include "device/CudaRng.hpp"

#include "tracing/PathGeneration.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <map>

#include <curand.h>
#include <curand_kernel.h>
#include <cub/device/device_select.cuh>


// Bumps the sample counter (the .w channel of the progressive accumulation
// buffer) once per frame. Radiance is summed into .xyz by the shade kernel
// across every bounce; the sample count must advance exactly once per frame.
__global__ void accumulateSampleCount(Vec4 *colors, int path_count)
{
    int idx = KERNEL_IDX(path_count);
    colors[idx].w += 1;
}

// void checkAliveRatio(int depth, int it, Size2i resolution, const DeviceBuffer<TriangleIdentifier> &gpuIds)
// {
//     const auto ids = gpuIds.toHost();
//     const auto invalids = std::ranges::count_if(ids, [](TriangleIdentifier id){return id.valid();});
//     const auto aliveRatio = static_cast<float>(invalids) / ids.size();

//     std::vector<uint32_t> pixels(ids.size());

//     for (int i=0;i<resolution.area();++i) {
//         Vec4 color = ids[i].valid() ? Vec4{0,0,0,1} : Vec4{1,0,0,1};
//         pixels[i] = color.to8bitColor();
//     }

//     savePNG("dead_paths/dead_at_depth_" + std::to_string(it) + "_" + std::to_string(depth) + ".png", pixels, resolution);

//     std::cout << "At " << depth << " alive: " << static_cast<int>(aliveRatio*1000)/10.0f << "%" << std::endl;
// }

void Renderer::scheduleDeviceRender()
{
    const int path_count = m_resolution.area();

    Camera localCamera;
    PixelSampling localPixelSampling;
    DebugOptions localDebug;

    {
        std::scoped_lock guard{m_renderMutex};
        localCamera = m_camera;
        localPixelSampling = m_pixel_sampling;
        localDebug = m_debug;
    }

    const auto objects = m_scene.objects.deviceSpan();
    const auto materials = m_scene.materials.deviceSpan();
    const auto light_table = m_light_sampler.entries.deviceSpan();

    WavefrontDataDevice wavefront{
        .rays = m_wavefront.rays.devicePtr(),
        .current_mat = m_wavefront.current_mat.devicePtr(),
        .sort_index = m_wavefront.sort_index.devicePtr(),
    };
    PathVertexDataDevice vertex{
        .t = m_pathVertices.t.devicePtr(),
        .bsdfPdfPrev = m_pathVertices.bsdfPdfPrev.devicePtr(),
        .throughput = m_pathVertices.throughput.devicePtr(),
        .prevSpecular = m_pathVertices.prevSpecular.devicePtr(),
        .ids = m_pathVertices.ids.devicePtr(),
        .alive = m_pathVertices.alive.devicePtr(),
    };
    RenderBuffersDevice output{
        .colors = m_buffers.color.devicePtr(),
        .resolution = m_resolution,
    };

    constexpr int blockSize = 256;
    const int gridSize = (path_count + blockSize - 1) / blockSize;

    curandState *rand = m_cuda_randoms.devicePtr();
    uint32_t *casts = m_buffers.casts.devicePtr();
    int *num_selected = m_binning.num_selected.devicePtr();

    if (localDebug != DebugOptions::Off)
    {
        m_binning.num_selected.copyFromHost(std::vector{path_count});

        initPaths<<<gridSize, blockSize>>>(vertex, path_count);
        initCameraRays<<<gridSize, blockSize>>>(wavefront, localPixelSampling, localCamera, rand);
        extendPaths<<<gridSize, blockSize>>>(wavefront, vertex, objects, num_selected, casts, 0);
        debugShade<<<gridSize, blockSize>>>(vertex, wavefront, objects, materials, localDebug, num_selected, output);
    }
    else
    {
        initPaths<<<gridSize, blockSize>>>(vertex, path_count);
        initCameraRays<<<gridSize, blockSize>>>(wavefront, localPixelSampling, localCamera, rand);

        for (int depth = 0; depth < Buffers::maxPathLength; ++depth)
        {
            m_binning.execute(vertex.alive, wavefront.sort_index);

            extendPaths<<<gridSize, blockSize>>>(wavefront, vertex, objects, num_selected, casts, depth);
            shade<<<gridSize, blockSize>>>(vertex, wavefront, rand, objects, materials, light_table, num_selected, m_scene.info, output);
            sampleBsdfDirection<<<gridSize, blockSize>>>(vertex, wavefront, rand, objects, materials, num_selected);
        }
    }

    accumulateSampleCount<<<gridSize, blockSize>>>(m_buffers.color.devicePtr(), path_count);

    CUDA_ERROR_CHECK();

    cudaDeviceSynchronize();

    CUDA_ERROR_CHECK();
}
