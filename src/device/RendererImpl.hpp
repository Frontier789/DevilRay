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

__global__ void cuda_render(Size2i size, Vec4 *pixels, uint32_t *casts, Camera camera, PixelSampling pixel_sampling, std::span<Object> objects, std::span<Material> materials, bool debug, curandState *randStates)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.width || y >= size.height) return;

    int idx = y * size.width + x;


    const int max_depth = debug ? 1 : 10;
    const auto iterations = debug ? 1 : 100;
    
    auto &pix = pixels[idx];


    for (int i=0;i<iterations;++i) {
        auto random = CudaRandom{randStates + idx};
        const auto ray = cameraRay(camera, Vec2{x, y}, pixel_sampling, pix.w, random);
        const auto sample = sampleColor(ray, max_depth, objects, materials, debug, random);
        
        pix.w++;
        pix = pix + sample.color;
        casts[idx] += sample.casts;
    }
}

void Renderer::schedule_device_render()
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid;
    dimGrid.x = (resolution.width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (resolution.height + dimBlock.y - 1) / dimBlock.y;

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

    const auto objects = std::span{scene.objects.devicePtr(), scene.objects.size()};
    const auto materials = std::span{scene.materials.devicePtr(), scene.materials.size()};

    cuda_render<<<dimGrid, dimBlock>>>(resolution, outputs.color.devicePtr(), outputs.casts.devicePtr(), camera, pixel_sampling, objects, materials, debug, cuda_randoms.ptr());
    CUDA_ERROR_CHECK();

    cudaDeviceSynchronize();
}
