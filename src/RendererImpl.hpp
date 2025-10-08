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

__global__ void cuda_render(Size2i size, Vec4 *pixels, Camera camera, Object *objects, size_t object_count, bool debug, curandState *randStates)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.width || y >= size.height) return;

    int idx = y * size.width + x;


    const int max_depth = debug ? 1 : 5;
    const auto iterations = debug ? 1 : 100;
    
    auto &pix = pixels[idx];
    pix.w += iterations;

    auto random = CudaRandom{randStates + idx};
    const auto ray = cameraRay(camera, Vec2{x, y});
    const auto sample = sampleColor(ray, max_depth, std::span{objects, object_count}, debug, iterations, random);

    pix = pix + sample.color;
}

const Material *getMaterial(const Object &obj)
{
    return std::visit([&](auto&& o) {return o.mat;}, obj);
}

void setMaterial(Object &obj, Material *material)
{
    std::visit([&](auto&& o) {o.mat = material;}, obj);
}

struct GpuData
{
    Object *objects;
    Material *materials;
};

GpuData copyToGpu(const std::vector<Object> &objects)
{
    std::vector<Material> materials;

    std::map<const Material*, int> mats;
    for (const auto &o : objects) {
        const auto m = getMaterial(o);
        mats[m] = 0;
    }

    for (auto &it : mats)
    {
        it.second = materials.size();
        materials.push_back(*it.first);
    }

    Material *materials_gpu;
    cudaMalloc(&materials_gpu, sizeof(*materials_gpu) * materials.size());
    cudaMemcpy(materials_gpu, materials.data(), sizeof(*materials_gpu) * materials.size(), cudaMemcpyHostToDevice);

    std::vector<Object> objects_relinked = objects;
    for (auto &o : objects_relinked) {
        setMaterial(o, materials_gpu + mats[getMaterial(o)]);
    }

    Object *objects_gpu;
    cudaMalloc(&objects_gpu, sizeof(*objects_gpu) * objects_relinked.size());
    cudaMemcpy(objects_gpu, objects_relinked.data(), sizeof(*objects_gpu) * objects_relinked.size(), cudaMemcpyHostToDevice);


    return GpuData{
        .objects = objects_gpu,
        .materials = materials_gpu,
    };
}

void Renderer::schedule_device_render()
{
    dim3 dimBlock(29, 29);
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

    auto data = copyToGpu(objects);

    cuda_render<<<dimGrid, dimBlock>>>(resolution, accumulator.devicePtr(), camera, data.objects, objects.size(), debug, cuda_randoms.ptr());
    CUDA_ERROR_CHECK();
}
