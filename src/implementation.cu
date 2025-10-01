#include <vector>
#include <iostream>

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/SampleScene.hpp"

#include "IntersectionTests.hpp"

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

__global__ void f(Camera *camera, Vec3 *pts, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        pts[i*N + j] = cameraRay(*camera, Vec2{i, j}).v;
    }
}

void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return;
    }

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "CUDA Device Info:" << std::endl;
    std::cout << "Name: " << deviceProp.name << std::endl;
    std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
}

void test_f()
{
    printCudaDeviceInfo();
    
    const int N = 128;
    
    Vec3 *gpu_data;

    cudaMalloc(&gpu_data, sizeof(*gpu_data) * N * N);

    Camera camera{
        .intrinsics = Intrinsics{
            .focal_length = 8e-3,
            .center = Vec2f{N, N} / 2 * 6e-3f,
        },
        .resolution = Size2i{N, N},
        .physical_pixel_size = Size2f{6e-3, 6e-3},
    };

    Camera *cudaCamera;
    cudaMalloc(&cudaCamera, sizeof(Camera));
    cudaMemcpy(cudaCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x) / threadsPerBlock.x, (N + threadsPerBlock.y) / threadsPerBlock.y);
    f<<<numBlocks, threadsPerBlock>>>(cudaCamera, gpu_data, N);

    std::vector<Vec3> cpu_data(N*N);

    cudaMemcpy(cpu_data.data(), gpu_data, sizeof(cpu_data[0])*N*N, cudaMemcpyDeviceToHost);

    std::cout << "[0]: " << cpu_data[0] << std::endl;
    std::cout << "[1]: " << cpu_data[1] << std::endl;
    std::cout << "[N-1]: " << cpu_data[N-1] << std::endl;
    std::cout << "[10*N]: " << cpu_data[10*N] << std::endl;
    std::cout << "[N*N-1]: " << cpu_data[N*N-1] << std::endl;
}
