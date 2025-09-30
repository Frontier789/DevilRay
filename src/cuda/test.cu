#include <vector>
#include <iostream>

__global__ void f(float *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        a[i] = i*i + 42;
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
    
    const int N = 100;
    
    float *gpu_data;

    cudaMalloc(&gpu_data, sizeof(float) * N);

    dim3 threadsPerBlock(16);
    dim3 numBlocks((N + threadsPerBlock.x) / threadsPerBlock.x);
    f<<<numBlocks, threadsPerBlock>>>(gpu_data, N);

    std::vector<float> cpu_data(N);

    cudaMemcpy(cpu_data.data(), gpu_data, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (float f : cpu_data)
    {
        std::cout << f << " ";
    }
    std::cout << std::endl;
}