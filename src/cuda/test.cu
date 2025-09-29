#include <vector>
#include <iostream>

__global__ void f(float *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        a[i] = i*i + 42;
    }
}

void test_f()
{
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