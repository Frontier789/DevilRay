cmake_minimum_required(VERSION 3.25.2)
project(devil_ray_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow constexpr calls in CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Print compilation notes about kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Headless benchmark: links only the core library, no GLFW/ImGui/OpenGL.
add_executable(${PROJECT_NAME}
               src/benchmark.cu
               src/BenchmarkGenerator.cpp
               src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 89)

target_link_libraries(${PROJECT_NAME}
    devil_ray_lib
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
