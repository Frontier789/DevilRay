cmake_minimum_required(VERSION 3.25.2)
project(devil_ray_lib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow constexpr calls in CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Print compilation notes about kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

find_package(stb REQUIRED)

add_library(${PROJECT_NAME}
            src/DistributionSamplers.cpp
            src/Utils.cpp
            src/Image.cpp
            src/Renderer.cpp
            src/Scene.cpp
            src/device/implementation.cu
            src/device/DevUtils.cu
            src/device/Array.cu
            src/device/Vector.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 89)

target_link_libraries(${PROJECT_NAME}
    stb::stb
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

add_subdirectory(renderer)
