cmake_minimum_required(VERSION 3.25.2)

# CUDA toolchain selection. Both variables have to be set before the CUDA
# language is enabled, and either can be overridden on the command line with
# -DCMAKE_CUDA_COMPILER=... / -DCMAKE_CUDA_HOST_COMPILER=... .
#
# CMake's own nvcc discovery can land on a distro-packaged /usr/bin/nvcc that is
# older than the toolkit actually on PATH, so prefer whatever `nvcc` PATH gives.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc)
endif()

project(devil_ray_lib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

# Allow constexpr calls in CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Print compilation notes about kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Save source line info 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(stb REQUIRED)
find_package(exiv2 REQUIRED)

option(BUILD_RENDERER     "Build the renderer application"        ON)
option(BUILD_BBOX_VIEWER  "Build the bbox viewer application"     ON)
option(BUILD_BENCHMARK    "Build the headless ray-cast benchmark" ON)
option(BUILD_SAMPLE_IMAGE "Build the headless image sampler"      ON)

if(BUILD_RENDERER OR BUILD_BBOX_VIEWER)
    find_package(glfw3 REQUIRED)
    find_package(imgui REQUIRED)
    find_package(glew REQUIRED)

    # Build the imgui GLFW+OpenGL3 backends once, from the conan package sources.
    # Both renderer and bbox_viewer link against this instead of keeping local copies.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _IMGUI_BUILD_TYPE)
    if(NOT _IMGUI_BUILD_TYPE)
        set(_IMGUI_BUILD_TYPE "RELEASE")
    endif()
    set(IMGUI_BACKENDS_DIR "${imgui_PACKAGE_FOLDER_${_IMGUI_BUILD_TYPE}}/res/bindings")

    add_library(imgui_backends OBJECT
        ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
        ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp)

    target_include_directories(imgui_backends PUBLIC ${IMGUI_BACKENDS_DIR})
    target_link_libraries(imgui_backends PUBLIC imgui::imgui glfw GLEW::GLEW OpenGL)
endif()

add_library(${PROJECT_NAME}
            src/CameraController.cpp
            src/DistributionSamplers.cpp
            src/Utils.cpp
            src/Image.cpp
            src/ImageMetadata.cpp
            src/Renderer.cpp
            src/RunningAverage.cpp
            src/Scene.cpp
            src/GpuTris.cpp
            src/models/Mesh.cpp
            src/models/BBH.cpp
            src/device/implementation.cu
            src/device/Binning.cu
            src/device/DevUtils.cu
            src/device/DeviceMemory.cu
            src/device/CudaTimer.cu
            src/tracing/TriangleMesh.cpp)

target_link_libraries(${PROJECT_NAME}
    stb::stb
    exiv2::exiv2
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

# Stamp the current git commit/branch into a generated header at build time.
include(cmake/GitVersion.cmake)
target_git_version(${PROJECT_NAME} src/ImageMetadata.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -Wmissing-field-initializers)

enable_testing()
add_subdirectory(test)
if(BUILD_RENDERER)
    add_subdirectory(renderer)
endif()
if(BUILD_BBOX_VIEWER)
    add_subdirectory(bbox_viewer)
endif()
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
if(BUILD_SAMPLE_IMAGE)
    add_subdirectory(sample_image)
endif()
