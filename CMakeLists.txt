cmake_minimum_required(VERSION 3.25.2)
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

find_package(stb REQUIRED)

option(BUILD_RENDERER    "Build the renderer application"    ON)
option(BUILD_BBOX_VIEWER "Build the bbox viewer application" ON)
option(BUILD_BENCHMARK   "Build the headless ray-cast benchmark" ON)

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
            src/Renderer.cpp
            src/RunningAverage.cpp
            src/Scene.cpp
            src/GpuTris.cpp
            src/models/Mesh.cpp
            src/models/BBH.cpp
            src/device/implementation.cu
            src/device/DevUtils.cu
            src/device/DeviceMemory.cu
            src/tracing/TriangleMesh.cpp)

target_link_libraries(${PROJECT_NAME}
    stb::stb
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

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
