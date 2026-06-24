cmake_minimum_required(VERSION 3.25.2)
project(devil_ray_bbox_viewer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow constexpr calls in CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Print compilation notes about kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(glew REQUIRED)

add_executable(${PROJECT_NAME}
               src/Application.cpp
               src/Shaders.cpp
               src/UiLayout.cpp
               src/benchmark.cu
               src/BenchmarkGenerator.cpp
               src/main.cpp)

target_link_libraries(${PROJECT_NAME}
    glfw
    imgui::imgui
    GLEW::GLEW
    OpenGL
    devil_ray_lib
    imgui_backends
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
