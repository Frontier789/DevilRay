cmake_minimum_required(VERSION 3.25.2)
project(sample_image LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(stb REQUIRED)

add_executable(${PROJECT_NAME}
               src/main.cpp)

target_link_libraries(${PROJECT_NAME}
    stb::stb
    devil_ray_lib
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
