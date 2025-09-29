cmake_minimum_required(VERSION 3.23)
project(devil_ray)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(glew REQUIRED)

add_executable(${PROJECT_NAME}
               src/main.cpp
               src/Application.cpp
               src/Shaders.cpp
               src/Utils.cpp
               imgui_backend/imgui_impl_glfw.cpp
               imgui_backend/imgui_impl_opengl3.cpp)

target_link_libraries(${PROJECT_NAME}
    glfw
    imgui::imgui
    GLEW::GLEW
    OpenGL
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
