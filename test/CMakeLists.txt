cmake_minimum_required(VERSION 3.14)
project(devil_ray_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_test_dr TEST_NAME)
  add_executable(
    ${TEST_NAME}
    src/${TEST_NAME}.cpp
  )
  target_link_libraries(
    ${TEST_NAME}
    GTest::gtest_main
    devil_ray_lib
  )
  gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

# Registers the whole GoogleTest binary as a single CTest entry so its cases run
# in one process. Tests that touch the CUDA runtime then share a single context
# init instead of paying it per case. Sources may be .cpp or .cu, and several can
# share one binary (and so one CUDA context).
function(add_shared_gpu_test_dr TEST_NAME)
  add_executable(
    ${TEST_NAME}
    ${ARGN}
  )
  target_link_libraries(
    ${TEST_NAME}
    GTest::gtest_main
    devil_ray_lib
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

# Fast, CPU-only unit tests (per-case CTest entries).
add_test_dr(test_alias)
add_test_dr(test_intersection)
add_test_dr(test_mesh)
add_test_dr(test_mesh_utils)
add_test_dr(test_matrix)
add_test_dr(test_math)
add_test_dr(test_transform)
add_test_dr(test_bbh)
add_test_dr(test_tracing)
add_test_dr(test_sampling)
add_test_dr(test_lighting)
add_test_dr(test_camera)
add_test_dr(test_scene)
add_test_dr(test_device)
add_test_dr(test_running_average)
add_test_dr(test_image)
add_test_dr(test_camera_controller)

add_test_dr(test_command_line)
target_include_directories(test_command_line PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../sample_image/include)

# GPU tests: one process each, so the CUDA context is initialized only once.
add_shared_gpu_test_dr(test_cuda    src/test_cuda.cu)
add_shared_gpu_test_dr(test_render  src/test_render.cpp src/test_render_accumulation.cpp)
