cmake_minimum_required(VERSION 3.14)
project(devil_ray_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

function(add_bench BENCH_NAME)
  add_executable(
    ${BENCH_NAME}
    src/${BENCH_NAME}.cu
  )
  set_target_properties(${BENCH_NAME} PROPERTIES CUDA_ARCHITECTURES 89)
  target_link_libraries(
    ${BENCH_NAME}
    devil_ray_lib
  )
endfunction()

add_bench(bench_tris)
