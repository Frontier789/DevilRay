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

enable_testing()
include(GoogleTest)

add_executable(
  test_alias
  src/test_alias.cpp
)
target_link_libraries(
  test_alias
  GTest::gtest_main
  devil_ray_lib
)
gtest_discover_tests(test_alias)
