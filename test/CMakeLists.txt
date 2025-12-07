cmake_minimum_required(VERSION 3.14)
project(devil_ray_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
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
