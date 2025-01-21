cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(hello CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(third_party)

add_executable(hello main.cpp)
target_link_libraries(
  hello
  gsl::gsl-lite-v1
  fmt::fmt
  range-v3::range-v3
  spdlog::spdlog
  argparse::argparse
  cpptrace_th::static)
