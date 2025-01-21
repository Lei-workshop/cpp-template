cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(hello CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(NCCL REQUIRED)

add_subdirectory(third_party)

add_executable(hello main.cpp)
target_link_libraries(
  hello
  MPI::MPI_CXX
  nccl::nccl
  CUDA::cudart
  gsl::gsl-lite-v1
  fmt::fmt
  range-v3::range-v3
  spdlog::spdlog
  argparse::argparse
  cpptrace_th::static)
