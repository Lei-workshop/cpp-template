#pragma once

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <driver_types.h>

#include <mpi.h>

#include <nccl.h>

#include <fmt/format.h>

#include <spdlog/spdlog.h>

template <typename... Args>
[[noreturn]] auto fail_fast(Args&&... args) -> void {
  spdlog::error(std::forward<Args>(args)...);
  std::terminate();
}

template <auto cudaFunc, typename... Args>
auto cuda_call(Args&&... args) -> void {
  auto rc = cudaFunc(std::forward<Args>(args)...);
  if (rc != cudaSuccess) {
    fail_fast("CUDA Error: {}", cudaGetErrorString(rc));
  }
}

template <auto MPI_Func, typename... Args>
auto mpi_call(Args&&... args) -> void {
  auto rc = MPI_Func(std::forward<Args>(args)...);
  if (rc != MPI_SUCCESS) {
    std::vector<char> error_string(MPI_MAX_ERROR_STRING);
    int length = 0;
    MPI_Error_string(rc, error_string.data(), &length);
    fail_fast("MPI Error: {}", std::string(error_string.data(), length));
  }
}

template <auto ncclFunc, typename... Args>
auto nccl_call(Args&&... args) -> void {
  auto rc = ncclFunc(std::forward<Args>(args)...);
  if (rc != ncclSuccess) {
    fail_fast("NCCL Error: {}", ncclGetErrorString(rc));
  }
}

template <typename... Args>
auto check_or_fail(bool condition, Args&&... args) -> void {
  if (!condition) {
    fail_fast("Check failed: {}", fmt::format(std::forward<Args>(args)...));
  }
}
