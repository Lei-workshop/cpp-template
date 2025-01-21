#include <algorithm>
#include <cstdio>
#include <string>

#include <cuda_runtime_api.h>

#include <nccl.h>

#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/span.hpp>
#include <range/v3/view/transform.hpp>

#include <argparse/argparse.hpp>

#include "buffer.hpp"
#include "checker.hpp"
#include "context.hpp"
#include "dtypes.hpp"

template <typename... Args>
auto master_info(Args&&... args) -> void {
  mpi_context::barrier();
  if (mpi_context::rank() == 0) {
    spdlog::info(std::forward<Args>(args)...);
  }
  mpi_context::barrier();
}

auto main(int argc, char* argv[]) -> int {
  argparse::ArgumentParser program("hello");
  program.parse_args(argc, argv);

  auto [rank, world_size] = mpi_context::rank_and_world_size();
  cuda_call<cudaSetDevice>(mpi_context::local_rank());
  master_info("world_size: {}", rank, world_size);
  spdlog::info("rank: {}", rank);

  constexpr int kN = 10;
  auto host_x = HostBuffer(kN, DTypeEnum::kF32);
  std::fill_n(host_x.as<float>(), kN, static_cast<float>(rank));
  auto cuda_x = host_x.to_cuda();

  nccl_call<ncclAllReduce>(cuda_x.data(), cuda_x.data(), kN,
                           cuda_x.dtype().nccl_type(), ncclSum,
                           nccl_context::world_comm(), nccl_context::stream());

  auto host_y = cuda_x.to_host();
  spdlog::info("rank {}: [{}]", rank,
               ranges::span<float>(host_y.as<float>(), kN) |
                   ranges::views::transform(
                       [](auto value) { return std::to_string(value); }) |
                   ranges::views::join(", ") | ranges::to<std::string>());
}
