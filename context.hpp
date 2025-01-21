#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <mpi.h>

#include <nccl.h>

#include "checker.hpp"

namespace cuda_helper {

class CudaStream {
 public:
  CudaStream() {
    cudaStream_t stream = nullptr;
    cuda_call<cudaStreamCreate>(&stream);
    stream_.reset(stream);
  }
  [[nodiscard]] auto get() const -> cudaStream_t { return stream_.get(); }

 private:
  class StreamDeleter {
   public:
    auto operator()(cudaStream_t stream) const noexcept -> void {
      cuda_call<cudaStreamDestroy>(stream);
    }
  };
  std::unique_ptr<CUstream_st, StreamDeleter> stream_;
};

}  // namespace cuda_helper

namespace mpi_context {

class MpiGuard {
 public:
  static auto instance() -> MpiGuard& {
    static MpiGuard instance;
    return instance;
  }
  ~MpiGuard() { mpi_call<MPI_Finalize>(); }
  MpiGuard(const MpiGuard&) = delete;
  MpiGuard(MpiGuard&&) = delete;
  auto operator=(const MpiGuard&) -> MpiGuard& = delete;
  auto operator=(MpiGuard&&) -> MpiGuard& = delete;

 private:
  MpiGuard() { mpi_call<MPI_Init>(nullptr, nullptr); }
};

auto rank_and_world_size() -> std::pair<int, int> {
  [[maybe_unused]] static decltype(auto) context = MpiGuard::instance();
  static auto rank_and_world_size = [] {
    int rank = 0;
    int world_size = 0;
    mpi_call<MPI_Comm_rank>(MPI_COMM_WORLD, &rank);
    mpi_call<MPI_Comm_size>(MPI_COMM_WORLD, &world_size);
    return std::pair{rank, world_size};
  }();
  return rank_and_world_size;
}

auto rank() -> int { return rank_and_world_size().first; }

auto world_size() -> int { return rank_and_world_size().second; }

auto local_rank() -> int {
  static int local_rank = [] {
    auto ompi_local_rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (ompi_local_rank != nullptr) {
      return std::stoi(ompi_local_rank);
    }
    auto slurm_local_id = std::getenv("SLURM_LOCALID");
    if (slurm_local_id != nullptr) {
      return std::stoi(slurm_local_id);
    }
    return rank();
  }();
  return local_rank;
}

auto barrier() -> void { mpi_call<MPI_Barrier>(MPI_COMM_WORLD); }

}  // namespace mpi_context

namespace nccl_context {

auto stream() -> cudaStream_t {
  static cuda_helper::CudaStream stream;
  return stream.get();
}

class NcclComm {
 public:
  static auto instance() -> const NcclComm& {
    static NcclComm instance;
    return instance;
  }
  [[nodiscard]] auto get() const -> ncclComm_t { return comm_.get(); }

 private:
  NcclComm() {
    ncclUniqueId id;
    if (mpi_context::rank() == 0) {
      nccl_call<ncclGetUniqueId>(&id);
    }
    mpi_call<MPI_Bcast>(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    cuda_call<cudaSetDevice>(mpi_context::local_rank());
    ncclComm_t comm = nullptr;
    nccl_call<ncclCommInitRank>(&comm, mpi_context::world_size(), id,
                                mpi_context::rank());
    comm_.reset(comm);
  }
  class CommDeleter {
   public:
    auto operator()(ncclComm_t comm) const noexcept -> void {
      nccl_call<ncclCommDestroy>(comm);
    }
  };
  std::unique_ptr<ncclComm, CommDeleter> comm_;
};

auto world_comm() -> ncclComm_t { return NcclComm::instance().get(); }

auto sync() -> void { cuda_call<cudaStreamSynchronize>(stream()); }

}  // namespace nccl_context
