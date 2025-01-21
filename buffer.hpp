#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "checker.hpp"
#include "dtypes.hpp"
#include "random.hpp"

template <auto ctor, auto dtor>
class Buffer {
 public:
  Buffer(std::size_t size, DType dtype)
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      : data_(ctor(size * dtype.size())), size_(size), dtype_(dtype) {
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
  }

  [[nodiscard]] auto data() -> void* { return data_.get(); }
  [[nodiscard]] auto data() const -> const void* { return data_.get(); }
  [[nodiscard]] auto size() const -> std::size_t { return size_; }
  [[nodiscard]] auto size_bytes() const -> std::size_t {
    return size_ * dtype_.size();
  }
  [[nodiscard]] auto dtype() const -> DType { return dtype_; }
  template <typename U>
  [[nodiscard]] auto as() -> U* {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<U*>(this->data());
  }
  template <typename U>
  [[nodiscard]] auto as() const -> const U* {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const U*>(this->data());
  }

 private:
  struct Deleter {
    auto operator()(void* ptr) const noexcept -> void { dtor(ptr); }
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<void, Deleter> data_;
  std::size_t size_;
  DType dtype_;
};

using HostBufferBase = Buffer<std::malloc, std::free>;

class HostBuffer : public HostBufferBase {
 public:
  HostBuffer(std::size_t size, DType dtype) : HostBufferBase(size, dtype) {}
  static auto random(std::size_t size, DType dtype) -> HostBuffer {
    auto buffer = HostBuffer(size, dtype);
    return buffer;
    switch (dtype.type()) {
// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define RANDOM_CASE(dtype, ctype)                                     \
  case DTypeEnum::dtype:                                              \
    random_n_parallel(buffer.as<ctype>(), size, DTypeEnum::dtype, 1); \
    break;
      RANDOM_CASE(kI8, std::int8_t)
      RANDOM_CASE(kI32, std::int32_t)
      RANDOM_CASE(kI64, std::int64_t)
      RANDOM_CASE(kU8, std::uint8_t)
      RANDOM_CASE(kU32, std::uint32_t)
      RANDOM_CASE(kU64, std::uint64_t)
      RANDOM_CASE(kF32, float)
      RANDOM_CASE(kF64, double)
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#undef RANDOM_CASE
      default:
        fail_fast("Unsupported DType");
    }
    return buffer;
  }
  [[nodiscard]] auto to_cuda() const -> class CudaBuffer;
  auto from_cuda(const CudaBuffer& cuda_buffer) -> void;
};

auto cuda_malloc(std::size_t size) -> void* {
  void* ptr = nullptr;
  cuda_call<cudaMalloc<void>>(&ptr, size);
  return ptr;
}

using CudaBufferBase = Buffer<cuda_malloc, cudaFree>;

class CudaBuffer : public CudaBufferBase {
 public:
  CudaBuffer(std::size_t size, DType dtype) : CudaBufferBase(size, dtype) {}
  [[nodiscard]] auto to_host() const -> HostBuffer {
    HostBuffer host_buffer(this->size(), this->dtype());
    cuda_call<cudaMemcpy>(host_buffer.data(), this->data(), this->size_bytes(),
                          cudaMemcpyDeviceToHost);
    return host_buffer;
  }
  auto from_host(const HostBuffer& host_buffer) -> void {
    check_or_fail(this->size() == host_buffer.size(), "Size mismatch");
    check_or_fail(this->dtype() == host_buffer.dtype(), "DType mismatch");
    cuda_call<cudaMemcpy>(this->data(), host_buffer.data(), this->size_bytes(),
                          cudaMemcpyHostToDevice);
  }
};

inline auto HostBuffer::to_cuda() const -> CudaBuffer {
  CudaBuffer cuda_buffer(this->size(), this->dtype());
  cuda_call<cudaMemcpy>(cuda_buffer.data(), this->data(), this->size_bytes(),
                        cudaMemcpyHostToDevice);
  return cuda_buffer;
}
inline auto HostBuffer::from_cuda(const CudaBuffer& cuda_buffer) -> void {
  check_or_fail(this->size() == cuda_buffer.size(), "Size mismatch");
  check_or_fail(this->dtype() == cuda_buffer.dtype(), "DType mismatch");
  cuda_call<cudaMemcpy>(this->data(), cuda_buffer.data(), this->size_bytes(),
                        cudaMemcpyDeviceToHost);
}

inline auto all_close(const HostBuffer& buf1, const HostBuffer& buf2,
                      double atol, double rtol) -> bool {
  check_or_fail(buf1.size() == buf2.size(), "Size mismatch");
  check_or_fail(buf1.dtype() == buf2.dtype(), "DType mismatch");
  switch (buf1.dtype().type()) {
// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-pointer-arithmetic)
#define ALL_CLOSE_CASE_INT(dtype, ctype)                                \
  case DTypeEnum::dtype:                                                \
    return std::equal(buf1.as<ctype>(), buf1.as<ctype>() + buf1.size(), \
                      buf2.as<ctype>());
#define ALL_CLOSE_CASE_FLOAT(dtype, ctype)                                 \
  case DTypeEnum::dtype:                                                   \
    return std::equal(buf1.as<ctype>(), buf1.as<ctype>() + buf1.size(),    \
                      buf2.as<ctype>(), [atol, rtol](ctype x1, ctype x2) { \
                        return std::abs(x1 - x2) <=                        \
                               (atol + rtol * std::abs(x1));               \
                      });
    ALL_CLOSE_CASE_INT(kI8, std::int8_t)
    ALL_CLOSE_CASE_INT(kI32, std::int32_t)
    ALL_CLOSE_CASE_INT(kI64, std::int64_t)
    ALL_CLOSE_CASE_INT(kU8, std::uint8_t)
    ALL_CLOSE_CASE_INT(kU32, std::uint32_t)
    ALL_CLOSE_CASE_INT(kU64, std::uint64_t)
    ALL_CLOSE_CASE_FLOAT(kF32, float)
    ALL_CLOSE_CASE_FLOAT(kF64, double)
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-pointer-arithmetic)
#undef ALL_CLOSE_CASE_INT
#undef ALL_CLOSE_CASE_FLOAT
    default:
      fail_fast("Unsupported DType");
  }
}
