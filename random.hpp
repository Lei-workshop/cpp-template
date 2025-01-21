#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <variant>
#include <vector>

#include "dtypes.hpp"

namespace random_detail {

using IntUnderlyingType = std::int64_t;
using FloatUnderlyingType = double;

class RandomValueProxy {
 public:
  template <typename T>
  // NOLINTNEXTLINE(google-explicit-constructor)
  RandomValueProxy(T value) {
    if constexpr (std::is_integral_v<T>) {
      value_ = static_cast<IntUnderlyingType>(value);
    } else if constexpr (std::is_floating_point_v<T>) {
      value_ = static_cast<FloatUnderlyingType>(value);
    } else {
      static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                    "T must be an integral or floating-point type");
    }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DEFINE_CAST_TO_TYPE(type, underlying_type)               \
  [[nodiscard]] operator type() const {                          \
    return static_cast<type>(std::get<underlying_type>(value_)); \
  }
  DEFINE_CAST_TO_TYPE(std::int8_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(std::int32_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(std::int64_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(std::uint8_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(std::uint32_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(std::uint64_t, IntUnderlyingType)
  DEFINE_CAST_TO_TYPE(float, FloatUnderlyingType)
  DEFINE_CAST_TO_TYPE(double, FloatUnderlyingType)
#undef DEFINE_CAST_TO_TYPE

 private:
  std::variant<std::int64_t, double> value_;
};

}  // namespace random_detail

inline auto random_host_value(DType dtype) -> random_detail::RandomValueProxy {
  thread_local static std::mt19937 gen(std::random_device{}());
  if (dtype.is_floating_point()) {
    return std::normal_distribution<random_detail::FloatUnderlyingType>{}(gen);
  }
  return std::uniform_int_distribution<random_detail::IntUnderlyingType>{
      std::numeric_limits<random_detail::IntUnderlyingType>::min(),
      std::numeric_limits<random_detail::IntUnderlyingType>::max()}(gen);
}

template <typename T>
auto random_n_parallel(T* first, std::size_t count, DType dtype, int nthreads)
    -> void {
  if (nthreads <= 1) {
    std::generate_n(first, count,
                    [dtype]() -> T { return random_host_value(dtype); });
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  constexpr std::size_t kChunkSize = 1024;
  for (int tid = 0; tid < nthreads; ++tid) {
    threads.emplace_back([tid, first, count, dtype, nthreads, kChunkSize]() {
      for (std::size_t i = tid * kChunkSize; i < count;
           i += nthreads * kChunkSize) {
        std::size_t real_chunk_size = std::min(kChunkSize, count - i);
        std::generate_n(first + i, real_chunk_size,
                        [dtype]() -> T { return random_host_value(dtype); });
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
