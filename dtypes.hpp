#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <nccl.h>

#include <fmt/ranges.h>

#include "checker.hpp"

enum class DTypeEnum : std::uint8_t {
  kI8 = 0,
  kI32,
  kI64,
  kU8,
  kU32,
  kU64,
  kF32,
  kF64,
  kTotalNumber,
};

template <typename E>
constexpr auto to_underlying(E en) noexcept -> std::underlying_type_t<E> {
  static_assert(std::is_enum_v<E>, "E must be an enum type");
  return static_cast<std::underlying_type_t<E>>(en);
}

struct DTypeInfo {
  std::size_t size;
  MPI_Datatype mpi_type;
  ncclDataType_t nccl_type;
  std::string str;
};
using DTypeMap = std::array<DTypeInfo, to_underlying(DTypeEnum::kTotalNumber)>;
const DTypeMap kDtypeMap = [] {
  DTypeMap dmap{};
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ADD_DTYPE(dtype, ctype, mpi_type, nccl_type, str) \
  dmap[to_underlying(dtype)] = {sizeof(ctype), mpi_type, nccl_type, str}
  ADD_DTYPE(DTypeEnum::kI8, std::int8_t, MPI_INT8_T, ncclInt8, "i8");
  ADD_DTYPE(DTypeEnum::kI32, std::int32_t, MPI_INT32_T, ncclInt32, "i32");
  ADD_DTYPE(DTypeEnum::kI64, std::int64_t, MPI_INT64_T, ncclInt64, "i64");
  ADD_DTYPE(DTypeEnum::kU8, std::uint8_t, MPI_UINT8_T, ncclUint8, "u8");
  ADD_DTYPE(DTypeEnum::kU32, std::uint32_t, MPI_UINT32_T, ncclUint32, "u32");
  ADD_DTYPE(DTypeEnum::kU64, std::uint64_t, MPI_UINT64_T, ncclUint64, "u64");
  ADD_DTYPE(DTypeEnum::kF32, float, MPI_FLOAT, ncclFloat32, "f32");
  ADD_DTYPE(DTypeEnum::kF64, double, MPI_DOUBLE, ncclFloat64, "f64");
#undef ADD_DTYPE
  return dmap;
}();
const std::vector<std::string> kDtypeStrs = [] {
  std::vector<std::string> dtype_strs(kDtypeMap.size());
  std::transform(kDtypeMap.begin(), kDtypeMap.end(), dtype_strs.begin(),
                 [](const DTypeInfo& info) { return info.str; });
  return dtype_strs;
}();

class DType {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  DType(DTypeEnum type) : type_(type) {
    check_or_fail(
        to_underlying(type) >= 0 &&
            to_underlying(type) < to_underlying(DTypeEnum::kTotalNumber),
        "Invalid DType (available: {})", fmt::join(kDtypeStrs, ", "));
  }
  explicit DType(std::string_view str) {
    if (str == "i8") {
      type_ = DTypeEnum::kI8;
    } else if (str == "i32") {
      type_ = DTypeEnum::kI32;
    } else if (str == "i64") {
      type_ = DTypeEnum::kI64;
    } else if (str == "u8") {
      type_ = DTypeEnum::kU8;
    } else if (str == "u32") {
      type_ = DTypeEnum::kU32;
    } else if (str == "u64") {
      type_ = DTypeEnum::kU64;
    } else if (str == "f32") {
      type_ = DTypeEnum::kF32;
    } else if (str == "f64") {
      type_ = DTypeEnum::kF64;
    } else {
      fail_fast("Invalid DType (available: {})", fmt::join(kDtypeStrs, ", "));
    }
  }

  [[nodiscard]] auto type() const -> DTypeEnum { return type_; }
  [[nodiscard]] auto size() const -> std::size_t {
    return kDtypeMap.at(to_underlying(type_)).size;
  }
  [[nodiscard]] auto mpi_type() const -> MPI_Datatype {
    return kDtypeMap.at(to_underlying(type_)).mpi_type;
  }
  [[nodiscard]] auto nccl_type() const -> ncclDataType_t {
    return kDtypeMap.at(to_underlying(type_)).nccl_type;
  }
  [[nodiscard]] auto str() const -> const std::string& {
    return kDtypeMap.at(to_underlying(type_)).str;
  }

  [[nodiscard]] auto operator==(const DType& other) const -> bool {
    return type_ == other.type_;
  }
  [[nodiscard]] auto operator!=(const DType& other) const -> bool {
    return type_ != other.type_;
  }

  [[nodiscard]] auto is_floating_point() const -> bool {
    return type_ == DTypeEnum::kF32 || type_ == DTypeEnum::kF64;
  }

 private:
  DTypeEnum type_;
};
