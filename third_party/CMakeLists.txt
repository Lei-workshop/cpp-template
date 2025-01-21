include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW) # allow setting options with `set`

FetchContent_Declare(
  gsl-lite
  GIT_REPOSITORY "https://github.com/gsl-lite/gsl-lite"
  GIT_TAG "v0.41.0"
  GIT_SHALLOW ON
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(gsl-lite)

FetchContent_Declare(
  cpptrace_th
  GIT_REPOSITORY "https://github.com/Lei-workshop/cpptrace_th"
  GIT_TAG "v0.1.0"
  GIT_SHALLOW ON
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(cpptrace_th)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY "https://github.com/fmtlib/fmt"
  GIT_TAG "11.0.2"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(fmt)

set(SPDLOG_FMT_EXTERNAL ON)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY "https://github.com/gabime/spdlog"
  GIT_TAG "v1.x"
  GIT_SHALLOW ON
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
  ranges-v3
  GIT_REPOSITORY "https://github.com/ericniebler/range-v3"
  GIT_TAG "0.12.0"
  GIT_SHALLOW ON
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(ranges-v3)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY "https://github.com/p-ranav/argparse"
  GIT_TAG "v3.1"
  GIT_SHALLOW ON
  GIT_PROGRESS TRUE
  SYSTEM)
FetchContent_MakeAvailable(fmt argparse)
