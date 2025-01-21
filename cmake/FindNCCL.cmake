# derived from cmake/Modules/FindNCCL.cmake

#[=======================================================================[.rst:
FindNCCL
-------

Finds the NCCL library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``nccl::nccl``
  The NCCL library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``NCCL_FOUND``
  True if the system has the NCCL library.
``NCCL_INCLUDE_DIRS``
  Include directories needed to use NCCL.
``NCCL_LIBRARIES``
  Libraries needed to link to NCCL.

#]=======================================================================]

set(NCCL_INCLUDE_DIR
    $ENV{NCCL_INCLUDE_DIR}
    CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR
    $ENV{NCCL_LIB_DIR}
    CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION
    $ENV{NCCL_VERSION}
    CACHE STRING "Version of NCCL to build with")

if($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()
list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(
  NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR})

if(USE_STATIC_NCCL)
  message(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  set(NCCL_LIBNAME "nccl_static")
  if(NCCL_VERSION) # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  set(NCCL_LIBNAME "nccl")
  if(NCCL_VERSION) # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(
  NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND) # obtaining NCCL version and some sanity checks
  set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message(STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")
  file(READ "${NCCL_HEADER_FILE}" NCCL_HEADER_CONTENTS)
  string(FIND "${NCCL_HEADER_CONTENTS}" "NCCL_VERSION_CODE" NCCL_VERSION_CODE_MATCHES)

  if(NOT NCCL_VERSION_DEFINED EQUAL -1)
    set(FILE "${PROJECT_BINARY_DIR}/detect_nccl_version.cu")
    file(
      WRITE ${FILE}
      "
      #include <iostream>
      #include <nccl.h>
      int main()
      {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;

        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
")
    try_run(
      NCCL_VERSION_MATCHED COMPILE_RESULT SOURCES ${FILE}
      RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
      COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${NCCL_INCLUDE_DIRS}" LINK_LIBRARIES ${NCCL_LIBRARIES} WORKING_DIRECTORY
                  ${PROJECT_BINARY_DIR})
    if(NOT COMPILE_RESULT)
      message(FATAL_ERROR "Failed to compile detect_nccl_version.cu: ${COMPILE_OUTPUT}")
    endif()
    if(NOT NCCL_VERSION_MATCHED)
      message(FATAL_ERROR "Found NCCL header version and library version do not match! \
(include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}) Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
    endif()
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  else()
    message(STATUS "NCCL version < 2.3.5-5")
  endif()

  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()

if(NCCL_FOUND AND NOT TARGET nccl::nccl)
  add_library(nccl::nccl UNKNOWN IMPORTED)
  set_target_properties(
    nccl::nccl
    PROPERTIES IMPORTED_LOCATION "${NCCL_LIBRARIES}"
               INTERFACE_COMPILE_DEFINITIONS "${PC_NCCL_CFLAGS_OTHER}"
               INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
               INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}")
endif()
