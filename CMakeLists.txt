# VC — a CUDA-like language targeting Vulkan via MLIR/SPIR-V
cmake_minimum_required(VERSION 3.20)

project(vc LANGUAGES CXX C VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LLVM/MLIR upstream are built with RTTI disabled; match that to link cleanly
# against the Ubuntu libmlir-18 / libLLVM-18 packages.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Default to position-independent code (needed for shared runtime).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan REQUIRED)

message(STATUS "Vulkan found: ${Vulkan_FOUND}")

# LLVM is always needed (frontend uses ADT/SourceMgr/raw_ostream). MLIR
# additionally provides the dialect/codegen/driver. We locate LLVM via
# its CMake package; it ships with llvm-18-dev (already installed).
set(LLVM_DIR "/usr/lib/llvm-18/lib/cmake/llvm" CACHE PATH "LLVM CMake dir")
find_package(LLVM REQUIRED CONFIG)
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

option(VC_ENABLE_MLIR "Build MLIR dialect/codegen/driver (needs libmlir-18-dev)" ON)
if(VC_ENABLE_MLIR)
  set(MLIR_DIR "/usr/lib/llvm-18/lib/cmake/mlir" CACHE PATH "MLIR CMake dir")
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "MLIR includes: ${MLIR_INCLUDE_DIRS}")
else()
  message(STATUS "MLIR disabled: skipping dialect/codegen/driver")
endif()

# Common include path for all VC targets.
set(VC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

enable_testing()

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(tools/vc-dump-ast)
if(VC_ENABLE_MLIR)
  add_subdirectory(tools)
endif()
