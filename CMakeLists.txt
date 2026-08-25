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

# ---------------------------------------------------------------------------
# LLVM / MLIR discovery
#
# LLVM is always needed (frontend uses ADT/SourceMgr/raw_ostream); MLIR
# provides the dialect/codegen/driver. Both are located via their CMake
# CONFIG packages. These are rarely on CMake's default search path because
# Homebrew kegs are not linked into /usr/local and Debian ships versioned
# /usr/lib/llvm-* trees, so we deliberately do not hardcode distro paths.
# Resolution order for the search roots:
#   1. Explicit overrides: -DLLVM_DIR=... and/or -DMLIR_DIR=...
#   2. An external CMAKE_PREFIX_PATH (the standard find_package mechanism)
#   3. Homebrew: `brew --prefix <formula>` (macOS; both /usr/local and
#      /opt/homebrew work) — the formula is `VC_LLVM_FORMULA`, defined only
#      in the APPLE branch, never baked into a versioned path
#   4. `llvm-config --cmakedir` if it is on PATH (Linux/other)
# ---------------------------------------------------------------------------
option(VC_ENABLE_MLIR "Build MLIR dialect/codegen/driver" ON)

if(NOT LLVM_DIR AND NOT MLIR_DIR AND NOT CMAKE_PREFIX_PATH)
  set(_vc_prefix_hint "")
  if(APPLE)
    # Homebrew-only: keg names use the "@" convention (e.g. llvm@18), so the
    # formula variable is scoped to Apple rather than leaking to Linux.
    set(VC_LLVM_FORMULA "llvm@18" CACHE STRING
        "Homebrew formula providing LLVM/MLIR (used as a search hint only)")
    execute_process(COMMAND brew --prefix "${VC_LLVM_FORMULA}"
                    OUTPUT_VARIABLE _vc_prefix_hint
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  else()
    find_program(_VC_LLVM_CONFIG NAMES llvm-config llvm-config-18 llvm-config-17)
    if(_VC_LLVM_CONFIG)
      execute_process(COMMAND "${_VC_LLVM_CONFIG}" --cmakedir
                      OUTPUT_VARIABLE _vc_cmakedir
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      # <...>/lib/cmake/llvm -> install root containing include/ and lib/
      get_filename_component(_vc_prefix_hint "${_vc_cmakedir}/../../.." ABSOLUTE)
    endif()
  endif()
  if(_vc_prefix_hint)
    list(PREPEND CMAKE_PREFIX_PATH "${_vc_prefix_hint}")
    message(STATUS "LLVM/MLIR prefix hint: ${_vc_prefix_hint}")
  endif()
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
# Provide tablegen()/add_public_tablegen_target() used by MLIR's AddMLIR.cmake.
include(${LLVM_CMAKE_DIR}/TableGen.cmake)
include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)

if(VC_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "MLIR includes: ${MLIR_INCLUDE_DIRS}")
  # Provides mlir_tablegen()/add_mlir_library() used by lib/Dialect/VC.
  include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
else()
  message(STATUS "MLIR disabled: skipping dialect/codegen/driver")
endif()

# Common include path for all VC targets.
set(VC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

enable_testing()

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(tools/vc-dump-ast)
add_subdirectory(tools/vc-glsl)
add_subdirectory(tools/vcc)
if(VC_ENABLE_MLIR)
  add_subdirectory(tools)
endif()

# ---------------------------------------------------------------------------
# Frontend regression tests
#
# `cmake --build build --target vc-check` runs every test/Frontend/*.vc
# through the vcc driver and validates its diagnostics with FileCheck
# (provided by the same LLVM install used for the build). Uses the plain
# python3 interpreter (stdlib only) as a tiny stand-in for lit.
# ---------------------------------------------------------------------------
find_package(Python3)
# FileCheck usually sits next to the LLVM install that provided LLVM_DIR
# (<prefix>/lib/cmake/llvm -> <prefix>/bin); fall back to the auto-detected
# prefix when the build was configured with explicitly-passed -DLLVM_DIR.
set(_vc_filecheck_hint "")
if(LLVM_DIR)
  # <prefix>/lib/cmake/llvm -> <prefix>/bin
  get_filename_component(_vc_filecheck_hint "${LLVM_DIR}/../../../bin"
                         ABSOLUTE)
elseif(_vc_prefix_hint)
  set(_vc_filecheck_hint "${_vc_prefix_hint}/bin")
endif()
find_program(VC_FILECHECK NAMES FileCheck HINTS "${_vc_filecheck_hint}"
             NO_CACHE)
message(STATUS "vc-check probe: python=${Python3_Interpreter_FOUND} "
               "filecheck='${VC_FILECHECK}' hint='${_vc_filecheck_hint}'")
if(Python3_Interpreter_FOUND AND VC_FILECHECK AND TARGET vcc)
  add_custom_target(vc-check
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/run_tests.py
            --vcc $<TARGET_FILE:vcc>
            --filecheck ${VC_FILECHECK}
            ${CMAKE_SOURCE_DIR}/test/Frontend
    DEPENDS vcc
    COMMENT "Running frontend regression tests (test/Frontend)"
    USES_TERMINAL)
  message(STATUS "vc-check: FileCheck=${VC_FILECHECK}")
else()
  message(STATUS "vc-check disabled (FileCheck or python3 not found)")
endif()
