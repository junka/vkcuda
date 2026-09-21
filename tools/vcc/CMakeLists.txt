# vcc: single-file CUDA-style compiler driver (no MLIR).
# Compiles a `.vc` file containing both __global__ kernels and host main() with
# kernel<<<g,b>>>(args) launches into a standalone executable, by emitting
# device SPIR-V (via the GLSL backend + glslc) and host C++ (embedding the
# SPIR-V) and driving g++ to link against libVCRuntime + Vulkan.

add_executable(vcc vcc.cpp)

target_include_directories(vcc PRIVATE
  ${VC_INCLUDE_DIR}
)
# LLVM headers are upstream; SYSTEM suppresses their build warnings.
target_include_directories(vcc SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(vcc PRIVATE cxx_std_20)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(vcc PRIVATE -fno-rtti)
endif()

target_link_libraries(vcc PRIVATE VCFrontend LLVM)

# Bake the locations the driver needs at build time so it can invoke g++ with
# the right -I / .a / -l flags without requiring an install step.
#   VC_INCLUDE_DIR  - absolute path to the vc/ headers root
#   VC_RUNTIME_LIB  - absolute path to libVCRuntime.a
#   VC_VULKAN_LIBS  - linker flags to pull in Vulkan (VCRuntime needs it)
# With -DVC_USE_INSTALL_PATHS=ON these are install-relative ($<INSTALL_PREFIX>
# based) so a packaged vcc finds the headers + .a under the install prefix.
target_compile_definitions(vcc PRIVATE
  VC_INCLUDE_DIR="${VC_DRIVER_INCLUDE_DIR}"
  VC_RUNTIME_LIB="${VC_DRIVER_RUNTIME_LIB}"
  VC_VULKAN_LIBS="${VC_DRIVER_VULKAN_LIBS}"
)
