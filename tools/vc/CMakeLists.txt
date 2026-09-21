# The vc compiler driver.
add_executable(vc vc.cpp)

target_include_directories(vc PRIVATE
  ${VC_INCLUDE_DIR}
)
# MLIR/LLVM headers are upstream; SYSTEM suppresses their build warnings
# (e.g. the C++20 TypeRange.h ambiguity note under GCC 13).
target_include_directories(vc SYSTEM PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(vc PRIVATE cxx_std_20)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(vc PRIVATE -fno-rtti)
endif()

target_link_libraries(vc PRIVATE
  VCCodegen
  VCFrontend
  VCDialect
  MLIRIR
  MLIRSPIRVTarget
  MLIRTranslateLib
  MLIRSupport
  LLVM
)

# Bake the locations -emit=full needs to invoke g++ with the right -I / .a /
# -l flags (mirrors tools/vcc/CMakeLists.txt). VCCodegen/VCFrontend already
# bring in the headers; VCRuntime itself is linked by g++ at executable link
# time, not by the vc binary. With -DVC_USE_INSTALL_PATHS=ON these are
# install-relative so a packaged vc finds them under the install prefix.
target_compile_definitions(vc PRIVATE
  VC_INCLUDE_DIR="${VC_DRIVER_INCLUDE_DIR}"
  VC_RUNTIME_LIB="${VC_DRIVER_RUNTIME_LIB}"
  VC_VULKAN_LIBS="${VC_DRIVER_VULKAN_LIBS}"
)
