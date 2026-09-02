# The vc compiler driver.
add_executable(vc vc.cpp)

target_include_directories(vc PRIVATE
  ${VC_INCLUDE_DIR}
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
# time, not by the vc binary.
file(TO_CMAKE_PATH "${VC_INCLUDE_DIR}" VC_INCLUDE_DIR_ABS)
set(VC_RUNTIME_LIB_PATH "$<TARGET_FILE:VCRuntime>")
set(VC_VULKAN_LIB_FLAG "-lvulkan")

target_compile_definitions(vc PRIVATE
  VC_INCLUDE_DIR="${VC_INCLUDE_DIR_ABS}"
  VC_RUNTIME_LIB="${VC_RUNTIME_LIB_PATH}"
  VC_VULKAN_LIBS="${VC_VULKAN_LIB_FLAG}"
)
