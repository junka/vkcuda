# Vulkan runtime: no MLIR dependency.
add_library(VCRuntime STATIC
  VCRuntime.cpp
  Buffer.cpp
  Pipeline.cpp
)

target_include_directories(VCRuntime PUBLIC
  ${VC_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}   # for RuntimeInternal.h
)

target_compile_features(VCRuntime PUBLIC cxx_std_20)

target_link_libraries(VCRuntime PUBLIC Vulkan::Vulkan)
