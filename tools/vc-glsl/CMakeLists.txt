# No-MLIR compiler driver: .vc -> AST -> GLSL -> glslc -> .spv
add_executable(vc-glsl vc.cpp)

target_include_directories(vc-glsl PRIVATE
  ${VC_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(vc-glsl PRIVATE cxx_std_20)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(vc-glsl PRIVATE -fno-rtti)
endif()

target_link_libraries(vc-glsl PRIVATE
  VCFrontend
  LLVM
)
