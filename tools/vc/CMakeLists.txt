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
  LLVMSupport
)
