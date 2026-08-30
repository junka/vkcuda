# Codegen depends on MLIR (IR + conversion passes) and the VC dialect.
add_library(VCCodegen STATIC
  ASTToMLIR.cpp
  LoweringPasses.cpp
  VCToGPU.cpp
)

target_include_directories(VCCodegen PUBLIC
  ${VC_INCLUDE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(VCCodegen PUBLIC cxx_std_20)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(VCCodegen PUBLIC -fno-rtti)
endif()

target_link_libraries(VCCodegen PUBLIC
  VCDialect
  VCFrontend
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRGPUToSPIRV
  MLIRFuncToSPIRV
  MLIRSCFToControlFlow
  MLIRArithToSPIRV
  MLIRMemRefToSPIRV
  MLIRControlFlowToSPIRV
  MLIRPass
  MLIRTransforms
)
