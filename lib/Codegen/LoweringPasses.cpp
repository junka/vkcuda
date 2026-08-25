//===- LoweringPasses.cpp - VC -> standard/gpu -> SPIR-V pipeline ---------===//
//
// Wires up the lowering pipeline. The first stage (VC -> standard/gpu) is a
// scaffold with a TODO. The second stage reuses MLIR's built-in SPIR-V
// conversions.
//
// NOTE: the exact pass factory names below must be verified against the
// installed MLIR 18 headers once libmlir-18-dev is available. They are
// written to the documented upstream names but have not been compiled yet.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"

#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace vc::codegen {

void runLoweringPipeline(ModuleOp module) {
  MLIRContext &ctx = *module.getContext();
  ctx.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

  PassManager pm(&ctx);

  // ---- Stage 1: VC -> standard/gpu  (TODO scaffold) -------------------
  // A dedicated conversion pass will be added here to lower:
  //   vc.thread_id   -> gpu.thread_id
  //   vc.block_id    -> gpu.block_id
  //   vc.block_dim   -> gpu.block_dim
  //   vc.grid_dim    -> gpu.grid_dim
  //   vc.barrier     -> gpu.barrier
  //   vc.kernel      -> gpu.module + gpu.func (with kernel attr)
  // Until that pass exists the pipeline cannot fully lower real kernels;
  // SPIR-V emission is therefore best-effort for the demo.

  // ---- Stage 2: standard/gpu -> SPIR-V --------------------------------
  // Reuse MLIR's built-in conversions. Pass factory names track upstream
  // MLIR 18; adjust to the installed headers if they differ.
  pm.addPass(createConvertGPUToSPIRVPass());
  pm.addPass(createConvertFuncToSPIRVPass());

  if (failed(pm.run(module))) {
    module.emitError("lowering pipeline failed");
  }
}

} // namespace vc::codegen
