//===- LoweringPasses.cpp - VC -> standard/gpu -> SPIR-V pipeline ---------===//
//
// Wires up the lowering pipeline:
//   Stage 1: VC -> GPU dialect   (lib/Codegen/VCToGPU.cpp)
//   Stage 2: GPU/standard -> SPIR-V (MLIR's built-in conversion passes)
//
// Stage 2 is intentionally one pass: the GPU-to-SPIR-V conversion itself
// clones each gpu.module and fully converts the clone down to spirv.module
// with the SCF/arith/memref/func patterns already baked in. It leaves the
// original gpu.module in place (it only exists for live gpu.launch_func
// hosts, of which a .spv output has none), so we drop it afterwards together
// with any residual host code and keep just the spirv.module.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <cstdlib>
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace vc::codegen {

void runLoweringPipeline(ModuleOp module) {
  MLIRContext &ctx = *module.getContext();
  ctx.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

  PassManager pm(&ctx);

  // ---- Stage 1: VC -> gpu -------------------------------------------------
  lowerVCToGPU(module);

  // SPIR-V modules carry device code only. Drop everything at module scope
  // that is not the gpu.module (host main, runtime API stubs, ...); the
  // gpu.module holding the kernels is the unit of translation.
  SmallVector<Operation *> nonDevice;
  module.walk([&](Operation *op) {
    if (op->getParentOp() == module.getOperation() &&
        !isa<gpu::GPUModuleOp>(op))
      nonDevice.push_back(op);
  });
  for (Operation *op : nonDevice)
    op->erase();

  if (std::getenv("VC_SPIRV_DEBUG")) {
    llvm::errs() << "=== after lowerVCToGPU ===\n";
    module->print(llvm::errs(), OpPrintingFlags().assumeVerified());
    llvm::errs() << "\n";
  }

  // ---- Stage 2: gpu/standard -> SPIR-V -----------------------------------
  // ConvertGPUToSPIRV clones every gpu.module and fully legalizes the clone
  // to a spirv.module (builtin-variable loads for thread/block ids, control
  // barrier, and the whole arith/scf/memref body via its bundled patterns).
  // spirv-lower-abi-attrs then materializes the spirv.entry_point_abi /
  // spirv.interface_var_abi attributes into SPIR-V globals and the entry
  // point operation.
  pm.addPass(createConvertGPUToSPIRVPass());
  // spirv-lower-abi-attrs runs on spirv.module (the op produced by the pass
  // above), not on the top-level module.
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  // The serializer requires spirv.module to carry a concrete vce_triple;
  // deduce it from the ops actually present in the module.
  pm.addNestedPass<spirv::ModuleOp>(
      spirv::createSPIRVUpdateVCEPass());

  if (failed(pm.run(module))) {
    module.emitError("lowering pipeline failed");
  }

  // Keep only the spirv.module: erase the original gpu.module (clone source)
  // and anything else left at module scope.
  SmallVector<Operation *> nonSpirv;
  module.walk([&](Operation *op) {
    if (op->getParentOp() == module.getOperation() &&
        !isa<spirv::ModuleOp>(op))
      nonSpirv.push_back(op);
  });
  for (Operation *op : nonSpirv)
    op->erase();
}

} // namespace vc::codegen