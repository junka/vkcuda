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

#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <cstdlib>
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace vc::codegen {

// Rewrite atomics that GPUToSPIRV cannot lower directly. MLIR 18's GPUToSPIRV
// has no lowering for memref.atomic_rmw 'assign' (atomicExch), so ASTToMLIR
// emits atomicExch on SSBO/global as a 'addi' memref.atomic_rmw carrying
// {vc.atomic_kind = "exch"}. GPUToSPIRV legalizes that to spirv.AtomicIAdd
// (dropping the marker attr), so we cannot find it by attribute afterwards.
//
// Strategy: before conversion, record per-function the ordinal positions of
// the marked atomics among ALL memref.atomic_rmw in that function. After
// conversion, the produced spirv.AtomicIAdd ops appear in the same relative
// order (only memref.atomic_rmw lowers to spirv.AtomicIAdd; arith.addi lowers
// to spirv.IAdd), so the k-th spirv.AtomicIAdd in a function corresponds to
// the k-th source memref.atomic_rmw. Turn each recorded position into a
// spirv.AtomicExchange (true swap semantics: same ptr/scope/semantics/value).
static llvm::StringMap<llvm::SmallVector<std::pair<unsigned, StringRef>>>
collectMarkedAtomicPositions(ModuleOp module) {
  // Keyed by the gpu.func symbol name; positions are indices among ALL
  // atomic ops (memref.atomic_rmw AND spirv.AtomicIAdd, since __shared__
  // atomics are emitted directly as spirv.AtomicIAdd at the gpu.func layer)
  // in source order. The post-conversion spirv.AtomicIAdd list has the same
  // relative order (only these two op kinds lower to spirv.AtomicIAdd;
  // arith.addi lowers to spirv.IAdd), so the ordinal identifies the op.
  // The second element is the marker kind ("exch" or "xor").
  llvm::StringMap<llvm::SmallVector<std::pair<unsigned, StringRef>>> out;
  module.walk([&](gpu::GPUFuncOp fn) {
    unsigned idx = 0;
    fn.walk([&](Operation *op) {
      bool isAtomic =
          isa<memref::AtomicRMWOp>(op) || isa<spirv::AtomicIAddOp>(op);
      if (!isAtomic)
        return;
      if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op)) {
        auto kind = rmw->getAttrOfType<StringAttr>("vc.atomic_kind");
        if (kind && (kind.getValue() == "exch" || kind.getValue() == "xor"))
          out[fn.getName()].push_back({idx, kind.getValue()});
      }
      ++idx;
    });
  });
  return out;
}

static void rewriteMarkedAtomics(
    ModuleOp module,
    const llvm::StringMap<llvm::SmallVector<std::pair<unsigned, StringRef>>>
        &positions) {
  module.walk([&](spirv::FuncOp fn) {
    auto it = positions.find(fn.getName());
    if (it == positions.end())
      return;
    const auto &marks = it->second;
    unsigned idx = 0;
    SmallVector<spirv::AtomicIAddOp> toConvert;
    fn.walk([&](spirv::AtomicIAddOp op) {
      if (marks.size() == toConvert.size())
        return;
      if (marks[toConvert.size()].first == idx)
        toConvert.push_back(op);
      ++idx;
    });
    for (unsigned i = 0; i < toConvert.size(); ++i) {
      spirv::AtomicIAddOp op = toConvert[i];
      StringRef kind = marks[i].second;
      OpBuilder b(op);
      Value result;
      if (kind == "xor")
        result = b.create<spirv::AtomicXorOp>(
            op.getLoc(), op.getType(), op.getPointer(), op.getMemoryScope(),
            op.getSemantics(), op.getValue());
      else // "exch"
        result = b.create<spirv::AtomicExchangeOp>(
            op.getLoc(), op.getType(), op.getPointer(), op.getMemoryScope(),
            op.getSemantics(), op.getValue());
      op.replaceAllUsesWith(result);
      op.erase();
    }
  });
}


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
  //
  // __shared__ workgroup memory is already emitted as spirv.GlobalVariable +
  // spirv.mlir.addressof + spirv.Load/Store by ASTToMLIR (see
  // getOrCreateSharedGlobal), so it needs no extra pass here; the
  // spirv.GlobalVariable hoisted into gpu.module is carried into the
  // spirv.module by the clone.
  //
  // ConvertFuncToSPIRV runs first to turn __device__ helper func.func (hoisted
  // into gpu.module by VCToGPU) + their func.call sites into spirv.func +
  // spirv.FunctionCall, which GPUToSPIRV then carries into the spirv.module.
  // Before conversion, record where the atomicExch markers are (see
  // rewriteMarkedAtomics): GPUToSPIRV drops the marker attr when it builds the
  // spirv.AtomicIAdd, so we match by ordinal position instead.
  auto markedPositions = collectMarkedAtomicPositions(module);

  pm.addNestedPass<gpu::GPUModuleOp>(createConvertFuncToSPIRVPass());
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

  // Post-conversion rewrite of atomics GPUToSPIRV could not lower directly
  // (atomicExch on SSBO/global, emitted as a marked AtomicIAdd).
  rewriteMarkedAtomics(module, markedPositions);

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