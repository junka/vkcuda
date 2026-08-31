//===- VCToGPU.cpp - Lower the vc dialect to the GPU dialect -------------===//
//
// Stage-1 of the lowering pipeline: maps the CUDA-shaped surface of the VC
// dialect onto MLIR's GPU dialect so the standard gpu -> spirv conversions
// can take over:
//
//   vc.kernel    -> gpu.module + gpu.func (kernel attr + spirv.entry_point_abi)
//   vc.thread_id -> gpu.thread_id        vc.block_id  -> gpu.block_id
//   vc.block_dim -> gpu.block_dim        vc.grid_dim  -> gpu.grid_dim
//   vc.barrier   -> gpu.barrier
//
// Only one gpu.module is created; every VC kernel is moved inside it as a
// gpu.func with the same name/signature. func.return terminators are rewritten
// to gpu.return.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"

#include "vc/Dialect/VC/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace vc;
using namespace mlir;

namespace {

gpu::Dimension toGpuDim(vc::Dim dim) {
  switch (dim) {
  case vc::Dim::x: return gpu::Dimension::x;
  case vc::Dim::y: return gpu::Dimension::y;
  case vc::Dim::z: return gpu::Dimension::z;
  }
  llvm_unreachable("unknown vc::Dim");
}

// The default target env only enables Shader without any extensions, while
// the Vulkan interface variables we generate for kernel arguments live in the
// StorageBuffer storage class, which SPIR-V 1.0 expresses via the
// SPV_KHR_storage_buffer_storage_class extension. Without it the GPUToSPIRV
// signature conversion cannot map the kernel arguments and memref.load/store
// on them fails to legalize (<UNKNOWN SSA VALUE>).
spirv::TargetEnvAttr getVCTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(triple,
                                   spirv::getDefaultResourceLimits(context));
}

// Rewrites vc.thread_id / block_id / block_dim / grid_dim / barrier.
void rewriteIndexingOps(ModuleOp module, IRRewriter &rw) {
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (isa<vc::ThreadIdOp, vc::BlockIdOp, vc::BlockDimOp, vc::GridDimOp,
            vc::BarrierOp>(op))
      ops.push_back(op);
  });
  for (Operation *op : ops) {
    rw.setInsertionPoint(op);
    if (auto tid = dyn_cast<vc::ThreadIdOp>(op))
      rw.replaceOpWithNewOp<gpu::ThreadIdOp>(op, tid.getType(),
                                             toGpuDim(tid.getDim()));
    else if (auto bid = dyn_cast<vc::BlockIdOp>(op))
      rw.replaceOpWithNewOp<gpu::BlockIdOp>(op, bid.getType(),
                                            toGpuDim(bid.getDim()));
    else if (auto bd = dyn_cast<vc::BlockDimOp>(op))
      rw.replaceOpWithNewOp<gpu::BlockDimOp>(op, bd.getType(),
                                             toGpuDim(bd.getDim()));
    else if (auto gd = dyn_cast<vc::GridDimOp>(op))
      rw.replaceOpWithNewOp<gpu::GridDimOp>(op, gd.getType(),
                                            toGpuDim(gd.getDim()));
    else if (isa<vc::BarrierOp>(op))
      rw.replaceOpWithNewOp<gpu::BarrierOp>(op);
  }
}

// Packs every vc.kernel (and its target func) into one gpu.module as gpu.func.
void packKernels(ModuleOp module, IRRewriter &rw) {
  SmallVector<vc::KernelOp> kernels;
  module.walk([&](vc::KernelOp k) { kernels.push_back(k); });
  if (kernels.empty()) return;

  gpu::GPUModuleOp gpuModule;
  for (vc::KernelOp k : kernels) {
    func::FuncOp fn = module.lookupSymbol<func::FuncOp>(k.getFunction());
    if (!fn) {
      k.emitError("kernel references unknown function '")
          << k.getFunction() << "'";
      continue;
    }

    if (!gpuModule) {
      rw.setInsertionPointToStart(module.getBody());
      gpuModule = rw.create<gpu::GPUModuleOp>(k.getLoc(), "vc_kernels");
    }

    // Each gpu.module carries the SPIR-V target environment; GPUToSPIRV
    // copies it onto the produced spirv.module (used by the serializer).
    gpuModule->setAttr(spirv::getTargetEnvAttrName(),
                       getVCTargetEnv(module.getContext()));

    rw.setInsertionPointToStart(gpuModule.getBody());
    auto gpuFn = rw.create<gpu::GPUFuncOp>(k.getLoc(), fn.getName(),
                                           fn.getFunctionType());
    // Mark the entry point: gpu.kernel + the SPIR-V launch ABI. The entry
    // point ABI needs the three fragment dims; use the launch's block size
    // (32x1x1 default) until host launch parameters are threaded through.
    gpuFn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), rw.getUnitAttr());
    gpuFn->setAttr(spirv::getEntryPointABIAttrName(),
                   spirv::getEntryPointABIAttr(module.getContext(),
                                               {32, 1, 1}));

    // KernelFuncSignatureConversion inside GPUToSPIRV splits each gpu.func
    // argument into a SPIR-V interface variable. Every argument must carry a
    // spirv.interface_var_abi, otherwise its memref value is left unconverted
    // and the body patterns fail on an unknown SSA value. Bindings count up
    // from 0. A memref carries its own storage class in the memref type, so
    // the ABI attr must omit it (the attr cannot specify a storage class on a
    // non-scalar); plain scalar args get an explicit StorageBuffer binding.
    for (auto [i, argType] : llvm::enumerate(fn.getFunctionType().getInputs())) {
      std::optional<spirv::StorageClass> storageClass =
          argType.isa<ShapedType>()
              ? std::optional<spirv::StorageClass>{}
              : spirv::StorageClass::StorageBuffer;
      gpuFn.setArgAttr(
          i, spirv::getInterfaceVarABIAttrName(),
          spirv::getInterfaceVarABIAttr(/*descSet=*/0, /*binding=*/i,
                                        storageClass, module.getContext()));
    }

    // Move the body over, then rewrite func.return -> gpu.return.
    gpuFn.getBody().takeBody(fn.getBody());
    SmallVector<func::ReturnOp> returns;
    gpuFn.walk([&](func::ReturnOp r) { returns.push_back(r); });
    for (func::ReturnOp r : returns) {
      rw.setInsertionPoint(r);
      rw.create<gpu::ReturnOp>(r.getLoc(), r.getOperands());
      r->erase();
    }
    if (gpuFn.getBody().front().empty()) {
      rw.setInsertionPointToEnd(&gpuFn.getBody().front());
      rw.create<gpu::ReturnOp>(k.getLoc());
    }

    fn.erase();
    k.erase();
  }

  // Hoist any module-scope workgroup-memory globals into the gpu.module.
  // GPUToSPIRV only legalizes symbols visible inside the gpu.module it clones,
  // and the stage-1 cleanup erases everything at the outer module scope that is
  // not the gpu.module itself. __shared__ variables are emitted (by ASTToMLIR)
  // as spirv.GlobalVariable in the Workgroup storage class at module scope;
  // moving them here (before the gpu.func) keeps spirv.mlir.addressof
  // references resolvable through Stage 2. (memref.global is hoisted too, in
  // case any remain, for the same reason.)
  SmallVector<Operation *> globals;
  module.walk([&](Operation *op) {
    if (isa<spirv::GlobalVariableOp, memref::GlobalOp>(op))
      globals.push_back(op);
  });
  for (Operation *g : globals) {
    g->moveBefore(&gpuModule.getBody()->front());
  }
}

} // namespace

void vc::codegen::lowerVCToGPU(ModuleOp module) {
  MLIRContext &ctx = *module.getContext();
  ctx.getOrLoadDialect<gpu::GPUDialect>();

  IRRewriter rw(&ctx);
  rewriteIndexingOps(module, rw);
  packKernels(module, rw);
}