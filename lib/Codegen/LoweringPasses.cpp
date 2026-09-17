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
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

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


// MLIR's MemRef→SPIRV storage-buffer type converter widens every sub-4-byte
// SSBO element to 4 bytes (f16→f32, i8→i32) with stride=4. For ordinary
// scalar loads that is patched up with extend/truncate, but a
// spirv.KHR.CooperativeMatrixLoad takes the SSBO pointer directly: it reads
// 4-byte elements into an f16 cooperative matrix, reinterpreting each float's
// 4 bytes as two halfs — silently wrong (spirv-val does not catch it; the
// pointer element type is not constrained to match the matrix element type).
//
// This post-conversion pass narrows the SSBO back to f16 for exactly the
// globals a cooperative-matrix f16 load/store touches. It does NOT touch the
// generic widening for scalar f16/i8 SSBO access (a separate, larger fix) —
// only the cooperative-matrix path, where there is no extend/truncate to
// rescue it and the tile layout must be contiguous f16.
//
// For each f16-target global, rebuild the dependent op chain with f16 types:
//   spirv.GlobalVariable  rtarray<f32,stride=4> -> rtarray<f16,stride=2>
//   spirv.mlir.addressof  ptr<struct<(rtarray<f32,4>)>> -> ...f16...
//   spirv.AccessChain     ptr<f32> -> ptr<f16>
// The spirv.KHR.CooperativeMatrixLoad/Store pointer operands are repointed to
// the rebuilt AccessChain results by RAUW — their result coopmatrix type is
// unchanged (f16 in, f16 out), and the pointer operand is an untyped Value
// slot, so the narrower pointer type is accepted as-is.
static void fixupF16StorageBuffers(ModuleOp module) {
  module.walk([&](spirv::ModuleOp spvModule) {
    // Collect the symbol names of SSBO globals that an f16 cooperative-matrix
    // load/store reaches. Trace: KHR.{Load,Store}.pointer -> AccessChain.base
    // -> AddressOf.variable. (Only the cooperative-matrix path needs this;
    // scalar spirv.Load/Store of f16 SSBOs are left to the generic widening
    // and would need extend/truncate insertion, which is out of scope here.)
    DenseSet<StringRef> f16Globals;
    spvModule.walk([&](Operation *op) {
      Value ptr;
      mlir::Type coopElemTy;
      if (auto load = dyn_cast<spirv::KHRCooperativeMatrixLoadOp>(op)) {
        coopElemTy = cast<spirv::CooperativeMatrixType>(load.getResult().getType())
                         .getElementType();
        ptr = load.getPointer();
      } else if (auto store = dyn_cast<spirv::KHRCooperativeMatrixStoreOp>(op)) {
        coopElemTy = cast<spirv::CooperativeMatrixType>(store.getObject().getType())
                         .getElementType();
        ptr = store.getPointer();
      } else {
        return;
      }
      if (!coopElemTy.isF16())
        return;
      // Trace pointer -> AccessChain -> AddressOf -> global symbol.
      auto access = ptr.getDefiningOp<spirv::AccessChainOp>();
      if (!access) return;
      auto addr = access.getBasePtr().getDefiningOp<spirv::AddressOfOp>();
      if (!addr) return;
      f16Globals.insert(addr.getVariable());
    });
    if (f16Globals.empty()) return;

    // Build the narrowed storage-buffer type for one f16 SSBO global:
    // !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>
    MLIRContext *ctx = spvModule->getContext();
    auto f16Ty = Float16Type::get(ctx);
    auto rtArr = spirv::RuntimeArrayType::get(f16Ty, /*stride=*/2);
    auto structTy = spirv::StructType::get({rtArr}, {0u});
    auto ptrTy = spirv::PointerType::get(structTy, spirv::StorageClass::StorageBuffer);
    auto elemPtrTy = spirv::PointerType::get(f16Ty, spirv::StorageClass::StorageBuffer);

    // Rewrite each f16-target global's type, then rebuild its addressof +
    // AccessChain chain so the dependent cooperative-matrix loads/stores see
    // a ptr<f16>. Collect-and-rebuild (rather than walk-and-mutate) because
    // recreating the addressof invalidates the AccessChain's base operand.
    for (auto gv : spvModule.getOps<spirv::GlobalVariableOp>()) {
      if (!f16Globals.contains(gv.getSymName())) continue;
      gv->setAttr("type", TypeAttr::get(ptrTy));

      // Gather addressof ops referencing this global, then rebuild each with
      // the narrowed pointer type. Record their AccessChain users to rebuild
      // next (the old AccessChain result type is ptr<f32>, stale).
      SmallVector<spirv::AddressOfOp> addrs;
      spvModule.walk([&](spirv::AddressOfOp a) {
        if (a.getVariable() == gv.getSymName()) addrs.push_back(a);
      });
      for (spirv::AddressOfOp a : addrs) {
        OpBuilder b(a);
        auto newAddr = b.create<spirv::AddressOfOp>(a.getLoc(), ptrTy,
                         FlatSymbolRefAttr::get(ctx, gv.getSymName()));
        // Rebuild every AccessChain on the old addressof with result ptr<f16>.
        SmallVector<spirv::AccessChainOp> chains;
        for (auto *u : a->getUsers())
          if (auto ac = dyn_cast<spirv::AccessChainOp>(u))
            chains.push_back(ac);
        for (spirv::AccessChainOp ac : chains) {
          OpBuilder b2(ac);
          auto newAc = b2.create<spirv::AccessChainOp>(
              ac.getLoc(), elemPtrTy, newAddr, ac.getIndices());
          ac.replaceAllUsesWith(newAc.getResult());
          ac.erase();
        }
        a.replaceAllUsesWith(newAddr.getResult());
        a.erase();
      }
    }
  });
}


// SPIR-V 1.4+ requires every statically-used interface variable to be listed
// in the OpEntryPoint interface list — including StorageBuffer/Uniform
// descriptor variables. spirv-lower-abi-attrs populates only the Input/Output
// builtins it creates (WorkgroupId, LocalInvocationId, ...), not the SSBO
// globals that gpu.func kernel args became. For 1.3-and-below shaders that is
// fine (the rule didn't apply), but the cooperative-matrix target is 1.6, so
// the SSBOs must be enumerated. Walk each spirv.func, collect the descriptor
// globals it references via spirv.mlir.addressof, and merge them into the
// matching spirv.EntryPoint's interface attribute.
static void populateEntryPointInterfaces(ModuleOp module) {
  module.walk([&](spirv::ModuleOp spvModule) {
    // Index module-scope global variables by symbol name -> FlatSymbolRefAttr.
    DenseMap<StringRef, FlatSymbolRefAttr> descriptorGlobals;
    for (auto gv : spvModule.getOps<spirv::GlobalVariableOp>()) {
      auto ptrTy = gv.getType().dyn_cast<spirv::PointerType>();
      if (!ptrTy) continue;
      auto sc = ptrTy.getStorageClass();
      if (sc == spirv::StorageClass::StorageBuffer ||
          sc == spirv::StorageClass::Uniform ||
          sc == spirv::StorageClass::UniformConstant ||
          sc == spirv::StorageClass::PushConstant)
        descriptorGlobals[gv.getSymName()] =
            FlatSymbolRefAttr::get(spvModule->getContext(), gv.getSymName());
    }
    if (descriptorGlobals.empty()) return;

    // For each entry point, find the spirv.func it references and collect
    // the descriptor globals that spirv.mlir.addressof appears in its body.
    for (auto ep : spvModule.getOps<spirv::EntryPointOp>()) {
      spirv::FuncOp fn = spvModule.lookupSymbol<spirv::FuncOp>(ep.getFn());
      if (!fn) continue;
      SmallVector<Attribute, 8> iface;
      if (auto existing = ep.getInterface())
        for (auto a : existing.getValue())
          iface.push_back(a);
      SmallPtrSet<Attribute, 16> seen;
      for (auto v : iface) seen.insert(v);
      fn.walk([&](spirv::AddressOfOp addr) {
        auto it = descriptorGlobals.find(addr.getVariable());
        if (it == descriptorGlobals.end()) return;
        if (seen.insert(it->second).second) iface.push_back(it->second);
      });
      if (iface.empty()) continue;
      ep->setAttr(
          "interface",
          ArrayAttr::get(spvModule->getContext(), iface));
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

  // Narrow f16 SSBOs widened by MemRef→SPIRV back to f16 for the globals an
  // f16 cooperative-matrix load/store touches (see fixupF16StorageBuffers).
  // Must run before populateEntryPointInterfaces so the entry-point interface
  // walk sees the final global types (it keys by symbol name, not type, but
  // keep the ordering stable).
  fixupF16StorageBuffers(module);

  // SPIR-V 1.4+ requires every statically-used interface variable (including
  // StorageBuffer/Uniform descriptor variables) to be listed in the entry
  // point's interface list. The KHR cooperative-matrix target is bumped to
  // 1.6, but spirv-lower-abi-attrs only populates Input/Output builtins
  // (WorkgroupId/LocalInvocationId), not the SSBO globals — so cooperative-
  // matrix shaders fail spirv-val ("interface variable used but not listed")
  // and pipeline creation. Walk each spirv.func and add the descriptor
  // globals it references (via spirv.mlir.addressof) to its EntryPoint.
  populateEntryPointInterfaces(module);

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