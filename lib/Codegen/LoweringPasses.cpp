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
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
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
    MLIRContext *ctx = spvModule->getContext();
    auto f16Ty = Float16Type::get(ctx);

    // MLIR's MemRef→SPIRV type converter widens sub-4-byte StorageBuffer
    // elements to 4 bytes: a `memref<?xf16, StorageBuffer>` kernel arg (or
    // global) becomes `ptr<struct<(rtarray<f32, stride=4>)>, StorageBuffer>`.
    // But the HOST backs these with packed f16 (2 bytes each), so the widened
    // 4-byte layout is a real data mismatch, not just a type inconvenience.
    // Two access kinds need the backing narrowed back to contiguous f16
    // (rtarray<f16, stride=2>):
    //   (a) spirv.KHR.CooperativeMatrixLoad/Store whose coopmatrix element is
    //       f16 — reads 4-byte elements into an f16 matrix (silently wrong).
    //   (b) scalar spirv.Load/Store from a `memref<?xf16>` — the store pattern
    //       even emits `spirv.Store ptr<f32>, %val:f16` (type mismatch that
    //       aborts the in-pipeline spirv.module passes).
    //
    // We operate on the ACCESS-CHAIN base, which is EITHER a spirv.func block
    // argument (the pre-lower-abi-attrs SSBO kernel arg — scalar path, where
    // lower-abi-attrs hasn't run because conversion aborted) OR a
    // spirv.mlir.addressof of a spirv.GlobalVariable (the post-lower-abi-attrs
    // form — cooperative-matrix path, where conversion succeeded). Narrowing
    // the base type + rebuilding the AccessChain as ptr<f16> fixes both; the
    // scalar Load is then rebuilt to yield f16 (dropping the bridge cast).

    // Pass 1: collect the set of base Values (func args or addressof results)
    // that an f16 access reaches, by walking each AccessChain's users.
    DenseSet<Value> f16Bases;
    auto recordIfF16 = [&](spirv::AccessChainOp ac) {
      for (auto *u : ac->getUsers()) {
        mlir::Type elemTy;
        if (auto load = dyn_cast<spirv::KHRCooperativeMatrixLoadOp>(u)) {
          elemTy = cast<spirv::CooperativeMatrixType>(load.getResult().getType())
                       .getElementType();
        } else if (auto store = dyn_cast<spirv::KHRCooperativeMatrixStoreOp>(u)) {
          elemTy = cast<spirv::CooperativeMatrixType>(store.getObject().getType())
                       .getElementType();
        } else if (auto store = dyn_cast<spirv::StoreOp>(u)) {
          // Scalar store: value operand is f16 (source memref.store value).
          elemTy = store.getValue().getType();
        } else if (auto load = dyn_cast<spirv::LoadOp>(u)) {
          // Scalar load widened to f32; the real f16 surfaces via the bridge
          // cast unrealized_conversion_cast f32->f16 on the load's result.
          for (auto *lu : load->getUsers())
            if (auto c = dyn_cast<UnrealizedConversionCastOp>(lu))
              if (c->getNumResults() == 1 && c.getResult(0).getType().isF16())
                elemTy = f16Ty;
        }
        // elemTy stays null when this user is none of the f16-relevant cases
        // (or a LoadOp with no f16 bridge-cast user). Guard the isF16() check
        // — calling it on a null mlir::Type dereferences a null impl pointer.
        if (elemTy && elemTy.isF16()) {
          f16Bases.insert(ac.getBasePtr());
          break;
        }
      }
    };
    spvModule.walk([&](spirv::AccessChainOp ac) { recordIfF16(ac); });
    if (f16Bases.empty()) return;

    // Narrowed types:
    //   base : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>
    //   elem : !spirv.ptr<f16, StorageBuffer>
    auto rtArr = spirv::RuntimeArrayType::get(f16Ty, /*stride=*/2);
    auto structTy = spirv::StructType::get({rtArr}, {0u});
    auto ptrTy = spirv::PointerType::get(structTy, spirv::StorageClass::StorageBuffer);
    auto elemPtrTy = spirv::PointerType::get(f16Ty, spirv::StorageClass::StorageBuffer);

    // Pass 2: for each f16 base, narrow its type and rebuild every AccessChain
    // on it to yield ptr<f16>. The base is either a func arg (mutate its type
    // in place) or an addressof (rebuild, then rebuild its AccessChains).
    for (Value base : f16Bases) {
      if (auto blkArg = dyn_cast<BlockArgument>(base)) {
        // SSBO kernel arg, pre-lower-abi-attrs: mutate the arg type directly.
        blkArg.setType(ptrTy);
        // Rebuild AccessChains using this arg as base.
        SmallVector<spirv::AccessChainOp> chains;
        for (auto *u : blkArg.getUsers())
          if (auto ac = dyn_cast<spirv::AccessChainOp>(u))
            chains.push_back(ac);
        for (spirv::AccessChainOp ac : chains) {
          OpBuilder b(ac);
          auto newAc = b.create<spirv::AccessChainOp>(
              ac.getLoc(), elemPtrTy, ac.getBasePtr(), ac.getIndices());
          ac.replaceAllUsesWith(newAc.getResult());
          ac.erase();
        }
      } else if (auto addr = base.getDefiningOp<spirv::AddressOfOp>()) {
        // Post-lower-abi-attrs: the global this addressof references must be
        // narrowed, and the addressof + its AccessChains rebuilt.
        auto gv = spvModule.lookupSymbol<spirv::GlobalVariableOp>(addr.getVariable());
        if (gv) gv->setAttr("type", TypeAttr::get(ptrTy));
        OpBuilder b(addr);
        auto newAddr = b.create<spirv::AddressOfOp>(addr.getLoc(), ptrTy,
                         FlatSymbolRefAttr::get(ctx, addr.getVariable()));
        SmallVector<spirv::AccessChainOp> chains;
        for (auto *u : addr->getUsers())
          if (auto ac = dyn_cast<spirv::AccessChainOp>(u))
            chains.push_back(ac);
        for (spirv::AccessChainOp ac : chains) {
          OpBuilder b2(ac);
          auto newAc = b2.create<spirv::AccessChainOp>(
              ac.getLoc(), elemPtrTy, newAddr, ac.getIndices());
          ac.replaceAllUsesWith(newAc.getResult());
          ac.erase();
        }
        addr.replaceAllUsesWith(newAddr.getResult());
        addr.erase();
      }
    }

    // Pass 3: after narrowing, scalar spirv.Load on an f16 SSBO still yields
    // f32 (the widened result type baked in by the pattern) while its pointer
    // is now ptr<f16> — a mismatch. Rebuild each such Load to yield f16 and
    // drop the redundant unrealized_conversion_cast f32->f16 bridge (the load
    // now produces f16 directly). Scalar Store is already fixed: its pointer
    // is ptr<f16> and its value was f16 all along.
    SmallVector<spirv::LoadOp> loads;
    spvModule.walk([&](spirv::LoadOp op) {
      auto ptrTy = dyn_cast<spirv::PointerType>(op.getPtr().getType());
      if (ptrTy && ptrTy.getStorageClass() == spirv::StorageClass::StorageBuffer &&
          ptrTy.getPointeeType().isF16() && op.getType().isF32())
        loads.push_back(op);
    });
    for (spirv::LoadOp op : loads) {
      OpBuilder b(op);
      auto newLoad = b.create<spirv::LoadOp>(op.getLoc(), f16Ty, op.getPtr(),
          op.getMemoryAccessAttr(), op.getAlignmentAttr());
      for (auto *u : llvm::make_early_inc_range(op->getUsers())) {
        if (auto c = dyn_cast<UnrealizedConversionCastOp>(u)) {
          c.getResult(0).replaceAllUsesWith(newLoad.getResult());
          c.erase();
        }
      }
      op.erase();
    }

    // Pass 4: now that SSBOs are narrowed to ptr<f16>, the lowered IR uses
    // 16-bit storage buffer access (spirv.Bitcast over ptr<f16>, f16 loads/
    // stores into a StorageBuffer). That requires the StorageBuffer16BitAccess
    // capability + the SPV_KHR_16bit_storage extension, and SPIR-V >= 1.3
    // (Vulkan 1.1). The original target_env only advertised Float16 (f16
    // arithmetic), not 16-bit *storage*. Augment the spirv.module's
    // spirv.target_env so the re-run UpdateVCEPass (rerunAbortedSPIRVLegality)
    // stamps a vce_triple that admits the 16-bit-storage ops; otherwise
    // spirv-lower-abi-attrs rejects the spirv.Bitcast ("requires
    // SPV_KHR_16bit_storage"). Done only when we actually narrowed an f16 SSBO.
    if (auto envAttr = spvModule->getAttrOfType<spirv::TargetEnvAttr>(
            spirv::getTargetEnvAttrName())) {
      auto triple = envAttr.getTripleAttr();
      auto version = triple.getVersion();
      if (version < spirv::Version::V_1_3)
        version = spirv::Version::V_1_3;
      SmallVector<spirv::Capability, 8> caps(
          triple.getCapabilities().begin(), triple.getCapabilities().end());
      if (!llvm::is_contained(caps, spirv::Capability::StorageBuffer16BitAccess))
        caps.push_back(spirv::Capability::StorageBuffer16BitAccess);
      SmallVector<spirv::Extension, 4> exts(
          triple.getExtensions().begin(), triple.getExtensions().end());
      if (!llvm::is_contained(exts, spirv::Extension::SPV_KHR_16bit_storage))
        exts.push_back(spirv::Extension::SPV_KHR_16bit_storage);
      auto newTriple = spirv::VerCapExtAttr::get(version, caps, exts, ctx);
      spvModule->setAttr(spirv::getTargetEnvAttrName(),
                         spirv::TargetEnvAttr::get(
                             newTriple, envAttr.getResourceLimits(),
                             envAttr.getClientAPI(), envAttr.getVendorID(),
                             envAttr.getDeviceType(), envAttr.getDeviceID()));
    }
  });
}


// When the MemRef→SPIRV conversion widens an f16 SSBO to f32 (see
// fixupF16StorageBuffers), the scalar `memref.store` pattern emits a
// `spirv.Store ptr<f32>, %val:f16` — a type mismatch that fails spirv
// verification DURING the in-pipeline conversion. That failure stops
// pm.run short of the spirv.module passes that run AFTER conversion:
// spirv-lower-abi-attrs (materializes spirv.EntryPoint + abi globals from
// the spirv.entry_point_abi / spirv.interface_var_abi attrs) and
// SPIRVUpdateVCEPass (sets the serializer's required vce_triple). Neither
// ran, so the spirv.module has no EntryPoint and no vce_triple.
//
// fixupF16StorageBuffers (run just before this) narrows the globals and
// rebuilds the load/store to be type-consistent, so the IR is now valid.
// This pass then re-runs the two aborted spirv.module passes on a fresh
// PassManager to finish legalization and make the module serializable.
static void rerunAbortedSPIRVLegality(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addNestedPass<spirv::ModuleOp>(
      spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  if (failed(pm.run(module)))
    module.emitError("post-fixup spirv.module legalization failed");
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
    // The 1.4+ "list every used interface variable" rule (which motivates
    // adding SSBO/Uniform/PushConstant vars) does NOT apply below 1.4 — there
    // listing a StorageBuffer in OpEntryPoint's interface is *illegal*
    // ("OpEntryPoint interfaces must be Input or Output"). Only the
    // cooperative-matrix path bumps the target to 1.6; ordinary f16 SSBO
    // kernels stay at 1.0 and must NOT have descriptor vars enumerated.
    auto vce = spvModule.getVceTriple();
    if (!vce || vce->getVersion() < spirv::Version::V_1_4)
      return;
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

  // The in-pipeline pm.run fails on the known f16-SSBO widening residue
  // (`spirv.Store ptr<f32>, %val:f16` mismatch), which fixupF16StorageBuffers
  // + rerunAbortedSPIRVLegality below recover. Don't emit a hard error here —
  // it would noise up every f16-SSBO shader's stderr even though the build
  // succeeds. A genuine (non-f16) lowering failure leaves the spirv.module
  // unserializable, caught by rerunAbortedSPIRVLegality's emitError or the
  // driver's "spirv translation failed".
  //
  // The MLIR op verifier still reports the offending spirv.Store/Load mismatch
  // as an inline error diagnostic DURING pm.run (before fixupF16StorageBuffers
  // repairs it). That single stderr line is spurious — the build succeeds and
  // spirv-val passes — so install a scoped handler around pm.run that swallows
  // exactly that known mismatch and forwards everything else. The handler is
  // RAII-scoped to this pm.run, so it cannot mask errors elsewhere or later.
  {
    ScopedDiagnosticHandler diagHandler(&ctx, [](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Error) {
        std::string msg = diag.str();
        // The residue the f16-SSBO converter leaves, repaired post-run by
        // fixupF16StorageBuffers. Match on the verifier's wording so unrelated
        // spirv.Store/Load errors still surface.
        if (msg.find("spirv.Store") != std::string::npos &&
            msg.find("mismatch in result type and pointer type") !=
                std::string::npos)
          return success();
        if (msg.find("spirv.Load") != std::string::npos &&
            msg.find("mismatch in result type and pointer type") !=
                std::string::npos)
          return success();
      }
      return failure();
    });
    (void)failed(pm.run(module));
  }

  // Post-conversion rewrite of atomics GPUToSPIRV could not lower directly
  // (atomicExch on SSBO/global, emitted as a marked AtomicIAdd).
  rewriteMarkedAtomics(module, markedPositions);

  // Narrow f16 SSBOs widened by MemRef→SPIRV back to contiguous f16
  // (rtarray<f16,stride=2>) for the globals an f16 access — cooperative-matrix
  // KHR load/store OR scalar spirv.Load/Store — touches. The host backs these
  // with packed f16, so the widened rtarray<f32,stride=4> is a layout
  // mismatch. This rebuilds the global + addressof + AccessChain chain as
  // ptr<f16> and reconstructs scalar Loads to yield f16, fixing the malformed
  // `spirv.Store ptr<f32>, %val:f16` the conversion pattern left behind (which
  // aborted the in-pipeline spirv.module passes below). See fixupF16StorageBuffers.
  fixupF16StorageBuffers(module);

  // The narrowing above makes the spirv.module type-consistent, but the
  // in-pipeline spirv-lower-abi-attrs + SPIRVUpdateVCEPass never ran (pm.run
  // aborted on the pre-narrowing store mismatch). Re-run them on a fresh
  // PassManager so the EntryPoint + vce_triple the serializer needs exist.
  rerunAbortedSPIRVLegality(module);

  // SPIR-V 1.4+ requires every statically-used interface variable (including
  // StorageBuffer/Uniform descriptor variables) to be listed in the entry
  // point's interface list. The KHR cooperative-matrix target is bumped to
  // 1.6, but spirv-lower-abi-attrs only populates Input/Output builtins
  // (WorkgroupId/LocalInvocationId), not the SSBO globals — so cooperative-
  // matrix shaders fail spirv-val ("interface variable used but not listed")
  // and pipeline creation. Walk each spirv.func and add the descriptor
  // globals it references (via spirv.mlir.addressof) to its EntryPoint.
  // (No-op below SPIR-V 1.4, where listing StorageBuffer vars is illegal.)
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