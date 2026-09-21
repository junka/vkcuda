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
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <cstdlib>
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
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
    //
    // A by-value f16 vector kernel arg (`__half2 v`) takes a second path: it
    // becomes a spirv.GlobalVariable in PushConstant storage whose struct
    // member is `vector<Nxf16>` (lowerScalarArgsToPushConstant). Loading f16
    // from a push constant requires StoragePushConstant16 (same
    // SPV_KHR_16bit_storage extension). Detect that case independently of the
    // SSBO narrowing — it occurs even when no f16 SSBO was narrowed — and add
    // StoragePushConstant16 alongside.
    bool hasF16PushConstant = false;
    spvModule.walk([&](spirv::GlobalVariableOp gv) {
      // Only PushConstant globals can carry the f16 vector member.
      auto ptrTy = gv.getType().dyn_cast<spirv::PointerType>();
      if (!ptrTy || ptrTy.getStorageClass() != spirv::StorageClass::PushConstant)
        return;
      // Walk the pointee struct for any f16 element (scalar or vector).
      if (auto structTy =
              ptrTy.getPointeeType().dyn_cast<spirv::StructType>()) {
        for (auto memberTy : structTy.getElementTypes()) {
          if (auto vecTy = memberTy.dyn_cast<mlir::VectorType>())
            memberTy = vecTy.getElementType();
          if (memberTy.isF16()) {
            hasF16PushConstant = true;
            return;
          }
        }
      }
    });
    if (!f16Bases.empty() || hasF16PushConstant) {
      if (auto envAttr = spvModule->getAttrOfType<spirv::TargetEnvAttr>(
              spirv::getTargetEnvAttrName())) {
        auto triple = envAttr.getTripleAttr();
        auto version = triple.getVersion();
        if (version < spirv::Version::V_1_3)
          version = spirv::Version::V_1_3;
        SmallVector<spirv::Capability, 8> caps(
            triple.getCapabilities().begin(),
            triple.getCapabilities().end());
        if (!f16Bases.empty() &&
            !llvm::is_contained(caps,
                                spirv::Capability::StorageBuffer16BitAccess))
          caps.push_back(spirv::Capability::StorageBuffer16BitAccess);
        if (hasF16PushConstant &&
            !llvm::is_contained(caps,
                                spirv::Capability::StoragePushConstant16))
          caps.push_back(spirv::Capability::StoragePushConstant16);
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


// ConvertMemRefToSPIRV lowers a Function-storage memref (a local array slot, or
// a reference parameter's slot) to a spirv.Variable + AccessChain, but it emits
// the access chain's index as `builtin.unrealized_conversion_cast %i : index to
// i32` — a *materialization* of the index operand, not the
// `arith.index_cast index -> i32` the SPIR-V conversion patterns know how to
// legalize. Nothing downstream folds or legalizes it: convert-index-to-spirv
// has no pattern for it and reconcile-unrealized-casts cannot eliminate it
// either (its operand is still an index value, so the pair never becomes a
// no-op). GPUToSPIRV then fails with "failed to legalize operation
// 'builtin.unrealized_conversion_cast'".
//
// The cast is exactly an index -> i32 truncation/sign-extend, so rewriting it
// into the arith op the conversion patterns *do* handle makes the remainder of
// the pipeline (convert-gpu-to-spirv + a trailing reconcile) complete. Only
// casts whose source type is index and whose result is an integer are touched —
// the ref-param / call-convention casts this pipeline relies on have
// memref/ptr types and must survive untouched.
namespace {
struct RewriteResidualIndexCastsPass
    : public PassWrapper<RewriteResidualIndexCastsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteResidualIndexCastsPass)

  StringRef getArgument() const final {
    return "vc-rewrite-residual-index-casts";
  }
  StringRef getDescription() const final {
    return "rewrite index->i32 unrealized_conversion_cast into arith.index_cast";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rw(&getContext());
    module.walk([&](UnrealizedConversionCastOp cast) {
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
        return;
      Type src = cast.getOperand(0).getType();
      Type dst = cast.getResult(0).getType();
      if (!src.isIndex() || !dst.isIntOrIndex())
        return;
      rw.setInsertionPoint(cast);
      rw.replaceOpWithNewOp<arith::IndexCastOp>(cast, dst, cast.getOperand(0));
    });
  }
};

// A `T&`-returning __device__ helper with an early return (`if (sel) return a;
// return b;`) lowers to an `scf.if` that yields a Function-storage memref (the
// reference slot). ConvertFuncToSPIRV turns the helper's signature into
// spirv.ptr, and ConvertMemRefToSPIRV lowers the body's memref ops — but the
// `scf.if` result is still a Function-memref, with its yield operands and the
// if-result all wrapped in `unrealized_conversion_cast ptr <-> memref`. The
// scf.if -> spirv.if conversion cannot legalize a memref-typed result (it has
// no SPIR-V type to lower to), so the residual cast around the if fails:
// "failed to legalize operation 'builtin.unrealized_conversion_cast'".
//
// This pass peels the casts through the scf.if: rewrite the if to yield the
// underlying spirv.ptr directly (each yield operand that is a `ptr -> memref`
// cast donates its ptr input; a bare memref operand is wrapped `memref -> ptr`),
// and re-wrap the if result back to memref for any downstream memref users. The
// new if yields a spirv.ptr, which GPUToSPIRV lowers to spirv.Select, and the
// residual casts reconcile away. Runs only in the reference-legalization stage.
struct HoistRefReturnIfYieldsPass
    : public PassWrapper<HoistRefReturnIfYieldsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistRefReturnIfYieldsPass)

  StringRef getArgument() const final {
    return "vc-hoist-ref-return-if-yields";
  }
  StringRef getDescription() const final {
    return "rewrite scf.if yielding a Function-memref to yield its spirv.ptr";
  }

  // Is `ty` a Function-storage memref (a reference slot)?
  static bool isFunctionMemRef(Type ty) {
    auto mt = ty.dyn_cast<MemRefType>();
    if (!mt)
      return false;
    auto sc = mt.getMemorySpace()
                  .dyn_cast_or_null<spirv::StorageClassAttr>();
    return sc && sc.getValue() == spirv::StorageClass::Function;
  }

  // The spirv.ptr type a Function-memref converts to (the other side of the
  // unrealized_conversion_cast ConvertFuncToSPIRV/MemRefToSPIRV emits). Find it
  // by scanning the unrealized casts flowing into/out of memref-typed values.
  static Type ptrForMemRef(Value memrefVal, IRRewriter &rw) {
    // Look at users: a cast `ptr -> memref` whose result reaches this value, or
    // a cast `memref -> ptr` whose operand is this value.
    for (auto *user : memrefVal.getUsers()) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast)
        continue;
      if (cast->getNumOperands() == 1 && cast.getOperand(0) == memrefVal &&
          cast.getNumResults() == 1)
        return cast.getResult(0).getType(); // memref -> ptr
    }
    // Look at the defining op: a cast `ptr -> memref`.
    if (auto cast = memrefVal.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast.getNumResults() == 1)
        return cast.getOperand(0).getType();
    }
    return Type();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rw(&getContext());
    // Collect first; rewriting mutates the use list.
    SmallVector<scf::IfOp, 8> ifs;
    module.walk([&](scf::IfOp op) {
      if (op.getNumResults() == 1 && isFunctionMemRef(op.getResult(0).getType()))
        ifs.push_back(op);
    });
    for (scf::IfOp op : ifs) {
      Value oldRes = op.getResult(0);
      Type ptrTy = ptrForMemRef(oldRes, rw);
      if (!ptrTy)
        continue;
      rw.setInsertionPoint(op);
      auto newIf = rw.create<scf::IfOp>(op.getLoc(), TypeRange{ptrTy},
                                       op.getCondition(), /*withElse=*/true);
      // Each new branch block starts with a default `scf.yield %undef`. Drop it
      // and merge the old branch's body (sans its yield) in, then append a fresh
      // yield of the ptr-form operand.
      auto fillBranch = [&](Region &newRegion, Region &oldRegion) {
        Block &oldB = oldRegion.front();
        scf::YieldOp oldYield = cast<scf::YieldOp>(oldB.getTerminator());
        Value v = oldYield.getOperand(0);
        // Detach the old yield so the remaining ops can be moved whole.
        oldYield->erase();
        Block &newB = newRegion.front();
        // Clear the new block's placeholder yield.
        newB.getOperations().clear();
        // Move the old body ops into the new block.
        newB.getOperations().splice(newB.begin(), oldB.getOperations());
        // Emit the ptr conversion inside this branch, then yield it.
        rw.setInsertionPointToEnd(&newB);
        Value pv = memRefToPtr(v, ptrTy, rw, op.getLoc());
        rw.create<scf::YieldOp>(op.getLoc(), ValueRange{pv});
      };
      fillBranch(newIf.getThenRegion(), op.getThenRegion());
      fillBranch(newIf.getElseRegion(), op.getElseRegion());
      // Re-wrap the new if's ptr result back to memref for the old users.
      rw.setInsertionPointAfter(newIf);
      auto wrap = rw.create<UnrealizedConversionCastOp>(
          op.getLoc(), oldRes.getType(), newIf.getResult(0));
      rw.replaceOp(op, {wrap.getResult(0)});
    }
  }

  // Convert a memref-typed value to the spirv.ptr form. If it is itself a
  // `ptr -> memref` cast, return the ptr input directly; otherwise emit a
  // `memref -> ptr` cast.
  Value memRefToPtr(Value v, Type ptrTy, IRRewriter &rw, Location l) {
    if (v.getType() == ptrTy)
      return v;
    if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast.getNumResults() == 1 &&
          cast.getOperand(0).getType() == ptrTy)
        return cast.getOperand(0);
    }
    auto c = rw.create<UnrealizedConversionCastOp>(l, ptrTy, v);
    return c.getResult(0);
  }
};
} // namespace

// Does any func.func in the module carry a Function-storage memref parameter or
// result? That is the shape a reference parameter (`void f(int &r)`, emitted as
// a Function-storage memref passed by value to func.call) AND a reference return
// (`int &f(...)`, emitted as a Function-storage memref result) have BEFORE
// ConvertFuncToSPIRV runs (after which it becomes a
// ptr<struct<array<1 x T>>, Function> spirv.func parameter/result). When none
// exists, the reference-parameter legalization stage below is both unnecessary
// and harmful: ConvertMemRefToSPIRV rebuilds a legality target that rejects a
// `spirv.ReturnValue : f16` left untouched in a pure f16-returning __device__
// helper's body (no memref ops to convert there), so the stage must not run on
// f16-device-helper-only modules like half.vc, which otherwise regressed here.
static bool hasFunctionStorageRefParams(ModuleOp module) {
  bool found = false;
  auto checkTy = [&](mlir::Type ty) {
    if (found)
      return;
    auto memrefTy = ty.dyn_cast<mlir::MemRefType>();
    if (!memrefTy)
      return;
    auto scAttr =
        memrefTy.getMemorySpace()
            .dyn_cast_or_null<spirv::StorageClassAttr>();
    if (scAttr && scAttr.getValue() == spirv::StorageClass::Function)
      found = true;
  };
  module.walk([&](func::FuncOp fn) {
    if (found)
      return;
    for (auto argTy : fn.getFunctionType().getInputs())
      checkTy(argTy);
    for (auto resTy : fn.getFunctionType().getResults())
      checkTy(resTy);
  });
  return found;
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
  // A reference parameter (`void f(int &r)`) is emitted as a Function-storage
  // memref passed by value to func.call. ConvertFuncToSPIRV cannot express
  // that: it wraps the parameter in
  // `!spirv.ptr<!spirv.struct<(!spirv.array<1 x i32>)>, Function>` (the
  // by-value call convention's aggregate shape) and leaves an
  // unrealized_conversion_cast back to the memref for the body — a cast whose
  // two sides have unrelated underlying types, so reconcile-unrealized-casts
  // alone cannot collapse it and nothing else legalizes it. Lowering the
  // memref ops to SPIR-V pointers *before* GPUToSPIRV keeps the memref type as
  // the single source of truth: the body's load/store/AccessChain already
  // operate on the SPIR-V pointer, and the leftover cast reconciles away. The
  // resulting signature keeps the ptr<struct<array<1 x T>>> spelling (SPIR-V
  // has no bare-pointer parameter type), which is exactly the shape the
  // call site then produces.
  //
  // Run this stage only when a reference parameter is actually present
  // (hasFunctionStorageRefParams). ConvertMemRefToSPIRV rebuilds a conversion
  // target whose legality check rejects a `spirv.ReturnValue : f16` left
  // untouched in a pure f16-returning __device__ helper's body (no memref ops
  // to convert there), so the stage must not run on f16-device-helper-only
  // modules like half.vc, which otherwise regressed here.
  if (hasFunctionStorageRefParams(module)) {
    pm.addPass(createConvertMemRefToSPIRVPass());
    pm.addPass(std::make_unique<HoistRefReturnIfYieldsPass>());
    pm.addPass(std::make_unique<RewriteResidualIndexCastsPass>());
    pm.addPass(createReconcileUnrealizedCastsPass());
  }
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