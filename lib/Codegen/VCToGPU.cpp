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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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

// The numeric widths a module's kernels actually use. Each non-f32/i32 width
// needs its own SPIR-V capability (Float64/Int64/Float16) for GPUToSPIRV to
// legalize the ops on it, and Vulkan gates the same widths behind device
// features (shaderFloat64/shaderInt64/shaderFloat16). So a capability is both
// required (by the IR) and optional (by the device): declaring one the source
// never uses makes vkCreateShaderModule reject the binary on a device that
// lacks the matching feature, which is why the list below is derived from the
// IR instead of being advertised unconditionally.
struct SPIRVWidthUse {
  bool f64 = false;
  bool i64 = false;
  bool f16 = false;
};

// Records the width of `t`, peeling container types (memref/vector/…) down to
// their element type and recursing through function signatures.
void noteWidthUse(Type t, SPIRVWidthUse &use) {
  while (auto shaped = t.dyn_cast<ShapedType>())
    t = shaped.getElementType();
  if (auto fn = t.dyn_cast<FunctionType>()) {
    for (Type input : fn.getInputs())
      noteWidthUse(input, use);
    for (Type result : fn.getResults())
      noteWidthUse(result, use);
    return;
  }
  if (auto intTy = t.dyn_cast<IntegerType>()) {
    if (intTy.getWidth() == 64) use.i64 = true;
    return;
  }
  if (t.isF64()) use.f64 = true;
  if (t.isF16()) use.f16 = true;
}

SPIRVWidthUse scanWidthUses(ModuleOp module) {
  SPIRVWidthUse use;
  module.walk([&](Operation *op) {
    for (Type t : op->getResultTypes())
      noteWidthUse(t, use);
    for (Value v : op->getOperands())
      noteWidthUse(v.getType(), use);
    // Kernel parameters are block arguments, not op operands.
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        for (BlockArgument arg : block.getArguments())
          noteWidthUse(arg.getType(), use);
  });
  return use;
}

// The default target env only enables Shader without any extensions, while
// the Vulkan interface variables we generate for kernel arguments live in the
// StorageBuffer storage class, which SPIR-V 1.0 expresses via the
// SPV_KHR_storage_buffer_storage_class extension. Without it the GPUToSPIRV
// signature conversion cannot map the kernel arguments and memref.load/store
// on them fails to legalize (<UNKNOWN SSA VALUE>).
spirv::TargetEnvAttr getVCTargetEnv(MLIRContext *context, bool usesSubgroup,
                                    bool usesCoopMatrix, bool usesF16Storage,
                                    const SPIRVWidthUse &widths) {
  // Float64/Int64/Float16 are advertised exactly when the kernels use a double
  // (`double`, `double2`), 64-bit integer (`long`, `long4`) or half (`__half`)
  // type — see scanWidthUses. Vulkan's core Shader capability already covers
  // i32/f32; the wider/narrower widths need their own capabilities declared or
  // the spirv.module's vce triple won't advertise them and GPUToSPIRV refuses to
  // legalize the ops (spirv.CompositeConstruct on vector<2xf64> fails with
  // "explicitly marked illegal"). The runtime enables the matching device
  // features opportunistically and refuses a kernel whose declared capability
  // the device did not enable (see VCRuntime device feature selection and
  // checkSpirvCapabilities).
  //
  // CUDA warp intrinsics lower to spirv.GroupNonUniform* ops, which require
  // SPIR-V 1.3 (vulkan1.1) + the GroupNonUniform{Ballot,Shuffle,
  // ShuffleRelative,Arithmetic,Vote} capabilities. When the AST→MLIR emitter
  // flagged subgroup usage (vc.uses_subgroup), bump the target version and
  // advertise those capabilities so GPUToSPIRV legalizes the subgroup ops.
  //
  // WMMA (wmma::*) lowers to spirv.KHR.CooperativeMatrix{Load,Store,MulAdd},
  // which require SPIR-V 1.6 (vulkan1.2) + the CooperativeMatrixKHR capability
  // + the SPV_KHR_cooperative_matrix extension. When the emitter flagged
  // cooperative-matrix usage (vc.uses_coopmatrix), bump the target version
  // (1.6 supersedes 1.3) and advertise the capability/extension.
  spirv::Version version = spirv::Version::V_1_0;
  SmallVector<spirv::Capability, 8> caps = {spirv::Capability::Shader};
  if (widths.f64) caps.push_back(spirv::Capability::Float64);
  if (widths.i64) caps.push_back(spirv::Capability::Int64);
  if (widths.f16) caps.push_back(spirv::Capability::Float16);
  SmallVector<spirv::Extension, 2> exts = {
      spirv::Extension::SPV_KHR_storage_buffer_storage_class};
  if (usesSubgroup) {
    if (version < spirv::Version::V_1_3) version = spirv::Version::V_1_3;
    caps.push_back(spirv::Capability::GroupNonUniform);
    caps.push_back(spirv::Capability::GroupNonUniformVote);
    caps.push_back(spirv::Capability::GroupNonUniformBallot);
    caps.push_back(spirv::Capability::GroupNonUniformShuffle);
    caps.push_back(spirv::Capability::GroupNonUniformShuffleRelative);
    caps.push_back(spirv::Capability::GroupNonUniformArithmetic);
  }
  if (usesCoopMatrix) {
    version = spirv::Version::V_1_6; // supersedes 1.3
    caps.push_back(spirv::Capability::CooperativeMatrixKHR);
    exts.push_back(spirv::Extension::SPV_KHR_cooperative_matrix);
  }
  // A by-value f16 vector kernel arg (`__half2 v`, `half4 v`) is pushed through
  // a PushConstant struct member `vector<Nxf16>` (lowerScalarArgsToPushConstant).
  // spirv-lower-abi-attrs legalizes that load only with StoragePushConstant16
  // (SPV_KHR_16bit_storage, SPIR-V >= 1.3). The SSBO f16 path is repaired
  // post-conversion by fixupF16StorageBuffers, but the push-constant path is
  // rejected DURING pm.run before a spirv.module even exists — so the cap must
  // be advertised up front here. `usesF16Storage` is set by packKernels when it
  // sees an f16 vector scalar param.
  if (usesF16Storage) {
    if (version < spirv::Version::V_1_3) version = spirv::Version::V_1_3;
    caps.push_back(spirv::Capability::StoragePushConstant16);
    if (!llvm::is_contained(exts, spirv::Extension::SPV_KHR_16bit_storage))
      exts.push_back(spirv::Extension::SPV_KHR_16bit_storage);
  }
  auto triple = spirv::VerCapExtAttr::get(version, caps, exts, context);
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

// Lower a gpu.func kernel's scalar (non-memref) arguments to a single SPIR-V
// push-constant struct, matching the GLSL backend + runtime ABI: pointer
// (memref) args get consecutive StorageBuffer bindings (skipping scalars);
// scalar args are packed into one `spirv.GlobalVariable` in the PushConstant
// storage class, read in the kernel prologue via addressof + AccessChain +
// Load. This is necessary because MLIR's GPUToSPIRV lowering has no path that
// materializes a by-value scalar arg as a push-constant global — it would
// leave the scalar as a StorageBuffer binding, which the runtime does not
// populate (it pushes scalars via vkCmdPushConstants, not descriptor writes),
// so the kernel would read garbage and run off the end.
//
// Field layout mirrors the runtime's buildLayout: scalar args appear in the
// push-constant struct in source order, each 4-byte-aligned (the runtime
// rounds scalar size up to 4 bytes). Pointer bindings count up from 0 over
// pointer args only.
static void lowerScalarArgsToPushConstant(gpu::GPUFuncOp gpuFn,
                                          OpBuilder &builder) {
  // Partition args: which are scalar (to move into the PC block) vs pointer
  // (memref, to keep as interface variables). A by-value vector arg
  // (vector<NxELEM>, e.g. `float4 v` / `__half2 v`) is NOT a pointer — it is
  // a value passed through push constants just like a scalar. ShapedType
  // covers both MemRefType and VectorType, so we must test MemRefType
  // specifically; otherwise a vector arg gets misclassified as a pointer,
  // handed a descriptor binding it cannot satisfy, and spirv.func legalization
  // fails ("explicitly marked illegal").
  SmallVector<Type> scalarTys;
  SmallVector<unsigned> scalarArgIdxs;
  SmallVector<unsigned> pointerArgIdxs;
  TypeRange inputTys = gpuFn.getFunctionType().getInputs();
  for (auto [i, ty] : llvm::enumerate(inputTys)) {
    if (isa<MemRefType>(ty))
      pointerArgIdxs.push_back(i);
    else {
      scalarArgIdxs.push_back(i);
      scalarTys.push_back(ty);
    }
  }
  if (scalarArgIdxs.empty()) {
    // No scalars: just (re-)assign pointer bindings in order.
    for (auto [slot, _] : llvm::enumerate(pointerArgIdxs))
      gpuFn.setArgAttr(slot, spirv::getInterfaceVarABIAttrName(),
                       spirv::getInterfaceVarABIAttr(/*descSet=*/0, slot,
                                                     std::nullopt,
                                                     gpuFn.getContext()));
    return;
  }

  // Build the push-constant struct type and a module-scope global for it,
  // placed inside the gpu.module so GPUToSPIRV carries it into the spirv.module
  // (same hoisting mechanism as __shared__ globals). The global's type is a
  // spirv.ptr<struct<...>, PushConstant>; storage class lives on the pointer.
  //
  // Member offsets are explicit (each scalar 4-byte-aligned, accumulated in
  // source order) to mirror the runtime's buildLayout packing and to satisfy
  // Vulkan's requirement that every Block struct member carry an Offset
  // decoration — spirv-lower-abi-attrs does not add them for us, and without
  // them the driver reads garbage (validation rejects the module).
  SmallVector<uint32_t> offsets;
  uint32_t off = 0;
  for (Type ty : scalarTys) {
    offsets.push_back(off);
    // Byte size of one push-constant member. Scalars use their bit width;
    // vectors pack element_count * (elem_bit_width/8) contiguously with no
    // per-lane padding (mirrors the host's sizeof of the emitted C++ vector
    // struct, e.g. half2 = 2*_Float16 = 4 bytes, float4 = 4*float = 16).
    uint32_t sz;
    if (auto vty = ty.dyn_cast<mlir::VectorType>())
      sz = (vty.getNumElements() * vty.getElementTypeBitWidth()) / 8;
    else
      sz = ty.isIntOrFloat() ? (ty.getIntOrFloatBitWidth() / 8) : 4;
    off += (sz + 3u) & ~3u; // round up to 4 bytes (std140-friendly)
  }
  Type pcStructTy = spirv::StructType::get(scalarTys, offsets);
  std::string pcName = ("__vc_pc_" + gpuFn.getName()).str();
  Type pcPtrTy =
      spirv::PointerType::get(pcStructTy, spirv::StorageClass::PushConstant);
  // spirv.GlobalVariableOp is module-scope: it must live in the gpu.module
  // body, NOT inside the gpu.func. The shared IRRewriter's insertion point is
  // whatever packKernels last left it at (often inside the gpu.func body after
  // the return rewrite), so set it explicitly to the gpu.module body before
  // creating the global, then reset into the gpu.func body for the prologue
  // loads below. Without this, the 3rd+ kernel's pcVar create segfaults
  // (inserting a module-scope op into a function body) — same stale-insertion-
  // point class of bug as lowerWorkgroupSizeToSpecConstants (see commit
  // 3281bec), surfacing here only when the shared rewriter's point drifts into
  // the func body across iterations.
  OpBuilder::InsertPoint saved = builder.saveInsertionPoint();
  builder.setInsertionPoint(gpuFn); // module-scope: just before the gpu.func
  auto pcVar = builder.create<spirv::GlobalVariableOp>(
      gpuFn.getLoc(), TypeAttr::get(pcPtrTy), builder.getStringAttr(pcName),
      /*initializer=*/FlatSymbolRefAttr());
  builder.restoreInsertionPoint(saved);

  // In the kernel prologue, load each scalar field and replace the original
  // block argument's uses with the loaded value. AccessChain takes SSA index
  // values, so materialize the field index as a spirv.Constant i32.
  OpBuilder body(gpuFn.getBody());
  body.setInsertionPointToStart(&gpuFn.getBody().front());
  Value pcAddr = body.create<spirv::AddressOfOp>(gpuFn.getLoc(), pcPtrTy, pcName);
  for (auto [fieldIdx, argIdx] : llvm::enumerate(scalarArgIdxs)) {
    Value idxVal = body.create<spirv::ConstantOp>(
        gpuFn.getLoc(), body.getI32Type(),
        body.getI32IntegerAttr(static_cast<int32_t>(fieldIdx)));
    Value fieldPtr = body.create<spirv::AccessChainOp>(
        gpuFn.getLoc(),
        spirv::PointerType::get(scalarTys[fieldIdx],
                                spirv::StorageClass::PushConstant),
        pcAddr, ValueRange{idxVal});
    Value loaded =
        body.create<spirv::LoadOp>(gpuFn.getLoc(), scalarTys[fieldIdx], fieldPtr);
    gpuFn.getArgument(argIdx).replaceAllUsesWith(loaded);
  }

  // Erase the scalar block args (back-to-front so indices stay valid) and
  // rebuild the function type over the remaining (pointer) args.
  llvm::BitVector erase(inputTys.size());
  for (unsigned i : scalarArgIdxs) erase.set(i);
  gpuFn.getBody().front().eraseArguments(erase);

  SmallVector<Type> newInputs;
  for (unsigned i : pointerArgIdxs) newInputs.push_back(inputTys[i]);
  Type newFnTy = FunctionType::get(builder.getContext(), newInputs,
                                   gpuFn.getFunctionType().getResults());
  gpuFn.setFunctionTypeAttr(TypeAttr::get(newFnTy));

  // Assign StorageBuffer bindings to the surviving pointer args in order.
  // After eraseArguments the pointer args occupy slots 0..N-1 in source order.
  for (auto [slot, _] : llvm::enumerate(pointerArgIdxs))
    gpuFn.setArgAttr(slot, spirv::getInterfaceVarABIAttrName(),
                     spirv::getInterfaceVarABIAttr(/*descSet=*/0, slot,
                                                   std::nullopt,
                                                   gpuFn.getContext()));
}

// Replaces the compile-time `gpu.block_dim` (which GPUToSPIRV would lower to a
// literal `spirv.Constant <abi size>`, baking the workgroup size at compile
// time) with a read of a SPIR-V specialization constant. The runtime sets the
// block size via Vulkan specialization constants with SpecId 0/1/2 = x/y/z
// (lib/Runtime/VCRuntime.cpp getPipeline), mirroring the GLSL backend's
// `layout(local_size_x_id = 0, ...)` + `gl_WorkGroupSize`. Without this, every
// MLIR kernel runs with the hardcoded LocalSize 32x1x1 regardless of the
// launch's block parameter, so any kernel launched with block != 32 computes
// the wrong blockDim and runs off the end (sync block=64, matmul 16x16, ...).
//
// Emits, per gpu.module (shared by all kernels):
//   spirv.SpecConstant @__vc_wg_x spec_id(0) = 32 : i32
//   spirv.SpecConstant @__vc_wg_y spec_id(1) = 1  : i32
//   spirv.SpecConstant @__vc_wg_z spec_id(2) = 1  : i32
// and rewrites each `gpu.block_dim <dim>` to:
//   %v = spirv.mlir.referenceof @__vc_wg_<dim> : i32
//   %idx = arith.index_cast %v : i32 to index
//
// The `OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize` that Vulkan requires
// to make the spec constants actually drive the hardware workgroup size is NOT
// expressible in MLIR's spirv dialect (SpecConstantCompositeOp carries no
// built_in decoration, and LocalSizeId's IdRef operands can't be serialized via
// ExecutionModeOp's ArrayAttr values). It is injected by patching the serialized
// SPIR-V binary in the vc driver (see vc.cpp). The placeholder LocalSize 1x1x1
// from the entry_point_abi is overridden by the WorkgroupSize builtin at run
// time, exactly as in the GLSL backend's shader.
static void lowerWorkgroupSizeToSpecConstants(gpu::GPUModuleOp gpuModule,
                                              OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  Type i32 = builder.getI32Type();

  // The spec constants and their composite are module-scope ops that must land
  // inside the gpu.module. Set the insertion point explicitly rather than
  // relying on the caller's builder state: packKernels shares one IRRewriter
  // across both kernel packing and this call, and after the per-kernel loop the
  // builder's insertion point can be left pointing at a stale/erased op (e.g.
  // the func::FuncOp that packKernels erases after takeBody). Without this,
  // builder.create below dereferences a dangling insertion point and segfaults
  // — this was the root cause of the multi-kernel + local-struct crash: any
  // second kernel left the builder's insertion block pointing into the just-
  // erased func body, so the first spec-constant create faulted.
  builder.setInsertionPointToStart(gpuModule.getBody());

  // Create the three scalar spec constants once per gpu.module.
  struct SpecConst {
    const char *name;
    unsigned specId;
    int32_t defVal;
  };
  const SpecConst scs[3] = {
      {"__vc_wg_x", 0, 32}, {"__vc_wg_y", 1, 1}, {"__vc_wg_z", 2, 1}};
  // Create in reverse order (z, y, x) and moveBefore front each time, so the
  // final module order is x, y, z (each moveBefore pushes to the front, so the
  // last-pushed x ends up first). Track z (the last in module order) so the
  // composite below can be placed after all three.
  spirv::SpecConstantOp zConst;
  for (const SpecConst &sc : llvm::reverse(scs)) {
    auto specConst = builder.create<spirv::SpecConstantOp>(
        gpuModule.getLoc(), builder.getStringAttr(sc.name),
        builder.getI32IntegerAttr(sc.defVal));
    specConst->setAttr("spec_id", builder.getI32IntegerAttr(sc.specId));
    specConst->moveBefore(&gpuModule.getBody()->front());
    if (sc.specId == 2) zConst = specConst;
  }
  // Module order is now: __vc_wg_x, __vc_wg_y, __vc_wg_z, <rest>.

  // The WorkgroupSize builtin: a SpecConstantComposite of the three spec
  // constants. Named gl_WorkGroupSize so the binary patch in the vc driver can
  // locate it by OpName and inject `OpDecorate <id> BuiltIn WorkgroupSize`
  // (the MLIR spirv dialect can't attach that decoration from IR). The actual
  // hardware workgroup size is driven by this builtin's spec-constant values,
  // which the runtime populates via specialization info at pipeline creation.
  // Placed *after* the three scalar spec constants: the spirv serializer
  // resolves the composite's constituent symbol-refs to already-emitted
  // constant result IDs, so all scalars must precede it.
  Type v3i32 = VectorType::get({3}, i32);
  SmallVector<Attribute> constituents = {
      FlatSymbolRefAttr::get(ctx, "__vc_wg_x"),
      FlatSymbolRefAttr::get(ctx, "__vc_wg_y"),
      FlatSymbolRefAttr::get(ctx, "__vc_wg_z")};
  auto wgComposite = builder.create<spirv::SpecConstantCompositeOp>(
      gpuModule.getLoc(), TypeAttr::get(v3i32),
      builder.getStringAttr("gl_WorkGroupSize"),
      builder.getArrayAttr(constituents));
  wgComposite->moveAfter(zConst);

  // Rewrite every gpu.block_dim in every gpu.func to a referenceof + index_cast.
  SmallVector<gpu::BlockDimOp> dims;
  gpuModule.walk([&](gpu::BlockDimOp op) { dims.push_back(op); });
  for (gpu::BlockDimOp op : dims) {
    gpu::Dimension dim = op.getDimension();
    const char *name =
        dim == gpu::Dimension::x ? "__vc_wg_x"
        : dim == gpu::Dimension::y ? "__vc_wg_y"
                                   : "__vc_wg_z";
    OpBuilder b(op);
    Value ref = b.create<spirv::ReferenceOfOp>(op.getLoc(), i32, name);
    Value idx = b.create<arith::IndexCastOp>(op.getLoc(),
                                             builder.getIndexType(), ref);
    op.replaceAllUsesWith(idx);
    op.erase();
  }
}

// Packs every vc.kernel (and its target func) into one gpu.module as gpu.func.
void packKernels(ModuleOp module, IRRewriter &rw) {
  SmallVector<vc::KernelOp> kernels;
  module.walk([&](vc::KernelOp k) { kernels.push_back(k); });
  if (kernels.empty()) return;

  gpu::GPUModuleOp gpuModule;
  // Pre-scan every kernel's signature for a by-value f16 vector param
  // (`__half2 v`): such a param becomes a PushConstant struct member
  // vector<Nxf16>, which needs StoragePushConstant16 advertised up front in
  // the target env (see getVCTargetEnv). Computed once over all kernels.
  auto kernelUsesF16VectorParam = [&]() {
    for (vc::KernelOp k : kernels) {
      func::FuncOp fn = module.lookupSymbol<func::FuncOp>(k.getFunction());
      if (!fn) continue;
      for (Type ty : fn.getFunctionType().getInputs()) {
        if (auto vty = ty.dyn_cast<mlir::VectorType>();
            vty && vty.getElementType().isF16())
          return true;
      }
    }
    return false;
  };
  bool usesF16Storage = kernelUsesF16VectorParam();
  // Which numeric widths the kernels use, computed once over the module: the
  // target env is one attribute on the single gpu.module every kernel lands in.
  SPIRVWidthUse widths = scanWidthUses(module);
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
    // Bump to 1.3 + subgroup capabilities when the module uses warp/subgroup
    // intrinsics (flagged by ASTToMLIR via the vc.uses_subgroup module attr).
    bool usesSubgroup = module->hasAttr("vc.uses_subgroup");
    bool usesCoopMatrix = module->hasAttr("vc.uses_coopmatrix");
    gpuModule->setAttr(spirv::getTargetEnvAttrName(),
                       getVCTargetEnv(module.getContext(), usesSubgroup,
                                      usesCoopMatrix, usesF16Storage, widths));

    rw.setInsertionPointToStart(gpuModule.getBody());
    auto gpuFn = rw.create<gpu::GPUFuncOp>(k.getLoc(), fn.getName(),
                                           fn.getFunctionType());
    // Mark the entry point: gpu.kernel + the SPIR-V launch ABI. The workgroup
    // size here is a placeholder (1x1x1): the actual block size is supplied at
    // run time via specialization constants (SpecId 0/1/2), wired up by
    // lowerWorkgroupSizeToSpecConstants + the vc driver's binary patch that
    // decorates the gl_WorkGroupSize spec-composite BuiltIn. LocalSize is
    // overridden by that builtin, exactly as in the GLSL backend's shader.
    gpuFn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), rw.getUnitAttr());
    gpuFn->setAttr(spirv::getEntryPointABIAttrName(),
                   spirv::getEntryPointABIAttr(module.getContext(),
                                               {1, 1, 1}));

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

    // Lower scalar (non-memref) args to a push-constant struct and assign
    // StorageBuffer bindings to pointer args in order. Done after takeBody
    // because it rewrites the kernel body (loads scalars from the PC global)
    // and reshapes the gpu.func signature. See lowerScalarArgsToPushConstant.
    lowerScalarArgsToPushConstant(gpuFn, rw);

    fn.erase();
    k.erase();
  }

  // Replace gpu.block_dim with specialization-constant reads so the block size
  // is set at run time (via Vulkan spec constants) rather than baked to the
  // placeholder LocalSize. Creates the spec constants + rewrites the ops.
  if (gpuModule)
    lowerWorkgroupSizeToSpecConstants(gpuModule, rw);

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

  // Hoist __device__ helper functions (func.func) into the gpu.module. Kernels
  // were consumed above (their func.func erased and re-emitted as gpu.func), so
  // any func.func left at module scope is a __device__ helper called from a
  // kernel body via func.call. GPUToSPIRV only sees symbols inside the gpu.module
  // it clones, and ConvertFuncToSPIRV (run before GPUToSPIRV) turns these
  // func.func + func.call into spirv.func + spirv.FunctionCall. Without hoisting
  // the callee, `func.call @helper` fails `does not reference a valid function`.
  SmallVector<func::FuncOp> helpers;
  module.walk([&](func::FuncOp f) {
    if (!f->hasAttr("vc.kernel"))
      helpers.push_back(f);
  });
  for (func::FuncOp f : helpers) {
    f->moveBefore(&gpuModule.getBody()->front());
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