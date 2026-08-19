//===- VC.td - VC dialect definition --------------------------*- TableGen -*-===//
//
// The VC dialect models CUDA-like GPU programming semantics (kernels, thread
// indexing, barriers, WMMA/tensor-core) so they can be lowered to SPIR-V /
// Vulkan. This is the device-side surface of the language.
//
//===----------------------------------------------------------------------===//

#ifndef VC_DIALECT
#define VC_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def VC_Dialect : Dialect {
  let name = "vc";
  let summary = "CUDA-like GPU dialect targeting Vulkan/SPIR-V";
  let description = [{
    The `vc` dialect is the device-side IR for the VC language. It models
    CUDA concepts (kernels, thread/block indices, shared memory, barriers,
    WMMA fragments) and lowers through the GPU and SPIR-V dialects to a
    Vulkan compute shader.
  }];
  let cppNamespace = "::vc";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Built-in dimension enum (x=0, y=1, z=2), mirrors CUDA threadIdx.x etc.
//===----------------------------------------------------------------------===//

def VCDim : I32EnumAttr<
  "Dim", "a thread/block dimension index (x/y/z)",
  [ I32EnumAttrCase<"x", 0>,
    I32EnumAttrCase<"y", 1>,
    I32EnumAttrCase<"z", 2> ]> {
  let cppNamespace = "::vc";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class VC_Op<string mnemonic, list<Trait> traits = []> :
  Op<VC_Dialect, mnemonic, traits>;

// Marks a function as a device entry point (CUDA __global__). The function
// body uses vc.* indexing ops. Lowers to spirv.EntryPoint(GLCompute).
def KernelOp : VC_Op<"kernel"> {
  let summary = "mark a function as a launchable device kernel";
  let description = [{
    Wraps a function symbol, declaring it an entry point callable from the
    host via vcLaunchKernel (which becomes vkCmdDispatch). Mirrors CUDA's
    `__global__`.
  }];
  let arguments = (ins SymbolRefAttr:$function);
  let assemblyFormat = "$function attr-dict";
}

// CUDA threadIdx.x / .y / .z  ->  spirv.Builtin LocalInvocationId
def ThreadIdOp : VC_Op<"thread_id"> {
  let summary = "index within a workgroup (CUDA threadIdx)";
  let arguments = (ins VCDim:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict";
}

// CUDA blockIdx.x / .y / .z  ->  spirv.Builtin WorkgroupId
def BlockIdOp : VC_Op<"block_id"> {
  let summary = "workgroup index (CUDA blockIdx)";
  let arguments = (ins VCDim:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict";
}

// CUDA blockDim.x / .y / .z  ->  spirv.Builtin WorkgroupSize
def BlockDimOp : VC_Op<"block_dim"> {
  let summary = "workgroup size (CUDA blockDim)";
  let arguments = (ins VCDim:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict";
}

// CUDA gridDim.x / .y / .z  ->  spirv.Builtin NumWorkgroups
def GridDimOp : VC_Op<"grid_dim"> {
  let summary = "grid size (CUDA gridDim)";
  let arguments = (ins VCDim:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict";
}

// CUDA __syncthreads()  ->  spirv.ControlBarrier(Workgroup)
def BarrierOp : VC_Op<"barrier", [MemoryEffectsOpInterface]> {
  let summary = "workgroup synchronization barrier (CUDA __syncthreads)";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// WMMA / Tensor Core placeholders (lowering deferred — user will extend).
// These map to spirv.KHR.CooperativeMatrix when implemented.
//===----------------------------------------------------------------------===//

def WMMALoadOp : VC_Op<"wmma.load"> {
  let summary = "load a WMMA fragment (placeholder)";
  let description = [{
    Loads a matrix fragment into tensor-core registers. Placeholder op;
    lowering to spirv.KHR.CooperativeMatrix is deferred.
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs AnyType:$fragment);
  let assemblyFormat = "$memref attr-dict type($fragment)";
}

def WMMAStoreOp : VC_Op<"wmma.store"> {
  let summary = "store a WMMA fragment (placeholder)";
  let arguments = (ins AnyType:$fragment, AnyMemRef:$memref);
  let assemblyFormat = "$fragment `,` $memref attr-dict";
}

def WMMAMmaOp : VC_Op<"wmma.mma"> {
  let summary = "WMMA multiply-accumulate (placeholder)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict type($result)";
}

#endif // VC_DIALECT
