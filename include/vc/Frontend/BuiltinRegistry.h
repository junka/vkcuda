//===- BuiltinRegistry.h - Central CUDA/GLSL builtin name registry --------===//
//
// One source of truth for "is this name a builtin" across Sema and the GLSL/
// MLIR backends. Sema uses isBuiltin() to skip the undeclared-function warning
// for builtin calls; backends use builtinClass() to dispatch lowering.
//
// The registry ONLY classifies names — it does not lower them. GLSL lowering
// (name -> string) and MLIR lowering (name -> spirv op) stay in their backends,
// because their output forms are incompatible. Centralizing the *names* here
// means adding a builtin is a one-place change instead of three parallel lists
// (Sema isMathBuiltin, GLSL isAtomicName/isWarpIntrinsicName/..., MLIR
// emitXxxBuiltin entry checks) that must be kept in sync by hand.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_BUILTINREGISTRY_H
#define VC_FRONTEND_BUILTINREGISTRY_H

#include "llvm/ADT/StringRef.h"

namespace vc {

enum class BuiltinClass {
  None,        // not a builtin
  Math,        // sin/cos/sqrt/__sinf/sinf/... + geometric (dot/cross/length/...)
  ThreadIndex, // threadIdx/blockIdx/blockDim/gridDim/warpSize
  Atomic,      // atomicAdd/atomicSub/atomicExch/atomicMin/atomicMax/atomicInc/
               // atomicDec/atomicCAS/atomicAnd/atomicOr/atomicXor
  Sync,        // __syncthreads/__threadfence/__threadfence_block/
               // __syncthreads_count/__syncthreads_and/__syncthreads_or
  Warp,        // __syncwarp/__ballot_sync/__anySync/__allSync/__activemask/
               // __shfl_sync/__shfl_up_sync/__shfl_down_sync/__shfl_xor_sync
  AsyncCopy,   // vcMemcpyAsync/vcPipelineProducerCommit/vcPipelineConsumerWait/
               // vcPipelineConsumerCommit
  VectorCtor,  // float4/int3/uint4/.../make_float4/...
  Dim,         // dim3 (launch grid/block constructor)
  Other,       // printf (kernel-internal debug output)
};

/// True if `name` is any recognized builtin (math, atomics, sync, warp,
/// thread-index, async-copy, vector ctor, dim3, printf). Sema uses this to
/// avoid flagging builtin calls as undeclared functions. Vector constructors
/// (float4, make_float4, ...) are recognized by pattern.
bool isBuiltin(llvm::StringRef name);

/// The class of a builtin name, or BuiltinClass::None if not a builtin.
/// Backends use this to dispatch lowering (each class has its own emit path).
BuiltinClass builtinClass(llvm::StringRef name);

/// True if `name` is a CUDA-style vector constructor: <base><2..4> where base
/// is float/int/uint/double/bool/long/ulong/half (e.g. float4, int3), or a
/// make_<vec> form (make_float4). Exposed so the GLSL backend can share the
/// recognition (it maps the matched base to the GLSL vec/ivec/... spelling).
bool isVectorCtorName(llvm::StringRef name);

} // namespace vc

#endif // VC_FRONTEND_BUILTINREGISTRY_H
