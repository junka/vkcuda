//===- BuiltinRegistry.cpp - Central CUDA/GLSL builtin name registry ------===//
//
// Implementation of the shared builtin name registry. The name lists here are
// the union of the three previously-duplicated sites (Sema isMathBuiltin /
// isThreadBuiltin, and the backend name-classification helpers), so this
// refactor changes which builtins are recognized by neither site.
//
//===----------------------------------------------------------------------===//

#include "vc/Frontend/BuiltinRegistry.h"

using namespace vc;

namespace {

// CUDA-style vector base names: <base><2..4> e.g. float4, int3, ulong2.
// Mirrors the set Parser::makeVectorType and Sema::isVectorCtorName used, so
// recognition stays identical. `__half` is alongside `half` so the legacy
// `__half2`/`__half3`/`__half4` spellings are recognized too.
const char *kVectorBases[] = {"float", "int", "uint", "double",
                              "bool",  "long", "ulong", "half", "__half"};

bool isPlainVectorCtorName(llvm::StringRef name) {
  for (const char *b : kVectorBases) {
    llvm::StringRef p = b;
    if (name.size() == p.size() + 1 && name.starts_with(p)) {
      char d = name.back();
      if (d >= '2' && d <= '4') return true;
    }
  }
  return false;
}

} // namespace

bool vc::isVectorCtorName(llvm::StringRef name) {
  if (isPlainVectorCtorName(name))
    return true;
  // make_<vec>(...) constructors -> vec constructors.
  if (name.starts_with("make_"))
    return isPlainVectorCtorName(name.substr(5));
  return false;
}

BuiltinClass vc::builtinClass(llvm::StringRef name) {
  // Thread-index builtins (implicit in every kernel).
  if (name == "threadIdx" || name == "blockIdx" || name == "blockDim" ||
      name == "gridDim" || name == "warpSize")
    return BuiltinClass::ThreadIndex;

  // CUDA atomics.
  if (name == "atomicAdd" || name == "atomicSub" || name == "atomicExch" ||
      name == "atomicMin" || name == "atomicMax" || name == "atomicInc" ||
      name == "atomicDec" || name == "atomicCAS" || name == "atomicAnd" ||
      name == "atomicOr" || name == "atomicXor")
    return BuiltinClass::Atomic;

  // CUDA synchronization primitives.
  if (name == "__syncthreads" || name == "__threadfence" ||
      name == "__threadfence_block" || name == "__syncthreads_count" ||
      name == "__syncthreads_and" || name == "__syncthreads_or")
    return BuiltinClass::Sync;

  // CUDA warp intrinsics.
  if (name == "__syncwarp" || name == "__ballot_sync" || name == "__anySync" ||
      name == "__allSync" || name == "__activemask" || name == "__shfl_sync" ||
      name == "__shfl_up_sync" || name == "__shfl_down_sync" ||
      name == "__shfl_xor_sync")
    return BuiltinClass::Warp;

  // VC async-copy approximation builtins.
  if (name == "vcMemcpyAsync" || name == "vcPipelineProducerCommit" ||
      name == "vcPipelineConsumerWait" || name == "vcPipelineConsumerCommit")
    return BuiltinClass::AsyncCopy;

  // Launch dimension constructor.
  if (name == "dim3")
    return BuiltinClass::Dim;

  // Kernel-internal printf.
  if (name == "printf")
    return BuiltinClass::Other;

  // Vector constructors (float4, make_float4, ...).
  if (isVectorCtorName(name))
    return BuiltinClass::VectorCtor;

  // Math / GLSL builtins. Permissive set so sinf(...)/__syncthreads()/etc.
  // don't get flagged as unknown functions. Real resolution would need a full
  // signature table (P3A overload work); this covers the kernels we ship.
  static const char *kMath[] = {
      "sin",   "cos",   "tan",   "asin",  "acos",  "atan",  "exp",
      "log",   "pow",   "sqrt",  "abs",   "fabs",  "fmin",  "fmax",
      "min",   "max",   "floor", "ceil",  "fract", "mix",   "clamp",
      "step",  "smoothstep", "mod", "fma", "trunc", "round", "sign",
      "inversesqrt", "isnan", "isinf", "exp2", "log2", "degrees", "radians",
      "sinf",  "cosf",  "tanf",  "asinf", "acosf", "atanf", "expf",
      "logf",  "powf",  "sqrtf", "fabsf", "fminf", "fmaxf", "floorf",
      "ceilf", "__sinf", "__cosf", "__expf", "__logf", "__powf", "__fabsf",
      // GLSL vector/geometric builtins.
      "dot",   "cross", "length", "normalize", "reflect", "refract",
      "distance", "faceforward", "all", "any", "lessThan", "greaterThan",
  };
  for (const char *m : kMath)
    if (name == m) return BuiltinClass::Math;

  return BuiltinClass::None;
}

bool vc::isBuiltin(llvm::StringRef name) {
  return builtinClass(name) != BuiltinClass::None;
}
