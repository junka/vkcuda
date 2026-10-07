//===- ASTToGLSL.h - Translate AST to a GLSL compute shader --------------===//
//
// Emits a GLSL `#version 460 compute` shader from a VC translation unit.
// This is the no-MLIR path: the GLSL is fed to glslc to produce SPIR-V,
// which the Vulkan runtime loads. Covers the vector-add subset.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_ASTTOGLSL_H
#define VC_CODEGEN_ASTTOGLSL_H

#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace vc {
class TranslationUnit;

namespace glsl {

/// One compiled GLSL compute unit. A `.vc` file may contain several
/// `__global__` kernels; each is lowered to its own independent `.comp`
/// (own `#version`/bindings/push-constants/`local_size`/entry function), since
/// GLSL forbids more than one of each per compilation unit. `entryName` is the
/// GLSL entry-function name (= the kernel's device symbol name, e.g. `vadd` or
/// `ns_kernel`), which the host passes to glslc's `-fentry-point` and to the
/// runtime's `vcLoadKernel`.
struct GLSLModule {
  std::string entryName;
  std::string source;
  // Comma-separated names of the `double`-argument transcendental calls this
  // unit contains (empty when there are none). Vulkan GLSL has no double
  // overload for them, so the driver reports them and stops rather than letting
  // glslc emit an opaque "no matching overloaded function found".
  std::string unsupportedF64Math;
};

/// Lower every `__global__` in `tu` to its own GLSL compute unit. Returns one
/// `GLSLModule` per kernel (empty if there are no kernels). The order matches
/// source order of the `__global__` declarations.
///
/// `allowF64MathF32` (-fallow-f64-math-f32) lowers a `double` transcendental
/// (`sin(d)`, `pow(d, e)`, ...) to its f32 form, `double(sin(float(d)))`.
/// Vulkan GLSL has no double overload for those builtins, so without the flag
/// the kernel carries a `#error` saying so instead of glslc's opaque one.
std::vector<GLSLModule> translateASTToGLSLSources(const TranslationUnit &tu,
                                                  bool allowF64MathF32 = false);

/// Convenience wrapper: emit the first `__global__` kernel to `os`. Returns
/// true on success. Kept for single-kernel / `-emit=glsl` debug use.
bool translateASTToGLSL(const TranslationUnit &tu, llvm::raw_ostream &os,
                        bool allowF64MathF32 = false);

} // namespace glsl
} // namespace vc

#endif // VC_CODEGEN_ASTTOGLSL_H
