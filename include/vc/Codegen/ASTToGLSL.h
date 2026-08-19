//===- ASTToGLSL.h - Translate AST to a GLSL compute shader --------------===//
//
// Emits a GLSL `#version 460 compute` shader from a VC translation unit.
// This is the no-MLIR path: the GLSL is fed to glslc to produce SPIR-V,
// which the Vulkan runtime loads. Covers the vector-add subset.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_ASTTOGLSL_H
#define VC_CODEGEN_ASTTOGLSL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace vc {
class TranslationUnit;

namespace glsl {

/// Emit a GLSL compute shader for the first __global__ kernel in `tu`.
/// `outWorkgroupSize` receives the workgroup size declared in the shader
/// (defaults to 32 for the scaffold). Returns true on success.
bool translateASTToGLSL(const TranslationUnit &tu, llvm::raw_ostream &os);

} // namespace glsl
} // namespace vc

#endif // VC_CODEGEN_ASTTOGLSL_H
