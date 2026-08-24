//===- ASTToHost.h - Lower host subset of AST to C++ source ---------------===//
//
// Single-file CUDA-style compilation: a `.vc` file may contain both
// `__global__` kernels (lowered to SPIR-V by the GLSL backend) and host code
// (`int main()`, `__host__` functions) using `kernel<<<g,b>>>(args)` launches.
// This backend lowers the host subset (`DeviceAttr::Host`/`None`) to a
// self-contained C++ source that calls the VC runtime, with the kernel SPIR-V
// embedded as a static byte array.
//
// The generated .cpp is fed to g++ and linked against libVCRuntime + Vulkan.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_ASTTOHOST_H
#define VC_CODEGEN_ASTTOHOST_H

#include <cstddef>
#include <cstdint>

#include "llvm/Support/raw_ostream.h"

namespace vc {
class TranslationUnit;

namespace host {

/// Emit a self-contained C++ source for the host subset of `tu` to `os`.
/// `spirvWords`/`wordCount` is the already-compiled SPIR-V for the device
/// subset (kernels), embedded as a `static const uint32_t[]` and loaded via
/// vcLoadKernel at the start of `main`. Returns false if there is no host
/// `main` function to emit.
bool translateASTToHost(const TranslationUnit &tu,
                        const uint32_t *spirvWords, size_t wordCount,
                        llvm::raw_ostream &os);

} // namespace host
} // namespace vc

#endif // VC_CODEGEN_ASTTOHOST_H
