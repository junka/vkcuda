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
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

namespace vc {
class TranslationUnit;

namespace host {

/// One device kernel's compiled SPIR-V, paired with its device symbol name
/// (the GLSL entry-function name = the `__global__` kernel's mangled device
/// name, e.g. `vadd` or `ns_kernel`). The host backend embeds each module as a
/// separate `static const uint32_t __vc_spirv_<name>[]` and loads it via
/// `vcLoadKernel(..., "<name>", ...)` at the matching launch site.
struct HostSpirvModule {
  std::string kernelName;
  const uint32_t *words = nullptr;
  size_t wordCount = 0;
};

/// Emit a self-contained C++ source for the host subset of `tu` to `os`.
/// `modules` is the per-kernel SPIR-V (one entry per `__global__`), embedded as
/// `static const uint32_t[]` arrays and loaded via vcLoadKernel at each launch.
/// Returns false if there is no host `main` function to emit.
bool translateASTToHost(const TranslationUnit &tu,
                        const std::vector<HostSpirvModule> &modules,
                        llvm::raw_ostream &os);

/// Single-module convenience overload (one kernel). Embeds the SPIR-V as
/// `__vc_spirv[]` and loads it with entry name `kernelName` (or `"main"` if
/// empty). Kept for compatibility.
bool translateASTToHost(const TranslationUnit &tu,
                        const uint32_t *spirvWords, size_t wordCount,
                        llvm::raw_ostream &os);

} // namespace host
} // namespace vc

#endif // VC_CODEGEN_ASTTOHOST_H
