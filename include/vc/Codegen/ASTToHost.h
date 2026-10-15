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
/// `vcLoadKernel(..., "<entryPoint>", ...)` at the matching launch site.
///
/// `entryPoint` is the SPIR-V entry-point name passed to vcLoadKernel. The
/// GLSL backend emits one .comp module per kernel with entry "main", so it
/// leaves `entryPoint` empty (resolved to "main"). The MLIR backend emits a
/// single SPIR-V binary with one OpEntryPoint per kernel named after the
/// kernel symbol, so it fills `entryPoint` with the kernel name.
struct HostSpirvModule {
  std::string kernelName;
  const uint32_t *words = nullptr;
  size_t wordCount = 0;
  std::string entryPoint;
};

/// One offline (Vulkan SC) pipeline the host code can load.
///
/// Vulkan SC has no runtime pipeline compilation and no SPIR-V in the process:
/// nvidia-pcc compiles every pipeline ahead of time into a pipeline cache, and
/// the runtime selects one by a 16-byte UUID. Everything the runtime needs to
/// describe the pipeline — and everything pcc needed when it compiled it — is in
/// this struct, so the two cannot drift: the driver builds one from the same
/// values it writes into the pcc JSON.
///
/// The host emitter turns each entry into a file-scope
/// `static const VCScPipelineDesc __vc_sc_<kernel>_b<x>_<y>_<z>`, a
/// `static VCKernelHandle __vc_k_<kernel>_b<x>_<y>_<z>` to load it into, and a
/// `vcLoadKernelOffline(&<desc>, &<handle>)` call at the launch site whose block
/// size matches, instead of the `vcLoadKernel` + embedded-SPIR-V prologue the
/// default backends use. The handle carries the block size because that is baked
/// into the pipeline: two block sizes are two handles. A kernel whose bare
/// `__vc_k_<kernel>` name the host source mentions (an explicit handle passed to
/// `vcLaunchKernelIndirect` / `vcOccupancyMaxPotentialBlockSize`) also gets a
/// reference alias to its first block size.
struct HostScPipeline {
  std::string kernelName; // device symbol (`vadd`, `ns_k`) — keys the handle
  std::string entryPoint; // SPIR-V OpEntryPoint name (the same symbol for MLIR)
  unsigned char uuid[16] = {}; // offline pipeline identifier (matches the JSON)
  uint32_t blockX = 1, blockY = 1, blockZ = 1;
  // Descriptor bindings: `storageBufferBindingCount` STORAGE_BUFFER bindings
  // (bindings 0..N-1) followed by `imageBindingCount` STORAGE_IMAGE ones
  // (N..N+M-1 — absolute binding numbers, so the image sequence starts where the
  // SSBO sequence ends).
  uint32_t storageBufferBindingCount = 0;
  uint32_t imageBindingCount = 0;
  // One push-constant range of this size at offset 0 (0 = no scalars).
  uint32_t pushConstantSize = 0;
};

/// Emit a self-contained C++ source for the host subset of `tu` to `os`.
/// `modules` is the per-kernel SPIR-V (one entry per `__global__`), embedded as
/// `static const uint32_t[]` arrays and loaded via vcLoadKernel at each launch.
/// Returns false if there is no host `main` function to emit.
///
/// `scPipelines`, when non-null, switches to the Vulkan SC shape: the SPIR-V is
/// not embedded at all and each launch loads a precompiled pipeline through
/// vcLoadKernelOffline. The vector must carry one entry per (kernel, block size)
/// the host launches — the driver produces it from the same table it writes the
/// pcc JSON from.
bool translateASTToHost(const TranslationUnit &tu,
                        const std::vector<HostSpirvModule> &modules,
                        llvm::raw_ostream &os,
                        const std::vector<HostScPipeline> *scPipelines = nullptr);

/// Single-module convenience overload (one kernel). Embeds the SPIR-V as
/// `__vc_spirv[]` and loads it with entry name `kernelName` (or `"main"` if
/// empty). Kept for compatibility.
bool translateASTToHost(const TranslationUnit &tu,
                        const uint32_t *spirvWords, size_t wordCount,
                        llvm::raw_ostream &os);

} // namespace host
} // namespace vc

#endif // VC_CODEGEN_ASTTOHOST_H
