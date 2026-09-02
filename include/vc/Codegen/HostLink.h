//===- HostLink.h - host C++ generation + g++ linking ---------------------===//
//
// Shared between the GLSL driver (`vcc`) and the MLIR driver (`vc`): given a
// translation unit and per-kernel SPIR-V, emit a self-contained host .cpp
// (via ASTToHost) and drive g++ to link it against libVCRuntime + Vulkan into
// a standalone executable. This is the last stage of single-file CUDA-style
// compilation; the device SPIR-V comes from whichever device backend is in
// use (GLSL+glslc for vcc, MLIR lowering+serialization for vc).
//
// The VC_INCLUDE_DIR / VC_RUNTIME_LIB / VC_VULKAN_LIBS locations are baked in
// at build time via target_compile_definitions so the driver can invoke g++
// with the right -I / .a / -l flags without requiring an install step.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_HOSTLINK_H
#define VC_CODEGEN_HOSTLINK_H

#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "vc/Codegen/ASTToHost.h"

namespace vc {
class TranslationUnit;

namespace host {

/// Generate the host .cpp (translateASTToHost) and, unless `emitHostOnly`,
/// drive g++ to link it against libVCRuntime + Vulkan into `outputFilename`.
///
/// `emitHostOnly=true` prints the generated .cpp to stdout and returns without
/// linking. Returns 0 on success, non-zero on failure (diagnostics already
/// printed). The `tu` must contain a host `int main()`; returns non-zero if
/// not. Requires the VC_INCLUDE_DIR / VC_RUNTIME_LIB / VC_VULKAN_LIBS macros
/// to be defined at compile time of the calling driver.
int linkHostExecutable(const TranslationUnit &tu,
                       const std::vector<HostSpirvModule> &modules,
                       llvm::StringRef outputFilename, bool emitHostOnly);

} // namespace host
} // namespace vc

#endif // VC_CODEGEN_HOSTLINK_H
