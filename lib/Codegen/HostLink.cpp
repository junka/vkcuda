//===- HostLink.cpp - host C++ generation + g++ linking -------------------===//
//
// Implements linkHostExecutable: the shared final stage of single-file
// CUDA-style compilation. Emits a self-contained host .cpp embedding the
// per-kernel SPIR-V (via ASTToHost) and drives g++ to link it against
// libVCRuntime + Vulkan into a standalone executable.
//
// The VC_INCLUDE_DIR / VC_RUNTIME_LIB / VC_VULKAN_LIBS macros must be defined
// at compile time (see tools/vcc/CMakeLists.txt and tools/vc/CMakeLists.txt).
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/HostLink.h"

#include "vc/Frontend/AST.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace vc;
using namespace llvm;

namespace vc::host {

// Locate g++; fall back to a couple of common absolute paths if PATH lookup
// fails. The driver needs a working g++ to produce an executable.
static std::string findGpp() {
  if (auto p = sys::findProgramByName("g++")) return *p;
  for (const char *fb : {"/usr/bin/g++", "/usr/local/bin/g++"})
    if (sys::fs::exists(fb)) return fb;
  return {};
}

int linkHostExecutable(const TranslationUnit &tu,
                       const std::vector<HostSpirvModule> &modules,
                       StringRef outputFilename, bool emitHostOnly) {
  // 1. Host subset -> C++ (embedding one SPIR-V module per kernel).
  std::string cppSource;
  raw_string_ostream cppOS(cppSource);
  if (!translateASTToHost(tu, modules, cppOS)) {
    errs() << "no host main() found in the input"
           << " (single-file mode needs an int main())\n";
    return 1;
  }
  cppOS.flush();

  if (emitHostOnly) {
    outs() << cppSource;
    return 0;
  }

  // 2. g++ compiles the .cpp -> executable, linking VCRuntime + Vulkan.
  //    VC_RUNTIME_LIB / VC_INCLUDE_DIR / VC_VULKAN_LIBS are baked in at build
  //    time so no install step is required.
#ifndef VC_RUNTIME_LIB
  errs() << "error: VC_RUNTIME_LIB not configured for this driver\n";
  return 1;
#endif
  std::string gpp = findGpp();
  if (gpp.empty() || !sys::fs::exists(gpp)) {
    errs() << "error: g++ not found\n";
    return 1;
  }

  SmallString<128> cppPath;
  sys::fs::createTemporaryFile("vchost", "cpp", cppPath);
  {
    std::error_code ec;
    raw_fd_ostream cppFile(cppPath, ec);
    if (ec) {
      errs() << "cannot write host temp: " << ec.message() << "\n";
      return 1;
    }
    cppFile << cppSource;
  }

  SmallVector<StringRef, 16> args;
  args.push_back(gpp);
  args.push_back("-std=c++20");
  args.push_back("-O2");
  args.push_back("-I" VC_INCLUDE_DIR);
  args.push_back(cppPath);
  args.push_back(VC_RUNTIME_LIB);
  args.push_back(VC_VULKAN_LIBS);
  args.push_back("-o");
  args.push_back(outputFilename);

  int grc = sys::ExecuteAndWait(gpp, args, std::nullopt, std::nullopt);
  if (grc != 0) {
    errs() << "g++ failed (rc=" << grc << ")\n--- generated host C++ ---\n"
           << cppSource;
    sys::fs::remove(cppPath);
    return 1;
  }
  sys::fs::remove(cppPath);

  size_t totalSpv = 0;
  for (const auto &m : modules) totalSpv += m.wordCount * sizeof(uint32_t);
  outs() << "built " << outputFilename << " (" << cppSource.size()
         << " bytes host C++, " << totalSpv << " bytes SPIR-V across "
         << modules.size() << " kernel"
         << (modules.size() == 1 ? "" : "s") << ")\n";
  return 0;
}

} // namespace vc::host
