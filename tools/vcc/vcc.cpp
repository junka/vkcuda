//===- vcc.cpp - Single-file CUDA-style compiler driver --------------------===//
//
// Like nvcc: one `.vc` file holds `__global__` kernels + host `int main()` +
// `kernel<<<g,b>>>(args)` launch sites. This driver:
//   1. lex+parse -> AST (+ Sema, host functions skipped)
//   2. device subset (__global__/__device__) -> GLSL -> glslc -> SPIR-V bytes
//   3. host subset (Host/None, incl main) -> self-contained C++ embedding SPIR-V
//   4. g++ compiles the .cpp, links libVCRuntime + Vulkan -> executable
//
// Options:
//   vcc <input.vc> -o <prog>          full pipeline -> executable
//   vcc <input.vc> -emit=host         print generated host .cpp, don't link
//   vcc <input.vc> -emit=glsl         print generated device GLSL
//   vcc <input.vc> -emit=spirv -o x.spv   write device SPIR-V
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/ASTToGLSL.h"
#include "vc/Codegen/ASTToHost.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Parser.h"
#include "vc/Frontend/Sema.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vc;
using namespace llvm;

namespace {
enum class EmitKind { Full, Host, GLSL, SPIRV };

static std::string findTool(const char *name) {
  if (auto p = sys::findProgramByName(name)) return *p;
  const char *fallbacks[] = {
      "/usr/bin/glslc", "/usr/local/bin/glslc",
      "/usr/bin/g++",   "/usr/local/bin/g++",
  };
  (void)fallbacks;
  return {};
}
} // namespace

int main(int argc, char **argv) {
  InitLLVM x(argc, argv);

  cl::opt<std::string> inputFilename(cl::Positional,
      cl::desc("<input .vc file>"), cl::Required);
  cl::opt<std::string> emitOpt("emit",
      cl::desc("output kind (host|glsl|spirv|full)"), cl::init("full"));
  cl::opt<std::string> outputFilename("o",
      cl::desc("output filename (executable or .spv)"), cl::init("a.out"));
  cl::opt<std::string> entryPoint("entry",
      cl::desc("SPIR-V entry point name"), cl::init("main"));
  cl::ParseCommandLineOptions(argc, argv, "VC single-file compiler\n");

  EmitKind kind = EmitKind::Full;
  if (emitOpt == "full") kind = EmitKind::Full;
  else if (emitOpt == "host") kind = EmitKind::Host;
  else if (emitOpt == "glsl") kind = EmitKind::GLSL;
  else if (emitOpt == "spirv") kind = EmitKind::SPIRV;
  else { errs() << "unknown -emit=" << emitOpt << "\n"; return 1; }

  // 1. Parse
  SourceMgr sm;
  auto buf = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = buf.getError()) {
    errs() << "error: cannot open " << inputFilename << ": " << ec.message()
           << "\n";
    return 1;
  }
  sm.AddNewSourceBuffer(std::move(*buf), SMLoc());

  Lexer lex(sm, sm.getMainFileID());
  SourceLocation start;
  TranslationUnit tu{start};
  Parser parser(lex, sm, tu);
  if (!parser.parseTranslationUnit()) {
    errs() << "parse failed\n";
    return 1;
  }
  Sema sema(tu);
  if (!sema.analyze()) {
    errs() << "sema: aborting due to errors\n";
    return 1;
  }

  if (emitOpt == "ast") { dumpAST(tu, outs()); return 0; }

  // 2. Device subset -> GLSL
  std::string glslSource;
  raw_string_ostream glslOS(glslSource);
  if (!glsl::translateASTToGLSL(tu, glslOS)) {
    errs() << "no __global__ kernel found in " << inputFilename << "\n";
    return 1;
  }
  glslOS.flush();

  if (kind == EmitKind::GLSL) {
    outs() << glslSource;
    return 0;
  }

  // 3. GLSL -> SPIR-V via glslc
  std::string glslc = findTool("glslc");
  if (glslc.empty()) {
    errs() << "error: glslc not found (install shaderc)\n";
    return 1;
  }
  SmallString<128> glslPath;
  sys::fs::createTemporaryFile("vckernel", "comp", glslPath);
  {
    std::error_code ec;
    raw_fd_ostream glslFile(glslPath, ec);
    if (ec) { errs() << "cannot write temp: " << ec.message() << "\n"; return 1; }
    glslFile << glslSource;
  }
  SmallString<128> spvPath;
  sys::fs::createTemporaryFile("vckernel", "spv", spvPath);
  std::string entryArg = std::string("-fentry-point=") + std::string(entryPoint);
  // Subgroup ops (CUDA warp intrinsics) require SPIR-V 1.3 = vulkan1.1 target.
  // The GLSL backend flags this with a "// vc:needs-spv1.3" header marker.
  bool needsSpv13 = glslSource.find("vc:needs-spv1.3") != std::string::npos;
  auto runGlslc = [&](bool withEntry) -> int {
    SmallVector<StringRef, 8> args;
    args.push_back(glslc);
    args.push_back("-fshader-stage=compute");
    if (needsSpv13) args.push_back("--target-env=vulkan1.1");
    args.push_back(glslPath);
    args.push_back("-o");
    args.push_back(spvPath);
    if (withEntry) args.push_back(entryArg);
    return sys::ExecuteAndWait(glslc, args, std::nullopt, std::nullopt);
  };
  int rc = runGlslc(/*withEntry=*/true);
  if (rc != 0) rc = runGlslc(/*withEntry=*/false);
  sys::fs::remove(glslPath);
  if (rc != 0) {
    errs() << "glslc failed (rc=" << rc << ")\n--- GLSL source ---\n"
           << glslSource;
    sys::fs::remove(spvPath);
    return 1;
  }

  // Read the SPIR-V bytes.
  auto spvBuf = MemoryBuffer::getFile(spvPath);
  if (auto ec = spvBuf.getError()) {
    errs() << "cannot read spirv: " << ec.message() << "\n";
    sys::fs::remove(spvPath);
    return 1;
  }
  size_t spvBytes = (*spvBuf)->getBufferSize();
  size_t wordCount = spvBytes / sizeof(uint32_t);
  const uint32_t *words =
      reinterpret_cast<const uint32_t *>((*spvBuf)->getBufferStart());

  if (kind == EmitKind::SPIRV) {
    std::error_code ec;
    raw_fd_ostream out(outputFilename, ec);
    if (ec) { errs() << "cannot write " << outputFilename << ": "
                     << ec.message() << "\n"; return 1; }
    out.write((*spvBuf)->getBufferStart(), spvBytes);
    sys::fs::remove(spvPath);
    outs() << "wrote " << outputFilename << " (" << spvBytes << " bytes)\n";
    return 0;
  }

  // 4. Host subset -> C++ (embedding SPIR-V)
  std::string cppSource;
  raw_string_ostream cppOS(cppSource);
  if (!host::translateASTToHost(tu, words, wordCount, cppOS)) {
    errs() << "no host main() found in " << inputFilename
           << " (single-file mode needs an int main())\n";
    sys::fs::remove(spvPath);
    return 1;
  }
  cppOS.flush();
  sys::fs::remove(spvPath);

  if (kind == EmitKind::Host) {
    outs() << cppSource;
    return 0;
  }

  // 5. g++ compiles the .cpp -> executable, linking VCRuntime + Vulkan.
  //    VC_RUNTIME_LIB / VC_INCLUDE_DIR / VC_VULKAN_LIBS are baked in at build
  //    time by CMake so the driver knows where libVCRuntime.a and the headers
  //    live without requiring an install.
  std::string gpp = findTool("g++");
  if (gpp.empty()) gpp = "/usr/bin/g++";
  if (!sys::fs::exists(gpp)) {
    errs() << "error: g++ not found\n";
    return 1;
  }

  SmallString<128> cppPath;
  sys::fs::createTemporaryFile("vchost", "cpp", cppPath);
  {
    std::error_code ec;
    raw_fd_ostream cppFile(cppPath, ec);
    if (ec) { errs() << "cannot write host temp: " << ec.message() << "\n";
              return 1; }
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

  rc = sys::ExecuteAndWait(gpp, args, std::nullopt, std::nullopt);
  if (rc != 0) {
    errs() << "g++ failed (rc=" << rc << ")\n--- generated host C++ ---\n"
           << cppSource;
    sys::fs::remove(cppPath);
    return 1;
  }
  sys::fs::remove(cppPath);
  outs() << "built " << outputFilename << " (" << cppSource.size()
         << " bytes host C++, " << spvBytes << " bytes SPIR-V)\n";
  return 0;
}
