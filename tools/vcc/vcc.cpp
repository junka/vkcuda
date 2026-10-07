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
//   vcc <input.vc> -emit=spirv -o x.spv   write device SPIR-V (x.spv, x.2.spv,
//                                         ... one module per kernel)
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/ASTToGLSL.h"
#include "vc/Codegen/ASTToHost.h"
#include "vc/Codegen/HostLink.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Parser.h"
#include "vc/Frontend/Sema.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
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
  cl::opt<bool> warningsAsErrors("Werror",
      cl::desc("treat warnings as errors"));
  cl::opt<bool> syntaxOnly("fsyntax-only",
      cl::desc("lex, parse and type-check only; emit no output"));
  cl::opt<bool> allowF64MathF32(
      "fallow-f64-math-f32",
      cl::desc("compute double transcendentals (sin/pow/...) in float "
               "instead of rejecting them"));
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
  Sema sema(tu, sm);
  sema.setWarningsAsErrors(warningsAsErrors);
  if (!sema.analyze()) {
    errs() << "sema: aborting due to errors\n";
    return 1;
  }

  // SPIR-V forbids recursion; rewrite bounded linear self-recursion in
  // __device__ functions into iterative loops before codegen. A recursive
  // function whose shape the pass can't handle is a hard error here.
  if (!unrollDeviceRecursion(tu)) {
    errs() << "recursion: aborting due to errors\n";
    return 1;
  }

  if (syntaxOnly) return 0;
  if (emitOpt == "ast") { dumpAST(tu, outs()); return 0; }

  // wmma:: / cooperative-matrix intrinsics are only lowered by the MLIR
  // backend (gpu.subgroup_mma -> spirv.KHR.CooperativeMatrix). The GLSL
  // backend has no tensor-core path, so refuse here with a clear message
  // rather than emitting broken GLSL.
  if (sema.usesCoopMatrix()) {
    errs() << inputFilename
           << ": error: wmma:: tensor-core intrinsics require the MLIR "
              "backend (-emit=full via `vc`); the GLSL backend (`vcc`) does "
              "not support cooperative matrices\n";
    return 1;
  }

  // 2. Device subset -> one GLSL compute unit per __global__ kernel.
  auto glslModules = glsl::translateASTToGLSLSources(tu, allowF64MathF32);
  if (glslModules.empty()) {
    errs() << "no __global__ kernel found in " << inputFilename << "\n";
    return 1;
  }

  // A `double` transcendental has no Vulkan GLSL spelling (glslang rejects
  // sin/cos/.../pow on a double even with the float64 extension, because
  // SPIR-V's GLSLstd450 entries are f16/f32-only). Report it here, in the
  // compiler's own words, instead of letting glslc answer with an opaque
  // "no matching overloaded function found".
  for (const auto &m : glslModules) {
    if (m.unsupportedF64Math.empty()) continue;
    errs() << "error: no double-precision form of " << m.unsupportedF64Math
           << " in kernel '" << m.entryName
           << "': SPIR-V's GLSLstd450 transcendentals are f16/f32-only and "
              "Vulkan GLSL has no double overload either. Compute it in float, "
              "e.g. `float r = sin((float)x);` (CUDA's `sinf`), or pass "
              "-fallow-f64-math-f32 to compute it in float and store the "
              "result as double\n";
    return 1;
  }

  if (kind == EmitKind::GLSL) {
    // Print every kernel's GLSL, separated by a marker comment.
    for (size_t i = 0; i < glslModules.size(); ++i) {
      outs() << "// === kernel " << glslModules[i].entryName << " ===\n";
      outs() << glslModules[i].source;
      if (i + 1 < glslModules.size()) outs() << "\n";
    }
    return 0;
  }

  // 3. GLSL -> SPIR-V via glslc, one module per kernel.
  std::string glslc = findTool("glslc");
  if (glslc.empty()) {
    errs() << "error: glslc not found (install shaderc)\n";
    return 1;
  }

  // Hold the SPIR-V buffers alive: HostSpirvModule.words points into these.
  std::vector<std::unique_ptr<MemoryBuffer>> spvBuffers;
  std::vector<host::HostSpirvModule> hostModules;

  for (const auto &mod : glslModules) {
    SmallString<128> glslPath;
    sys::fs::createTemporaryFile("vckernel", "comp", glslPath);
    {
      std::error_code ec;
      raw_fd_ostream glslFile(glslPath, ec);
      if (ec) { errs() << "cannot write temp: " << ec.message() << "\n"; return 1; }
      glslFile << mod.source;
    }
    SmallString<128> spvPath;
    sys::fs::createTemporaryFile("vckernel", "spv", spvPath);
    // Each kernel's .comp uses `void main()` as its entry (see ASTToGLSL
    // emitBody), so glslc's default entry point applies — no -fentry-point.
    // (shaderc's -fentry-point is broken on common distro builds.) Subgroup
    // ops (CUDA warp intrinsics) require SPIR-V 1.3 = vulkan1.1 target; the
    // GLSL backend flags this with a "// vc:needs-spv1.3" header marker.
    bool needsSpv13 = mod.source.find("vc:needs-spv1.3") != std::string::npos;
    auto runGlslc = [&]() -> int {
      SmallVector<StringRef, 8> args;
      args.push_back(glslc);
      args.push_back("-fshader-stage=compute");
      if (needsSpv13) args.push_back("--target-env=vulkan1.1");
      args.push_back(glslPath);
      args.push_back("-o");
      args.push_back(spvPath);
      return sys::ExecuteAndWait(glslc, args, std::nullopt, std::nullopt);
    };
    int rc = runGlslc();
    sys::fs::remove(glslPath);
    if (rc != 0) {
      errs() << "glslc failed for kernel '" << mod.entryName
             << "' (rc=" << rc << ")\n--- GLSL source ---\n" << mod.source;
      sys::fs::remove(spvPath);
      return 1;
    }

    auto spvBuf = MemoryBuffer::getFile(spvPath);
    if (auto ec = spvBuf.getError()) {
      errs() << "cannot read spirv: " << ec.message() << "\n";
      sys::fs::remove(spvPath);
      return 1;
    }
    size_t spvBytes = (*spvBuf)->getBufferSize();
    const uint32_t *words =
        reinterpret_cast<const uint32_t *>((*spvBuf)->getBufferStart());
    hostModules.push_back({mod.entryName, words, spvBytes / sizeof(uint32_t)});
    spvBuffers.push_back(std::move(*spvBuf));
    sys::fs::remove(spvPath);

    if (kind == EmitKind::SPIRV) {
      // SPIR-V cannot concatenate entry points, so each kernel is written to its
      // own module: `out.spv`, `out.2.spv`, ... The first keeps the plain name, so
      // single-kernel input is unaffected. Anything that wants to validate every
      // kernel (test/check_spirv_val.py) could not see past the first otherwise.
      size_t idx = hostModules.size() - 1;
      SmallString<128> path(outputFilename);
      if (idx)
        sys::path::replace_extension(path,
                                     "." + std::to_string(idx + 1) + ".spv");
      std::error_code ec;
      raw_fd_ostream out(path, ec);
      if (ec) {
        errs() << "cannot write " << path << ": " << ec.message() << "\n";
        return 1;
      }
      out.write(reinterpret_cast<const char *>(hostModules[idx].words),
                hostModules[idx].wordCount * sizeof(uint32_t));
      outs() << "wrote " << path << " (" << spvBytes << " bytes, kernel "
             << hostModules[idx].kernelName << ")\n";
      continue;
    }
  }

  if (kind == EmitKind::SPIRV) return 0;

  // 4 + 5. Host subset -> C++ (embedding one SPIR-V module per kernel) ->
  //       g++ links it against libVCRuntime + Vulkan into the executable.
  //       Shared with the MLIR driver via host::linkHostExecutable.
  return host::linkHostExecutable(tu, hostModules, outputFilename,
                                  kind == EmitKind::Host);
}
