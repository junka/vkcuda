//===- vc.cpp - Compiler driver (GLSL backend, no MLIR) ------------------===//
//
// Usage:
//   vc <input.vc> -o <out.spv>
//   vc <input.vc> -emit=glsl            # print GLSL source
//   vc <input.vc> -emit=ast             # print AST
//   vc <input.vc> -o out.spv            # GLSL -> glslc -> SPIR-V binary
//
// Pipeline: lex+parse -> AST -> GLSL compute shader -> glslc -> .spv.
// This driver has no MLIR dependency, so it builds without libmlir-18-dev.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/ASTToGLSL.h"
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

#include <cstdlib>
#include <string>

using namespace vc;
using namespace llvm;

namespace {
enum class EmitKind { AST, GLSL, SPIRV };
} // namespace

static std::string findGlslc() {
  // Prefer PATH lookup; fall back to common SDK locations.
  if (auto p = sys::findProgramByName("glslc")) return *p;
  const char *fallbacks[] = {
    "/usr/bin/glslc",
    "/usr/local/bin/glslc",
  };
  for (auto *f : fallbacks)
    if (sys::fs::exists(f)) return f;
  return {};
}

int main(int argc, char **argv) {
  InitLLVM x(argc, argv);

  cl::opt<std::string> inputFilename(cl::Positional,
      cl::desc("<input .vc file>"), cl::Required);
  cl::opt<std::string> emitOpt("emit",
      cl::desc("output kind (ast|glsl|spirv)"), cl::init("spirv"));
  cl::opt<std::string> outputFilename("o",
      cl::desc("output filename"), cl::init("-"));
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
  cl::ParseCommandLineOptions(argc, argv, "VC compiler (GLSL backend)\n");

  EmitKind kind = EmitKind::SPIRV;
  if (emitOpt == "ast") kind = EmitKind::AST;
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

  if (syntaxOnly) return 0;

  if (kind == EmitKind::AST) {
    dumpAST(tu, outs());
    return 0;
  }

  // 2. AST -> one GLSL compute unit per __global__ kernel.
  auto glslModules = glsl::translateASTToGLSLSources(tu, allowF64MathF32);
  if (glslModules.empty()) {
    errs() << "no kernel to emit\n";
    return 1;
  }

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
    for (size_t i = 0; i < glslModules.size(); ++i) {
      outs() << "// === kernel " << glslModules[i].entryName << " ===\n";
      outs() << glslModules[i].source;
      if (i + 1 < glslModules.size()) outs() << "\n";
    }
    return 0;
  }

  // 3. GLSL -> SPIR-V via glslc, one module per kernel.
  std::string glslc = findGlslc();
  if (glslc.empty()) {
    errs() << "error: glslc not found (install glslc / shaderc)\n";
    return 1;
  }

  std::string outPath = std::string(outputFilename) == "-"
                            ? "vadd.spv"
                            : std::string(outputFilename);

  // -o writes the first kernel's SPIR-V (SPIR-V can't merge multiple entries).
  for (size_t mi = 0; mi < glslModules.size(); ++mi) {
    const auto &mod = glslModules[mi];
    SmallString<128> glslPath;
    sys::fs::createTemporaryFile("vckernel", "comp", glslPath);
    {
      std::error_code ec;
      raw_fd_ostream glslFile(glslPath, ec);
      if (ec) { errs() << "cannot write temp: " << ec.message() << "\n"; return 1; }
      glslFile << mod.source;
    }

    // glslc -fshader-stage=compute <glsl> -o <out.spv>. Each kernel's .comp
    // uses `void main()` as its entry (see ASTToGLSL emitBody), so glslc's
    // default entry point applies — no -fentry-point (shaderc's flag is broken
    // on common distro builds). Subgroup ops require SPIR-V 1.3 = vulkan1.1.
    bool needsSpv13 = mod.source.find("vc:needs-spv1.3") != std::string::npos;

    auto runGlslc = [&]() -> int {
      SmallVector<StringRef, 8> args;
      args.push_back(glslc);
      args.push_back("-fshader-stage=compute");
      if (needsSpv13) args.push_back("--target-env=vulkan1.1");
      args.push_back(glslPath);
      // Only the first kernel writes to the user's -o path; others go to a
      // temp (vc-glsl is a debug driver — multi-kernel users use -emit=glsl).
      std::string target = outPath;
      SmallString<128> tmpOut;
      if (mi > 0) {
        sys::fs::createTemporaryFile("vckernel", "spv", tmpOut);
        target = std::string(tmpOut.str());
      }
      args.push_back("-o");
      args.push_back(target);
      return sys::ExecuteAndWait(glslc, args, std::nullopt, std::nullopt);
    };

    int rc = runGlslc();

    sys::fs::remove(glslPath);
    if (rc != 0) {
      errs() << "glslc failed for kernel '" << mod.entryName
             << "' (rc=" << rc << ")\n--- GLSL source ---\n" << mod.source;
      return 1;
    }
  }

  outs() << "wrote " << outPath << " (" << glslModules.front().source.size()
         << " bytes GLSL, " << glslModules.size() << " kernel"
         << (glslModules.size() == 1 ? "" : "s") << ")\n";
  return 0;
}
