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

  // 2. AST -> GLSL
  std::string glslSource;
  raw_string_ostream glslOS(glslSource);
  if (!glsl::translateASTToGLSL(tu, glslOS)) {
    errs() << "no kernel to emit\n";
    return 1;
  }
  glslOS.flush();

  if (kind == EmitKind::GLSL) {
    outs() << glslSource;
    return 0;
  }

  // 3. GLSL -> SPIR-V via glslc
  std::string glslc = findGlslc();
  if (glslc.empty()) {
    errs() << "error: glslc not found (install glslc / shaderc)\n";
    return 1;
  }

  // Write GLSL to a temp file.
  SmallString<128> glslPath;
  sys::fs::createTemporaryFile("vckernel", "comp", glslPath);
  {
    std::error_code ec;
    raw_fd_ostream glslFile(glslPath, ec);
    if (ec) { errs() << "cannot write temp: " << ec.message() << "\n"; return 1; }
    glslFile << glslSource;
  }

  // glslc -fshader-stage=compute <glsl> -o <out.spv> -fentry-point=<name>
  // -fentry-point requires a recent shaderc; if unsupported it errors and
  // we fall back by retrying without it below.
  std::string outPath = std::string(outputFilename) == "-" ? "vadd.spv"
                                                        : std::string(outputFilename);
  std::string entryArg = std::string("-fentry-point=") + std::string(entryPoint);

  // The GLSL backend writes a "// vc:needs-spv1.3" marker into the header when
  // the kernel uses Vulkan subgroup ops (CUDA warp intrinsics), which require
  // SPIR-V 1.3 (= vulkan1.1 target env). Default is vulkan1.0/spv1.0.
  bool needsSpv13 = glslSource.find("vc:needs-spv1.3") != std::string::npos;

  auto runGlslc = [&](bool withEntry) -> int {
    SmallVector<StringRef, 8> args;
    args.push_back(glslc);
    args.push_back("-fshader-stage=compute");
    if (needsSpv13) args.push_back("--target-env=vulkan1.1");
    args.push_back(glslPath);
    args.push_back("-o");
    args.push_back(outPath);
    if (withEntry) args.push_back(entryArg);

    std::string errMsg;
    return sys::ExecuteAndWait(glslc, args, std::nullopt, std::nullopt);
  };

  int rc = runGlslc(/*withEntry=*/true);
  if (rc != 0)
    rc = runGlslc(/*withEntry=*/false); // retry without -fentry-point

  sys::fs::remove(glslPath);
  if (rc != 0) {
    errs() << "glslc failed (rc=" << rc << ")\n--- GLSL source ---\n"
           << glslSource;
    return 1;
  }

  outs() << "wrote " << outPath << " (" << glslSource.size()
         << " bytes GLSL)\n";
  return 0;
}
