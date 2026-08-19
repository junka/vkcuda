//===- vc.cpp - Compiler driver -------------------------------------------===//
//
// Usage:
//   vc <input.vc> [-emit=ast|mlir|spirv] [-o <out.spv>]
//
// Pipeline: lex+parse -> AST -> (Sema) -> MLIR(VC) -> lower -> SPIR-V.
// The SPIR-V emission reuses MLIR's spirv-to-binary translation. With the
// VC->gpu lowering still scaffolded, -emit=mlir is the most useful mode
// until that pass lands.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"
#include "vc/Dialect/VC/Dialect.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Parser.h"
#include "vc/Frontend/Sema.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/SPIRV/Target.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

using namespace vc;
using namespace mlir;

namespace {
enum class EmitKind { AST, MLIR, SPIRV };
} // namespace

static bool loadSource(const std::string &path, llvm::SourceMgr &sm) {
  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code ec = buf.getError()) {
    llvm::errs() << "error: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  sm.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  return true;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input .vc file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> emit("emit",
      llvm::cl::desc("output kind (ast|mlir|spirv)"),
      llvm::cl::init("mlir"));
  llvm::cl::opt<std::string> outputFilename("o",
      llvm::cl::desc("output filename"), llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "VC compiler\n");

  EmitKind kind = EmitKind::MLIR;
  if (emit == "ast") kind = EmitKind::AST;
  else if (emit == "spirv") kind = EmitKind::SPIRV;
  else if (emit != "mlir") {
    llvm::errs() << "unknown -emit=" << emit << "\n";
    return 1;
  }

  llvm::SourceMgr sm;
  if (!loadSource(inputFilename, sm)) return 1;

  int mainBuf = sm.getMainFileID();
  Lexer lex(sm, mainBuf);
  TranslationUnit tu(llvm::SMLoc());
  Parser parser(lex, sm, tu);
  if (!parser.parseTranslationUnit()) {
    llvm::errs() << "parse failed\n";
    return 1;
  }
  Sema sema(tu);
  sema.analyze();

  if (kind == EmitKind::AST) {
    dumpAST(tu, llvm::outs());
    return 0;
  }

  // MLIR / SPIRV
  MLIRContext ctx;
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<vc::VCDialect, SPIRVDialect>();
  ctx.appendDialectRegistry(registry);

  auto module = codegen::translateASTToMLIR(tu, ctx);
  if (!module) {
    llvm::errs() << "codegen failed\n";
    return 1;
  }

  if (kind == EmitKind::MLIR) {
    module->print(llvm::outs());
    return 0;
  }

  // SPIRV: run lowering then translate to binary.
  codegen::runLoweringPipeline(*module);

  // Write the spirv.module as assembly text (binary emission to .spv uses
  // mlir::translateModuleToSPIRVBinary).
  SmallVector<uint32_t, 0> binary;
  if (failed(spirv::translateModuleToBinary(
          *module, binary, /*emitDebugInfo=*/false))) {
    llvm::errs() << "spirv translation failed\n";
    return 1;
  }

  std::error_code ec;
  auto out = std::make_unique<llvm::ToolOutputFile>(outputFilename, ec,
                                                    llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "cannot open " << outputFilename << ": " << ec.message()
                 << "\n";
    return 1;
  }
  out->os().write(reinterpret_cast<const char *>(binary.data()),
                  binary.size() * sizeof(uint32_t));
  out->keep();
  return 0;
}
