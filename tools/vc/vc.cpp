//===- vc.cpp - Compiler driver -------------------------------------------===//
//
// Usage:
//   vc <input.vc> [-emit=ast|mlir|spirv|host|full] [-o <out>]
//
// Pipeline: lex+parse -> AST -> (Sema) -> MLIR(VC) -> lower -> SPIR-V.
// The SPIR-V emission reuses MLIR's spirv-to-binary translation.
//   -emit=ast    dump the parsed AST
//   -emit=mlir   dump the VC-dialect MLIR (default)
//   -emit=spirv  lower + serialize to a .spv binary
//   -emit=host   lower + serialize, then print the embedded host C++ (no link)
//   -emit=full   lower + serialize + embed + drive g++ -> standalone executable
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"
#include "vc/Codegen/HostLink.h"
#include "vc/Dialect/VC/Dialect.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Parser.h"
#include "vc/Frontend/Sema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Target/SPIRV/Target.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

using namespace vc;
using namespace mlir;

namespace {
enum class EmitKind { AST, MLIR, SPIRV, Host, Full };
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
      llvm::cl::desc("output kind (ast|mlir|spirv|host|full)"),
      llvm::cl::init("mlir"));
  llvm::cl::opt<std::string> outputFilename("o",
      llvm::cl::desc("output filename"), llvm::cl::init("-"));
  llvm::cl::opt<bool> warningsAsErrors("Werror",
      llvm::cl::desc("treat warnings as errors"));
  llvm::cl::opt<bool> syntaxOnly("fsyntax-only",
      llvm::cl::desc("lex, parse and type-check only; emit no output"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "VC compiler\n");

  EmitKind kind = EmitKind::MLIR;
  if (emit == "ast") kind = EmitKind::AST;
  else if (emit == "spirv") kind = EmitKind::SPIRV;
  else if (emit == "host") kind = EmitKind::Host;
  else if (emit == "full") kind = EmitKind::Full;
  else if (emit != "mlir") {
    llvm::errs() << "unknown -emit=" << emit << "\n";
    return 1;
  }

  llvm::SourceMgr sm;
  if (!loadSource(inputFilename, sm)) return 1;

  int mainBuf = sm.getMainFileID();
  Lexer lex(sm, mainBuf);
  SourceLocation start;
  TranslationUnit tu{start};
  Parser parser(lex, sm, tu);
  if (!parser.parseTranslationUnit()) {
    llvm::errs() << "parse failed\n";
    return 1;
  }
  Sema sema(tu, sm);
  sema.setWarningsAsErrors(warningsAsErrors);
  if (!sema.analyze()) {
    llvm::errs() << "sema: aborting due to errors\n";
    return 1;
  }

  if (syntaxOnly) return 0;

  if (kind == EmitKind::AST) {
    dumpAST(tu, llvm::outs());
    return 0;
  }

  // MLIR / SPIRV
  MLIRContext ctx;
  DialectRegistry registry;
  registry.insert<vc::VCDialect, mlir::spirv::SPIRVDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::index::IndexDialect>();
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

  // Serialize the spirv.module to a SPIR-V binary.
  SmallVector<uint32_t, 0> binary;
  mlir::spirv::ModuleOp spirvModule;
  module->walk([&](mlir::spirv::ModuleOp m) {
    if (!spirvModule)
      spirvModule = m;
  });
  if (!spirvModule || failed(mlir::spirv::serialize(spirvModule, binary))) {
    llvm::errs() << "spirv translation failed\n";
    return 1;
  }

  // -emit=spirv: write the single SPIR-V binary and stop. (A SPIR-V file
  // can't concatenate multiple entry points across kernels, so like the GLSL
  // driver we only write the first kernel's module — but MLIR emits one
  // binary with one OpEntryPoint per kernel, so the whole binary is written.)
  if (kind == EmitKind::SPIRV) {
    std::error_code ec;
    auto out = std::make_unique<llvm::ToolOutputFile>(
        outputFilename, ec, llvm::sys::fs::OF_None);
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

  // -emit=host / -emit=full: build one HostSpirvModule per __global__ kernel.
  // MLIR emits a single SPIR-V binary carrying one OpEntryPoint per kernel
  // named after the kernel symbol, so every module shares the same `binary`
  // (its lifetime outlives linkHostExecutable) and sets `entryPoint` to the
  // kernel name — vcLoadKernel then loads that entry point out of the shared
  // binary. (The GLSL backend instead emits one .spv per kernel with entry
  // "main"; see ASTToHost/HostLink.)
  std::vector<host::HostSpirvModule> hostModules;
  for (const auto &d : tu.decls) {
    if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
    auto *fn = static_cast<FunctionDecl *>(d.get());
    if (fn->deviceAttr != DeviceAttr::Global)
      continue;
    hostModules.push_back({fn->name.str(), binary.data(), binary.size(),
                           /*entryPoint=*/fn->name.str()});
  }
  if (hostModules.empty()) {
    llvm::errs() << "no __global__ kernel found in " << inputFilename << "\n";
    return 1;
  }

  return host::linkHostExecutable(tu, hostModules, outputFilename,
                                  kind == EmitKind::Host);
}
