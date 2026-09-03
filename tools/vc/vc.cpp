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
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <functional>
#include <string>

using namespace vc;
using namespace mlir;

namespace {
enum class EmitKind { AST, MLIR, SPIRV, Host, Full };
} // namespace

// Patch the serialized SPIR-V binary to decorate the `gl_WorkGroupSize`
// SpecConstantComposite with BuiltIn WorkgroupSize. This is what makes the
// runtime-supplied specialization constants (SpecId 0/1/2 = block x/y/z) drive
// the actual hardware workgroup size, mirroring the GLSL backend's
// `layout(local_size_x_id = 0, ...)` -> `gl_WorkGroupSize` builtin.
//
// Why a binary patch: MLIR's spirv dialect has no way to attach a BuiltIn
// decoration to a SpecConstantComposite from IR (SpecConstantCompositeOp
// carries no built_in attr; only GlobalVariableOp does, and WorkgroupSize must
// be a spec-constant composite, not a variable). The serializer thus emits the
// composite + its scalar SpecId decorations but omits the WorkgroupSize builtin
// decoration. We inject it by name: locate the OpName record for
// "gl_WorkGroupSize", read its target <id>, then insert an OpDecorate
// <id> BuiltIn WorkgroupSize instruction.
//
// Insertion point: SPIR-V's logical layout puts all OpDecorate records in an
// annotation section that follows the debug section (OpName / ...) and the
// entry-point header (OpCapability ... OpExecutionMode). The serializer already
// emits a valid block of OpDecorate records there, so we insert ours
// immediately before the first existing OpDecorate to keep the layout valid
// (verified by spirv-val).
static void patchWorkgroupSizeBuiltin(SmallVectorImpl<uint32_t> &binary) {
  // OpName = opcode 5; OpDecorate = opcode 71; BuiltIn decoration = 11;
  // BuiltIn.WorkgroupSize = 25.
  constexpr uint32_t kOpName = 5;
  constexpr uint32_t kOpDecorate = 71;
  constexpr uint32_t kDecorationBuiltIn = 11;
  constexpr uint32_t kBuiltinWorkgroupSize = 25;

  if (binary.size() < 5)
    return;

  // Walk instructions to find `OpName <id> "gl_WorkGroupSize"`.
  // OpName record: [wordCount|opcode, target_id, name...]. The name is the
  // remaining (wordCount - 2) words, null-padded to a word boundary.
  size_t i = 5;
  std::optional<uint32_t> wgsizeId;
  while (i < binary.size()) {
    uint32_t word = binary[i];
    uint32_t opcode = word & 0xFFFF;
    uint32_t wordCount = word >> 16;
    if (wordCount == 0)
      break; // malformed
    if (opcode == kOpName && wordCount >= 3 && i + wordCount <= binary.size()) {
      uint32_t targetId = binary[i + 1];
      // Reassemble the name string from the trailing words.
      std::string name;
      for (uint32_t w = 2; w < wordCount; ++w) {
        uint32_t val = binary[i + w];
        for (int b = 0; b < 4; ++b) {
          char c = static_cast<char>((val >> (8 * b)) & 0xFF);
          if (c == 0)
            goto name_done;
          name.push_back(c);
        }
      }
    name_done:
      if (name == "gl_WorkGroupSize") {
        wgsizeId = targetId;
        break;
      }
    }
    i += wordCount;
  }

  if (!wgsizeId)
    return; // No gl_WorkGroupSize in this module (no kernels / not patched).

  // Build the OpDecorate instruction: OpDecorate <target_id> BuiltIn
  // WorkgroupSize -> 4 words: [4<<16|71, target_id, 11, 25].
  uint32_t decor[4] = {(4u << 16) | kOpDecorate, *wgsizeId,
                       kDecorationBuiltIn, kBuiltinWorkgroupSize};
  // SPIR-V layout (logical): OpCapability, OpExtension, OpExtInstImport,
  // OpMemoryModel, OpEntryPoint, OpExecutionMode, then a debug section
  // (OpName / OpMemberName / OpString / OpLine / ...), then annotations
  // (OpDecorate / OpMemberDecorate / OpGroupDecorate / ...), then types &
  // constants, then functions. OpDecorate must land in the annotation section,
  // i.e. after all OpName lines. The serializer already emits a valid block of
  // OpDecorate records there, so the simplest correct insertion is right
  // before the first existing OpDecorate.
  size_t insertPos = 0;
  size_t j = 5;
  while (j < binary.size()) {
    uint32_t w = binary[j];
    uint32_t op = w & 0xFFFF;
    uint32_t wc = w >> 16;
    if (wc == 0)
      break;
    if (op == kOpDecorate) {
      insertPos = j;
      break;
    }
    j += wc;
  }
  if (insertPos == 0)
    return; // No existing OpDecorate — unexpected; don't risk a bad patch.
  binary.insert(binary.begin() + insertPos, decor, decor + 4);
}

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

  // Decorate the gl_WorkGroupSize spec-composite BuiltIn WorkgroupSize so the
  // runtime's specialization constants (SpecId 0/1/2) drive the hardware
  // workgroup size. See patchWorkgroupSizeBuiltin for why this is a binary
  // patch rather than an IR-level decoration.
  patchWorkgroupSizeBuiltin(binary);

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
  //
  // The kernel symbol is the MANGLED device name (`ns::kernel` -> `ns_kernel`)
  // because the MLIR backend's buildFunction registers the func.func under that
  // mangled name (mirroring the host launchHandleName), and the serialized
  // OpEntryPoint carries the same symbol. A bare `fn->name` would mismatch the
  // spirv entry point and the host's launch handle. Kernels nested in
  // namespaces are found by recursing into NamespaceDecl bodies.
  auto deviceSymName = [](const FunctionDecl *fn) -> std::string {
    if (!fn->nsName.empty()) {
      std::string out;
      llvm::StringRef rest = fn->nsName;
      while (!rest.empty()) {
        auto pair = rest.split("::");
        if (!out.empty()) out += '_';
        out += pair.first.str();
        rest = pair.second;
      }
      return out + "_" + fn->name.str();
    }
    return fn->name.str();
  };
  std::vector<host::HostSpirvModule> hostModules;
  std::function<void(const std::vector<NodePtr> &)> collectKernels =
      [&](const std::vector<NodePtr> &decls) {
        for (const auto &d : decls) {
          if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
            collectKernels(static_cast<const NamespaceDecl *>(d.get())->decls);
            continue;
          }
          if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
          auto *fn = static_cast<FunctionDecl *>(d.get());
          if (fn->deviceAttr != DeviceAttr::Global)
            continue;
          std::string sym = deviceSymName(fn);
          hostModules.push_back({sym, binary.data(), binary.size(),
                                 /*entryPoint=*/sym});
        }
      };
  collectKernels(tu.decls);
  if (hostModules.empty()) {
    llvm::errs() << "no __global__ kernel found in " << inputFilename << "\n";
    return 1;
  }

  return host::linkHostExecutable(tu, hostModules, outputFilename,
                                  kind == EmitKind::Host);
}
