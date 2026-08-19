// Standalone frontend smoke test: parse + dump AST (no MLIR needed).
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Parser.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace vc;

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0] << " <file.vc>\n";
    return 1;
  }
  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (auto ec = buf.getError()) {
    llvm::errs() << "cannot open " << argv[1] << ": " << ec.message() << "\n";
    return 1;
  }
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  Lexer lex(sm, sm.getMainFileID());
  SourceLocation start;
  TranslationUnit tu{start};
  Parser p(lex, sm, tu);
  bool ok = p.parseTranslationUnit();
  llvm::outs() << "parse " << (ok ? "ok" : "with errors") << "\n";
  dumpAST(tu, llvm::outs());
  return ok ? 0 : 1;
}
