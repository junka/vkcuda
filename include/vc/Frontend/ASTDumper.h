//===- ASTDumper.h - Pretty-print the AST ---------------------------------===//

#ifndef VC_FRONTEND_ASTDUMPER_H
#define VC_FRONTEND_ASTDUMPER_H

#include "llvm/Support/raw_ostream.h"

namespace vc {

class TranslationUnit;

/// Dump the translation unit AST to `os` in a human-readable indented form.
void dumpAST(const TranslationUnit &tu, llvm::raw_ostream &os);

} // namespace vc

#endif // VC_FRONTEND_ASTDUMPER_H
