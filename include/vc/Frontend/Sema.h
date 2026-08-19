//===- Sema.h - Semantic analysis / symbol table --------------------------===//
//
// Minimal semantic analysis: builds a symbol table, resolves names, and
// checks that __global__ kernels have the expected shape. Extended later.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_SEMA_H
#define VC_FRONTEND_SEMA_H

#include "vc/Frontend/AST.h"

#include "llvm/ADT/StringMap.h"

namespace vc {

class Sema {
  TranslationUnit &tu;
  // name -> VarDecl*/ParamDecl* for the current scope (single scope for now)
  llvm::StringMap<ASTNode *> symbols;

public:
  Sema(TranslationUnit &unit) : tu(unit) {}

  /// Run analysis; returns true if no errors were added.
  bool analyze();

  void declare(StringRef name, ASTNode *node) { symbols[name] = node; }
  ASTNode *lookup(StringRef name) {
    auto it = symbols.find(name);
    return it == symbols.end() ? nullptr : it->second;
  }
};

} // namespace vc

#endif // VC_FRONTEND_SEMA_H
