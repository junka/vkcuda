//===- Sema.cpp - Semantic analysis (placeholder) -------------------------===//

#include "vc/Frontend/Sema.h"

using namespace vc;

bool Sema::analyze() {
  // TODO: real symbol table per scope, type checking, kernel shape checks.
  // For the scaffold we just record top-level function names.
  for (auto &d : tu.decls) {
    if (d->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
      auto *f = static_cast<FunctionDecl *>(d.get());
      declare(f->name, f);
    }
  }
  return tu.diagnostics.empty();
}
