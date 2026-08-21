//===- Sema.h - Semantic analysis / symbol table --------------------------===//
//
// Semantic analysis: builds a scoped symbol table, resolves names, infers
// expression types into a side table (exprTypes), and emits diagnostics for
// undeclared identifiers, bad struct fields, and illegal swizzles. The AST is
// not retrofitted with type fields — types live in a DenseMap keyed by node.
//
// The analysis is deliberately conservative about hard errors: only clearly
// wrong programs (undeclared identifier, unknown struct field, malformed
// swizzle, call to unknown non-builtin function with wrong arity) produce
// errors that fail the build. Type-mismatch is a warning so existing kernels
// keep compiling while still surfacing issues.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_SEMA_H
#define VC_FRONTEND_SEMA_H

#include "vc/Frontend/AST.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace vc {

class Sema {
  TranslationUnit &tu;

  // Scope stack: innermost last. Each frame maps a name to the ASTNode that
  // declared it (VarDecl* or ParamDecl*). A CompoundStmt/for-init pushes a
  // frame; leaving it pops.
  std::vector<llvm::StringMap<ASTNode *>> scopes;

  // Top-level function table: name -> FunctionDecl*.
  llvm::StringMap<FunctionDecl *> functions;

  // Top-level type names: struct/typedef name -> Type* (RecordType/TypedefType).
  llvm::StringMap<Type *> typeNames;

  // Side table of inferred expression types, keyed by AST node identity.
  llvm::DenseMap<const ASTNode *, Type *> exprTypes;

public:
  Sema(TranslationUnit &unit) : tu(unit) {}

  /// Run analysis; returns true if no Error-severity diagnostics were added.
  bool analyze();

  // Public for testing / introspection.
  Type *typeOf(const ASTNode *n) const {
    auto it = exprTypes.find(n);
    return it == exprTypes.end() ? nullptr : it->second;
  }

private:
  // Scope management.
  void pushScope() { scopes.emplace_back(); }
  void popScope() { scopes.pop_back(); }
  void declare(StringRef name, ASTNode *node);
  ASTNode *lookup(StringRef name);

  // Diagnostics.
  void error(const ASTNode *at, std::string msg);
  void warn(const ASTNode *at, std::string msg);

  // Two passes.
  void collectTopLevel();
  void checkFunctions();

  // Statement / expression checking.
  void checkStmt(const ASTNode *n);
  Type *checkExpr(const ASTNode *n);

  // Helpers.
  bool isThreadBuiltin(StringRef name) const;
  bool isMathBuiltin(StringRef name) const;
  static bool isValidSwizzle(StringRef s);
  Type *builtin(BuiltinTypeKind k) const;
};

} // namespace vc

#endif // VC_FRONTEND_SEMA_H
