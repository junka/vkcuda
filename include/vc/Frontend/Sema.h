//===- Sema.h - Semantic analysis / symbol table --------------------------===//
//
// Semantic analysis: builds a scoped symbol table, resolves names, infers
// expression types into a side table (exprTypes), and emits diagnostics for
// undeclared identifiers, bad struct fields, and illegal swizzles. The AST is
// not retrofitted with type fields — types live in a DenseMap keyed by node.
//
// The analysis is deliberately conservative about hard errors: only clearly
// wrong programs (undeclared identifier, unknown struct field, malformed
// swizzle, call to unknown non-builtin function with wrong arity, redefinition)
// produce errors that fail the build. Everything else — type mismatches,
// unused variables/parameters, division by zero, bad subscripts, duplicate
// case labels, calling a host function from device code, etc. — is a warning
// so existing kernels keep compiling while still surfacing issues.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_SEMA_H
#define VC_FRONTEND_SEMA_H

#include "vc/Frontend/AST.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

namespace vc {

class Sema {
  TranslationUnit &tu;
  llvm::SourceMgr &srcMgr;
  // -Werror mode: warnings are reported (and count) as errors.
  bool warningsAsErrors = false;

  // Scope stack: innermost last. Each frame maps a name to the ASTNode that
  // declared it (VarDecl* or ParamDecl*). A CompoundStmt/for-init pushes a
  // frame; leaving it pops.
  std::vector<llvm::StringMap<ASTNode *>> scopes;

  // Top-level function table: mangled device name -> overload set. The key is
  // the namespace-mangled name WITHOUT parameter encoding (so `f(int)` and
  // `f(float)` share a key and form one overload set). Overload resolution
  // picks among the SmallVector at each call site.
  llvm::StringMap<llvm::SmallVector<FunctionDecl *, 2>> functions;

  // Top-level global variables (__constant__ decls): name -> VarDecl*. These
  // are file-scope, so DeclRefExpr resolves against this map (they aren't in
  // any lexical scope frame).
  llvm::StringMap<VarDecl *> globalVars;

  // Top-level type names: struct/typedef name -> Type* (RecordType/TypedefType).
  llvm::StringMap<Type *> typeNames;

  // Unscoped enum constants: name -> integer value. Populated from EnumDecls
  // in collectTopLevel; DeclRefExpr resolves against this so enum names are not
  // flagged as undeclared.
  llvm::StringMap<int64_t> enumConstants;

  // Side table of inferred expression types, keyed by AST node identity.
  llvm::DenseMap<const ASTNode *, Type *> exprTypes;

  // Context for the function body currently being checked (drives the
  // return-type consistency check). Null outside a __global__/__device__ body.
  FunctionDecl *currentFunc = nullptr;

  // Break/continue legality: break is allowed inside a loop or a switch,
  // continue only inside a loop. Maintained by checkStmt as it descends.
  unsigned loopDepth = 0;
  unsigned breakableDepth = 0;

  // Unused-variable tracking. declare() seeds a VarDecl/ParamDecl as unused;
  // every DeclRefExpr to it flips the flag. When a scope pops, entries left
  // unused are reported as warnings.
  llvm::DenseMap<const ASTNode *, bool> used;

  // Per-switch case constants so `case 3:` twice — and a repeated `default` —
  // is caught. One entry per active switch, maintained by checkStmt.
  struct SwitchCases {
    llvm::DenseSet<int64_t> values;
    bool sawDefault = false;
  };
  std::vector<SwitchCases> switchStack;

public:
  Sema(TranslationUnit &unit, llvm::SourceMgr &sm) : tu(unit), srcMgr(sm) {}

  /// -Werror: upgrade warnings to errors during reporting.
  void setWarningsAsErrors(bool on) { warningsAsErrors = on; }

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
  void popScope() {
    checkUnusedInFrame(scopes.back());
    scopes.pop_back();
  }
  void declare(StringRef name, ASTNode *node);
  ASTNode *lookup(StringRef name);
  // Warn about VarDecl/ParamDecl declared in `frame` that were never read.
  void checkUnusedInFrame(const llvm::StringMap<ASTNode *> &frame);
  // Type-check a call (or kernel launch) against a callee's parameter list.
  // Checks each argument against its parameter and reports structural
  // mismatches and lossy float->int conversions as warnings.
  void checkCallArgs(const ASTNode *call, StringRef calleeName,
                     FunctionDecl *f, const std::vector<NodePtr> &args);

  // Overload resolution: pick the best candidate from `candidates` for a call
  // with the given (already type-inferred) argument types. Returns the chosen
  // FunctionDecl, or null if no viable candidate. On ambiguity (two or more
  // candidates tie for best), reports an error listing the candidates and
  // returns null. `outRank` (if set) receives the winning rank.
  FunctionDecl *resolveOverload(const ASTNode *call, StringRef calleeName,
                                const llvm::SmallVector<FunctionDecl *, 2> &cands,
                                const std::vector<NodePtr> &args,
                                int *outRank = nullptr);

  // Conversion rank for a single argument->parameter binding. Lower is better.
  //   0 exact, 1 promotion, 2 standard, 3 lossy, 4 incompatible.
  static int conversionRank(const Type *param, const Type *arg);

  // Structural type equality (typedefs resolved). Used for redefinition
  // detection and overload-set dedup; not a full canonical Type.
  static bool sameType(const Type *a, const Type *b);
  // Two functions collide in an overload set iff their parameter type lists
  // are structurally equal (return type is NOT part of the signature).
  static bool sameSignature(const FunctionDecl *a, const FunctionDecl *b);

  // Diagnostics.
  void error(const ASTNode *at, std::string msg);
  void warn(const ASTNode *at, std::string msg);

  // Two passes.
  void collectTopLevel();
  // Recursive helper: collect decls from a namespace body, prefixing registered
  // names with `nsPrefix` (e.g. `ns_f` for `ns::f`). An empty prefix means
  // top-level. The device backend mangles scoped names to underscore-joined
  // form, so Sema registers them under the same mangled key call sites resolve.
  void collectDecls(const std::vector<NodePtr> &decls, StringRef nsPrefix);
  void checkFunctions();
  // Recursive helper: type-check device function bodies, descending into
  // namespace bodies (mirrors collectDecls' recursion).
  void checkFunctionsIn(const std::vector<NodePtr> &decls);

  // Statement / expression checking.
  void checkStmt(const ASTNode *n);
  Type *checkExpr(const ASTNode *n);
  void checkCond(const ASTNode *cond, const char *what);

  // Helpers.
  bool isThreadBuiltin(StringRef name) const;
  bool isMathBuiltin(StringRef name) const;
  static bool isValidSwizzle(StringRef s);
  Type *builtin(BuiltinTypeKind k) const;

  // Type taxonomy used by the compatibility checks. Arithmetic covers the
  // builtin scalars between which C-style implicit conversions are legal;
  // anything else (pointer/vector/record) does not implicitly convert.
  static bool isArithmetic(const Type *t);
  static bool isIntegerType(const Type *t);
  static bool isFloatType(const Type *t);
  static std::string typeName(const Type *t);
  // Coarse byte size of a type for `sizeof` folding (no alignment padding).
  static int64_t sizeOfType(const Type *t);
  static const char *opName(BinaryOp op);
  // Strip typedef aliases down to the underlying type for comparison and
  // arithmetic classification. A TypedefType is never directly comparable.
  static const Type *resolveTypedefs(const Type *t);
  // Two operands with the same TypeKind convert implicitly (builtin->builtin,
  // vector->vector, pointer->pointer); differing kinds only convert via an
  // explicit cast or for pointer+index Add/Sub.
  static bool isCompatibleKinds(const Type *a, const Type *b);
  static bool isCompatibleForAssign(const Type *dst, const Type *src);
};

} // namespace vc

#endif // VC_FRONTEND_SEMA_H
