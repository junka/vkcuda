//===- Sema.cpp - Semantic analysis ---------------------------------------===//
//
// Scoped symbol table + expression type inference (side table) + diagnostics.
// Conservative by design: only undeclared identifiers, unknown struct fields,
// malformed swizzles, redefinitions, and calls to unknown non-builtin functions
// are hard errors. Type mismatches, unused variables, division by zero, bad
// subscripts and similar are warnings so existing kernels keep compiling.
//
// Thread index builtins (threadIdx, blockIdx, blockDim, gridDim) and math
// builtins (sinf, __syncthreads, ...) are treated as implicitly declared so
// real kernels don't trip false-positive errors.

#include "vc/Frontend/Sema.h"

#include "vc/Frontend/BuiltinRegistry.h"
#include "vc/Frontend/Mangle.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;

namespace {
// Build the source spelling of a `::`-qualified MemberAccessExpr chain as
// "A::B::C". The chain is left-nested: MemberAccessExpr(base=MemberAccessExpr(
// base=DeclRefExpr("A"), member="B"), member="C"). Non-scope bases (an object
// expression, not a name chain) stop the walk — returns empty in that case.
std::string scopeChainStr(const MemberAccessExpr *ma) {
  if (!ma) return {};
  std::vector<std::string> parts;
  parts.push_back(ma->member.str());
  const ASTNode *cur = ma->base.get();
  while (cur) {
    if (cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
      auto *sub = static_cast<const MemberAccessExpr *>(cur);
      if (!sub->isScope) break; // base is an object expression, not a name
      parts.push_back(sub->member.str());
      cur = sub->base.get();
    } else if (cur->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      parts.push_back(static_cast<const DeclRefExpr *>(cur)->name.str());
      break;
    } else {
      break;
    }
  }
  std::reverse(parts.begin(), parts.end());
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out += "::";
    out += parts[i];
  }
  return out;
}

// Fold a constant integer expression to its value: plain literals and
// unary-minus literals (e.g. -1). Returns false if the node isn't one, so
// checks that need a statically-known integer can skip dynamic expressions.
bool constIntValue(const ASTNode *n, int64_t &out) {
  if (!n) return false;
  if (n->getNodeType() == ASTNode::NodeKind::IntegerLiteral) {
    out = static_cast<const IntegerLiteral *>(n)->value;
    return true;
  }
  if (n->getNodeType() == ASTNode::NodeKind::UnaryExpr) {
    auto *u = static_cast<const UnaryExpr *>(n);
    if (u->op == UnaryOp::Neg &&
        u->operand->getNodeType() == ASTNode::NodeKind::IntegerLiteral) {
      out = -static_cast<const IntegerLiteral *>(u->operand.get())->value;
      return true;
    }
  }
  return false;
}
} // namespace

// Builtin recognition is centralized in BuiltinRegistry so Sema and both
// backends share one name list. These forward to it.

bool Sema::isThreadBuiltin(StringRef name) const {
  return builtinClass(name) == BuiltinClass::ThreadIndex;
}

// True for any builtin (math, atomics, sync, warp, vector ctors, ...). Used at
// call sites to skip the undeclared-function warning — the backend lowers it.
bool Sema::isMathBuiltin(StringRef name) const {
  return isBuiltin(name);
}

// A legal GLSL swizzle: 1-4 chars drawn from only xyzw, only rgba, or only stpq,
// with no repeats.
bool Sema::isValidSwizzle(StringRef s) {
  if (s.empty() || s.size() > 4) return false;
  char set = 0;
  for (char c : s) {
    char g;
    if (c == 'x' || c == 'y' || c == 'z' || c == 'w') g = 'x';
    else if (c == 'r' || c == 'g' || c == 'b' || c == 'a') g = 'r';
    else if (c == 's' || c == 't' || c == 'p' || c == 'q') g = 's';
    else return false;
    if (set == 0) set = g;
    else if (set != g) return false;
  }
  // No duplicate component within the swizzle.
  for (unsigned i = 0; i < s.size(); ++i)
    for (unsigned j = i + 1; j < s.size(); ++j)
      if (s[i] == s[j]) return false;
  return true;
}

const Type *Sema::resolveTypedefs(const Type *t) {
  while (t && t->getKind() == TypeKind::Typedef) {
    const TypedefType *td = static_cast<const TypedefType *>(t);
    t = td->decl ? td->decl->underlying : nullptr;
  }
  return t;
}

bool Sema::isArithmetic(const Type *t) {
  if (!t || t->getKind() != TypeKind::Builtin) return false;
  switch (static_cast<const BuiltinType *>(t)->builtin) {
  case BuiltinTypeKind::Bool:
  case BuiltinTypeKind::Int32:
  case BuiltinTypeKind::UInt32:
  case BuiltinTypeKind::Int64:
  case BuiltinTypeKind::UInt64:
  case BuiltinTypeKind::Float32:
  case BuiltinTypeKind::Float64:
    return true;
  default:
    return false;
  }
}

bool Sema::isIntegerType(const Type *t) {
  if (!t || t->getKind() != TypeKind::Builtin) return false;
  switch (static_cast<const BuiltinType *>(t)->builtin) {
  case BuiltinTypeKind::Bool:
  case BuiltinTypeKind::Int32:
  case BuiltinTypeKind::UInt32:
  case BuiltinTypeKind::Int64:
  case BuiltinTypeKind::UInt64:
    return true;
  default:
    return false;
  }
}

bool Sema::isFloatType(const Type *t) {
  if (!t || t->getKind() != TypeKind::Builtin) return false;
  switch (static_cast<const BuiltinType *>(t)->builtin) {
  case BuiltinTypeKind::Float32:
  case BuiltinTypeKind::Float64:
    return true;
  default:
    return false;
  }
}

std::string Sema::typeName(const Type *t) {
  if (!t) return "<unknown>";
  switch (t->getKind()) {
  case TypeKind::Builtin:
    switch (static_cast<const BuiltinType *>(t)->builtin) {
    case BuiltinTypeKind::Void:    return "void";
    case BuiltinTypeKind::Bool:    return "bool";
    case BuiltinTypeKind::Int32:   return "int";
    case BuiltinTypeKind::UInt32:  return "uint";
    case BuiltinTypeKind::Int64:   return "long";
    case BuiltinTypeKind::UInt64:  return "ulong";
    case BuiltinTypeKind::Float32: return "float";
    case BuiltinTypeKind::Float64: return "double";
    case BuiltinTypeKind::Float16: return "half";
    }
    return "<builtin>";
  case TypeKind::Pointer:
    return typeName(static_cast<const PointerType *>(t)->pointee) + "*";
  case TypeKind::Reference:
    return typeName(static_cast<const ReferenceType *>(t)->pointee) + "&";
  case TypeKind::Vector:
    return "vector<" +
           typeName(static_cast<const VectorType *>(t)->elem) + ">";
  case TypeKind::Record:
    return static_cast<const RecordType *>(t)->decl->name.str();
  case TypeKind::Typedef:
    return static_cast<const TypedefType *>(t)->decl->name.str();
  }
  return "<unknown>";
}

// Coarse byte-size of a type for `sizeof` folding. Scalars follow the usual
// widths (half=2, int/float=4, long/double/pointer=8, bool=1). Vectors are
// elem*count. Records sum field sizes WITHOUT alignment padding (not ABI-exact
// — sufficient for kernel buffer sizing). Unknown/void => 0.
int64_t Sema::sizeOfType(const Type *t) {
  if (!t) return 0;
  switch (t->getKind()) {
  case TypeKind::Builtin:
    switch (static_cast<const BuiltinType *>(t)->builtin) {
    case BuiltinTypeKind::Void:    return 0;
    case BuiltinTypeKind::Bool:    return 1;
    case BuiltinTypeKind::Float16: return 2;
    case BuiltinTypeKind::Int32:
    case BuiltinTypeKind::UInt32:
    case BuiltinTypeKind::Float32: return 4;
    case BuiltinTypeKind::Int64:
    case BuiltinTypeKind::UInt64:
    case BuiltinTypeKind::Float64: return 8;
    }
    return 0;
  case TypeKind::Pointer:
  case TypeKind::Reference:
    return 8;
  case TypeKind::Vector: {
    const auto *v = static_cast<const VectorType *>(t);
    return sizeOfType(resolveTypedefs(v->elem)) * v->count;
  }
  case TypeKind::Record: {
    int64_t sz = 0;
    for (const FieldDecl *f :
         static_cast<const RecordType *>(t)->decl->fields) {
      int64_t fsz = sizeOfType(resolveTypedefs(f->type));
      for (int64_t dim : f->arrayDims) fsz *= dim > 0 ? dim : 1;
      sz += fsz;
    }
    return sz;
  }
  case TypeKind::Typedef:
    return 0; // resolveTypedefs should have stripped this
  }
  return 0;
}

const char *Sema::opName(BinaryOp op) {
  switch (op) {
  case BinaryOp::Add:   return "+";
  case BinaryOp::Sub:   return "-";
  case BinaryOp::Mul:   return "*";
  case BinaryOp::Div:   return "/";
  case BinaryOp::Mod:   return "%";
  case BinaryOp::Assign: return "=";
  case BinaryOp::Eq:    return "==";
  case BinaryOp::NEq:   return "!=";
  case BinaryOp::Lt:    return "<";
  case BinaryOp::Gt:    return ">";
  case BinaryOp::Le:    return "<=";
  case BinaryOp::Ge:    return ">=";
  case BinaryOp::Shl:   return "<<";
  case BinaryOp::Shr:   return ">>";
  case BinaryOp::And:   return "&";
  case BinaryOp::Or:    return "|";
  case BinaryOp::Xor:   return "^";
  case BinaryOp::LAnd:  return "&&";
  case BinaryOp::LOr:   return "||";
  }
  return "?";
}

bool Sema::isCompatibleKinds(const Type *a, const Type *b) {
  if (!a || !b) return true; // unknown side: don't complain
  return resolveTypedefs(a)->getKind() == resolveTypedefs(b)->getKind();
}

bool Sema::isCompatibleForAssign(const Type *dst, const Type *src) {
  if (!dst || !src) return true; // unknown side: don't complain
  const Type *d = resolveTypedefs(dst);
  const Type *s = resolveTypedefs(src);
  if (isArithmetic(d) && isArithmetic(s)) return true; // implicit conversions
  // Pointer to pointer needs identical pointees: int* vs float* is a mismatch
  // even though both share the pointer kind. void* accepts any pointee,
  // mirroring C's void* conversions.
  if (d->getKind() == TypeKind::Pointer && s->getKind() == TypeKind::Pointer) {
    const Type *dp =
        resolveTypedefs(static_cast<const PointerType *>(d)->pointee);
    const Type *sp =
        resolveTypedefs(static_cast<const PointerType *>(s)->pointee);
    if (dp && dp->getKind() == TypeKind::Builtin &&
        static_cast<const BuiltinType *>(dp)->builtin == BuiltinTypeKind::Void)
      return true;
    if (!dp || !sp) return true; // unknown pointee side: don't complain
    if (dp->getKind() == TypeKind::Builtin && sp->getKind() == TypeKind::Builtin)
      // Unlike scalar assignment, builtin pointees must match exactly —
      // float* does not implicitly convert to int*.
      return static_cast<const BuiltinType *>(dp)->builtin ==
             static_cast<const BuiltinType *>(sp)->builtin;
    return dp->getKind() == sp->getKind();
  }
  // Same structural kind (vector/vector, record/record, pointer/pointer)
  // assigns cleanly; differing kinds need an explicit cast.
  return d->getKind() == s->getKind();
}

Type *Sema::builtin(BuiltinTypeKind k) const {
  return new BuiltinType(k);
}

void Sema::declare(StringRef name, ASTNode *node) {
  if (scopes.empty()) pushScope();
  auto &frame = scopes.back();
  // Re-declaring the same name in the same scope is an error; shadowing an
  // outer scope's name is fine (C blocking semantics).
  if (frame.find(name) != frame.end()) {
    error(node, "redefinition of '" + std::string(name) + "'");
    return;
  }
  frame[name] = node;
  // A fresh local starts unused; any DeclRefExpr flips the flag.
  if (node->getNodeType() == ASTNode::NodeKind::VarDecl ||
      node->getNodeType() == ASTNode::NodeKind::ParamDecl)
    used[node] = false;
}

ASTNode *Sema::lookup(StringRef name) {
  for (auto it = scopes.rbegin(), end = scopes.rend(); it != end; ++it) {
    auto found = it->find(name);
    if (found != it->end()) return found->second;
  }
  return nullptr;
}

// A scope is about to pop: any variable/parameter declared in it that was
// never referenced is almost certainly a bug worth surfacing (e.g. a typo'd
// name, or a value you meant to use).
void Sema::checkUnusedInFrame(const llvm::StringMap<ASTNode *> &frame) {
  for (const auto &entry : frame) {
    ASTNode *node = entry.second;
    auto it = used.find(node);
    if (it != used.end() && !it->second)
      warn(node,
           node->getNodeType() == ASTNode::NodeKind::ParamDecl
               ? "unused parameter '" + std::string(entry.getKey()) + "'"
               : "unused variable '" + std::string(entry.getKey()) + "'");
  }
}

// Match a call's argument types against a callee's parameter types (each
// argument is type-checked here, so callers must not pre-check them).
// Structural mismatches (e.g. passing a float* where an int was expected) are
// warnings; so is an implicit float->int conversion, which is legal C but
// almost always a silent precision-loss bug in a kernel.
void Sema::checkCallArgs(const ASTNode *call, StringRef calleeName,
                         FunctionDecl *f, const std::vector<NodePtr> &args) {
  // Default arguments: a call may omit trailing parameters that have defaults.
  // `f(a, b)` for `void f(int a, int b, int c = 10)` is fine — c is defaulted.
  if (args.size() > f->params.size()) {
    error(call, "call to '" + std::string(calleeName) + "' has " +
                    std::to_string(args.size()) + " args, expected " +
                    std::to_string(f->params.size()));
    return;
  }
  // Each omitted trailing argument MUST have a default value.
  for (size_t i = args.size(); i < f->params.size(); ++i) {
    if (!f->params[i]->defaultVal) {
      error(call, "call to '" + std::string(calleeName) + "' has " +
                      std::to_string(args.size()) + " args, expected " +
                      std::to_string(f->params.size()) + " (parameter '" +
                      std::string(f->params[i]->name) + "' has no default)");
      return;
    }
  }
  for (size_t i = 0; i < args.size(); ++i) {
    Type *paramTy = f->params[i]->type;
    Type *argTy = checkExpr(args[i].get());
    std::string msg = "argument " + std::to_string(i + 1) + " of '" +
                      std::string(calleeName) + "' has type " +
                      typeName(argTy) + ", expected " + typeName(paramTy);
    if (!isCompatibleForAssign(paramTy, argTy))
      warn(args[i].get(), msg);
    else if (isFloatType(resolveTypedefs(argTy)) &&
             isIntegerType(resolveTypedefs(paramTy)))
      warn(args[i].get(), msg + " (implicit conversion loses precision)");
  }
}

// Structural type equality after stripping typedefs. Used by redefinition
// detection and overload-set dedup; not a full canonical Type (that's a later
// milestone). Two types are equal iff their resolved kinds match and their
// distinguishing sub-fields recurse equal: builtins compare their BuiltinTypeKind,
// pointers/references recurse on pointee, vectors on (elem, count), records on
// the underlying StructDecl identity, typedefs never reach here (resolved above).
bool Sema::sameType(const Type *a, const Type *b) {
  if (!a || !b) return a == b; // null == null, null != non-null
  const Type *ra = resolveTypedefs(a);
  const Type *rb = resolveTypedefs(b);
  if (ra->getKind() != rb->getKind()) return false;
  switch (ra->getKind()) {
  case TypeKind::Builtin:
    return static_cast<const BuiltinType *>(ra)->builtin ==
           static_cast<const BuiltinType *>(rb)->builtin;
  case TypeKind::Pointer:
    return sameType(static_cast<const PointerType *>(ra)->pointee,
                    static_cast<const PointerType *>(rb)->pointee);
  case TypeKind::Reference:
    return sameType(static_cast<const ReferenceType *>(ra)->pointee,
                    static_cast<const ReferenceType *>(rb)->pointee);
  case TypeKind::Vector: {
    const auto *va = static_cast<const VectorType *>(ra);
    const auto *vb = static_cast<const VectorType *>(rb);
    return va->count == vb->count && sameType(va->elem, vb->elem);
  }
  case TypeKind::Record:
    // Identity by decl: two RecordTypes are the same type iff they name the
    // same StructDecl. (Distinct decls with identical fields are distinct
    // types, matching C++ struct identity.)
    return static_cast<const RecordType *>(ra)->decl ==
           static_cast<const RecordType *>(rb)->decl;
  case TypeKind::Typedef:
    return false; // resolved above; unreachable
  }
  return false;
}

// Two functions have the same signature (and so collide in an overload set) iff
// their parameter counts match and each parameter pair is the same type. Return
// type is intentionally NOT compared: C++ allows overloads differing only in
// return type to be a redefinition error (return type is not part of the
// signature for overload resolution).
bool Sema::sameSignature(const FunctionDecl *a, const FunctionDecl *b) {
  if (a->params.size() != b->params.size()) return false;
  for (size_t i = 0; i < a->params.size(); ++i)
    if (!sameType(a->params[i]->type, b->params[i]->type))
      return false;
  return true;
}

// Conversion rank for binding an argument of type `arg` to a parameter of type
// `param`. Lower is better. Mirrors C/CUDA's implicit-conversion ladder:
//   0 exact          — same type (typedefs resolved)
//   1 promotion      — int->wider int, float->double (lossless widening)
//   2 standard       — int<->float same-or-narrower width, cross-kind numeric
//   3 lossy          — float->int, large-int->small-int (narrowing)
//   4 incompatible   — pointer mismatch, vector/record mismatch, kind mismatch
// `sameType` and the arithmetic taxonomy (isArithmetic/isIntegerType/isFloatType)
// are reused so this stays consistent with the existing assignment checks.
int Sema::conversionRank(const Type *param, const Type *arg) {
  if (!param || !arg) return 0; // unknown side: treat as exact (don't penalize)
  if (sameType(param, arg)) return 0; // exact match
  const Type *p = resolveTypedefs(param);
  const Type *a = resolveTypedefs(arg);
  // Arithmetic conversions: rank by direction of conversion.
  if (isArithmetic(p) && isArithmetic(a)) {
    bool pInt = isIntegerType(p), aInt = isIntegerType(a);
    bool pFlt = isFloatType(p), aFlt = isFloatType(a);
    // float -> int is lossy.
    if (pInt && aFlt) return 3;
    // int -> float: standard (legal but may lose int precision for large ints).
    if (pFlt && aInt) return 2;
    // int -> int or float -> float: rank by width change.
    int pw = static_cast<int>(sizeOfType(p));
    int aw = static_cast<int>(sizeOfType(a));
    if (pw == aw) return 1;            // same width, different signedness: promotion-ish
    if (pw < aw) return 1;             // narrowing the value range is a promotion? no —
    // pw > aw: widening (int -> long, float -> double): lossless promotion.
    return 1; // any same-kind numeric that isn't exact is at worst promotion/standard
    // (kept simple: real C++ ranks int->long as promotion, int->short as narrowing,
    //  but VC has no short type, so width diffs are benign here.)
  }
  // Pointer conversions: void* binds to any pointer (rank 2); otherwise pointees
  // must match exactly (rank 0 handled above, so a mismatch here is incompatible).
  if (p->getKind() == TypeKind::Pointer && a->getKind() == TypeKind::Pointer) {
    const Type *pp = resolveTypedefs(static_cast<const PointerType *>(p)->pointee);
    if (pp && pp->getKind() == TypeKind::Builtin &&
        static_cast<const BuiltinType *>(pp)->builtin == BuiltinTypeKind::Void)
      return 2;
    return 4;
  }
  // Same-kind non-arithmetic (vector/record/reference) that isn't exactly equal
  // is incompatible — no implicit conversions exist for those.
  return 4;
}

// Pick the best overload candidate for a call with the given argument types.
// Each candidate is scored by the WORST conversion rank across its parameters
// (after applying default arguments for omitted trailing params); the candidate
// with the lowest worst-rank wins. A tie at the lowest rank is ambiguous and
// reported as an error listing the tied candidates. No viable candidate (every
// candidate has an arity mismatch or an incompatible binding) is also an error.
// On success returns the chosen FunctionDecl and (if outRank set) its rank.
FunctionDecl *Sema::resolveOverload(const ASTNode *call, StringRef calleeName,
                                    const llvm::SmallVector<FunctionDecl *, 2> &cands,
                                    const std::vector<NodePtr> &args,
                                    int *outRank) {
  FunctionDecl *best = nullptr;
  int bestRank = 0;
  bool ambiguous = false;
  llvm::SmallVector<FunctionDecl *, 2> tied; // candidates tied with `best`

  for (FunctionDecl *f : cands) {
    // Arity with default arguments: args may omit trailing defaulted params.
    if (args.size() > f->params.size()) continue; // too many args: not viable
    bool viable = true;
    for (size_t i = args.size(); i < f->params.size(); ++i)
      if (!f->params[i]->defaultVal) { viable = false; break; }
    if (!viable) continue;

    int worst = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      Type *argTy = checkExpr(args[i].get());
      int r = conversionRank(f->params[i]->type, argTy);
      if (r > worst) worst = r;
    }
    // rank 4 (incompatible) still counts as "viable-but-bad" so that a single
    // candidate produces a soft diagnostic rather than a hard no-viable error;
    // ambiguity is only declared among candidates whose worst rank is < 4.
    if (!best || worst < bestRank) {
      best = f;
      bestRank = worst;
      ambiguous = false;
      tied.clear();
      tied.push_back(f);
    } else if (worst == bestRank) {
      tied.push_back(f);
      if (bestRank < 4) ambiguous = true;
    }
  }

  if (!best) {
    error(call, "no matching function for call to '" + std::string(calleeName) +
                    "' with " + std::to_string(args.size()) + " argument(s)");
    return nullptr;
  }
  if (ambiguous) {
    error(call, "call to '" + std::string(calleeName) + "' is ambiguous");
    for (FunctionDecl *f : tied)
      warn(f, "candidate: '" + std::string(f->name) + "' (rank " +
                  std::to_string(bestRank) + ")");
    return nullptr;
  }
  if (outRank) *outRank = bestRank;
  return best;
}

void Sema::error(const ASTNode *at, std::string msg) {
  Diagnostic d{DiagnosticKind::Error, at ? at->getLoc() : SourceLocation{},
               std::move(msg)};
  tu.diagnostics.push_back(d);
}
void Sema::warn(const ASTNode *at, std::string msg) {
  Diagnostic d{DiagnosticKind::Warning, at ? at->getLoc() : SourceLocation{},
               std::move(msg)};
  tu.diagnostics.push_back(d);
}

bool Sema::analyze() {
  collectTopLevel();
  checkFunctions();

  // Render diagnostics through the SourceMgr so they get the same
  // caret/source-line treatment as parser errors. Parser errors are already
  // rendered when they occur (their `rendered` flag is set), so reporting
  // them again here is skipped — this makes each problem print exactly once.
  // With -Werror, warnings are upgraded to errors for both output and the
  // exit status.
  bool hadError = false;
  for (const Diagnostic &d : tu.diagnostics) {
    DiagnosticKind kind = d.kind;
    if (warningsAsErrors && kind == DiagnosticKind::Warning)
      kind = DiagnosticKind::Error;
    if (!d.rendered) {
      llvm::SourceMgr::DiagKind smKind = kind == DiagnosticKind::Error
                                       ? llvm::SourceMgr::DK_Error
                                       : kind == DiagnosticKind::Warning
                                             ? llvm::SourceMgr::DK_Warning
                                             : llvm::SourceMgr::DK_Note;
      if (d.where.loc.getPointer())
        srcMgr.PrintMessage(d.where.loc, smKind, d.message);
      else
        llvm::errs() << (smKind == llvm::SourceMgr::DK_Error
                             ? "error: "
                             : smKind == llvm::SourceMgr::DK_Warning
                                   ? "warning: "
                                   : "note: ")
                     << d.message << "\n";
    }
    if (kind == DiagnosticKind::Error) hadError = true;
  }
  return !hadError;
}

void Sema::collectTopLevel() {
  collectDecls(tu.decls, StringRef());
}

// Mangle a plain name (struct/typedef/var/enum-const) under a namespace prefix.
static std::string mangleScoped(StringRef prefix, StringRef name) {
  if (prefix.empty()) return name.str();
  return prefix.str() + "_" + name.str();
}

void Sema::collectDecls(const std::vector<NodePtr> &decls, StringRef nsPrefix) {
  for (auto &d : decls) {
    switch (d->getNodeType()) {
    case ASTNode::NodeKind::NamespaceDecl: {
      // Recurse into the namespace body with an extended prefix. Nested
      // namespaces chain: `namespace a { namespace b { ... } }` -> `a_b`.
      auto *ns = static_cast<NamespaceDecl *>(d.get());
      std::string inner = mangleScoped(nsPrefix, ns->name);
      // Persist the inner prefix string for the recursive StringRefs.
      static std::vector<std::unique_ptr<std::string>> pstore;
      pstore.push_back(std::make_unique<std::string>(std::move(inner)));
      collectDecls(ns->decls, *pstore.back());
      break;
    }
    case ASTNode::NodeKind::FunctionDecl: {
      auto *f = static_cast<FunctionDecl *>(d.get());
      // Register under the namespace-mangled BASE name (no parameter suffix) so
      // overloaded functions (f(int), f(float)) share a key and form one
      // overload set. The backend emits each under its full deviceMangledName
      // (with parameter suffix) for distinct symbols. The key is derived from
      // f->nsName (colon form) via the shared deviceBaseName.
      std::string key = deviceBaseName(f);
      static std::vector<std::unique_ptr<std::string>> fstore;
      fstore.push_back(std::make_unique<std::string>(std::move(key)));
      StringRef keyRef = *fstore.back();
      auto &overloadSet = functions[keyRef];
      // Redefinition check: an identical signature (same param types) in the
      // same overload set is a redefinition error. Distinct signatures are
      // legal overloads.
      for (FunctionDecl *existing : overloadSet) {
        if (sameSignature(f, existing)) {
          error(f, "redefinition of '" + std::string(f->name) + "'");
          break;
        }
      }
      overloadSet.push_back(f);
      // Default-argument contiguity check (same as top-level free functions).
      bool seenDefault = false;
      for (ParamDecl *p : f->params) {
        if (p->defaultVal) seenDefault = true;
        else if (seenDefault) {
          error(p, "default argument missing for parameter '" +
                       std::string(p->name) + "' of '" +
                       std::string(f->name) + "' (parameters with defaults "
                       "must be right-to-left contiguous)");
        }
      }
      break;
    }
    case ASTNode::NodeKind::StructDecl: {
      auto *s = static_cast<StructDecl *>(d.get());
      // A struct/class inside a namespace is registered under its mangled name
      // (`ns_Class`) so scoped type references resolve. The StructDecl's own
      // `name` is left intact for host emission; only the typeNames key is
      // mangled (the device backend emits the struct under the mangled name).
      if (!nsPrefix.empty()) {
        std::string key = mangleScoped(nsPrefix, s->name);
        static std::vector<std::unique_ptr<std::string>> tstore;
        tstore.push_back(std::make_unique<std::string>(std::move(key)));
        typeNames[*tstore.back()] = new RecordType(s);
      } else {
        typeNames[s->name] = new RecordType(s);
      }
      break;
    }
    case ASTNode::NodeKind::TypedefDecl: {
      auto *t = static_cast<TypedefDecl *>(d.get());
      if (!nsPrefix.empty()) {
        std::string key = mangleScoped(nsPrefix, t->name);
        static std::vector<std::unique_ptr<std::string>> ttstore;
        ttstore.push_back(std::make_unique<std::string>(std::move(key)));
        typeNames[*ttstore.back()] = new TypedefType(t);
      } else {
        typeNames[t->name] = new TypedefType(t);
      }
      break;
    }
    case ASTNode::NodeKind::EnumDecl: {
      auto *e = static_cast<EnumDecl *>(d.get());
      // Enum constants inside a namespace are registered under a mangled key
      // (`ns_NAME`) so `ns::CONST` resolves.
      for (auto &c : e->constants) {
        if (!nsPrefix.empty()) {
          std::string key = mangleScoped(nsPrefix, c.name);
          static std::vector<std::unique_ptr<std::string>> estore;
          estore.push_back(std::make_unique<std::string>(std::move(key)));
          enumConstants[*estore.back()] = c.value;
        } else {
          enumConstants[c.name] = c.value;
        }
      }
      break;
    }
    case ASTNode::NodeKind::VarDecl: {
      auto *v = static_cast<VarDecl *>(d.get());
      if (!nsPrefix.empty()) {
        std::string key = mangleScoped(nsPrefix, v->name);
        static std::vector<std::unique_ptr<std::string>> vstore;
        vstore.push_back(std::make_unique<std::string>(std::move(key)));
        globalVars[*vstore.back()] = v;
      } else {
        globalVars[v->name] = v;
      }
      break;
    }
    default:
      break;
    }
  }
}

void Sema::checkFunctions() {
  checkFunctionsIn(tu.decls);
}

// Recursively walk top-level decls (and namespace bodies) and type-check every
// device function body. Mirrors collectDecls' recursion so methods/helpers
// declared inside a namespace are still checked.
void Sema::checkFunctionsIn(const std::vector<NodePtr> &decls) {
  for (auto &d : decls) {
    if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
      checkFunctionsIn(static_cast<NamespaceDecl *>(d.get())->decls);
      continue;
    }
    if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
    auto *f = static_cast<FunctionDecl *>(d.get());
    if (!f->body) continue;
    // Only check device code (kernels and __device__ helpers). Host functions
    // (DeviceAttr::Host or None, e.g. `int main()` calling printf/vcMalloc)
    // are lowered to C++ by the host backend, where printf/vcMalloc/etc. are
    // real symbols — checking them here would produce false "undeclared
    // function" warnings. Host-side expression checking is intentionally lax.
    if (f->deviceAttr != DeviceAttr::Global &&
        f->deviceAttr != DeviceAttr::Device)
      continue;
    pushScope();
    currentFunc = f;
    for (ParamDecl *p : f->params) declare(p->name, p);
    // A device method references `this` (spelled "_this" by the parser). The
    // device backend lowers the method to `Class_method(Class _this, ...)`,
    // so synthesize a `_this` parameter typed as the class record so the body
    // type-checks. Host methods are not checked here (host path is lax), so
    // this only applies to __device__/__global__ methods.
    if (f->isMethod && !f->className.empty()) {
      auto it = typeNames.find(f->className);
      if (it != typeNames.end()) {
        auto *thisParam = new ParamDecl(f->getLoc(), it->second, "_this");
        declare("_this", thisParam);
      }
    }
    checkStmt(f->body.get());
    currentFunc = nullptr;
    popScope();
  }
}

void Sema::checkStmt(const ASTNode *n) {
  if (!n) return;
  switch (n->getNodeType()) {
  case ASTNode::NodeKind::CompoundStmt: {
    pushScope();
    for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
      checkStmt(s.get());
    popScope();
    return;
  }
  case ASTNode::NodeKind::DeclStmt: {
    auto *ds = static_cast<const DeclStmt *>(n);
    for (VarDecl *v : ds->decls) {
      if (!v) continue;
      // Function-local `static`/`extern` in device code has no GLSL lowering:
      // GLSL has no function-local static storage (a `static` local would need
      // to persist across invocations, which SPIR-V function storage doesn't),
      // and `extern` locals have no device analogue. `__shared__` is the device
      // persistent-storage mechanism, not `static`. Host functions are lowered
      // to C++ where these are legal, so only reject inside device code.
      bool inDevice = currentFunc &&
                      (currentFunc->deviceAttr == DeviceAttr::Global ||
                       currentFunc->deviceAttr == DeviceAttr::Device);
      if (inDevice && v->storageClass == StorageClass::Static)
        error(v, "'static' local variable is not allowed in device code "
                 "(use __shared__ for block-local persistent storage)");
      // `extern` is allowed in device code ONLY as the CUDA dynamic-shared
      // spelling `extern __shared__ T s[]` (an unsized workgroup array sized
      // at launch). A bare `extern` local (no __shared__) still has no device
      // lowering.
      if (inDevice && v->storageClass == StorageClass::Extern && !v->isShared)
        error(v, "'extern' local variable is not allowed in device code "
                 "(only `extern __shared__ T s[]` is supported)");
      declare(v->name, v);
      if (v->init) {
        Type *initTy = checkExpr(v->init.get());
        if (initTy && initTy->getKind() == TypeKind::Builtin &&
            static_cast<BuiltinType *>(initTy)->builtin ==
                BuiltinTypeKind::Void)
          warn(v, "initializer has type void");
        else if (v->type && !isCompatibleForAssign(v->type, initTy))
          warn(v, "initializer of type " + typeName(initTy) +
                      " does not match declared type " + typeName(v->type));
      }
    }
    // Back-compat: a DeclStmt built with a single decl also sets `decl`.
    return;
  }
  case ASTNode::NodeKind::ExprStmt:
    checkExpr(static_cast<const ExprStmt *>(n)->expr.get());
    return;
  case ASTNode::NodeKind::ReturnStmt: {
    auto *r = static_cast<const ReturnStmt *>(n);
    Type *valTy = checkExpr(r->value.get());
    if (!currentFunc) return;
    Type *retTy = currentFunc->returnType;
    bool isVoid = retTy && retTy->getKind() == TypeKind::Builtin &&
                  static_cast<BuiltinType *>(retTy)->builtin ==
                      BuiltinTypeKind::Void;
    if (isVoid) {
      if (r->value)
        error(n, "void function '" + std::string(currentFunc->name) +
                     "' cannot return a value");
    } else {
      if (!r->value)
        warn(n, "non-void function '" + std::string(currentFunc->name) +
                    "' returns no value");
      else if (retTy && !isCompatibleForAssign(retTy, valTy))
        warn(n, "return type " + typeName(valTy) +
                    " does not match declared return type " + typeName(retTy));
    }
    return;
  }
  case ASTNode::NodeKind::IfStmt: {
    auto *iff = static_cast<const IfStmt *>(n);
    checkCond(iff->cond.get(), "if");
    checkStmt(iff->thenStmt.get());
    checkStmt(iff->elseStmt.get());
    return;
  }
  case ASTNode::NodeKind::ForStmt: {
    auto *fs = static_cast<const ForStmt *>(n);
    pushScope();
    checkStmt(fs->init.get());
    checkCond(fs->cond.get(), "for");
    checkExpr(fs->step.get());
    ++loopDepth;
    ++breakableDepth;
    checkStmt(fs->body.get());
    --breakableDepth;
    --loopDepth;
    popScope();
    return;
  }
  case ASTNode::NodeKind::WhileStmt: {
    auto *ws = static_cast<const WhileStmt *>(n);
    checkCond(ws->cond.get(), "while");
    ++loopDepth;
    ++breakableDepth;
    checkStmt(ws->body.get());
    --breakableDepth;
    --loopDepth;
    return;
  }
  case ASTNode::NodeKind::DoStmt: {
    auto *ds = static_cast<const DoStmt *>(n);
    ++loopDepth;
    ++breakableDepth;
    checkStmt(ds->body.get());
    --breakableDepth;
    --loopDepth;
    checkCond(ds->cond.get(), "do-while");
    return;
  }
  case ASTNode::NodeKind::SwitchStmt: {
    auto *sw = static_cast<const SwitchStmt *>(n);
    checkCond(sw->cond.get(), "switch");
    switchStack.push_back({});
    ++breakableDepth;
    checkStmt(sw->body.get());
    --breakableDepth;
    switchStack.pop_back();
    return;
  }
  case ASTNode::NodeKind::CaseStmt: {
    auto *cs = static_cast<const CaseStmt *>(n);
    checkExpr(cs->value.get());
    // Duplicate-case detection: constant labels only need comparing values.
    if (!switchStack.empty()) {
      SwitchCases &ctx = switchStack.back();
      if (!cs->value) {
        if (ctx.sawDefault)
          warn(cs, "multiple default labels in one switch");
        ctx.sawDefault = true;
      } else if (cs->value->getNodeType() ==
                 ASTNode::NodeKind::IntegerLiteral) {
        int64_t v = static_cast<const IntegerLiteral *>(cs->value.get())->value;
        if (!ctx.values.insert(v).second)
          warn(cs, "duplicate case value " + std::to_string(v));
      }
    }
    checkStmt(cs->sub.get());
    return;
  }
  case ASTNode::NodeKind::BreakStmt:
    if (breakableDepth == 0)
      error(n, "'break' statement not inside a loop or switch");
    return;
  case ASTNode::NodeKind::ContinueStmt:
    if (loopDepth == 0)
      error(n, "'continue' statement not inside a loop");
    return;
  default:
    return;
  }
}

// Conditions must be usable as a boolean: a scalar builtin. Vectors, pointers
// and structs in a condition slot are almost certainly a bug.
void Sema::checkCond(const ASTNode *cond, const char *what) {
  if (!cond) return;
  Type *t = checkExpr(cond);
  if (t && !isArithmetic(t))
    warn(cond, std::string(what) + " condition must be a scalar (got " +
                   typeName(t) + ")");
}

Type *Sema::checkExpr(const ASTNode *n) {
  if (!n) return nullptr;
  switch (n->getNodeType()) {
  case ASTNode::NodeKind::IntegerLiteral:
    exprTypes[n] = builtin(BuiltinTypeKind::Int32);
    return exprTypes[n];
  case ASTNode::NodeKind::FloatLiteral:
    exprTypes[n] = builtin(BuiltinTypeKind::Float32);
    return exprTypes[n];
  case ASTNode::NodeKind::CharLiteral:
    // C promotes char to int; treat as Int32.
    exprTypes[n] = builtin(BuiltinTypeKind::Int32);
    return exprTypes[n];
  case ASTNode::NodeKind::BoolLiteral:
    exprTypes[n] = builtin(BuiltinTypeKind::Bool);
    return exprTypes[n];
  case ASTNode::NodeKind::SizeOfExpr: {
    auto *s = static_cast<SizeOfExpr *>(const_cast<ASTNode *>(n));
    // Determine the type whose size we want: the explicit target type, or the
    // type of the operand expression.
    Type *ty = nullptr;
    int64_t arrMul = 1; // for `sizeof(arr)`, multiply elem size by total elems
    if (s->isType) {
      ty = s->target;
    } else {
      ty = checkExpr(s->sub.get());
      // If the operand is a bare array variable, checkExpr returns its element
      // type; fold in the array dimensions so `sizeof(arr)` gives the total
      // byte count (matching C semantics for stack arrays).
      if (s->sub &&
          s->sub->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        StringRef nm = static_cast<const DeclRefExpr *>(s->sub.get())->name;
        ASTNode *sym = lookup(nm);
        if (sym && sym->getNodeType() == ASTNode::NodeKind::VarDecl) {
          for (int64_t dim :
               static_cast<VarDecl *>(sym)->arrayDims)
            arrMul *= dim > 0 ? dim : 1;
        }
      }
    }
    s->folded = sizeOfType(resolveTypedefs(ty)) * arrMul;
    if (s->folded == 0)
      warn(n, "sizeof unable to compute size of type '" +
                  typeName(ty) + "' (folded to 0)");
    exprTypes[n] = builtin(BuiltinTypeKind::Int32);
    return exprTypes[n];
  }
  case ASTNode::NodeKind::StringLiteral:
    // No string type in kernels; legal but unusable as a value.
    return nullptr;
  case ASTNode::NodeKind::DeclRefExpr: {
    auto *d = static_cast<const DeclRefExpr *>(n);
    if (isThreadBuiltin(d->name)) {
      // Implicit thread-index identifier; backend lowers it. The index
      // builtins (threadIdx/blockIdx/...) are struct-like and typed unknown;
      // warpSize is a scalar int.
      if (d->name == "warpSize")
        return builtin(BuiltinTypeKind::Int32);
      return nullptr;
    }
    // Unscoped enum constant — an integer value.
    auto it = enumConstants.find(d->name);
    if (it != enumConstants.end())
      return builtin(BuiltinTypeKind::Int32);
    // Top-level __constant__ global.
    auto gv = globalVars.find(d->name);
    if (gv != globalVars.end())
      return gv->second->type;
    if (typeNames.count(d->name)) return nullptr; // a type name used as a value?
    ASTNode *sym = lookup(d->name);
    if (!sym) {
      error(n, "use of undeclared identifier '" + std::string(d->name) + "'");
      return nullptr;
    }
    // Any reference counts as a use for the unused-variable check.
    if (sym->getNodeType() == ASTNode::NodeKind::VarDecl ||
        sym->getNodeType() == ASTNode::NodeKind::ParamDecl)
      used[sym] = true;
    if (sym->getNodeType() == ASTNode::NodeKind::VarDecl)
      return static_cast<VarDecl *>(sym)->type;
    if (sym->getNodeType() == ASTNode::NodeKind::ParamDecl)
      return static_cast<ParamDecl *>(sym)->type;
    return nullptr;
  }
  case ASTNode::NodeKind::BinaryExpr: {
    auto *b = static_cast<const BinaryExpr *>(n);
    Type *lt = checkExpr(b->lhs.get());
    Type *rt = checkExpr(b->rhs.get());
    const Type *rl = resolveTypedefs(lt);
    const Type *rr = resolveTypedefs(rt);
    // Dividing or taking modulo by a literal zero is always a bug.
    if (b->op == BinaryOp::Div || b->op == BinaryOp::Mod) {
      int64_t zero = 0;
      if (constIntValue(b->rhs.get(), zero) && zero == 0)
        warn(b->rhs.get(), "division by zero");
    }
    // '%' and the shifts only make sense for integer operands.
    if (b->op == BinaryOp::Mod) {
      const Type *bad = nullptr;
      if (rl && !isIntegerType(rl) && isArithmetic(rl)) bad = rl;
      else if (rr && !isIntegerType(rr) && isArithmetic(rr)) bad = rr;
      if (bad)
        warn(n, "operand of '%' has non-integer type " + typeName(bad));
    }
    if (b->op == BinaryOp::Shl || b->op == BinaryOp::Shr) {
      if (rl && !isIntegerType(rl) && isArithmetic(rl))
        warn(b->lhs.get(), "left operand of '" +
                               std::string(opName(b->op)) +
                               "' must be an integer (got " + typeName(rl) + ")");
      if (rr && !isIntegerType(rr) && isArithmetic(rr))
        warn(b->rhs.get(), "right operand of '" +
                               std::string(opName(b->op)) +
                               "' must be an integer (got " + typeName(rr) + ")");
    }
    // p + q (two pointers) has no meaning; p - q is the only legal pair.
    if (b->op == BinaryOp::Add && rl && rr &&
        rl->getKind() == TypeKind::Pointer &&
        rr->getKind() == TypeKind::Pointer)
      warn(n, "invalid operands to binary '+' (two pointers)");
    if (b->op == BinaryOp::Assign) {
      // LHS must be an lvalue.
      auto k = b->lhs->getNodeType();
      if (k != ASTNode::NodeKind::DeclRefExpr &&
          k != ASTNode::NodeKind::IndexExpr &&
          k != ASTNode::NodeKind::MemberAccessExpr)
        warn(n, "assignment to non-lvalue");
      if (lt && rt && !isCompatibleForAssign(lt, rt))
        warn(n, "assigning " + typeName(rt) + " to variable of type " +
                    typeName(lt));
      // Writing to a variable is not "using" it: checkExpr() above flagged the
      // LHS DeclRefExpr as used, but a store-only variable is still unused
      // (mirrors GCC's -Wunused-but-set-variable). A read through an index or
      // member (`a[0] = x`, `s.f = x`) still counts as using the base.
      if (k == ASTNode::NodeKind::DeclRefExpr) {
        ASTNode *sym =
            lookup(static_cast<const DeclRefExpr *>(b->lhs.get())->name);
        if (sym &&
            (sym->getNodeType() == ASTNode::NodeKind::VarDecl ||
             sym->getNodeType() == ASTNode::NodeKind::ParamDecl))
          used[sym] = false;
        // Writing to a `const`-qualified variable is illegal.
        if (sym) {
          if (sym->getNodeType() == ASTNode::NodeKind::VarDecl &&
              static_cast<VarDecl *>(sym)->isConst)
            error(n, "assignment to const variable '" +
                         std::string(
                             static_cast<const DeclRefExpr *>(b->lhs.get())->name) +
                         "'");
          else if (sym->getNodeType() == ASTNode::NodeKind::ParamDecl &&
                   static_cast<ParamDecl *>(sym)->isConst)
            error(n, "assignment to const parameter '" +
                         std::string(
                             static_cast<const DeclRefExpr *>(b->lhs.get())->name) +
                         "'");
        }
      }
      return rt ? rt : lt;
    }
    // Plain operators (incl. comparisons and shifts): operands of different
    // kinds only combine through an explicit cast, except pointer+index Add/Sub
    // which is the standard way to step through an array.
    bool logical = b->op == BinaryOp::LAnd || b->op == BinaryOp::LOr;
    bool ptrArith = (b->op == BinaryOp::Add || b->op == BinaryOp::Sub) &&
                    ((lt && lt->getKind() == TypeKind::Pointer &&
                      rt && rt->getKind() == TypeKind::Builtin) ||
                     (lt && lt->getKind() == TypeKind::Builtin &&
                      rt && rt->getKind() == TypeKind::Pointer));
    if (!logical && !ptrArith && !isCompatibleKinds(lt, rt))
      warn(n, "operands of '" + std::string(opName(b->op)) +
                  "' have incompatible types (" + typeName(lt) + " and " +
                  typeName(rt) + ")");
    return rt ? rt : lt;
  }
  case ASTNode::NodeKind::UnaryExpr: {
    auto *u = static_cast<const UnaryExpr *>(n);
    Type *t = checkExpr(u->operand.get());
    switch (u->op) {
    case UnaryOp::Deref:
      if (t && t->getKind() != TypeKind::Pointer)
        warn(n, "indirection requires pointer operand (got " + typeName(t) +
                    ")");
      return t && t->getKind() == TypeKind::Pointer
                 ? static_cast<PointerType *>(t)->pointee
                 : t;
    case UnaryOp::Neg:
      if (t && !isArithmetic(t) && t->getKind() != TypeKind::Vector)
        warn(n, "unary '-' on non-numeric type " + typeName(t));
      return t;
    default:
      return t;
    }
  }
  case ASTNode::NodeKind::ConditionalExpr: {
    auto *c = static_cast<const ConditionalExpr *>(n);
    checkCond(c->cond.get(), "ternary");
    Type *tt = checkExpr(c->thenExpr.get());
    Type *et = checkExpr(c->elseExpr.get());
    if (tt && et && !isCompatibleKinds(tt, et))
      warn(n, "incompatible operand types (" + typeName(tt) + " and " +
                  typeName(et) + ") in ternary expression");
    return tt ? tt : et;
  }
  case ASTNode::NodeKind::CommaExpr: {
    auto *c = static_cast<const CommaExpr *>(n);
    checkExpr(c->lhs.get());
    return checkExpr(c->rhs.get());
  }
  case ASTNode::NodeKind::CStyleCastExpr: {
    auto *cc = static_cast<const CStyleCastExpr *>(n);
    // Visit the sub-expression so uses inside a cast count (e.g. `(void)x`)
    // and its own diagnostics still fire.
    checkExpr(cc->sub.get());
    return cc->target;
  }
  case ASTNode::NodeKind::InitListExpr: {
    auto *il = static_cast<const InitListExpr *>(n);
    Type *first = nullptr;
    for (auto &e : il->elements) {
      Type *t = checkExpr(e.get());
      if (!first) first = t;
    }
    return first;
  }
  case ASTNode::NodeKind::CallExpr: {
    auto *c = static_cast<const CallExpr *>(n);
    // Resolve callee name if it's a plain DeclRefExpr.
    StringRef calleeName;
    bool calleeIsScoped = false;
    if (c->callee &&
        c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      calleeName = static_cast<const DeclRefExpr *>(c->callee.get())->name;
    else if (c->callee && c->callee->getNodeType() ==
                               ASTNode::NodeKind::MemberAccessExpr) {
      // Scoped call `ns::func(...)` or member call `obj.method(...)`. For a
      // scope (`::`) the callee spelling is mangled to `ns_func` so it resolves
      // against the device symbol table.
      auto *ma = static_cast<const MemberAccessExpr *>(c->callee.get());
      if (ma->isScope) {
        calleeIsScoped = true;
        std::string chain = scopeChainStr(ma);       // "A::B::func"
        std::string mangled = mangleScopeName(chain); // "A_B_func"
        static std::vector<std::unique_ptr<std::string>> cstore;
        cstore.push_back(std::make_unique<std::string>(std::move(mangled)));
        calleeName = *cstore.back();
      }
    }

    // Vector constructors (float4(...)) and math builtins pass through.
    if (!calleeName.empty()) {
      if (isMathBuiltin(calleeName)) {
        for (auto &a : c->args) checkExpr(a.get());
        return nullptr;
      }
      auto it = functions.find(calleeName);
      if (it != functions.end()) {
        const auto &overloadSet = it->second;
        int rank = 0;
        FunctionDecl *f = resolveOverload(n, calleeName, overloadSet, c->args, &rank);
        if (f) {
          const_cast<CallExpr *>(c)->resolvedCallee =
              f; // backend emits this callee's mangled symbol
          // GLSL forbids recursion: a device function calling itself (directly)
          // would lower to a recursive GLSL function, which is invalid. Reject it
          // here rather than emit illegal GLSL. Indirect recursion (A->B->A) is
          // not detected — TODO: needs a call-graph closure.
          if (currentFunc && f == currentFunc)
            error(n, "recursive function '" + std::string(calleeName) +
                         "' is not allowed in GLSL (device functions cannot "
                         "call themselves)");
          // Device code may only call __device__/__global__ functions; a plain
          // (host) function is not callable from a kernel.
          if ((f->deviceAttr == DeviceAttr::None ||
               f->deviceAttr == DeviceAttr::Host) &&
              currentFunc &&
              (currentFunc->deviceAttr == DeviceAttr::Global ||
               currentFunc->deviceAttr == DeviceAttr::Device))
            warn(n, "call to host function '" + std::string(calleeName) +
                        "' from device code");
          // checkCallArgs re-type-checks each arg (cheap) and emits the lossy
          // float->int warning for the chosen candidate.
          checkCallArgs(n, calleeName, f, c->args);
          return f->returnType;
        }
        // resolveOverload already reported no-viable/ambiguous; nothing more.
        for (auto &a : c->args) checkExpr(a.get());
        return nullptr;
      }
      // Unknown callee: could be a GLSL builtin we didn't list (e.g. a
      // vector constructor). Warn softly rather than hard-error, so we don't
      // break valid kernels using less-common builtins.
      for (auto &a : c->args) checkExpr(a.get());
      warn(n, "call to undeclared function '" + std::string(calleeName) +
                  "' (assuming builtin)");
      return nullptr;
    }
    // Object method call `obj.method(args)` (MemberAccessExpr callee, not a
    // scope): resolve the method against the object's struct type. The device
    // backend lowers `Class::method` to a free function `Class_method`, and with
    // parameter mangling the emitted symbol is `Class_method_<params>`; the
    // backend cannot recover the param types itself, so Sema must pick the
    // overload and mark resolvedCallee. Scoped calls (`Class::method(...)`,
    // `ns::f(...)`) were handled above via calleeName.
    if (c->callee &&
        c->callee->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
      auto *ma = static_cast<const MemberAccessExpr *>(c->callee.get());
      if (!ma->isScope) {
        Type *baseTy = checkExpr(ma->base.get());
        const Type *rec = resolveTypedefs(baseTy);
        if (rec && rec->getKind() == TypeKind::Record) {
          StructDecl *sd = static_cast<const RecordType *>(rec)->decl;
          // Gather same-named methods into an overload set and resolve. The
          // object (`this`) is the implicit first parameter, so the explicit
          // args bind to params[1..].
          llvm::SmallVector<FunctionDecl *, 2> methods;
          for (FunctionDecl *m : sd->methods)
            if (m->name == ma->member) methods.push_back(m);
          if (!methods.empty()) {
            FunctionDecl *f = resolveOverload(n, ma->member, methods, c->args);
            if (f) {
              const_cast<CallExpr *>(c)->resolvedCallee = f;
              checkCallArgs(n, ma->member, f, c->args);
              return f->returnType;
            }
            for (auto &a : c->args) checkExpr(a.get());
            return nullptr;
          }
        }
      }
    }
    checkExpr(c->callee.get());
    for (auto &a : c->args) checkExpr(a.get());
    return nullptr;
  }
  case ASTNode::NodeKind::IndexExpr: {
    auto *ie = static_cast<const IndexExpr *>(n);
    Type *base = checkExpr(ie->base.get());
    Type *idx = checkExpr(ie->index.get());
    const Type *rb = resolveTypedefs(base);

    // An array variable (`float a[16]`) is typed as its element type but
    // carries trailing array dims — legal to subscript, unlike a plain scalar.
    VarDecl *arrayVar = nullptr;
    if (ie->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      auto *bref = static_cast<const DeclRefExpr *>(ie->base.get());
      ASTNode *sym = lookup(bref->name);
      // Global (__constant__ / file-scope) arrays aren't in the local scope;
      // fall back to the translation-unit symbol table.
      if (!sym) {
        auto gv = globalVars.find(bref->name);
        if (gv != globalVars.end()) sym = gv->second;
      }
      if (sym && sym->getNodeType() == ASTNode::NodeKind::VarDecl) {
        auto *vd = static_cast<VarDecl *>(sym);
        if (!vd->arrayDims.empty()) arrayVar = vd;
      }
    }

    if (rb && rb->getKind() != TypeKind::Pointer &&
        rb->getKind() != TypeKind::Vector && !arrayVar)
      warn(ie->base.get(),
           "subscripted value is not an array, pointer, or vector");
    if (idx && !isIntegerType(idx))
      warn(ie->index.get(),
           "array index is not an integer (got " + typeName(idx) + ")");

    // Constant index out of bounds: vector element counts and array sizes are
    // both statically known, so an out-of-range literal is determinable.
    int64_t iv = 0;
    if (constIntValue(ie->index.get(), iv)) {
      if (rb && rb->getKind() == TypeKind::Vector &&
          (iv < 0 ||
           (unsigned long long)iv >=
               static_cast<const VectorType *>(rb)->count))
        warn(ie->index.get(), "array index " + std::to_string(iv) +
                                  " out of bounds (vector size " +
                                  std::to_string(
                                      static_cast<const VectorType *>(rb)->count) +
                                  ")");
      else if (arrayVar && !arrayVar->arrayDims.empty() &&
               arrayVar->arrayDims.back() > 0 &&
               (iv < 0 || iv >= arrayVar->arrayDims.back()))
        warn(ie->index.get(),
             "array index " + std::to_string(iv) +
                 " out of bounds (declared size " +
                 std::to_string(arrayVar->arrayDims.back()) + ")");
    }

    // Indexing yields the pointee / element type.
    if (rb) {
      if (rb->getKind() == TypeKind::Pointer)
        return static_cast<const PointerType *>(rb)->pointee;
      if (rb->getKind() == TypeKind::Vector)
        return static_cast<const VectorType *>(rb)->elem;
    }
    return base;
  }
  case ASTNode::NodeKind::MemberAccessExpr: {
    auto *m = static_cast<const MemberAccessExpr *>(n);
    Type *baseTy = checkExpr(m->base.get());
    if (!baseTy) return nullptr;
    if (baseTy->getKind() == TypeKind::Builtin) {
      warn(n, "request for member '" + std::string(m->member) +
                  "' in non-class type " + typeName(baseTy));
      return nullptr;
    }
    if (baseTy->getKind() == TypeKind::Vector) {
      if (!isValidSwizzle(m->member))
        error(n, "invalid swizzle '." + std::string(m->member) + "'");
      // Result is a vector of the swizzle length (1 -> scalar).
      return nullptr;
    }
    if (baseTy->getKind() == TypeKind::Record) {
      auto *sd = static_cast<RecordType *>(baseTy)->decl;
      for (FieldDecl *fd : sd->fields)
        if (fd->name == m->member) return fd->type;
      // A class method accessed as `obj.method` (typically a call target). We
      // don't model a method type; just accept the member so the device backend
      // can lower the call to `Class_method(obj, ...)`.
      for (FunctionDecl *meth : sd->methods)
        if (meth->name == m->member) return nullptr;
      error(n, "no member named '" + std::string(m->member) + "' in struct '" +
                   std::string(sd->name) + "'");
    }
    if (baseTy->getKind() == TypeKind::Typedef) {
      // Resolve typedef and re-check the member against the underlying type.
      auto *td = static_cast<TypedefType *>(baseTy)->decl;
      // Tail-recurse by swapping in the underlying type.
      Type *under = td->underlying;
      if (under) {
        if (under->getKind() == TypeKind::Record) {
          auto *sd = static_cast<RecordType *>(under)->decl;
          for (FieldDecl *fd : sd->fields)
            if (fd->name == m->member) return fd->type;
          for (FunctionDecl *meth : sd->methods)
            if (meth->name == m->member) return nullptr;
          error(n, "no member named '" + std::string(m->member) +
                       "' in struct '" + std::string(sd->name) + "'");
        }
      }
    }
    return nullptr;
  }
  case ASTNode::NodeKind::LaunchExpr: {
    auto *l = static_cast<const LaunchExpr *>(n);
    StringRef name;
    if (l->callee &&
        l->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      name = static_cast<const DeclRefExpr *>(l->callee.get())->name;
    if (!name.empty()) {
      auto it = functions.find(name);
      if (it != functions.end()) {
        const auto &overloadSet = it->second;
        // Kernels cannot be overloaded (CUDA disallows __global__ overloads; the
        // launch syntax `k<<<...>>>(args)` cannot disambiguate). Pick the unique
        // __global__ candidate; a __device__ of the same base name is not a
        // launch target.
        FunctionDecl *f = nullptr;
        for (FunctionDecl *cand : overloadSet)
          if (cand->deviceAttr == DeviceAttr::Global) { f = cand; break; }
        if (!f) {
          error(n, "launch target '" + std::string(name) +
                       "' is not a __global__ kernel");
          for (auto &a : l->args) checkExpr(a.get());
        } else if (overloadSet.size() > 1) {
          // A kernel sharing a base name with a device helper is ambiguous from
          // a launch site (no param info to disambiguate). CUDA forbids this.
          error(n, "launch target '" + std::string(name) +
                       "' is ambiguous (shares name with device functions)");
        } else {
          checkCallArgs(n, name, f, l->args);
        }
      } else {
        warn(n, "launch of undeclared kernel '" + std::string(name) + "'");
        for (auto &a : l->args) checkExpr(a.get());
      }
    }
    checkExpr(l->gridDim.get());
    checkExpr(l->blockDim.get());
    checkExpr(l->gridDimY.get());
    checkExpr(l->blockDimY.get());
    checkExpr(l->stream.get());
    return nullptr;
  }
  default:
    return nullptr;
  }
}
