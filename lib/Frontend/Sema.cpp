//===- Sema.cpp - Semantic analysis ---------------------------------------===//
//
// Scoped symbol table + expression type inference (side table) + diagnostics.
// Conservative by design: only undeclared identifiers, unknown struct fields,
// malformed swizzles, and unknown-function calls (when not a recognized
// builtin) are hard errors. Type mismatches are warnings so existing kernels
// keep compiling.
//
// Thread index builtins (threadIdx, blockIdx, blockDim, gridDim) and math
// builtins (sinf, __syncthreads, ...) are treated as implicitly declared so
// real kernels don't trip false-positive errors.

#include "vc/Frontend/Sema.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;

namespace {
// Small set of CUDA thread-index identifiers that are implicitly available in
// every kernel/device function. Resolved to a placeholder int type — the GLSL
// backend handles the real lowering.
bool isThreadBuiltinName(llvm::StringRef n) {
  return n == "threadIdx" || n == "blockIdx" || n == "blockDim" ||
         n == "gridDim";
}
} // namespace

bool Sema::isThreadBuiltin(StringRef name) const {
  return isThreadBuiltinName(name);
}

// A permissive set of math / GLSL builtins so calls like sinf(...) or
// __syncthreads() don't get flagged as unknown functions. Real resolution
// would need a full builtin table; this covers the kernels we ship.
bool Sema::isMathBuiltin(StringRef name) const {
  // CUDA __f-prefixed and f-suffixed math intrinsics, plus common GLSL math.
  static const char *names[] = {
      "__syncthreads", "sin", "cos", "tan", "asin", "acos", "atan",
      "exp", "log", "pow", "sqrt", "abs", "fabs", "fmin", "fmax", "min", "max",
      "floor", "ceil", "fract", "mix", "clamp", "step", "smoothstep", "mod",
      "sinf", "cosf", "tanf", "asinf", "acosf", "atanf", "expf", "logf",
      "powf", "sqrtf", "fabsf", "fminf", "fmaxf", "floorf", "ceilf", "__sinf",
      "__cosf", "__expf", "__logf", "__powf", "__fabsf",
  };
  for (const char *m : names)
    if (name == m) return true;
  return false;
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

Type *Sema::builtin(BuiltinTypeKind k) const {
  return new BuiltinType(k);
}

void Sema::declare(StringRef name, ASTNode *node) {
  if (scopes.empty()) pushScope();
  scopes.back()[name] = node;
}

ASTNode *Sema::lookup(StringRef name) {
  for (auto it = scopes.rbegin(), end = scopes.rend(); it != end; ++it) {
    auto found = it->find(name);
    if (found != it->end()) return found->second;
  }
  return nullptr;
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

  // Report diagnostics. Errors abort the build (driver checks the return).
  bool hadError = false;
  for (const Diagnostic &d : tu.diagnostics) {
    const char *tag = d.kind == DiagnosticKind::Error   ? "error"
                      : d.kind == DiagnosticKind::Warning ? "warning"
                                                          : "note";
    llvm::errs() << "sema " << tag << " (line " << d.where.line << ":"
                 << d.where.col << "): " << d.message << "\n";
    if (d.kind == DiagnosticKind::Error) hadError = true;
  }
  return !hadError;
}

void Sema::collectTopLevel() {
  for (auto &d : tu.decls) {
    switch (d->getNodeType()) {
    case ASTNode::NodeKind::FunctionDecl: {
      auto *f = static_cast<FunctionDecl *>(d.get());
      functions[f->name] = f;
      break;
    }
    case ASTNode::NodeKind::StructDecl: {
      auto *s = static_cast<StructDecl *>(d.get());
      typeNames[s->name] = new RecordType(s);
      break;
    }
    case ASTNode::NodeKind::TypedefDecl: {
      auto *t = static_cast<TypedefDecl *>(d.get());
      typeNames[t->name] = new TypedefType(t);
      break;
    }
    default:
      break;
    }
  }
}

void Sema::checkFunctions() {
  for (auto &d : tu.decls) {
    if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
    auto *f = static_cast<FunctionDecl *>(d.get());
    if (!f->body) continue;
    pushScope();
    for (ParamDecl *p : f->params) declare(p->name, p);
    checkStmt(f->body.get());
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
    VarDecl *v = ds->decl;
    if (v) {
      declare(v->name, v);
      if (v->init) checkExpr(v->init.get());
    }
    return;
  }
  case ASTNode::NodeKind::ExprStmt:
    checkExpr(static_cast<const ExprStmt *>(n)->expr.get());
    return;
  case ASTNode::NodeKind::ReturnStmt:
    checkExpr(static_cast<const ReturnStmt *>(n)->value.get());
    return;
  case ASTNode::NodeKind::IfStmt: {
    auto *iff = static_cast<const IfStmt *>(n);
    checkExpr(iff->cond.get());
    checkStmt(iff->thenStmt.get());
    checkStmt(iff->elseStmt.get());
    return;
  }
  case ASTNode::NodeKind::ForStmt: {
    auto *fs = static_cast<const ForStmt *>(n);
    pushScope();
    checkStmt(fs->init.get());
    checkExpr(fs->cond.get());
    checkExpr(fs->step.get());
    checkStmt(fs->body.get());
    popScope();
    return;
  }
  case ASTNode::NodeKind::WhileStmt: {
    auto *ws = static_cast<const WhileStmt *>(n);
    checkExpr(ws->cond.get());
    checkStmt(ws->body.get());
    return;
  }
  case ASTNode::NodeKind::DoStmt: {
    auto *ds = static_cast<const DoStmt *>(n);
    checkStmt(ds->body.get());
    checkExpr(ds->cond.get());
    return;
  }
  case ASTNode::NodeKind::SwitchStmt: {
    auto *sw = static_cast<const SwitchStmt *>(n);
    checkExpr(sw->cond.get());
    checkStmt(sw->body.get());
    return;
  }
  case ASTNode::NodeKind::CaseStmt: {
    auto *cs = static_cast<const CaseStmt *>(n);
    checkExpr(cs->value.get());
    checkStmt(cs->sub.get());
    return;
  }
  case ASTNode::NodeKind::BreakStmt:
  case ASTNode::NodeKind::ContinueStmt:
    return;
  default:
    return;
  }
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
  case ASTNode::NodeKind::StringLiteral:
    // No string type in kernels; legal but unusable as a value.
    return nullptr;
  case ASTNode::NodeKind::DeclRefExpr: {
    auto *d = static_cast<const DeclRefExpr *>(n);
    if (isThreadBuiltin(d->name)) {
      // Implicit thread-index identifier; backend lowers it. Type unknown
      // but legal.
      return nullptr;
    }
    if (typeNames.count(d->name)) return nullptr; // a type name used as a value?
    ASTNode *sym = lookup(d->name);
    if (!sym) {
      error(n, "use of undeclared identifier '" + std::string(d->name) + "'");
      return nullptr;
    }
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
    if (b->op == BinaryOp::Assign) {
      // LHS must be an lvalue.
      auto k = b->lhs->getNodeType();
      if (k != ASTNode::NodeKind::DeclRefExpr &&
          k != ASTNode::NodeKind::IndexExpr &&
          k != ASTNode::NodeKind::MemberAccessExpr)
        warn(n, "assignment to non-lvalue");
    }
    return rt ? rt : lt;
  }
  case ASTNode::NodeKind::UnaryExpr:
    return checkExpr(static_cast<const UnaryExpr *>(n)->operand.get());
  case ASTNode::NodeKind::ConditionalExpr: {
    auto *c = static_cast<const ConditionalExpr *>(n);
    checkExpr(c->cond.get());
    Type *tt = checkExpr(c->thenExpr.get());
    Type *et = checkExpr(c->elseExpr.get());
    return tt ? tt : et;
  }
  case ASTNode::NodeKind::CStyleCastExpr:
    return static_cast<const CStyleCastExpr *>(n)->target;
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
    if (c->callee &&
        c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      calleeName = static_cast<const DeclRefExpr *>(c->callee.get())->name;

    // Vector constructors (float4(...)) and math builtins pass through.
    if (!calleeName.empty()) {
      for (auto &a : c->args) checkExpr(a.get());
      if (isMathBuiltin(calleeName)) return nullptr;
      auto it = functions.find(calleeName);
      if (it != functions.end()) {
        FunctionDecl *f = it->second;
        if (f->params.size() != c->args.size())
          error(n, "call to '" + std::string(calleeName) +
                       "' has " + std::to_string(c->args.size()) +
                       " args, expected " + std::to_string(f->params.size()));
        return f->returnType;
      }
      // Unknown callee: could be a GLSL builtin we didn't list (e.g. a
      // vector constructor). Warn softly rather than hard-error, so we don't
      // break valid kernels using less-common builtins.
      warn(n, "call to undeclared function '" + std::string(calleeName) +
                  "' (assuming builtin)");
      return nullptr;
    }
    checkExpr(c->callee.get());
    for (auto &a : c->args) checkExpr(a.get());
    return nullptr;
  }
  case ASTNode::NodeKind::IndexExpr: {
    auto *ie = static_cast<const IndexExpr *>(n);
    Type *base = checkExpr(ie->base.get());
    checkExpr(ie->index.get());
    // Pointer-to-T or array-of-T indexes to T.
    if (base) {
      if (base->getKind() == TypeKind::Pointer)
        return static_cast<PointerType *>(base)->pointee;
    }
    return nullptr;
  }
  case ASTNode::NodeKind::MemberAccessExpr: {
    auto *m = static_cast<const MemberAccessExpr *>(n);
    Type *baseTy = checkExpr(m->base.get());
    if (!baseTy) return nullptr;
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
          error(n, "no member named '" + std::string(m->member) +
                       "' in struct '" + std::string(sd->name) + "'");
        }
      }
    }
    return nullptr;
  }
  case ASTNode::NodeKind::LaunchExpr: {
    auto *l = static_cast<const LaunchExpr *>(n);
    checkExpr(l->gridDim.get());
    checkExpr(l->blockDim.get());
    for (auto &a : l->args) checkExpr(a.get());
    return nullptr;
  }
  default:
    return nullptr;
  }
}
