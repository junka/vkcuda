//===- AstTransforms.cpp - AST lowering passes (recursion elimination) ----===//
//
// unrollDeviceRecursion: SPIR-V forbids recursive function calls (GLSL
// explicitly rejects them; spirv.FunctionCall cannot form a cycle, as GPU
// shaders have no call stack). CUDA device code nonetheless uses bounded
// self-recursion (factorial, sum-to-n, power-by-repeated-multiply). This pass
// rewrites a recognized bounded *linear* self-recursion into an iterative
// accumulator loop before the backends see it, so neither backend has to
// model recursion. Shapes the pass cannot rewrite are left in place and
// reported as errors (so we never silently emit illegal SPIR-V).
//
// Recognized shape (single parameter, one recursive call, optional combine):
//
//   T f(P p) {
//     if (C) return B;                       // base case (must dominate)
//     return combine(X, f(A));               // OR  return f(A);   (tail form)
//   }
//
// where `combine` is `X op f(A)` or `f(A) op X` for op in {+,-,*,/,%,&,|,^,
// <<,>>} (and the bare `return f(A);` tail form). C/A/X are expressions over
// p (and other params); A must be a compile-time-recognizable strict decrease
// of p (p - k, p / k, p >> k, p & (p-1)) so the loop terminates.
//
// Rewrite (accumulator loop, MAXDEPTH guards runaway):
//
//   T f(P p) {
//     P __vc_p = p;
//     T __vc_acc = <identity for op>;        // tail form: T __vc_acc; (uninit)
//     for (int __vc_d = 0; __vc_d < MAXDEPTH; ++__vc_d) {
//       if (C) break;
//       __vc_acc = <combine X with __vc_acc>;  // tail form: skip
//       __vc_p = A;
//     }
//     return C ? B : __vc_acc;                 // tail form: return B (C holds)
//   }
//
// Branching recursion (fib: two self-calls), indirect recursion (A->B->A),
// and recursion through a non-base return path are rejected with an error.
//
//===----------------------------------------------------------------------===//

#include "vc/Frontend/AST.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;

namespace {

// Depth cap: bounds the synthesized loop so a base case that never fires
// (e.g. called with a negative argument) cannot spin forever. Matches C's
// "runaway recursion is undefined" — partial accumulator is returned.
constexpr int MAXDEPTH = 1024;

SourceLocation locOf(const ASTNode *n) { return n ? n->getLoc() : SourceLocation{}; }

// Intern a synthesized name into the function's durable storage so the
// StringRef returned here (and stored in DeclRefExpr/VarDecl) stays valid
// through codegen. DeclRefExpr names are StringRefs into stable storage
// (source buffer or string literals); synthesized names have no source, so
// they live on the FunctionDecl.
StringRef internName(FunctionDecl *f, const std::string &s) {
  f->synthNames.push_back(s);
  return StringRef(f->synthNames.back());
}

// Recursively rename every DeclRefExpr naming `from` to `to` within an
// expression tree (in place). Used to substitute the parameter `p` with the
// loop-carried variable `__vc_p` inside cloned copies of C / A / X.
void renameDeclRef(ASTNode *n, StringRef from, StringRef to) {
  if (!n) return;
  switch (n->getNodeType()) {
  using NK = ASTNode::NodeKind;
  case NK::DeclRefExpr:
    if (static_cast<DeclRefExpr *>(n)->name == from)
      static_cast<DeclRefExpr *>(n)->name = to;
    return;
  case NK::BinaryExpr: {
    auto *b = static_cast<BinaryExpr *>(n);
    renameDeclRef(b->lhs.get(), from, to);
    renameDeclRef(b->rhs.get(), from, to);
    return;
  }
  case NK::UnaryExpr:
    renameDeclRef(static_cast<UnaryExpr *>(n)->operand.get(), from, to);
    return;
  case NK::ConditionalExpr: {
    auto *c = static_cast<ConditionalExpr *>(n);
    renameDeclRef(c->cond.get(), from, to);
    renameDeclRef(c->thenExpr.get(), from, to);
    renameDeclRef(c->elseExpr.get(), from, to);
    return;
  }
  case NK::CommaExpr: {
    auto *c = static_cast<CommaExpr *>(n);
    renameDeclRef(c->lhs.get(), from, to);
    renameDeclRef(c->rhs.get(), from, to);
    return;
  }
  case NK::CStyleCastExpr:
    renameDeclRef(static_cast<CStyleCastExpr *>(n)->sub.get(), from, to);
    return;
  case NK::IndexExpr: {
    auto *i = static_cast<IndexExpr *>(n);
    renameDeclRef(i->base.get(), from, to);
    renameDeclRef(i->index.get(), from, to);
    return;
  }
  case NK::MemberAccessExpr:
    renameDeclRef(static_cast<MemberAccessExpr *>(n)->base.get(), from, to);
    return;
  case NK::CallExpr: {
    auto *c = static_cast<CallExpr *>(n);
    renameDeclRef(c->callee.get(), from, to);
    for (auto &a : c->args) renameDeclRef(a.get(), from, to);
    return;
  }
  case NK::InitListExpr:
    for (auto &e : static_cast<InitListExpr *>(n)->elements)
      renameDeclRef(e.get(), from, to);
    return;
  case NK::SizeOfExpr:
    renameDeclRef(static_cast<SizeOfExpr *>(n)->sub.get(), from, to);
    return;
  default:
    return;
  }
}

// Clone an expression and rename DeclRefExpr(from)->to in the clone.
NodePtr cloneRenamed(const ASTNode *n, StringRef from, StringRef to) {
  NodePtr c = cloneExpr(n);
  if (c) renameDeclRef(c.get(), from, to);
  return c;
}

// Does expression `n` contain a CallExpr whose resolvedCallee is `target`?
bool containsSelfCall(const ASTNode *n, const FunctionDecl *target) {
  if (!n) return false;
  switch (n->getNodeType()) {
  using NK = ASTNode::NodeKind;
  case NK::CallExpr: {
    auto *c = static_cast<const CallExpr *>(n);
    if (c->resolvedCallee == target) return true;
    for (auto &a : c->args)
      if (containsSelfCall(a.get(), target)) return true;
    return false; // callee of a non-self call is a DeclRefExpr, not a call
  }
  case NK::BinaryExpr: {
    auto *b = static_cast<const BinaryExpr *>(n);
    return containsSelfCall(b->lhs.get(), target) ||
           containsSelfCall(b->rhs.get(), target);
  }
  case NK::UnaryExpr:
    return containsSelfCall(static_cast<const UnaryExpr *>(n)->operand.get(), target);
  case NK::ConditionalExpr: {
    auto *c = static_cast<const ConditionalExpr *>(n);
    return containsSelfCall(c->cond.get(), target) ||
           containsSelfCall(c->thenExpr.get(), target) ||
           containsSelfCall(c->elseExpr.get(), target);
  }
  case NK::CommaExpr: {
    auto *c = static_cast<const CommaExpr *>(n);
    return containsSelfCall(c->lhs.get(), target) ||
           containsSelfCall(c->rhs.get(), target);
  }
  case NK::CStyleCastExpr:
    return containsSelfCall(static_cast<const CStyleCastExpr *>(n)->sub.get(), target);
  case NK::IndexExpr: {
    auto *i = static_cast<const IndexExpr *>(n);
    return containsSelfCall(i->base.get(), target) ||
           containsSelfCall(i->index.get(), target);
  }
  case NK::MemberAccessExpr:
    return containsSelfCall(static_cast<const MemberAccessExpr *>(n)->base.get(), target);
  case NK::InitListExpr:
    for (auto &e : static_cast<const InitListExpr *>(n)->elements)
      if (containsSelfCall(e.get(), target)) return true;
    return false;
  case NK::SizeOfExpr:
    return containsSelfCall(static_cast<const SizeOfExpr *>(n)->sub.get(), target);
  default:
    return false;
  }
}

// Count self-calls anywhere in `n`.
unsigned countSelfCalls(const ASTNode *n, const FunctionDecl *target) {
  if (!n) return 0;
  unsigned c = 0;
  switch (n->getNodeType()) {
  using NK = ASTNode::NodeKind;
  case NK::CallExpr: {
    auto *call = static_cast<const CallExpr *>(n);
    if (call->resolvedCallee == target) ++c;
    for (auto &a : call->args) c += countSelfCalls(a.get(), target);
    return c;
  }
  case NK::BinaryExpr:
    return countSelfCalls(static_cast<const BinaryExpr *>(n)->lhs.get(), target) +
           countSelfCalls(static_cast<const BinaryExpr *>(n)->rhs.get(), target);
  case NK::UnaryExpr:
    return countSelfCalls(static_cast<const UnaryExpr *>(n)->operand.get(), target);
  case NK::ConditionalExpr: {
    auto *ce = static_cast<const ConditionalExpr *>(n);
    return countSelfCalls(ce->cond.get(), target) +
           countSelfCalls(ce->thenExpr.get(), target) +
           countSelfCalls(ce->elseExpr.get(), target);
  }
  case NK::CommaExpr:
    return countSelfCalls(static_cast<const CommaExpr *>(n)->lhs.get(), target) +
           countSelfCalls(static_cast<const CommaExpr *>(n)->rhs.get(), target);
  case NK::CStyleCastExpr:
    return countSelfCalls(static_cast<const CStyleCastExpr *>(n)->sub.get(), target);
  case NK::IndexExpr:
    return countSelfCalls(static_cast<const IndexExpr *>(n)->base.get(), target) +
           countSelfCalls(static_cast<const IndexExpr *>(n)->index.get(), target);
  case NK::MemberAccessExpr:
    return countSelfCalls(static_cast<const MemberAccessExpr *>(n)->base.get(), target);
  case NK::InitListExpr:
    for (auto &e : static_cast<const InitListExpr *>(n)->elements)
      c += countSelfCalls(e.get(), target);
    return c;
  case NK::SizeOfExpr:
    return countSelfCalls(static_cast<const SizeOfExpr *>(n)->sub.get(), target);
  default:
    return 0;
  }
}

// Recognize a strict decrease of parameter `p` in expression `a`: returns the
// BinaryOp of the decrease (Sub/Div/Shr) or And (for p & (p-1)), else None.
// Conservative: only p - const, p / const, p >> const, p & (p-1).
bool isDecrease(const ASTNode *a, StringRef p) {
  if (!a) return false;
  using NK = ASTNode::NodeKind;
  if (a->getNodeType() == NK::BinaryExpr) {
    auto *b = static_cast<const BinaryExpr *>(a);
    // p - k  /  p / k  /  p >> k  : lhs is p, rhs is a positive const.
    if (b->lhs->getNodeType() == NK::DeclRefExpr &&
        static_cast<const DeclRefExpr *>(b->lhs.get())->name == p &&
        b->rhs->getNodeType() == NK::IntegerLiteral) {
      int64_t k = static_cast<const IntegerLiteral *>(b->rhs.get())->value;
      if ((b->op == BinaryOp::Sub || b->op == BinaryOp::Div ||
           b->op == BinaryOp::Shr) && k > 0)
        return true;
    }
    // p & (p-1)  : clears lowest set bit (popcount-style decrease).
    if (b->op == BinaryOp::And &&
        b->lhs->getNodeType() == NK::DeclRefExpr &&
        static_cast<const DeclRefExpr *>(b->lhs.get())->name == p &&
        b->rhs->getNodeType() == NK::BinaryExpr) {
      auto *r = static_cast<const BinaryExpr *>(b->rhs.get());
      if (r->op == BinaryOp::Sub &&
          r->lhs->getNodeType() == NK::DeclRefExpr &&
          static_cast<const DeclRefExpr *>(r->lhs.get())->name == p &&
          r->rhs->getNodeType() == NK::IntegerLiteral &&
          static_cast<const IntegerLiteral *>(r->rhs.get())->value == 1)
        return true;
    }
  }
  return false;
}

// Diagnostic helper: emit an error mentioning the function and return false.
bool failRecursion(FunctionDecl *f, const std::string &why) {
  SourceLocation l = locOf(f);
  llvm::errs() << "<vc>:" << l.line << ":" << l.col
               << ": error: cannot rewrite recursive function '"
               << f->name.str() << "': " << why
               << " (SPIR-V forbids recursion; use an iterative form)\n";
  return false;
}

// Build `name = rhs;` as a NodePtr ExprStmt wrapping a BinaryExpr(Assign).
NodePtr makeAssignStmt(SourceLocation l, StringRef name, NodePtr rhs) {
  auto lhs = NodePtr(new DeclRefExpr(l, name));
  auto *es = new ExprStmt(l);
  es->expr = NodePtr(new BinaryExpr(l, BinaryOp::Assign, std::move(lhs),
                                    std::move(rhs)));
  return NodePtr(es);
}

// Build `++name` (pre-increment) as a NodePtr ExprStmt.
NodePtr makeIncStmt(SourceLocation l, StringRef name) {
  auto operand = NodePtr(new DeclRefExpr(l, name));
  auto *es = new ExprStmt(l);
  es->expr = NodePtr(new UnaryExpr(l, UnaryOp::PreInc, std::move(operand)));
  return NodePtr(es);
}

// Try to rewrite one recursive function. Returns true on success (body
// replaced), false if not handled (caller may report an error if the function
// is recursive but unhandled).
bool tryRewrite(FunctionDecl *f) {
  if (!f->body) return false;
  auto *body = static_cast<CompoundStmt *>(f->body.get());
  auto &stmts = body->statements;
  if (stmts.size() < 2) return false;

  // First statement must be `if (C) return B;` (base case dominating).
  if (stmts[0]->getNodeType() != ASTNode::NodeKind::IfStmt) return false;
  auto *baseIf = static_cast<IfStmt *>(stmts[0].get());
  if (!baseIf->thenStmt || baseIf->elseStmt) return false;
  if (baseIf->thenStmt->getNodeType() != ASTNode::NodeKind::ReturnStmt) return false;
  auto *baseRet = static_cast<ReturnStmt *>(baseIf->thenStmt.get());
  if (!baseRet->value) return false; // `return;` is not a value base case
  NodePtr condClone = cloneExpr(baseIf->cond.get()); // C
  NodePtr baseClone = cloneExpr(baseRet->value.get()); // B

  // Last statement must be `return R;` carrying the recursion. Any statement
  // between base and the recursive return must NOT contain a self-call.
  if (stmts.back()->getNodeType() != ASTNode::NodeKind::ReturnStmt) return false;
  auto *recRet = static_cast<ReturnStmt *>(stmts.back().get());
  if (!recRet->value) return false;
  for (size_t i = 1; i + 1 < stmts.size(); ++i)
    if (containsSelfCall(stmts[i].get(), f))
      return false; // self-call outside the final return — not our shape

  const ASTNode *R = recRet->value.get();
  unsigned selfCalls = countSelfCalls(R, f);
  if (selfCalls == 0) return false; // not actually recursive in the return
  if (selfCalls > 1)
    return failRecursion(f, "branching recursion (more than one self-call in "
                            "the return expression)");

  // The recursive return is either:
  //   (a) `return f(A);`               — tail form, no combine
  //   (b) `return X op f(A);`          — combine on the left
  //   (c) `return f(A) op X;`          — combine on the right
  // where X has no self-call.
  BinaryOp combineOp = BinaryOp::Add; // placeholder
  const ASTNode *combineOther = nullptr; // X (no self-call), or null (tail form)
  const CallExpr *selfCall = nullptr;

  if (R->getNodeType() == ASTNode::NodeKind::CallExpr &&
      static_cast<const CallExpr *>(R)->resolvedCallee == f) {
    selfCall = static_cast<const CallExpr *>(R); // tail form
  } else if (R->getNodeType() == ASTNode::NodeKind::BinaryExpr) {
    auto *b = static_cast<const BinaryExpr *>(R);
    const CallExpr *lc = nullptr, *rc = nullptr;
    if (b->lhs->getNodeType() == ASTNode::NodeKind::CallExpr &&
        static_cast<const CallExpr *>(b->lhs.get())->resolvedCallee == f)
      lc = static_cast<const CallExpr *>(b->lhs.get());
    if (b->rhs->getNodeType() == ASTNode::NodeKind::CallExpr &&
        static_cast<const CallExpr *>(b->rhs.get())->resolvedCallee == f)
      rc = static_cast<const CallExpr *>(b->rhs.get());
    if (lc && !rc && !containsSelfCall(b->rhs.get(), f)) {
      selfCall = lc; combineOp = b->op; combineOther = b->rhs.get(); // f(A) op X
    } else if (rc && !lc && !containsSelfCall(b->lhs.get(), f)) {
      selfCall = rc; combineOp = b->op; combineOther = b->lhs.get(); // X op f(A)
    }
  }
  if (!selfCall)
    return failRecursion(f, "self-call is not the whole return nor a simple "
                            "`X op f(A)` / `f(A) op X` combine");

  // Only the simple arithmetic/logical combine ops have an identity; others
  // (shift, mod, comparisons) are rejected to keep the accumulator sound.
  bool tailForm = (combineOther == nullptr);
  if (!tailForm) {
    switch (combineOp) {
    case BinaryOp::Add: case BinaryOp::Sub: case BinaryOp::Mul:
    case BinaryOp::Div: case BinaryOp::And: case BinaryOp::Or:
    case BinaryOp::Xor: case BinaryOp::LAnd: case BinaryOp::LOr:
      break;
    default:
      return failRecursion(f, "combine operator is not supported for the "
                              "accumulator rewrite");
    }
  }

  // Multi-parameter recursion: the self-call must pass exactly f->params.size()
  // arguments. Exactly one argument must be a strict decrease of its parameter
  // (the recursion driver); the rest must pass the corresponding parameter
  // through unchanged (a DeclRefExpr naming that param) — those don't vary
  // across iterations, so they can keep referencing the original parameter.
  if (selfCall->args.size() != f->params.size())
    return failRecursion(f, "self-call argument count does not match the "
                            "function parameter count");
  int decIdx = -1;
  for (unsigned i = 0; i < f->params.size(); ++i) {
    const ASTNode *a = selfCall->args[i].get();
    StringRef pn = f->params[i]->name;
    if (isDecrease(a, pn)) {
      if (decIdx >= 0)
        return failRecursion(f, "more than one parameter decreases in the "
                                "recursive call (unsupported)");
      decIdx = (int)i;
    } else {
      // Pass-through: must be a bare reference to the same parameter.
      if (!(a && a->getNodeType() == ASTNode::NodeKind::DeclRefExpr &&
            static_cast<const DeclRefExpr *>(a)->name == pn))
        return failRecursion(f, "non-decreasing recursive argument must pass "
                                "its parameter through unchanged");
    }
  }
  if (decIdx < 0)
    return failRecursion(f, "no parameter strictly decreases in the recursive "
                            "call (would not terminate)");
  StringRef param = f->params[decIdx]->name;
  const ASTNode *argExpr = selfCall->args[decIdx].get();

  SourceLocation l = locOf(f);
  Type *retTy = f->returnType;
  Type *paramTy = f->params[decIdx]->type;
  // Intern the synthesized names into durable storage on the FunctionDecl:
  // DeclRefExpr/VarDecl hold StringRefs, which must point into stable memory
  // (source buffer or literals). These names have neither, so they live on f.
  StringRef pVar = internName(f, std::string("_vc_") + param.str() + "_p");
  StringRef accVar = internName(f, "_vc_acc");
  StringRef depthVar = internName(f, "_vc_d");

  // Build the new body.
  auto *newBody = new CompoundStmt(l);

  // P __vc_p = p;
  auto *pDecl = new VarDecl(l, paramTy, pVar);
  pDecl->init = NodePtr(new DeclRefExpr(l, param));
  newBody->statements.push_back(NodePtr(new DeclStmt(l, pDecl)));

  // T __vc_acc = B;   -- the base-case return value seeds the accumulator.
  // The recursion f(p) = combine(X, f(A)) with f(base)=B unrolls to
  // B op X(p) op X(A(p)) op ... , so the accumulator starts at B and each
  // loop iteration prepends the per-step contribution X. (For the tail form
  // `return f(A);` the result is just B once the base is reached; the
  // accumulator is unused but kept for a uniform shape.) B is cloned with
  // the parameter renamed to the loop-carried variable.
  auto *accDecl = new VarDecl(l, retTy, accVar);
  accDecl->init = cloneRenamed(baseRet->value.get(), param, pVar);
  newBody->statements.push_back(NodePtr(new DeclStmt(l, accDecl)));

  // for (int __vc_d = 0; __vc_d < MAXDEPTH; ++__vc_d) { if (C) break;
  //   __vc_acc = combine(X, __vc_acc);  __vc_p = A; }
  auto *forInit = new VarDecl(l,
      new BuiltinType(BuiltinTypeKind::Int32), depthVar);
  forInit->init = NodePtr(new IntegerLiteral(l, 0));
  auto *forCond = new BinaryExpr(l, BinaryOp::Lt,
      NodePtr(new DeclRefExpr(l, depthVar)),
      NodePtr(new IntegerLiteral(l, MAXDEPTH)));
  NodePtr forStep = NodePtr(new UnaryExpr(l, UnaryOp::PreInc,
      NodePtr(new DeclRefExpr(l, depthVar))));

  auto *loopBody = new CompoundStmt(l);
  // if (C) break;   -- C with p renamed to __vc_p
  auto *breakIf = new IfStmt(l);
  breakIf->cond = cloneRenamed(baseIf->cond.get(), param, pVar);
  breakIf->thenStmt = NodePtr(new BreakStmt(l));
  loopBody->statements.push_back(NodePtr(breakIf));

  if (!tailForm) {
    // __vc_acc = X op __vc_acc   (or  __vc_acc op X), with p->__vc_p in X.
    // The original return was `X op f(A)` (combineOther on lhs) or `f(A) op X`
    // (rhs); we replace the self-call f(A) with the running __vc_acc.
    NodePtr xRenamed = cloneRenamed(combineOther, param, pVar);
    NodePtr combine;
    if (combineOther == static_cast<const BinaryExpr *>(R)->rhs.get())
      // was f(A) op X  ->  __vc_acc op X
      combine = NodePtr(new BinaryExpr(l, combineOp,
          NodePtr(new DeclRefExpr(l, accVar)), std::move(xRenamed)));
    else
      // was X op f(A)  ->  X op __vc_acc
      combine = NodePtr(new BinaryExpr(l, combineOp, std::move(xRenamed),
          NodePtr(new DeclRefExpr(l, accVar))));
    loopBody->statements.push_back(
        makeAssignStmt(l, accVar, std::move(combine)));
  }
  // __vc_p = A;   (A with p renamed to __vc_p)
  loopBody->statements.push_back(
      makeAssignStmt(l, pVar, cloneRenamed(argExpr, param, pVar)));

  auto *forStmt = new ForStmt(l);
  forStmt->init = NodePtr(new DeclStmt(l, forInit));
  forStmt->cond = NodePtr(forCond);
  forStmt->step = std::move(forStep);
  forStmt->body = NodePtr(loopBody);
  newBody->statements.push_back(NodePtr(forStmt));

  // return __vc_acc;   -- the loop broke when C held, so __vc_acc is either B
  // (zero iterations: base case was true at entry) or the fully combined
  // result. No ternary needed.
  auto *finalRet = new ReturnStmt(l);
  finalRet->value = NodePtr(new DeclRefExpr(l, accVar));
  newBody->statements.push_back(NodePtr(finalRet));

  f->body = NodePtr(newBody);
  f->isRecursive = false; // rewritten; no longer recursive
  return true;
}

} // namespace

namespace vc {

// Rewrite bounded linear self-recursion in __device__ functions into
// iterative accumulator loops. Returns false if a recursive function could
// not be rewritten (a diagnostic is emitted in that case).
bool unrollDeviceRecursion(TranslationUnit &tu) {
  bool ok = true;
  // Walk all decls (top-level functions + methods reached via namespaces/
  // structs). collectDecls-style flattening is overkill; recursion only
  // happens in FunctionDecls, which appear at top level or in namespaces.
  std::vector<ASTNode *> work;
  for (auto &d : tu.decls) work.push_back(d.get());
  // Also descend into NamespaceDecls (methods/functions defined in a ns).
  // StructDecl::methods are non-owning pointers already in the decl list, so
  // the top-level walk covers them.
  std::vector<FunctionDecl *> recursiveFns;
  for (auto *n : work) {
    if (!n || n->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
    auto *f = static_cast<FunctionDecl *>(n);
    if (f->isRecursive) recursiveFns.push_back(f);
  }
  for (FunctionDecl *f : recursiveFns) {
    if (!tryRewrite(f)) ok = false;
  }
  return ok;
}

} // namespace vc
