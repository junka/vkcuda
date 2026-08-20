//===- ASTHelpers.cpp - AST utilities (expr cloning) ----------------------===//
//
// cloneExpr: deep-clone of the expression subset that can appear as an
// lvalue/rvalue when the parser expands compound assignment (`a += b` ->
// `a = a + b`) and `++`/`--` (`a++` -> `a = a + 1`). Keeping these lowers
// in the parser means no new AST node kinds are needed for them.
//
//===----------------------------------------------------------------------===//

#include "vc/Frontend/AST.h"

using namespace vc;

NodePtr vc::cloneExpr(const ASTNode *n) {
  if (!n) return nullptr;
  using NK = ASTNode::NodeKind;
  auto L = n->getLoc();
  switch (n->getNodeType()) {
  case NK::IntegerLiteral:
    return NodePtr(new IntegerLiteral(L, static_cast<const IntegerLiteral *>(n)->value));
  case NK::FloatLiteral:
    return NodePtr(new FloatLiteral(L, static_cast<const FloatLiteral *>(n)->value));
  case NK::DeclRefExpr:
    return NodePtr(new DeclRefExpr(L, static_cast<const DeclRefExpr *>(n)->name));
  case NK::BinaryExpr: {
    auto *b = static_cast<const BinaryExpr *>(n);
    return NodePtr(new BinaryExpr(L, b->op, cloneExpr(b->lhs.get()),
                                  cloneExpr(b->rhs.get())));
  }
  case NK::UnaryExpr: {
    auto *u = static_cast<const UnaryExpr *>(n);
    return NodePtr(new UnaryExpr(L, u->op, cloneExpr(u->operand.get())));
  }
  case NK::ConditionalExpr: {
    auto *c = static_cast<const ConditionalExpr *>(n);
    return NodePtr(new ConditionalExpr(L, cloneExpr(c->cond.get()),
                                       cloneExpr(c->thenExpr.get()),
                                       cloneExpr(c->elseExpr.get())));
  }
  case NK::IndexExpr: {
    auto *i = static_cast<const IndexExpr *>(n);
    return NodePtr(new IndexExpr(L, cloneExpr(i->base.get()),
                                 cloneExpr(i->index.get())));
  }
  case NK::MemberAccessExpr: {
    auto *m = static_cast<const MemberAccessExpr *>(n);
    return NodePtr(new MemberAccessExpr(L, cloneExpr(m->base.get()), m->member));
  }
  case NK::CallExpr: {
    auto *c = static_cast<const CallExpr *>(n);
    auto *out = new CallExpr(L, cloneExpr(c->callee.get()));
    for (auto &a : c->args) out->args.push_back(cloneExpr(a.get()));
    return NodePtr(out);
  }
  default:
    // Statements, decls, launch — not clonable here (not needed).
    return nullptr;
  }
}
