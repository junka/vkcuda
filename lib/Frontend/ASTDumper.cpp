//===- ASTDumper.cpp - Pretty-print the AST -------------------------------===//

#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/AST.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;
using namespace llvm;

namespace {

class ASTDumper {
  raw_ostream &os;
  unsigned indent = 0;

  void pad() { for (unsigned i = 0; i < indent; ++i) os << "  "; }

public:
  ASTDumper(raw_ostream &o) : os(o) {}

  void dump(const TranslationUnit &tu) {
    os << "TranslationUnit\n";
    for (auto &d : tu.decls)
      dumpNode(d.get());
  }

  void dumpNode(const ASTNode *n) {
    if (!n) { pad(); os << "(null)\n"; return; }
    pad();
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::TranslationUnit: os << "TU\n"; break;
    case ASTNode::NodeKind::FunctionDecl: {
      const auto *f = cast<FunctionDecl>(n);
      os << "Function " << f->name << " (";
      const char *attr = "host";
      if (f->deviceAttr == DeviceAttr::Global) attr = "global";
      else if (f->deviceAttr == DeviceAttr::Device) attr = "device";
      os << attr << ")\n";
      ++indent;
      if (f->body) dumpNode(f->body.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::KernelDecl: {
      const auto *k = cast<KernelDecl>(n);
      os << "Kernel -> " << (k->func ? k->func->name : "?") << "\n";
      break;
    }
    case ASTNode::NodeKind::CompoundStmt: {
      os << "{\n";
      ++indent;
      for (auto &s : cast<CompoundStmt>(n)->statements) dumpNode(s.get());
      --indent;
      pad(); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::ReturnStmt: {
      os << "return\n";
      ++indent;
      if (auto *r = cast<ReturnStmt>(n)->value.get()) dumpNode(r);
      --indent;
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      const auto *v = cast<DeclStmt>(n)->decl;
      os << "Var " << (v ? v->name : "?")
         << (v && v->isShared ? " [shared]" : "") << "\n";
      ++indent;
      if (v && v->init) dumpNode(v->init.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::ExprStmt: {
      os << "ExprStmt\n";
      ++indent;
      if (auto *e = cast<ExprStmt>(n)->expr.get()) dumpNode(e);
      --indent;
      break;
    }
    case ASTNode::NodeKind::IfStmt: {
      const auto *iff = cast<IfStmt>(n);
      os << "If\n";
      ++indent;
      dumpNode(iff->cond.get());
      if (iff->thenStmt) dumpNode(iff->thenStmt.get());
      if (iff->elseStmt) dumpNode(iff->elseStmt.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      const auto *fs = cast<ForStmt>(n);
      os << "For\n";
      ++indent;
      if (fs->init) dumpNode(fs->init.get());
      if (fs->cond) dumpNode(fs->cond.get());
      if (fs->step) dumpNode(fs->step.get());
      if (fs->body) dumpNode(fs->body.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      const auto *ws = cast<WhileStmt>(n);
      os << "While\n";
      ++indent;
      dumpNode(ws->cond.get());
      if (ws->body) dumpNode(ws->body.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      const auto *b = cast<BinaryExpr>(n);
      const char *op = "?";
      switch (b->op) {
      case BinaryOp::Add: op = "+"; break;
      case BinaryOp::Sub: op = "-"; break;
      case BinaryOp::Mul: op = "*"; break;
      case BinaryOp::Div: op = "/"; break;
      case BinaryOp::Assign: op = "="; break;
      case BinaryOp::Lt: op = "<"; break;
      case BinaryOp::Gt: op = ">"; break;
      case BinaryOp::Le: op = "<="; break;
      case BinaryOp::Ge: op = ">="; break;
      case BinaryOp::Eq: op = "=="; break;
      case BinaryOp::NEq: op = "!="; break;
      case BinaryOp::And: op = "&"; break;
      case BinaryOp::Or: op = "|"; break;
      case BinaryOp::LAnd: op = "&&"; break;
      case BinaryOp::LOr: op = "||"; break;
      default: break;
      }
      os << "BinOp " << op << "\n";
      ++indent;
      dumpNode(b->lhs.get());
      dumpNode(b->rhs.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::IndexExpr: {
      const auto *i = cast<IndexExpr>(n);
      os << "Index\n";
      ++indent;
      dumpNode(i->base.get());
      dumpNode(i->index.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      const auto *m = cast<MemberAccessExpr>(n);
      os << "Member ." << m->member << "\n";
      ++indent;
      dumpNode(m->base.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::CallExpr: {
      const auto *c = cast<CallExpr>(n);
      os << "Call\n";
      ++indent;
      dumpNode(c->callee.get());
      for (auto &a : c->args) dumpNode(a.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::LaunchExpr: {
      const auto *l = cast<LaunchExpr>(n);
      os << "Launch <<<...>>>\n";
      ++indent;
      dumpNode(l->callee.get());
      pad(); os << "grid:\n"; ++indent; dumpNode(l->gridDim.get()); --indent;
      pad(); os << "block:\n"; ++indent; dumpNode(l->blockDim.get()); --indent;
      for (auto &a : l->args) dumpNode(a.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::DeclRefExpr:
      os << "Ref " << cast<DeclRefExpr>(n)->name << "\n";
      break;
    case ASTNode::NodeKind::IntegerLiteral:
      os << "Int " << cast<IntegerLiteral>(n)->value << "\n";
      break;
    case ASTNode::NodeKind::FloatLiteral:
      os << "Float " << cast<FloatLiteral>(n)->value << "\n";
      break;
    case ASTNode::NodeKind::UnaryExpr: {
      const auto *u = cast<UnaryExpr>(n);
      os << "Unary\n";
      ++indent; dumpNode(u->operand.get()); --indent;
      break;
    }
    case ASTNode::NodeKind::ParamDecl:
    case ASTNode::NodeKind::VarDecl:
      break;
    }
  }

  // Small helper: reinterpret a node by kind (no RTTI).
  template <typename T> const T *cast(const ASTNode *n) {
    return static_cast<const T *>(n);
  }
};

} // namespace

namespace vc {

void dumpAST(const TranslationUnit &tu, raw_ostream &os) {
  ASTDumper(os).dump(tu);
}

} // namespace vc
