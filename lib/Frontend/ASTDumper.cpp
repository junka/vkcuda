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
    case ASTNode::NodeKind::StructDecl: {
      const auto *s = cast<StructDecl>(n);
      os << "Struct " << s->name << "\n";
      ++indent;
      for (const auto *f : s->fields) {
        pad(); os << "Field " << f->name << "\n";
      }
      --indent;
      break;
    }
    case ASTNode::NodeKind::FieldDecl:
      break; // handled inline by StructDecl
    case ASTNode::NodeKind::TypedefDecl: {
      const auto *t = cast<TypedefDecl>(n);
      os << "Typedef " << t->name << "\n";
      break;
    }
    case ASTNode::NodeKind::EnumDecl: {
      const auto *e = cast<EnumDecl>(n);
      os << "Enum " << e->name << "\n";
      ++indent;
      for (auto &c : e->constants) { pad(); os << c.name << " = " << c.value << "\n"; }
      --indent;
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
      const auto *ds = cast<DeclStmt>(n);
      os << "DeclStmt\n";
      ++indent;
      for (const VarDecl *v : ds->decls) {
        pad(); os << "Var " << (v ? v->name : std::string("?"))
                  << (v && v->isShared ? " [shared]" : "") << "\n";
        if (v && v->init) { ++indent; dumpNode(v->init.get()); --indent; }
      }
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
    case ASTNode::NodeKind::DoStmt: {
      const auto *ds = cast<DoStmt>(n);
      os << "DoWhile\n";
      ++indent;
      if (ds->body) dumpNode(ds->body.get());
      dumpNode(ds->cond.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::BreakStmt:
      os << "Break\n";
      break;
    case ASTNode::NodeKind::ContinueStmt:
      os << "Continue\n";
      break;
    case ASTNode::NodeKind::SwitchStmt: {
      const auto *sw = cast<SwitchStmt>(n);
      os << "Switch\n";
      ++indent;
      if (sw->cond) dumpNode(sw->cond.get());
      if (sw->body) dumpNode(sw->body.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      const auto *cs = cast<CaseStmt>(n);
      os << (cs->value ? "Case\n" : "Default\n");
      ++indent;
      if (cs->value) dumpNode(cs->value.get());
      if (cs->sub) dumpNode(cs->sub.get());
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
      case BinaryOp::Mod: op = "%"; break;
      case BinaryOp::Assign: op = "="; break;
      case BinaryOp::Lt: op = "<"; break;
      case BinaryOp::Gt: op = ">"; break;
      case BinaryOp::Le: op = "<="; break;
      case BinaryOp::Ge: op = ">="; break;
      case BinaryOp::Eq: op = "=="; break;
      case BinaryOp::NEq: op = "!="; break;
      case BinaryOp::Shl: op = "<<"; break;
      case BinaryOp::Shr: op = ">>"; break;
      case BinaryOp::And: op = "&"; break;
      case BinaryOp::Or: op = "|"; break;
      case BinaryOp::Xor: op = "^"; break;
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
      if (l->gridDimY) { pad(); os << "gridY:\n"; ++indent; dumpNode(l->gridDimY.get()); --indent; }
      pad(); os << "block:\n"; ++indent; dumpNode(l->blockDim.get()); --indent;
      if (l->blockDimY) { pad(); os << "blockY:\n"; ++indent; dumpNode(l->blockDimY.get()); --indent; }
      if (l->stream) { pad(); os << "stream:\n"; ++indent; dumpNode(l->stream.get()); --indent; }
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
    case ASTNode::NodeKind::BoolLiteral:
      os << "Bool " << (cast<BoolLiteral>(n)->value ? "true" : "false") << "\n";
      break;
    case ASTNode::NodeKind::SizeOfExpr: {
      const auto *s = cast<SizeOfExpr>(n);
      os << "SizeOf " << (s->isType ? "(type)" : "(expr)") << "\n";
      ++indent;
      dumpNode(s->sub.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::CharLiteral:
      os << "Char " << cast<CharLiteral>(n)->value << "\n";
      break;
    case ASTNode::NodeKind::StringLiteral: {
      // Note: `StringLiteral` is ambiguous here (llvm::StringLiteral vs
      // vc::StringLiteral due to `using namespace llvm`), so qualify it.
      const auto *sl = static_cast<const vc::StringLiteral *>(n);
      os << "String \"" << sl->value << "\"\n";
      break;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      const auto *u = cast<UnaryExpr>(n);
      const char *opn = "?";
      switch (u->op) {
      case UnaryOp::Neg: opn = "-"; break;
      case UnaryOp::Not: opn = "~"; break;
      case UnaryOp::LNot: opn = "!"; break;
      case UnaryOp::Deref: opn = "*"; break;
      case UnaryOp::AddrOf: opn = "&"; break;
      case UnaryOp::PreInc: opn = "++(pre)"; break;
      case UnaryOp::PostInc: opn = "++(post)"; break;
      case UnaryOp::PreDec: opn = "--(pre)"; break;
      case UnaryOp::PostDec: opn = "--(post)"; break;
      }
      os << "Unary " << opn << "\n";
      ++indent; dumpNode(u->operand.get()); --indent;
      break;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      const auto *c = cast<ConditionalExpr>(n);
      os << "Conditional\n";
      ++indent;
      dumpNode(c->cond.get());
      dumpNode(c->thenExpr.get());
      dumpNode(c->elseExpr.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::CommaExpr: {
      const auto *c = cast<CommaExpr>(n);
      os << "Comma\n";
      ++indent;
      dumpNode(c->lhs.get());
      dumpNode(c->rhs.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::CStyleCastExpr: {
      const auto *c = cast<CStyleCastExpr>(n);
      os << "Cast\n";
      ++indent;
      dumpNode(c->sub.get());
      --indent;
      break;
    }
    case ASTNode::NodeKind::InitListExpr: {
      const auto *il = cast<InitListExpr>(n);
      os << "InitList\n";
      ++indent;
      for (auto &e : il->elements) dumpNode(e.get());
      --indent;
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
