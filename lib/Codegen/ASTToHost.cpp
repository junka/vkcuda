//===- ASTToHost.cpp - Lower host subset of AST to C++ source -------------===//
//
// Companion to ASTToGLSL: where the GLSL backend lowers the device subset
// (`__global__`/`__device__`) to a compute shader, this backend lowers the
// host subset (`DeviceAttr::Host`/`None`, including `int main()`) to C++.
// CUDA-style launches `kernel<<<grid,block>>>(args)` are translated into
// vcLoadKernel + VCKernelArg[] + vcLaunchKernel sequences. The device SPIR-V
// is embedded as a static uint32_t array so the output is a single .cpp.
//
// Design: host statements/expressions are emitted as near-verbatim C++
// (declarations, control flow, printf, arithmetic pass straight through).
// Only LaunchExpr and the kernel-handle prologue need real translation.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/ASTToHost.h"

#include "vc/Frontend/AST.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace vc;
using namespace llvm;

namespace {

class HostEmitter {
  raw_ostream &os;
  const uint32_t *spirvWords = nullptr;
  size_t wordCount = 0;
  // Name -> Type* for every variable declared in the host functions being
  // emitted. Used to classify launch args as Pointer vs Scalar: a DeclRefExpr
  // arg whose declaration has a PointerType is a device pointer (vcMalloc
  // handle) and becomes VCKernelArg::Pointer; otherwise it's a by-value
  // scalar becoming VCKernelArg::Scalar.
  StringMap<const Type *> hostVarTypes;
  // Kernel names referenced by any LaunchExpr in main(). Drives the
  // handle-declaration + vcLoadKernel prologue emitted at the top of main.
  StringSet<> launchedKernels;

public:
  HostEmitter(raw_ostream &o, const uint32_t *w, size_t n)
      : os(o), spirvWords(w), wordCount(n) {}

  bool emit(const TranslationUnit &tu) {
    const FunctionDecl *mainFn = nullptr;
    // First pass: collect host var types across all host functions so launch
    // arg classification works even if a var is declared in an outer scope.
    for (auto &d : tu.decls) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global ||
          f->deviceAttr == DeviceAttr::Device)
        continue;
      // Params are host vars too.
      for (ParamDecl *p : f->params)
        if (p->type) hostVarTypes[p->name] = p->type;
      if (f->body) collectHostVars(f->body.get());
    }

    // Locate the host main().
    for (auto &d : tu.decls) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global ||
          f->deviceAttr == DeviceAttr::Device)
        continue;
      if (f->name == "main") { mainFn = f; break; }
    }
    if (!mainFn) return false;

    // Find every kernel launched from main so we can preload handles.
    if (mainFn->body) scanLaunches(mainFn->body.get());

    emitPreamble(tu);
    emitSpirvEmbed();

    // Emit top-level struct/typedef definitions so host code can name those
    // types (e.g. `Point hPts[64]`). These are shared with the device subset
    // but the host output needs its own copy visible before main().
    emitHostTypeDecls(tu);

    // Emit host functions (Host/None). main is one of them; its body gets the
    // kernel-handle prologue prepended.
    for (auto &d : tu.decls) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global ||
          f->deviceAttr == DeviceAttr::Device)
        continue;
      emitHostFunction(f, /*isMain=*/f == mainFn);
    }
    return true;
  }

private:
  // --------------------------------------------------------------------- //
  // Preamble + SPIR-V embedding
  // --------------------------------------------------------------------- //

  void emitPreamble(const TranslationUnit &tu) {
    // User preprocessor lines first (their #includes win for ordering), then
    // the runtime header so VCKernelArg / vc* are always available.
    for (const std::string &line : tu.hostPpLines)
      os << line << "\n";
    os << "#include \"vc/Runtime/VCRuntime.h\"\n";
    os << "#include <cstdio>\n";
    // The runtime symbols (vcInit, VCKernelArg, ...) live in namespace vc.
    // Bring them into scope so the generated launch calls resolve.
    os << "using namespace vc;\n";
    // `dim3` is parsed as a plain identifier (CallExpr); provide a host-side
    // definition so a `dim3(N,N)` expression type-checks even though the launch
    // translator normally splits it into X/Y before emitting it.
    os << "struct dim3 { unsigned x=1,y=1,z=1; "
          "dim3(unsigned X=1,unsigned Y=1,unsigned Z=1):x(X),y(Y),z(Z){} };\n\n";
  }

  void emitSpirvEmbed() {
    os << "// Embedded device SPIR-V (kernel subset), little-endian words.\n";
    os << "static const uint32_t __vc_spirv[] = {";
    for (size_t i = 0; i < wordCount; ++i) {
      if ((i % 8) == 0) os << "\n  ";
      else os << " ";
      char buf[16];
      std::snprintf(buf, sizeof(buf), "0x%08x", spirvWords[i]);
      os << buf;
      if (i + 1 < wordCount) os << ",";
    }
    if (wordCount) os << "\n";
    os << "};\n";
    os << "static const size_t __vc_spirv_len = " << wordCount << ";\n\n";
  }

  // Emit top-level struct/typedef declarations as plain C++. These mirror the
  // device-side definitions (the GLSL backend emits its own copy); the host
  // needs them visible so host code can declare variables of these types.
  void emitHostTypeDecls(const TranslationUnit &tu) {
    for (auto &d : tu.decls) {
      if (d->getNodeType() == ASTNode::NodeKind::StructDecl) {
        auto *s = static_cast<const StructDecl *>(d.get());
        os << "struct " << s->name << " {\n";
        for (FieldDecl *f : s->fields) {
          os << "  " << cppType(f->type) << " " << f->name;
          for (int64_t dim : f->arrayDims)
            os << "[" << dim << "]";
          os << ";\n";
        }
        os << "};\n";
      } else if (d->getNodeType() == ASTNode::NodeKind::TypedefDecl) {
        auto *td = static_cast<const TypedefDecl *>(d.get());
        os << "typedef " << cppType(td->underlying) << " " << td->name << ";\n";
      }
    }
    os << "\n";
  }

  // --------------------------------------------------------------------- //
  // Host var type collection + launch scanning
  // --------------------------------------------------------------------- //

  void collectHostVars(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        collectHostVars(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt: {
      for (VarDecl *d : static_cast<const DeclStmt *>(n)->decls)
        if (d && d->type) hostVarTypes[d->name] = d->type;
      return;
    }
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      collectHostVars(iff->thenStmt.get());
      collectHostVars(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      collectHostVars(fs->init.get());
      collectHostVars(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt:
      collectHostVars(static_cast<const WhileStmt *>(n)->body.get());
      return;
    case ASTNode::NodeKind::DoStmt:
      collectHostVars(static_cast<const DoStmt *>(n)->body.get());
      return;
    case ASTNode::NodeKind::SwitchStmt:
      collectHostVars(static_cast<const SwitchStmt *>(n)->body.get());
      return;
    default:
      return;
    }
  }

  void scanLaunches(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::LaunchExpr: {
      auto *l = static_cast<const LaunchExpr *>(n);
      if (l->callee &&
          l->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
        launchedKernels.insert(
            static_cast<const DeclRefExpr *>(l->callee.get())->name);
      scanLaunches(l->gridDim.get());
      scanLaunches(l->blockDim.get());
      scanLaunches(l->gridDimY.get());
      scanLaunches(l->blockDimY.get());
      scanLaunches(l->stream.get());
      for (auto &a : l->args) scanLaunches(a.get());
      return;
    }
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanLaunches(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt: {
      auto *d = static_cast<const DeclStmt *>(n);
      for (VarDecl *v : d->decls)
        if (v && v->init) scanLaunches(v->init.get());
      return;
    }
    case ASTNode::NodeKind::ExprStmt:
      scanLaunches(static_cast<const ExprStmt *>(n)->expr.get());
      return;
    case ASTNode::NodeKind::ReturnStmt:
      scanLaunches(static_cast<const ReturnStmt *>(n)->value.get());
      return;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      scanLaunches(iff->cond.get());
      scanLaunches(iff->thenStmt.get());
      scanLaunches(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanLaunches(fs->init.get());
      scanLaunches(fs->cond.get());
      scanLaunches(fs->step.get());
      scanLaunches(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      scanLaunches(ws->cond.get());
      scanLaunches(ws->body.get());
      return;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      scanLaunches(ds->body.get());
      scanLaunches(ds->cond.get());
      return;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      scanLaunches(sw->cond.get());
      scanLaunches(sw->body.get());
      return;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      scanLaunches(cs->value.get());
      scanLaunches(cs->sub.get());
      return;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      scanLaunches(b->lhs.get());
      scanLaunches(b->rhs.get());
      return;
    }
    case ASTNode::NodeKind::UnaryExpr:
      scanLaunches(static_cast<const UnaryExpr *>(n)->operand.get());
      return;
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      scanLaunches(c->cond.get());
      scanLaunches(c->thenExpr.get());
      scanLaunches(c->elseExpr.get());
      return;
    }
    case ASTNode::NodeKind::CStyleCastExpr:
      scanLaunches(static_cast<const CStyleCastExpr *>(n)->sub.get());
      return;
    case ASTNode::NodeKind::InitListExpr:
      for (auto &e : static_cast<const InitListExpr *>(n)->elements)
        scanLaunches(e.get());
      return;
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      scanLaunches(c->callee.get());
      for (auto &a : c->args) scanLaunches(a.get());
      return;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      scanLaunches(ie->base.get());
      scanLaunches(ie->index.get());
      return;
    }
    case ASTNode::NodeKind::MemberAccessExpr:
      scanLaunches(static_cast<const MemberAccessExpr *>(n)->base.get());
      return;
    default:
      return;
    }
  }

  // --------------------------------------------------------------------- //
  // Function emission
  // --------------------------------------------------------------------- //

  void emitHostFunction(const FunctionDecl *f, bool isMain) {
    os << cppType(f->returnType) << " " << f->name << "(";
    for (unsigned i = 0; i < f->params.size(); ++i) {
      if (i) os << ", ";
      os << cppType(f->params[i]->type) << " " << f->params[i]->name;
    }
    os << ") {\n";
    if (isMain) emitMainPrologue();
    if (f->body &&
        f->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(f->body.get());
      for (auto &s : cs->statements)
        emitHostStmt(s.get(), 1);
    }
    os << "}\n\n";
  }

  // Prepend kernel-handle declarations + loads to main. The user's own
  // vcInit()/vcShutdown() calls are left in their body (passed through).
  // The GLSL backend emits a single __global__ with SPIR-V entry "main", so
  // every launched kernel's handle loads that same embedded module.
  void emitMainPrologue() {
    if (launchedKernels.empty()) return;
    // Declare kernel handles only — the actual vcLoadKernel call is deferred
    // to a lazy guard at each launch site. vcInit() may be called anywhere in
    // the user's main(); loading eagerly here would run before vcInit() and
    // silently fail (loadKernel checks init_). The guard loads on first use.
    for (const auto &kv : launchedKernels)
      os << "  VCKernelHandle __vc_k_" << kv.getKey() << " = nullptr;\n";
  }

  void pad(unsigned n) { for (unsigned i = 0; i < n; ++i) os << "  "; }

  // --------------------------------------------------------------------- //
  // Statements
  // --------------------------------------------------------------------- //

  void emitForInit(const ASTNode *n) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::DeclStmt) {
      // for-init must be a single declaration. Emit the base type once, then
      // each declarator with its own pointer `*`s (so `int *a, b` stays
      // correct). Rare in practice but kept correct.
      auto *ds = static_cast<const DeclStmt *>(n);
      bool first = true;
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        if (first) {
          os << cppBaseType(d->type) << " ";
        } else {
          os << ", ";
        }
        first = false;
        os << std::string(pointerDepth(d->type), '*') << d->name;
        for (int64_t dim : d->arrayDims)
          os << "[" << dim << "]";
        if (d->init) { os << " = "; emitHostExpr(d->init.get()); }
      }
      return;
    }
    if (n->getNodeType() == ASTNode::NodeKind::ExprStmt) {
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitHostExpr(e);
    }
  }

  // Depth of pointer wrapping (0 for non-pointers), for for-init emission.
  static unsigned pointerDepth(const Type *t) {
    unsigned d = 0;
    while (t && t->getKind() == TypeKind::Pointer) {
      ++d;
      t = static_cast<const PointerType *>(t)->pointee;
    }
    return d;
  }

  // The base (non-pointer) type spelling, for shared-type for-init emission.
  std::string cppBaseType(const Type *t) {
    while (t && t->getKind() == TypeKind::Pointer)
      t = static_cast<const PointerType *>(t)->pointee;
    return cppType(t);
  }

  void emitHostStmt(const ASTNode *n, unsigned indent) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt: {
      pad(indent); os << "{\n";
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        emitHostStmt(s.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *ds = static_cast<const DeclStmt *>(n);
      if (ds->decls.empty()) break;
      // Emit each declarator on its own line. (C's comma form `T* a, b;`
      // would make `b` a non-pointer, which differs from the per-declarator
      // pointer the parser attached; separate declarations are always
      // correct and stay in the enclosing scope.)
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        pad(indent);
        os << cppType(d->type) << " " << d->name;
        for (int64_t dim : d->arrayDims)
          os << "[" << dim << "]";
        if (d->init) { os << " = "; emitHostExpr(d->init.get()); }
        os << ";\n";
      }
      break;
    }
    case ASTNode::NodeKind::ExprStmt: {
      auto *es = static_cast<const ExprStmt *>(n);
      // A kernel launch appears wrapped in an ExprStmt (the `;` belongs to the
      // statement). Translate it to the runtime call sequence instead of
      // trying to emit it as a plain expression.
      if (es->expr &&
          es->expr->getNodeType() == ASTNode::NodeKind::LaunchExpr) {
        emitLaunch(static_cast<const LaunchExpr *>(es->expr.get()), indent);
        break;
      }
      pad(indent);
      if (auto *e = es->expr.get())
        emitHostExpr(e);
      os << ";\n";
      break;
    }
    case ASTNode::NodeKind::ReturnStmt:
      pad(indent); os << "return";
      if (auto *r = static_cast<const ReturnStmt *>(n)->value.get()) {
        os << " "; emitHostExpr(r);
      }
      os << ";\n";
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      pad(indent); os << "if ("; emitHostExpr(iff->cond.get()); os << ") {\n";
      if (iff->thenStmt) emitHostStmt(iff->thenStmt.get(), indent + 1);
      pad(indent); os << "}\n";
      if (iff->elseStmt) {
        pad(indent); os << "else {\n";
        emitHostStmt(iff->elseStmt.get(), indent + 1);
        pad(indent); os << "}\n";
      }
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      pad(indent); os << "for (";
      if (fs->init) emitForInit(fs->init.get());
      os << "; ";
      if (fs->cond) emitHostExpr(fs->cond.get());
      os << "; ";
      if (fs->step) emitHostExpr(fs->step.get());
      os << ") {\n";
      if (fs->body) emitHostStmt(fs->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      pad(indent); os << "while ("; emitHostExpr(ws->cond.get()); os << ") {\n";
      if (ws->body) emitHostStmt(ws->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      pad(indent); os << "do {\n";
      if (ds->body) emitHostStmt(ds->body.get(), indent + 1);
      pad(indent); os << "} while ("; emitHostExpr(ds->cond.get()); os << ");\n";
      break;
    }
    case ASTNode::NodeKind::BreakStmt:
      pad(indent); os << "break;\n";
      break;
    case ASTNode::NodeKind::ContinueStmt:
      pad(indent); os << "continue;\n";
      break;
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      pad(indent); os << "switch (";
      emitHostExpr(sw->cond.get());
      os << ") {\n";
      if (sw->body &&
          sw->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
        auto *cs = static_cast<const CompoundStmt *>(sw->body.get());
        for (auto &s : cs->statements) emitHostStmt(s.get(), indent + 1);
      } else if (sw->body) {
        emitHostStmt(sw->body.get(), indent + 1);
      }
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      pad(indent);
      if (cs->value) {
        os << "case "; emitHostExpr(cs->value.get()); os << ":\n";
      } else {
        os << "default:\n";
      }
      if (cs->sub) emitHostStmt(cs->sub.get(), indent + 1);
      break;
    }
    default:
      // Expressions appearing as statements (e.g. LaunchExpr) fall here.
      if (n && n->getNodeType() == ASTNode::NodeKind::LaunchExpr) {
        emitLaunch(static_cast<const LaunchExpr *>(n), indent);
        break;
      }
      break;
    }
  }

  // --------------------------------------------------------------------- //
  // Expressions
  // --------------------------------------------------------------------- //

  void emitHostExpr(const ASTNode *n) {
    if (!n) { os << "/*null*/"; return; }
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral:
      os << static_cast<const IntegerLiteral *>(n)->value;
      return;
    case ASTNode::NodeKind::FloatLiteral:
      os << static_cast<const FloatLiteral *>(n)->value;
      return;
    case ASTNode::NodeKind::CharLiteral:
      os << static_cast<const CharLiteral *>(n)->value;
      return;
    case ASTNode::NodeKind::StringLiteral:
      emitCString(static_cast<const vc::StringLiteral *>(n)->value);
      return;
    case ASTNode::NodeKind::DeclRefExpr:
      os << static_cast<const DeclRefExpr *>(n)->name;
      return;
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      os << "(";
      emitHostExpr(b->lhs.get());
      os << " " << binopStr(b->op) << " ";
      emitHostExpr(b->rhs.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      auto *u = static_cast<const UnaryExpr *>(n);
      const char *op = unaryopStr(u->op);
      // Postfix ++/-- print operand then operator; prefix the reverse.
      bool postfix = (u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec);
      if (!postfix) os << op;
      emitHostExpr(u->operand.get());
      if (postfix) os << op;
      return;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      os << "(";
      emitHostExpr(c->cond.get());
      os << " ? ";
      emitHostExpr(c->thenExpr.get());
      os << " : ";
      emitHostExpr(c->elseExpr.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::CStyleCastExpr: {
      auto *c = static_cast<const CStyleCastExpr *>(n);
      // Emit functional-cast form: T(expr). Valid for scalars and pointers.
      os << "(" << cppType(c->target) << ")(";
      emitHostExpr(c->sub.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::InitListExpr: {
      auto *il = static_cast<const InitListExpr *>(n);
      os << "{";
      for (unsigned i = 0; i < il->elements.size(); ++i) {
        if (i) os << ", ";
        emitHostExpr(il->elements[i].get());
      }
      os << "}";
      return;
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      emitHostExpr(c->callee.get());
      os << "(";
      for (unsigned i = 0; i < c->args.size(); ++i) {
        if (i) os << ", ";
        emitHostExpr(c->args[i].get());
      }
      os << ")";
      return;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      emitHostExpr(ie->base.get());
      os << "[";
      emitHostExpr(ie->index.get());
      os << "]";
      return;
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      auto *m = static_cast<const MemberAccessExpr *>(n);
      emitHostExpr(m->base.get());
      // The parser lowers both `.` and `::` to MemberAccessExpr; `isScope`
      // records which. A scoped enum/namespace access (VCMemcpyKind::HostToDevice)
      // must emit `::` — `.` is a hard error for those in C++.
      os << (m->isScope ? "::" : ".") << m->member;
      return;
    }
    default:
      os << "/*unhandled expr " << static_cast<unsigned>(n->getNodeType())
         << "*/";
      return;
    }
  }

  // Emit a C string literal with proper escaping.
  void emitCString(const std::string &s) {
    os << "\"";
    for (char c : s) {
      switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\t': os << "\\t"; break;
      case '\r': os << "\\r"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
          os << buf;
        } else {
          os << c;
        }
        break;
      }
    }
    os << "\"";
  }

  // --------------------------------------------------------------------- //
  // Launch translation: kernel<<<grid,block>>>(args)
  // --------------------------------------------------------------------- //

  void emitLaunch(const LaunchExpr *l, unsigned indent) {
    pad(indent);
    os << "{\n";
    // Lazy-load the kernel handle on first launch. vcInit() may be called
    // anywhere in the user's main(); this guard runs after it.
    std::string kname = "main";
    if (l->callee &&
        l->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      kname = static_cast<const DeclRefExpr *>(l->callee.get())->name;
    pad(indent + 1);
    os << "if (!__vc_k_" << kname
       << ") vcLoadKernel(__vc_spirv, __vc_spirv_len, \"main\", &__vc_k_"
       << kname << ");\n";
    pad(indent + 1);
    os << "VCKernelArg __args[" << l->args.size() << "] = {";
    for (unsigned i = 0; i < l->args.size(); ++i) {
      if (i) os << ", ";
      emitLaunchArg(l->args[i].get(), indent + 1);
    }
    os << "};\n";

    // Choose the runtime launch API by dimensionality and stream presence:
    //   1D, default stream  -> vcLaunchKernel(k, gridX, blockX, args, n)
    //   1D, explicit stream -> vcLaunchKernelS(k, gridX, blockX, args, n, s)
    //   2D, default stream  -> vcLaunchKernel2D(k, gridX, gridY, blockX, blockY, args, n)
    //   2D, explicit stream -> vcLaunchKernel2DS(k, gridX, gridY, blockX, blockY, args, n, s)
    bool is2D = l->gridDimY || l->blockDimY;
    bool hasStream = static_cast<bool>(l->stream);
    pad(indent + 1);
    os << "vcLaunchKernel";
    if (is2D) os << "2D";
    if (hasStream) os << "S";
    os << "(__vc_k_" << kname << ", ";
    emitHostExpr(l->gridDim.get());
    if (is2D) {
      os << ", ";
      if (l->gridDimY) emitHostExpr(l->gridDimY.get());
      else os << "1";
    }
    os << ", ";
    emitHostExpr(l->blockDim.get());
    if (is2D) {
      os << ", ";
      if (l->blockDimY) emitHostExpr(l->blockDimY.get());
      else os << "1";
    }
    os << ", __args, " << l->args.size();
    if (hasStream) {
      os << ", ";
      emitHostExpr(l->stream.get());
    }
    os << ");\n";
    pad(indent);
    os << "}\n";
  }

  // Classify one launch arg and emit its VCKernelArg initializer.
  void emitLaunchArg(const ASTNode *arg, unsigned /*indent*/) {
    // Pointer arg: a DeclRefExpr whose declared type is PointerType. The
    // variable holds a vcMalloc device handle; pass its value directly.
    if (arg && arg->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      auto *d = static_cast<const DeclRefExpr *>(arg);
      auto it = hostVarTypes.find(d->name);
      if (it != hostVarTypes.end() && it->second &&
          it->second->getKind() == TypeKind::Pointer) {
        os << "{VCKernelArg::Pointer, ";
        emitHostExpr(arg); // the handle value
        os << ", 0}";
        return;
      }
    }
    // Scalar arg: pass address + size. Use sizeof on the declared type when
    // known; fall back to sizeof(int) for literals/expressions.
    os << "{VCKernelArg::Scalar, &(";
    emitHostExpr(arg);
    os << "), sizeof(";
    if (arg && arg->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      auto *d = static_cast<const DeclRefExpr *>(arg);
      auto it = hostVarTypes.find(d->name);
      if (it != hostVarTypes.end() && it->second) {
        os << cppType(it->second);
      } else {
        os << "int";
      }
    } else {
      os << "int";
    }
    os << ")}";
  }

  // --------------------------------------------------------------------- //
  // Type + operator spelling
  // --------------------------------------------------------------------- //

  std::string cppType(const Type *t) {
    if (!t) return "int";
    if (t->getKind() == TypeKind::Builtin) {
      switch (static_cast<const BuiltinType *>(t)->builtin) {
      case BuiltinTypeKind::Void: return "void";
      case BuiltinTypeKind::Bool: return "bool";
      case BuiltinTypeKind::Int32: return "int";
      case BuiltinTypeKind::UInt32: return "unsigned int";
      case BuiltinTypeKind::Int64: return "long";
      case BuiltinTypeKind::UInt64: return "unsigned long";
      case BuiltinTypeKind::Float32: return "float";
      case BuiltinTypeKind::Float64: return "double";
      }
    }
    if (t->getKind() == TypeKind::Pointer) {
      // void* stays void*; otherwise T*. Host needs the real pointer type
      // (unlike GLSL, which strips pointers to SSBO element types).
      const Type *p = static_cast<const PointerType *>(t)->pointee;
      if (p && p->getKind() == TypeKind::Builtin &&
          static_cast<const BuiltinType *>(p)->builtin == BuiltinTypeKind::Void)
        return "void*";
      return cppType(p) + "*";
    }
    if (t->getKind() == TypeKind::Reference) {
      // C++ lvalue reference `T&` (host-only, e.g. helper `const Point &p`).
      return cppType(static_cast<const ReferenceType *>(t)->pointee) + "&";
    }
    if (t->getKind() == TypeKind::Vector) {
      // float4 etc. aren't real host types; emit the name best-effort. Host
      // code shouldn't use these (documented limitation).
      auto *v = static_cast<const VectorType *>(t);
      return "int/*vec" + std::to_string(v->count) + "*/";
    }
    if (t->getKind() == TypeKind::Record) {
      auto *r = static_cast<const RecordType *>(t);
      return std::string(r->decl->name);
    }
    if (t->getKind() == TypeKind::Typedef) {
      auto *td = static_cast<const TypedefType *>(t);
      return cppType(td->decl->underlying);
    }
    return "int";
  }

  const char *binopStr(BinaryOp op) const {
    switch (op) {
    case BinaryOp::Add: return "+";
    case BinaryOp::Sub: return "-";
    case BinaryOp::Mul: return "*";
    case BinaryOp::Div: return "/";
    case BinaryOp::Mod: return "%";
    case BinaryOp::Assign: return "=";
    case BinaryOp::Lt: return "<";
    case BinaryOp::Gt: return ">";
    case BinaryOp::Le: return "<=";
    case BinaryOp::Ge: return ">=";
    case BinaryOp::Eq: return "==";
    case BinaryOp::NEq: return "!=";
    case BinaryOp::Shl: return "<<";
    case BinaryOp::Shr: return ">>";
    case BinaryOp::And: return "&";
    case BinaryOp::Or: return "|";
    case BinaryOp::Xor: return "^";
    case BinaryOp::LAnd: return "&&";
    case BinaryOp::LOr: return "||";
    }
    return "?";
  }

  const char *unaryopStr(UnaryOp op) const {
    switch (op) {
    case UnaryOp::Neg: return "-";
    case UnaryOp::Not: return "~";
    case UnaryOp::LNot: return "!";
    case UnaryOp::Deref: return "*";
    case UnaryOp::AddrOf: return "&";
    case UnaryOp::PreInc: case UnaryOp::PostInc: return "++";
    case UnaryOp::PreDec: case UnaryOp::PostDec: return "--";
    }
    return "?";
  }
};

} // namespace

namespace vc {
namespace host {

bool translateASTToHost(const TranslationUnit &tu,
                        const uint32_t *spirvWords, size_t wordCount,
                        llvm::raw_ostream &os) {
  HostEmitter e(os, spirvWords, wordCount);
  return e.emit(tu);
}

} // namespace host
} // namespace vc
