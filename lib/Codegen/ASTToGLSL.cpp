//===- ASTToGLSL.cpp - Emit GLSL compute shader from AST -----------------===//

#include "vc/Codegen/ASTToGLSL.h"

#include "vc/Frontend/AST.h"
#include "vc/Frontend/BuiltinRegistry.h"
#include "vc/Frontend/Mangle.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>

using namespace vc;
using namespace llvm;

namespace {

class GLSLEmitter {
  raw_ostream *os;
  // The kernel function being emitted.
  const FunctionDecl *kernel = nullptr;
  // Track which params are pointers vs scalars to declare bindings.
  SmallVector<const ParamDecl *, 8> params;
  // Which workgroup-index dimensions (x/y/z) the kernel reads. Drives the
  // local_size layout: a 1D kernel uses local_size_x only; a 2D kernel
  // gets local_size_x + local_size_y, etc. The actual sizes are supplied
  // by the host via specialization constants.
  bool useY = false;
  bool useZ = false;
  // Names of scalar parameters, which are accessed as `pc.<name>` since they
  // live in the push-constant block.
  SmallVector<StringRef, 8> scalarParams;
  // Names of pointer (SSBO buffer) parameters. CUDA `atomicAdd(counter, v)`
  // treats `counter` as a pointer to a single value, but in GLSL it's an SSBO
  // array — so a bare reference must be lowered to `counter[0]`.
  SmallVector<StringRef, 8> ssboParams;
  // Whether the kernel references any Vulkan subgroup primitive (CUDA warp
  // intrinsics or warpSize). Subgroup ops require SPIR-V 1.3 + the
  // GL_KHR_shader_subgroup_* extensions; emitHeader drives the extension list
  // and the driver bumps the glslc target to vulkan1.1 when this is set.
  bool usesSubgroup = false;
  // Whether the kernel references any `double`/`doubleN` type. GLSL's
  // GL_EXT_shader_explicit_arithmetic_types (enabled unconditionally) gives us
  // the double *type* and the any-width builtins (sqrt/floor/fabs/min/max work
  // on double without extra extensions), but the transcendental overloads
  // (sin/cos/tan/asin/.../pow/exp/log on a double arg) require the separate
  // _float64 extension. Set during emitBindings/emitSharedDecls by checking
  // each param's and shared var's glslType for `double`/`dvec`.
  bool usesDouble = false;
  // Name -> FunctionDecl index of all top-level functions (kernels + __device__
  // helpers). Used to look up default-argument expressions when completing call
  // sites that omit trailing defaulted parameters.
  llvm::StringMap<const FunctionDecl *> funcDecls;
  // Whether the kernel uses a block-wide vote intrinsic
  // (__syncthreads_count/and/or). These lower to a shared-array reduction +
  // barriers and need a pre-statement hoisting buffer plus a scratch shared
  // array declared at the top.
  bool usesVote = false;
  // kernel-internal printf -> debugPrintfEXT (GL_EXT_debug_printf extension +
  // NonSemantic.DebugPrintf SPIR-V). Flipped by scanDims when a `printf(...)`
  // call is seen in the kernel body or any __device__ helper it can call.
  bool usesPrintf = false;
  // Set by scanSubArrayDecay when a call passes a sub-array (`b[i]` of `int
  // b[2][3]`) where a pointer parameter is expected — unsupported on the GLSL
  // backend (no pointer type). emitHeader emits a `#error` so glslc fails with
  // a clear message instead of an opaque parse error from the broken body.
  bool hasSubArrayDecayCall = false;
  // Hoisted statements to emit before the current enclosing statement. The
  // vote intrinsics push their reduction prologue here; emitStmt flushes and
  // clears this before emitting each statement node.
  std::string preStmts;
  // During emitStmt's real emit pass, vote calls emit only the value ref
  // (their prologue was already flushed by the pre-pass). During the pre-pass
  // this is false and vote calls push the prologue into preStmts.
  bool emittingVoteRef = false;
  // Name -> type of every in-scope scalar value (kernel params, __shared__
  // decls, and locals as they are emitted). GLSL's explicit-arithmetic-types
  // extension forbids mixed-width arithmetic (`float16_t * int` is a hard
  // error), so binary expressions must mirror C's usual arithmetic conversions
  // by wrapping one operand in a constructor cast. The AST stores no result
  // type on expression nodes, so we reconstruct it from this map plus the
  // literal/cast structure of the expression subtree (see exprType).
  llvm::StringMap<const Type *> localTypes;
  // Name -> VarDecl for in-scope locals/params/shared decls. Mirrors
  // localTypes but retains the VarDecl so its arrayDims (which the Type does
  // not carry) are available — used to detect a partial subscript of a
  // multi-dim local array (sub-array-to-pointer decay), which the GLSL
  // backend cannot lower.
  llvm::StringMap<const VarDecl *> localVarDecls;

  // CUDA pointer-typed locals (`int *q = out + i;`) have no GLSL
  // representation: glslType strips Pointer->pointee, so a plain decl would
  // emit `int q = out_ + i` (a scalar, useless for `*q`/`q[k]`). Instead we
  // track each such local as a (baseName, offsetExpr) pair and rewrite uses
  // to `base[offset (+k)]`. baseName is the GLSL SSBO/local-array name; the
  // offset is a GLSL expression string (or "0" for a bare alias). Cleared
  // per function alongside localTypes (see emitFunction).
  struct PtrLocal { std::string base; std::string offset; };
  llvm::StringMap<PtrLocal> pointerLocals;

  // Reference locals (`int &r = x;`) have no GLSL counterpart: there is no way
  // to declare a second name for an existing variable. Since every use of `r`
  // means the lvalue it was bound to, we record the bound lvalue's GLSL text
  // and substitute it at each use — `r = r + 3` becomes `x = x + 3`. The
  // initializer must be an lvalue expression (the parser accepts any
  // expression, so a non-lvalue falls back to a plain by-value decl, which is
  // what the previous behavior was). Cleared per function, like pointerLocals.
  llvm::StringMap<std::string> refLocals;

public:
  GLSLEmitter(raw_ostream &o) : os(&o) {}

  bool emit(const TranslationUnit &tu) {
    // Flatten the translation unit: namespaces are transparent on the device
    // side (GLSL has none), so their body decls are spliced into the top-level
    // list. Structs/functions inside a namespace are emitted with mangled names
    // (the Sema pass already registered them under those keys; here we just
    // walk the flattened list).
    std::vector<const ASTNode *> flat;
    flattenDecls(tu.decls, flat);

    // Index all functions (kernels + __device__ helpers, including those inside
    // namespaces) under their mangled device name so call sites resolve. Shared
    // across all kernels (a kernel may call any device helper).
    for (auto *d : flat) {
      if (d->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
        auto *f = static_cast<const FunctionDecl *>(d);
        funcDecls[deviceMangledName(f)] = f;
      }
    }

    // Find the first __global__ function (single-kernel legacy path).
    const FunctionDecl *fn = nullptr;
    for (auto *d : flat) {
      if (d->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
        auto *f = static_cast<const FunctionDecl *>(d);
        if (f->deviceAttr == DeviceAttr::Global) { fn = f; break; }
      }
    }
    if (!fn) {
      (*os) << "// no __global__ kernel found\n";
      return false;
    }
    emitOneForKernel(fn, flat);
    return true;
  }

  // Emit one complete GLSL compute unit for `k`. Resets the per-kernel state
  // (params, dim flags, subgroup/vote flags, ssbo/scalar param lists) so the
  // same emitter can produce multiple independent units for a multi-kernel TU.
  void emitOneForKernel(const FunctionDecl *k,
                        const std::vector<const ASTNode *> &flat) {
    kernel = k;
    params.assign(k->params.begin(), k->params.end());
    useY = false; useZ = false;
    scalarParams.clear();
    ssboParams.clear();
    usesSubgroup = false;
    usesVote = false;
    usesDouble = false;
    usesPrintf = false;
    hasSubArrayDecayCall = false;
    preStmts.clear();
    emittingVoteRef = false;
    if (k->body) scanDims(k->body.get());
    scanDoubleUsage(k);
    // A printf in a __device__ helper also needs the extension. Walk every
    // device function body reachable from this kernel's translation unit.
    for (const auto &kv : funcDecls)
      if (kv.second->body) scanDims(kv.second->body.get());
    // Detect sub-array-to-pointer decay calls (unsupported on this backend)
    // before emitting the header, so emitHeader can raise a clear `#error`.
    if (k->body) scanSubArrayDecay(k->body.get());

    emitHeader();
    emitStructDecls(flat);
    emitEnumDecls(flat);
    emitConstantDecls(flat);
    emitBindings();
    emitSharedDecls();
    emitVoteDecls();
    emitDeviceFunctions(flat);
    emitBody();
  }

  // Emit a complete unit for every __global__ in `tu`, each into its own
  // string buffer. Returns (entryName, source) per kernel in source order.
  std::vector<glsl::GLSLModule> emitAll(const TranslationUnit &tu) {
    std::vector<const ASTNode *> flat;
    flattenDecls(tu.decls, flat);
    for (auto *d : flat) {
      if (d->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
        auto *f = static_cast<const FunctionDecl *>(d);
        funcDecls[deviceMangledName(f)] = f;
      }
    }
    std::vector<glsl::GLSLModule> out;
    for (auto *d : flat) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d);
      if (f->deviceAttr != DeviceAttr::Global) continue;
      std::string src;
      raw_string_ostream buf(src);
      raw_ostream *saved = os;
      os = &buf;
      emitOneForKernel(f, flat);
      os = saved;
      buf.flush();
      out.push_back({deviceMangledName(f), std::move(src)});
    }
    return out;
  }

  // Recursively splice NamespaceDecl bodies into `out` (namespaces are
  // transparent on the device side). Nested namespaces recurse.
  static void flattenDecls(const std::vector<NodePtr> &decls,
                           std::vector<const ASTNode *> &out) {
    for (auto &d : decls) {
      if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
        flattenDecls(static_cast<const NamespaceDecl *>(d.get())->decls, out);
        continue;
      }
      out.push_back(d.get());
    }
  }

private:
  // Detect any `double`/`doubleN` usage in the kernel's params or body so
  // emitHeader can enable GL_EXT_shader_explicit_arithmetic_types_float64 (the
  // transcendental overloads sin/cos/.../pow on a double arg need it; the base
  // explicit-arithmetic extension only provides the double *type* and the
  // any-width builtins). Checks params, shared decls, and walks the body for
  // VarDecls whose glslType comes out `double`/`dvecN`.
  void scanDoubleUsage(const FunctionDecl *k) {
    auto mark = [&](const char *ty) {
      if (ty && (StringRef(ty).starts_with("double") ||
                 StringRef(ty).starts_with("dvec")))
        usesDouble = true;
    };
    for (const auto &p : k->params)
      if (p->type) mark(glslType(p->type));
    if (k->body) {
      SmallVector<const VarDecl *, 8> shared;
      collectShared(k->body.get(), shared);
      for (const VarDecl *v : shared)
        if (v->type) mark(glslType(v->type));
      scanDoubleDecls(k->body.get(), mark);
    }
  }

  // Recursively walk the body marking any VarDecl (local or shared) whose type
  // lowers to a GLSL double. Covers locals that collectShared skips (non-shared
  // decls). `mark` is the lambda from scanDoubleUsage.
  void scanDoubleDecls(const ASTNode *n,
                       const std::function<void(const char *)> &mark) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::DeclStmt:
      for (VarDecl *d : static_cast<const DeclStmt *>(n)->decls)
        if (d && d->type) mark(glslType(d->type));
      break;
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanDoubleDecls(s.get(), mark);
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      scanDoubleDecls(iff->thenStmt.get(), mark);
      scanDoubleDecls(iff->elseStmt.get(), mark);
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanDoubleDecls(fs->init.get(), mark);
      scanDoubleDecls(fs->body.get(), mark);
      break;
    }
    case ASTNode::NodeKind::WhileStmt:
      scanDoubleDecls(static_cast<const WhileStmt *>(n)->body.get(), mark);
      break;
    case ASTNode::NodeKind::DoStmt:
      scanDoubleDecls(static_cast<const DoStmt *>(n)->body.get(), mark);
      break;
    case ASTNode::NodeKind::SwitchStmt:
      if (auto *sw = static_cast<const SwitchStmt *>(n)->body.get())
        scanDoubleDecls(sw, mark);
      break;
    default:
      break;
    }
  }

  // Pre-scan for sub-array-to-pointer decay calls (e.g. `sumRow(b[i], 3)` with
  // `int *row` and `int b[2][3]`). The GLSL backend can't lower these (no
  // pointer type); set hasSubArrayDecayCall so emitHeader raises a clear
  // `#error` before the broken body reaches glslc. Self-contained: builds a
  // body-wide name->VarDecl table so the decay check resolves locals without
  // the emit pass's localVarDecls (not populated yet).
  void scanSubArrayDecay(const ASTNode *n) {
    if (!n) return;
    llvm::StringMap<const VarDecl *> vars;
    collectBodyVarDecls(n, vars);
    auto resolve = [&](llvm::StringRef name) -> const VarDecl * {
      auto it = vars.find(name);
      return it == vars.end() ? nullptr : it->second;
    };
    scanDecayCalls(n, resolve);
  }

  // Recursively collect every VarDecl declared in `n`'s subtree (DeclStmt
  // declarators) into `out`. Mirrors the local seeding the emit pass does, but
  // body-wide and read-only.
  void collectBodyVarDecls(
      const ASTNode *n,
      llvm::StringMap<const VarDecl *> &out) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::DeclStmt) {
      auto *ds = static_cast<const DeclStmt *>(n);
      for (VarDecl *vd : ds->decls)
        if (vd) out[vd->name] = vd;
    }
    // Recurse into every child AST node. We use the same structural cases as
    // scanDims (compound/control-flow) plus expression children for completeness.
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        collectBodyVarDecls(s.get(), out);
      break;
    case ASTNode::NodeKind::DeclStmt:
      if (auto *d = static_cast<const DeclStmt *>(n)->decl)
        collectBodyVarDecls(d->init.get(), out);
      break;
    case ASTNode::NodeKind::ExprStmt:
      collectBodyVarDecls(static_cast<const ExprStmt *>(n)->expr.get(), out);
      break;
    case ASTNode::NodeKind::ReturnStmt:
      collectBodyVarDecls(static_cast<const ReturnStmt *>(n)->value.get(), out);
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      collectBodyVarDecls(iff->cond.get(), out);
      collectBodyVarDecls(iff->thenStmt.get(), out);
      collectBodyVarDecls(iff->elseStmt.get(), out);
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      collectBodyVarDecls(fs->init.get(), out);
      collectBodyVarDecls(fs->cond.get(), out);
      collectBodyVarDecls(fs->step.get(), out);
      collectBodyVarDecls(fs->body.get(), out);
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      collectBodyVarDecls(ws->cond.get(), out);
      collectBodyVarDecls(ws->body.get(), out);
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      collectBodyVarDecls(ds->cond.get(), out);
      collectBodyVarDecls(ds->body.get(), out);
      break;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      collectBodyVarDecls(sw->cond.get(), out);
      collectBodyVarDecls(sw->body.get(), out);
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      collectBodyVarDecls(cs->value.get(), out);
      collectBodyVarDecls(cs->sub.get(), out);
      break;
    }
    default:
      break;
    }
  }

  // Walk `n` for CallExprs; for each, resolve the callee (via resolvedCallee or
  // funcDecls by bare/scoped name) and check whether any argument bound to a
  // pointer parameter is a sub-array decay. Sets hasSubArrayDecayCall on hit.
  void scanDecayCalls(
      const ASTNode *n,
      llvm::function_ref<const VarDecl *(llvm::StringRef)> resolveVar) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::CallExpr) {
      auto *c = static_cast<const CallExpr *>(n);
      const FunctionDecl *calleeFn = c->resolvedCallee;
      if (!calleeFn) {
        // Resolve by bare name (DeclRefExpr callee) or scoped name.
        std::string nm;
        if (c->callee &&
            c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
          nm = static_cast<const DeclRefExpr *>(c->callee.get())->name.str();
        else if (c->callee && c->callee->getNodeType() ==
                                  ASTNode::NodeKind::MemberAccessExpr)
          nm = mangleScopeChainGLSL(
              static_cast<const MemberAccessExpr *>(c->callee.get()));
        if (!nm.empty()) {
          auto fit = funcDecls.find(nm);
          if (fit != funcDecls.end()) calleeFn = fit->second;
        }
      }
      if (calleeFn) {
        unsigned nArgs = std::min(c->args.size(), calleeFn->params.size());
        for (unsigned i = 0; i < nArgs; ++i) {
          const ParamDecl *p = calleeFn->params[i];
          if (!p->type || p->type->getKind() != TypeKind::Pointer) continue;
          if (isSubArrayDecayArg(c->args[i].get(), resolveVar)) {
            hasSubArrayDecayCall = true;
            break;
          }
        }
      }
      // Still recurse into args (a decay call may nest inside another's arg).
      for (auto &a : c->args) scanDecayCalls(a.get(), resolveVar);
      if (c->callee) scanDecayCalls(c->callee.get(), resolveVar);
      return;
    }
    // Recurse structurally (same cases as collectBodyVarDecls).
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanDecayCalls(s.get(), resolveVar);
      break;
    case ASTNode::NodeKind::DeclStmt:
      if (auto *d = static_cast<const DeclStmt *>(n)->decl)
        scanDecayCalls(d->init.get(), resolveVar);
      break;
    case ASTNode::NodeKind::ExprStmt:
      scanDecayCalls(static_cast<const ExprStmt *>(n)->expr.get(), resolveVar);
      break;
    case ASTNode::NodeKind::ReturnStmt:
      scanDecayCalls(static_cast<const ReturnStmt *>(n)->value.get(), resolveVar);
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      scanDecayCalls(iff->cond.get(), resolveVar);
      scanDecayCalls(iff->thenStmt.get(), resolveVar);
      scanDecayCalls(iff->elseStmt.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanDecayCalls(fs->init.get(), resolveVar);
      scanDecayCalls(fs->cond.get(), resolveVar);
      scanDecayCalls(fs->step.get(), resolveVar);
      scanDecayCalls(fs->body.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      scanDecayCalls(ws->cond.get(), resolveVar);
      scanDecayCalls(ws->body.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      scanDecayCalls(ds->cond.get(), resolveVar);
      scanDecayCalls(ds->body.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      scanDecayCalls(sw->cond.get(), resolveVar);
      scanDecayCalls(sw->body.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      scanDecayCalls(cs->value.get(), resolveVar);
      scanDecayCalls(cs->sub.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      scanDecayCalls(b->lhs.get(), resolveVar);
      scanDecayCalls(b->rhs.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::UnaryExpr:
      scanDecayCalls(static_cast<const UnaryExpr *>(n)->operand.get(),
                     resolveVar);
      break;
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      scanDecayCalls(ie->base.get(), resolveVar);
      scanDecayCalls(ie->index.get(), resolveVar);
      break;
    }
    case ASTNode::NodeKind::MemberAccessExpr:
      scanDecayCalls(static_cast<const MemberAccessExpr *>(n)->base.get(),
                     resolveVar);
      break;
    default:
      break;
    }
  }

  // Walk the AST noting which .y/.z components of threadIdx/blockIdx/
  // blockDim/gridDim are referenced, so we emit the matching local_size.
  void scanDims(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::MemberAccessExpr: {
      auto *m = static_cast<const MemberAccessExpr *>(n);
      if (m->base &&
          m->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *b = static_cast<const DeclRefExpr *>(m->base.get());
        if (b->name == "threadIdx" || b->name == "blockIdx" ||
            b->name == "blockDim" || b->name == "gridDim") {
          if (m->member == "y") useY = true;
          else if (m->member == "z") useZ = true;
        }
      }
      scanDims(m->base.get());
      return;
    }
    case ASTNode::NodeKind::DeclRefExpr: {
      // warpSize is a bare value reference (no .x member like threadIdx),
      // so detect it here. Any subgroup primitive flips usesSubgroup so the
      // header enables the KHR extensions and the driver targets vulkan1.1.
      StringRef name = static_cast<const DeclRefExpr *>(n)->name;
      if (name == "warpSize") usesSubgroup = true;
      return;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      scanDims(b->lhs.get());
      scanDims(b->rhs.get());
      return;
    }
    case ASTNode::NodeKind::UnaryExpr:
      scanDims(static_cast<const UnaryExpr *>(n)->operand.get());
      return;
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      scanDims(ie->base.get());
      scanDims(ie->index.get());
      return;
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      // A warp intrinsic call flips usesSubgroup (header + driver target).
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        StringRef callee =
            static_cast<const DeclRefExpr *>(c->callee.get())->name;
        if (isWarpIntrinsicName(callee)) usesSubgroup = true;
        // Block-wide vote intrinsics lower to a shared-array reduction.
        if (callee == "__syncthreads_count" || callee == "__syncthreads_and" ||
            callee == "__syncthreads_or")
          usesVote = true;
        // kernel-internal printf lowers to debugPrintfEXT, which needs the
        // GL_EXT_debug_printf extension (and, at runtime, the validation layer
        // + VK_KHR_shader_non_semantic_info to forward the output).
        if (callee == "printf") usesPrintf = true;
      }
      scanDims(c->callee.get());
      for (auto &a : c->args) scanDims(a.get());
      return;
    }
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanDims(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt:
      scanDims(static_cast<const DeclStmt *>(n)->decl->init.get());
      return;
    case ASTNode::NodeKind::ExprStmt:
      scanDims(static_cast<const ExprStmt *>(n)->expr.get());
      return;
    case ASTNode::NodeKind::ReturnStmt:
      scanDims(static_cast<const ReturnStmt *>(n)->value.get());
      return;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      scanDims(iff->cond.get());
      scanDims(iff->thenStmt.get());
      scanDims(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanDims(fs->init.get());
      scanDims(fs->cond.get());
      scanDims(fs->step.get());
      scanDims(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      scanDims(ws->cond.get());
      scanDims(ws->body.get());
      return;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      scanDims(ds->cond.get());
      scanDims(ds->body.get());
      return;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      scanDims(sw->cond.get());
      scanDims(sw->body.get());
      return;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      scanDims(cs->value.get());
      scanDims(cs->sub.get());
      return;
    }
    default:
      return;
    }
  }

  // GLSL reserves a handful of words (out, in, uniform, buffer, layout, ...)
  // that are legal as C/CUDA identifiers and thus as VC kernel parameter /
  // variable names. When such a name is used as an SSBO field or shared global,
  // glslc rejects it ("unexpected OUT"). Mangle by appending an underscore so
  // the emitted GLSL is always valid. References in the body are rewritten to
  // match via glslName() at every emit site, so the user-visible name is
  // unchanged.
  static bool isReservedGlslWord(llvm::StringRef n) {
    // Only the words most likely to collide with real kernel param names.
    static const char *reserved[] = {
        "out", "in", "uniform", "buffer", "layout", "input", "output",
        "image", "sampler", "patch", "centroid", "flat", "smooth", "noperspective",
        "invariant", "precise", "coherent", "volatile", "restrict", "readonly",
        "writeonly", "atomic_uint", "active", "filter", "rows", "columns",
        "sample", "subroutine", "common", "partition", "hit", "hitObject"};
    for (const char *r : reserved)
      if (n == r) return true;
    return false;
  }
  // Is `t` a `T&` (possibly spelled through a typedef)? Reference parameters
  // emit as GLSL `inout`.
  static bool isReferenceType(const Type *t) {
    while (t && t->getKind() == TypeKind::Typedef)
      t = static_cast<const TypedefType *>(t)->decl
              ? static_cast<const TypedefType *>(t)->decl->underlying
              : nullptr;
    return t && t->getKind() == TypeKind::Reference;
  }

  // Return a GLSL-safe spelling of `name`, mangling reserved words.
  std::string glslName(llvm::StringRef name) const {
    if (isReservedGlslWord(name))
      return std::string(name) + "_";
    return name.str();
  }

  // Map a VC builtin type to a GLSL scalar type string.
  const char *glslType(const Type *t) {
    if (!t) return "float";
    if (t->getKind() == TypeKind::Builtin) {
      switch (static_cast<const BuiltinType *>(t)->builtin) {
      case BuiltinTypeKind::Void: return "void";
      case BuiltinTypeKind::Bool: return "bool";
      case BuiltinTypeKind::Int32: return "int";
      case BuiltinTypeKind::UInt32: return "uint";
      case BuiltinTypeKind::Int64: return "int64_t";
      case BuiltinTypeKind::UInt64: return "uint64_t";
      case BuiltinTypeKind::Float32: return "float";
      case BuiltinTypeKind::Float64: return "double";
      case BuiltinTypeKind::Float16: return "float16_t";
      }
    }
    // pointer-to-T -> T (SSBO element type)
    if (t->getKind() == TypeKind::Pointer)
      return glslType(static_cast<const PointerType *>(t)->pointee);
    // Reference types are host-only and should never reach the GLSL backend;
    // resolve to the underlying type defensively if they do.
    if (t->getKind() == TypeKind::Reference)
      return glslType(static_cast<const ReferenceType *>(t)->pointee);
    // vector: float4 -> vec4, int3 -> ivec3, uint2 -> uvec2,
    // double2 -> dvec2, bool4 -> bvec4.
    if (t->getKind() == TypeKind::Vector) {
      auto *v = static_cast<const VectorType *>(t);
      const char *p = "vec";
      if (v->elem->getKind() == TypeKind::Builtin) {
        switch (static_cast<const BuiltinType *>(v->elem)->builtin) {
        case BuiltinTypeKind::Float32: p = "vec"; break;
        case BuiltinTypeKind::Int32: p = "ivec"; break;
        case BuiltinTypeKind::UInt32: p = "uvec"; break;
        case BuiltinTypeKind::Float64: p = "dvec"; break;
        case BuiltinTypeKind::Float16: p = "f16vec"; break;
        case BuiltinTypeKind::Bool: p = "bvec"; break;
        case BuiltinTypeKind::Int64: p = "i64vec"; break;
        case BuiltinTypeKind::UInt64: p = "u64vec"; break;
        default: break;
        }
      }
      // Build "vecN" / "ivecN" ... into a small static buffer.
      static char buf[16];
      unsigned n = v->count;
      if (n < 2) n = 2;
      if (n > 4) n = 4;
      snprintf(buf, sizeof(buf), "%s%u", p, n);
      return buf;
    }
    // A named struct: emit its name as the GLSL struct name. The struct
    // definition is emitted separately (emitStructDecls) before main(). The
    // StringRef may not be NUL-terminated, so copy into a static buffer.
    if (t->getKind() == TypeKind::Record) {
      auto *r = static_cast<const RecordType *>(t);
      static char buf[128];
      StringRef n = r->decl->name;
      unsigned len = n.size() < sizeof(buf) - 1 ? n.size() : sizeof(buf) - 1;
      memcpy(buf, n.data(), len);
      buf[len] = '\0';
      return buf;
    }
    // A typedef alias resolves to its underlying type's GLSL spelling — GLSL
    // has no typedef, so the alias is erased at emit time.
    if (t->getKind() == TypeKind::Typedef) {
      auto *td = static_cast<const TypedefType *>(t);
      return glslType(td->decl->underlying);
    }
    return "float";
  }

  void emitHeader() {
    (*os) << "#version 460 core\n";
    (*os) << "#extension GL_EXT_shader_explicit_arithmetic_types : enable\n";
    if (hasSubArrayDecayCall) {
      // A call passes a sub-array (`b[i]` of `int b[2][3]`) where a pointer
      // parameter is expected. GLSL has no pointer type, so this can't lower
      // here; the MLIR backend inlines the callee instead. Fail loudly with a
      // clear message at the top of the source so glslc reports THIS error,
      // not an opaque parse failure from the broken body below.
      (*os) << "#error sub-array to pointer parameter is not supported on the "
               "GLSL backend (use the MLIR backend, or pass a flat pointer)\n";
    }
    if (usesDouble) {
      // The base extension gives us the double *type* and any-width builtins
      // (sqrt/floor/fabs/min/max on a double), but the transcendental overloads
      // (sin/cos/tan/asin/.../pow/exp/log on a double arg) require this
      // separate float64 extension. Runtime advertises shaderFloat64.
      (*os) << "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n";
    }
    (*os) << "// generated by vc (GLSL backend)\n";
    if (usesPrintf) {
      // Kernel printf lowers to debugPrintfEXT. The Vulkan validation layer
      // (enabled by vcEnableKernelPrintf before vcInit) captures the
      // NonSemantic.DebugPrintf SPIR-V instructions and forwards the formatted
      // text to the runtime's debug messenger -> stderr.
      (*os) << "#extension GL_EXT_debug_printf : enable\n";
    }
    if (usesSubgroup) {
      // CUDA warp intrinsics lower to Vulkan subgroup ops, which require
      // SPIR-V 1.3 (vulkan1.1). The driver detects this marker comment and
      // bumps glslc's target env accordingly. The KHR subgroup extensions
      // are split by capability; enable the subsets we actually use.
      (*os) << "// vc:needs-spv1.3\n";
      (*os) << "#extension GL_KHR_shader_subgroup_basic : enable\n";
      (*os) << "#extension GL_KHR_shader_subgroup_ballot : enable\n";
      (*os) << "#extension GL_KHR_shader_subgroup_shuffle : enable\n";
      (*os) << "#extension GL_KHR_shader_subgroup_shuffle_relative : enable\n";
      (*os) << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n";
      (*os) << "#extension GL_KHR_shader_subgroup_vote : enable\n";
    }
    // Workgroup size is a specialization constant so the host can choose the
    // block dimensions at launch time (1D or 2D) without recompiling. Constant
    // IDs: 0 -> local_size_x, 1 -> local_size_y, 2 -> local_size_z. Unused
    // axes default to 1.
    (*os) << "layout(local_size_x_id = 0";
    if (useY) (*os) << ", local_size_y_id = 1";
    if (useZ) (*os) << ", local_size_z_id = 2";
    (*os) << ") in;\n\n";
  }

  void emitBindings() {
    // Pointer params get consecutive SSBO bindings (0,1,2,...). Scalar
    // params are packed into a single push-constant block instead of SSBOs,
    // so the runtime can pass them via vkCmdPushConstants without staging
    // buffers. Binding indices skip scalars, matching the runtime's layout.
    SmallVector<const ParamDecl *, 8> scalars;
    unsigned bindIdx = 0;
    for (unsigned i = 0; i < params.size(); ++i) {
      const ParamDecl *p = params[i];
      bool isPtr = p->type && p->type->getKind() == TypeKind::Pointer;
      if (!isPtr) { scalars.push_back(p); scalarParams.push_back(p->name); continue; }
      ssboParams.push_back(p->name);
      const char *ty = glslType(p->type);
      (*os) << "layout(set = 0, binding = " << bindIdx << ") buffer B" << bindIdx
         << " {\n  " << ty << " " << glslName(p->name) << "[];\n};\n\n";
      ++bindIdx;
    }
    if (!scalars.empty()) {
      (*os) << "layout(push_constant) uniform PC {\n";
      for (const ParamDecl *p : scalars)
        (*os) << "  " << glslType(p->type) << " " << glslName(p->name) << ";\n";
      (*os) << "} pc;\n\n";
    }
  }

  void emitSharedDecls() {
    // Recursively collect __shared__ VarDecls and declare them as GLSL
    // `shared` (workgroup) globals, hoisted out of main().
    SmallVector<const VarDecl *, 8> sharedVars;
    if (kernel->body)
      collectShared(kernel->body.get(), sharedVars);
    for (const VarDecl *v : sharedVars) {
      const char *ty = glslType(v->type);
      (*os) << "shared " << ty << " " << glslName(v->name);
      // Emit explicit array dimensions if present (e.g. As[16][16]). A declared
      // but unsized dimension (extern __shared__ T s[], parsed as arrayDims={0})
      // defaults to the workgroup x size. A plain scalar (__shared__ int s;)
      // has no arrayDims and must NOT get a trailing [].
      if (!v->arrayDims.empty()) {
        for (int64_t d : v->arrayDims) {
          if (d > 0) (*os) << "[" << d << "]";
          else (*os) << "[gl_WorkGroupSize.x]"; // extern __shared__ T s[]
        }
      }
      if (v->init) { (*os) << " = "; emitExpr(v->init.get()); }
      (*os) << ";\n";
    }
    if (!sharedVars.empty()) (*os) << "\n";
  }

  // Scratch storage for block-wide vote intrinsics. One slot per lane to hold
  // each thread's (booleanized) predicate, plus a single result cell. Declared
  // at workgroup scope so all threads see the same storage.
  void emitVoteDecls() {
    if (!usesVote) return;
    (*os) << "shared int _vc_vote[gl_WorkGroupSize.x];\n";
    (*os) << "shared int _vc_vote_result;\n\n";
  }

  void collectShared(const ASTNode *n,
                     SmallVector<const VarDecl *, 8> &out) const {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::DeclStmt: {
      // All declarators in one statement share isShared; check each.
      for (VarDecl *d : static_cast<const DeclStmt *>(n)->decls)
        if (d && d->isShared) out.push_back(d);
      break;
    }
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        collectShared(s.get(), out);
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      collectShared(iff->thenStmt.get(), out);
      collectShared(iff->elseStmt.get(), out);
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      collectShared(fs->init.get(), out);
      collectShared(fs->body.get(), out);
      break;
    }
    case ASTNode::NodeKind::WhileStmt:
      collectShared(static_cast<const WhileStmt *>(n)->body.get(), out);
      break;
    case ASTNode::NodeKind::DoStmt:
      collectShared(static_cast<const DoStmt *>(n)->body.get(), out);
      break;
    case ASTNode::NodeKind::SwitchStmt:
      if (auto *sw = static_cast<const SwitchStmt *>(n)->body.get())
        collectShared(sw, out);
      break;
    default:
      break;
    }
  }

  // Emit every __device__ function as a GLSL top-level function, before
  // main(). These are callable helpers (CUDA __device__ functions); the
  // __global__ kernel becomes main(). Forward references are handled if the
  // helper is declared before use (CUDA source is typically top-down). A
  // __device__ class method is lowered to a free function `Class_method`
  // with the synthesized `this` (spelled `_this` in the body) as the first
  // parameter, since GLSL structs have no member functions.
  void emitDeviceFunctions(const std::vector<const ASTNode *> &flat) {
    for (auto *d : flat) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d);
      if (f->deviceAttr != DeviceAttr::Device) continue;
      emitFunction(f);
    }
  }

  // Emit every top-level `struct Name { ... };` as a GLSL struct definition,
  // before any function body that uses the type. GLSL requires types be
  // declared before use, so all structs are emitted up front regardless of
  // source order (mutual references via pointers are uncommon in kernels).
  // A `class` is emitted the same way (fields only — GLSL structs have no
  // methods or access control; methods are lowered to free functions).
  void emitStructDecls(const std::vector<const ASTNode *> &flat) {
    for (auto *d : flat) {
      if (d->getNodeType() != ASTNode::NodeKind::StructDecl) continue;
      auto *sd = static_cast<const StructDecl *>(d);
      (*os) << "struct " << sd->name << " {\n";
      for (const FieldDecl *fd : sd->fields) {
        (*os) << "  " << glslType(fd->type) << " " << glslName(fd->name);
        for (int64_t dim : fd->arrayDims)
          (*os) << "[" << dim << "]";
        (*os) << ";\n";
      }
      (*os) << "};\n\n";
    }
  }

  // Emit unscoped enum constants as `const int NAME = <value>;`. GLSL has no
  // enum type, so each constant becomes a compile-time int. Anonymous enums
  // (no name) are flattened the same way — their constants are still named.
  void emitEnumDecls(const std::vector<const ASTNode *> &flat) {
    for (auto *d : flat) {
      if (d->getNodeType() != ASTNode::NodeKind::EnumDecl) continue;
      auto *ed = static_cast<const EnumDecl *>(d);
      for (auto &c : ed->constants)
        (*os) << "const int " << glslName(c.name) << " = " << c.value << ";\n";
      if (!ed->constants.empty()) (*os) << "\n";
    }
  }

  // Emit __constant__ globals as `const` GLSL globals before main(). CUDA
  // __constant__ variables are device-resident read-only globals; GLSL has no
  // separate constant address space, so a plain `const` global is the correct
  // lowering (it lives in the shader's constant data and is read-only). VC
  // supports compile-time initializers only — no runtime symbol copy.
  void emitConstantDecls(const std::vector<const ASTNode *> &flat) {
    for (auto *d : flat) {
      if (d->getNodeType() != ASTNode::NodeKind::VarDecl) continue;
      auto *v = static_cast<const VarDecl *>(d);
      if (!v->isConstant) continue;
      (*os) << "const " << glslType(v->type) << " " << glslName(v->name);
      for (int64_t dim : v->arrayDims)
        (*os) << "[" << dim << "]";
      if (v->init) {
        (*os) << " = ";
        emitExpr(v->init.get());
      }
      (*os) << ";\n";
    }
    (*os) << "\n";
  }

  // Emit one function signature + body. A method (isMethod) is lowered to a
  // free function `Class_method` with a synthesized leading `Class _this`
  // parameter — the body already references `_this` (the parser mapped `this`
  // to that name), so no body rewrite is needed. Free functions emit as-is.
  void emitFunction(const FunctionDecl *f) {
    // Rebuild the in-scope type table for this function's locals: its own
    // params plus its __shared__ decls. The kernel path seeds localTypes in
    // emitOneForKernel, but __device__ helpers are emitted via this entry
    // point and must not inherit the kernel's param names (a helper `a` is a
    // different variable than the kernel's `a`).
    localTypes.clear();
    pointerLocals.clear();
    refLocals.clear();
    localVarDecls.clear();
    // A helper's parameters are ordinary GLSL function parameters, NOT the
    // kernel's push-constant scalars. The kernel path (emitOneForKernel)
    // populates scalarParams with the kernel's scalar arg names so they emit
    // as `pc.name`; that set must not leak into a helper body, or a helper
    // parameter that happens to share a name with a kernel scalar (e.g. both
    // `int n`) would wrongly emit as `pc.n`. Save and clear it for this emit.
    SmallVector<StringRef, 8> savedScalarParams;
    savedScalarParams = scalarParams;
    scalarParams.clear();
    for (const auto &p : f->params)
      if (p->type) localTypes[p->name] = p->type;
    // `_this` (synthesized method receiver) is a struct lvalue that never
    // participates in arithmetic; leaving it unregistered makes exprType
    // return null for it, which is safe (no promotion attempted).
    if (f->body) {
      SmallVector<const VarDecl *, 8> shared;
      collectShared(f->body.get(), shared);
      for (const VarDecl *v : shared)
        if (v && v->type) {
          localTypes[v->name] = v->type;
          localVarDecls[v->name] = v;
        }
    }
    (*os) << glslType(f->returnType) << " " << deviceMangledName(f) << "(";
    bool emittedParam = false;
    if (f->isMethod && !f->className.empty()) {
      // `this` as the first parameter, typed as the class record. Build a
      // RecordType whose decl carries the class name so glslType spells it.
      // It is `inout`: a method mutating `this->field` (e.g. `this->sum += x`)
      // must propagate back to the caller's object, mirroring C++ reference
      // semantics. GLSL `inout` is the exact counterpart (passed by reference).
      auto *sd = new StructDecl(f->getLoc(), f->className);
      (*os) << "inout " << glslType(new RecordType(sd)) << " _this";
      emittedParam = true;
    }
    for (unsigned i = 0; i < f->params.size(); ++i) {
      if (emittedParam) (*os) << ", ";
      emittedParam = true;
      // `T &r` is a CUDA reference parameter: calls must see the caller's
      // object, and GLSL `inout` is the exact counterpart (passed by
      // reference). Without the qualifier the parameter silently binds by
      // value and the callee's writes are lost. `const T&` must drop the
      // `const` here — GLSL rejects `const inout` ("too many storage
      // qualifiers"), and `inout` already grants the write the callee is
      // entitled to; the constness is a caller-side promise Sema already
      // enforces.
      if (isReferenceType(f->params[i]->type)) {
        (*os) << "inout ";
      } else if (f->params[i]->isConst) {
        (*os) << "const ";
      }
      (*os) << glslType(f->params[i]->type) << " " << glslName(f->params[i]->name);
    }
    (*os) << ") {\n";
    if (f->body &&
        f->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(f->body.get());
      for (auto &s : cs->statements)
        emitStmt(s.get(), 1);
    }
    (*os) << "}\n\n";
    scalarParams = savedScalarParams;
  }

  // Map a CUDA/math builtin name to its GLSL equivalent. Returns the name
  // unchanged if no mapping is known (lets user helpers and GLSL builtins
  // pass through verbatim). CUDA single-precision math intrinsics (__sinf,
  // __expf, ...) lower to GLSL's float math builtins.
  StringRef lowerBuiltinCall(StringRef name) {
    // Kernel-internal printf -> Vulkan's debugPrintfEXT (captured by the
    // validation layer when vcEnableKernelPrintf was called before vcInit).
    // The first argument is a printf-style format string literal; debugPrintfEXT
    // accepts the same %d/%u/%f/%s specifiers and emits them as SPIR-V
    // NonSemantic.DebugPrintf instructions.
    if (name == "printf") return "debugPrintfEXT";
    // Vector constructors: float4(...) -> vec4(...), int3(...) -> ivec3(...).
    // (Recognize the CUDA-style vector name and emit the GLSL constructor.)
    if (auto *glslName = glslVectorCtorName(name))
      return glslName;
    // CUDA make_<vec>(...) constructors -> GLSL vec(...) constructors.
    // e.g. make_float4 -> vec4, make_int3 -> ivec3, make_uint4 -> uvec4.
    if (name.starts_with("make_")) {
      StringRef base = name.substr(5); // drop "make_"
      if (auto *glslName = glslVectorCtorName(base))
        return glslName;
    }
    // CUDA __f-prefixed intrinsics -> GLSL float builtin (drop leading __,
    // trailing f).
    if (name.starts_with("__") && name.ends_with("f") && name.size() > 3) {
      // e.g. __sinf -> sin, __expf -> exp, __powf -> pow
      return name.substr(2, name.size() - 3);
    }
    // CUDA f-suffixed intrinsics without __ : sinf -> sin, sqrtf -> sqrt.
    if (name.ends_with("f") && name.size() > 2) {
      // Only strip if the result is a plausible GLSL builtin; we don't have a
      // full table, so only strip the common math set.
      StringRef base = name.drop_back();
      if (base == "sin" || base == "cos" || base == "tan" || base == "asin" ||
          base == "acos" || base == "atan" || base == "exp" || base == "log" ||
          base == "exp2" || base == "log2" || base == "sqrt" || base == "abs" ||
          base == "pow" || base == "floor" || base == "ceil" || base == "round" ||
          base == "trunc" || base == "fract")
        return base;
    }
    // CUDA double-precision math names (no f suffix) that have no direct GLSL
    // spelling: fabs/fmin/fmax -> abs/min/max. GLSL's abs/min/max are generic
    // over float/double under GL_EXT_shader_explicit_arithmetic_types, so these
    // work on double without needing the _float64 extension (unlike sin/pow,
    // which DO need it — handled below).
    if (name == "fabs") return "abs";
    if (name == "fmin") return "min";
    if (name == "fmax") return "max";
    // CUDA fabsf/fminf/fmaxf already handled above by f-strip or builtin pass.
    return name;
  }

  // If `name` is a CUDA-style vector name (float4, int3, ...), return the
  // matching GLSL constructor name (vec4, ivec3, ...) in a static buffer.
  // Otherwise return null.
  static const char *glslVectorCtorName(StringRef name) {
    struct Base { const char *prefix; const char *glsl; };
    static constexpr Base bases[] = {
        {"float", "vec"}, {"int", "ivec"}, {"uint", "uvec"},
        {"double", "dvec"}, {"bool", "bvec"},
        {"long", "i64vec"}, {"ulong", "u64vec"},
        {"half", "f16vec"}, {"__half", "f16vec"},
    };
    for (const Base &b : bases) {
      StringRef p = b.prefix;
      if (name.size() == p.size() + 1 && name.starts_with(p)) {
        char d = name.back();
        if (d >= '2' && d <= '4') {
          static char buf[16];
          snprintf(buf, sizeof(buf), "%s%c", b.glsl, d);
          return buf;
        }
      }
    }
    return nullptr;
  }

  // --- CUDA atomic -> GLSL atomic lowering --------------------------------
  // CUDA atomics take a pointer (`atomicAdd(&ptr, v)` or `atomicAdd(ptr, v)`
  // where `ptr` is a `T*` kernel param) and return the old value. GLSL atomics
  // take an lvalue reference to a single scalar and return the old value, so:
  //   - `atomicX(&e, ...)`  -> `atomicX(e, ...)`
  //   - `atomicX(ssboArr, ...)` (bare SSBO array param, no index) -> `atomicX(ssboArr[0], ...)`
  //   - `atomicX(ssboArr[i], ...)` / `atomicX(sharedVar, ...)` -> unchanged
  // atomicInc/atomicDec (no direct GLSL form) map to atomicAdd/atomicSub by 1.
  // atomicExch -> atomicExchange, atomicCAS -> atomicCompSwap.
  static bool isAtomicName(StringRef name) {
    return builtinClass(name) == BuiltinClass::Atomic;
  }

  // --- CUDA warp -> Vulkan subgroup lowering --------------------------------
  // CUDA's warp model (fixed 32 lanes) maps to Vulkan subgroups (variable
  // size). warpSize becomes gl_SubgroupSize so kernels adapt to the hardware.
  // The __shfl_*/__ballot_sync/etc intrinsics map to subgroup* functions;
  // CUDA's mask argument (the active-lane bitmask) has no subgroup equivalent
  // and is dropped — subgroup ops act on the currently-active invocations,
  // which matches the common `mask = 0xffffffff` usage.
  static bool isWarpIntrinsicName(StringRef name) {
    return builtinClass(name) == BuiltinClass::Warp;
  }

  // Map a CUDA atomic name to its GLSL counterpart. atomicInc/atomicDec are
  // special-cased by the caller (they change arity, not just the name).
  static StringRef lowerAtomicName(StringRef name) {
    if (name == "atomicExch") return "atomicExchange";
    if (name == "atomicCAS") return "atomicCompSwap";
    return name; // atomicAdd/Sub/Min/Max/And/Or/Xor are identical in GLSL
  }

  // Is `n` a bare DeclRefExpr naming an SSBO (pointer) kernel parameter with
  // no indexing applied? CUDA writes `atomicAdd(counter, v)` for a `T* counter`
  // param; in GLSL that buffer is `counter[]`, so the single value is [0].
  bool isBareSsboRef(const ASTNode *n) const {
    if (!n || n->getNodeType() != ASTNode::NodeKind::DeclRefExpr) return false;
    StringRef name = static_cast<const DeclRefExpr *>(n)->name;
    for (StringRef s : ssboParams)
      if (s == name) return true;
    return false;
  }

  // Emit the atomic target expression: strip a leading `&` (AddrOf) from a
  // CUDA `atomicX(&e, ...)`, and index a bare SSBO param to `[0]`.
  void emitAtomicTarget(const ASTNode *target) {
    if (target &&
        target->getNodeType() == ASTNode::NodeKind::UnaryExpr) {
      auto *u = static_cast<const UnaryExpr *>(target);
      if (u->op == UnaryOp::AddrOf) {
        // `&e` -> `e`
        emitAtomicTarget(u->operand.get());
        return;
      }
    }
    if (isBareSsboRef(target)) {
      // bare SSBO array param -> first element
      emitExpr(target);
      (*os) << "[0]";
      return;
    }
    emitExpr(target);
  }

  void emitAtomicCall(StringRef name, const std::vector<NodePtr> &args) {
    if (args.empty()) { (*os) << lowerAtomicName(name) << "()"; return; }
    // atomicInc(a) / atomicDec(a) -> atomicAdd(a, 1) / atomicSub(a, 1).
    if (name == "atomicInc" || name == "atomicDec") {
      (*os) << (name == "atomicInc" ? "atomicAdd" : "atomicSub") << "(";
      emitAtomicTarget(args[0].get());
      (*os) << ", 1)";
      return;
    }
    (*os) << lowerAtomicName(name) << "(";
    emitAtomicTarget(args[0].get());
    for (unsigned i = 1; i < args.size(); ++i) {
      (*os) << ", ";
      emitExpr(args[i].get());
    }
    (*os) << ")";
  }

  // Lower a CUDA warp intrinsic to its Vulkan subgroup counterpart. CUDA warp
  // intrinsics take a leading `mask` argument (active-lane bitmask); subgroups
  // have no such concept (ops apply to active invocations), so the mask is
  // dropped. Layout per intrinsic:
  //   __syncwarp(mask)                  -> subgroupBarrier()
  //   __activemask()                    -> subgroupBallot(true).x
  //   __ballot_sync(mask, pred)         -> subgroupBallot(bool(pred)).x
  //   __anySync(mask, pred)             -> subgroupAny(bool(pred))
  //   __allSync(mask, pred)             -> subgroupAll(bool(pred))
  //   __shfl_sync(mask, v, lane)        -> subgroupShuffle(v, lane)
  //   __shfl_up_sync(mask, v, d)        -> subgroupShuffleUp(v, d)
  //   __shfl_down_sync(mask, v, d)      -> subgroupShuffleDown(v, d)
  //   __shfl_xor_sync(mask, v, lm)      -> subgroupShuffleXor(v, lm)
  // __ballot_sync/__activemask return a 32-bit bitmask in CUDA; subgroupBallot
  // returns uvec4, so `.x` extracts the low 32 bits (correct for subgroup<=32).
  void emitWarpIntrinsic(StringRef name, const std::vector<NodePtr> &args) {
    if (name == "__syncwarp") {
      (*os) << "subgroupBarrier()";
      return;
    }
    if (name == "__activemask") {
      (*os) << "subgroupBallot(true).x";
      return;
    }
    if (name == "__ballot_sync") {
      // args[0]=mask (dropped), args[1]=predicate
      (*os) << "subgroupBallot(bool(";
      emitExpr(args[1].get());
      (*os) << ")).x";
      return;
    }
    if (name == "__anySync") {
      (*os) << "subgroupAny(bool(";
      emitExpr(args[1].get());
      (*os) << "))";
      return;
    }
    if (name == "__allSync") {
      (*os) << "subgroupAll(bool(";
      emitExpr(args[1].get());
      (*os) << "))";
      return;
    }
    // __shfl* family: args[0]=mask (dropped), args[1]=value, args[2]=index/delta,
    // optional args[3]=width (dropped).
    StringRef glslFn = "subgroupShuffle";
    if (name == "__shfl_up_sync") glslFn = "subgroupShuffleUp";
    else if (name == "__shfl_down_sync") glslFn = "subgroupShuffleDown";
    else if (name == "__shfl_xor_sync") glslFn = "subgroupShuffleXor";
    (*os) << glslFn << "(";
    emitExpr(args[1].get());
    (*os) << ", ";
    emitExpr(args[2].get());
    (*os) << ")";
  }

  // Lower a block-wide vote intrinsic (__syncthreads_count/and/or) to a
  // shared-array reduction. The reduction prologue is hoisted into `preStmts`
  // (flushed by emitStmt before the enclosing statement); the inline value is
  // a bare reference to the result cell. See the CallExpr emit site above for
  // the uniform-control-flow requirement.
  //
  // CUDA semantics: count = #threads with pred!=0; and = all threads pred!=0;
  // or = any thread pred!=0. All three derive from the per-lane count.
  void emitVoteCall(StringRef name, const std::vector<NodePtr> &args) {
    // During the real emit pass the prologue has already been flushed (by the
    // emitStmt pre-pass); just emit the value reference.
    if (emittingVoteRef) {
      (*os) << "_vc_vote_result";
      return;
    }
    // Pre-pass: build the reduction prologue into `preStmts`.
    // Capture the predicate expression as text by emitting into a string.
    std::string predStr;
    {
      raw_string_ostream predOS(predStr);
      raw_ostream *saved = os;
      os = &predOS;
      if (!args.empty()) emitExpr(args[0].get());
      else (*os) << "0";
      os = saved;
    }

    // Build the reduction prologue. `_vc_acc` accumulates the count of threads
    // whose predicate is nonzero; the result cell is derived per op. The
    // predicate may be bool or int, so wrap in bool(...) before the ?: to get a
    // well-typed 0/1 in both cases (GLSL rejects `bool != int`).
    preStmts += "  _vc_vote[int(gl_LocalInvocationID.x)] = (bool(";
    preStmts += predStr;
    preStmts += ")) ? 1 : 0;\n";
    preStmts += "  barrier();\n";
    preStmts += "  if ((int(gl_LocalInvocationID.x) == 0)) {\n";
    preStmts += "    int _vc_acc = 0;\n";
    preStmts += "    for (int _vc_i = 0; _vc_i < int(gl_WorkGroupSize.x); "
                "_vc_i++) { _vc_acc += _vc_vote[_vc_i]; }\n";
    preStmts += "    _vc_vote_result = ";
    if (name == "__syncthreads_count") {
      preStmts += "_vc_acc;\n";
    } else if (name == "__syncthreads_and") {
      // nonzero iff every lane contributed (count == block size).
      preStmts += "(_vc_acc == int(gl_WorkGroupSize.x)) ? 1 : 0;\n";
    } else { // __syncthreads_or
      preStmts += "(_vc_acc != 0) ? 1 : 0;\n";
    }
    preStmts += "  }\n";
    preStmts += "  barrier();\n";

    // Inline value: the reduced result, now visible to all lanes.
    (*os) << "_vc_vote_result";
  }

  // --- VC async-copy approximation (software cooperative copy) ---------------
  // Vulkan/SPIR-V has no TMA hardware (no cp.async.bulk, no mbarrier, no
  // TensorMap descriptor), so vcMemcpyAsync is lowered to a software
  // cooperative copy: every workgroup thread copies a contiguous segment of
  // the source into the shared destination, followed by a barrier(). This is
  // synchronous (there is no real DMA), but the API shape mirrors CUDA's
  // __pipeline_memcpy_async so tiled double-buffered kernels can be written
  // correctly today and re-lowered to a hardware path if one appears.
  //
  // vcPipeline* are pure barrier() wrappers (no mbarrier / transactional
  // barrier exists in Vulkan). They must sit in uniform control flow because
  // GLSL barrier() is only legal when all invocations reach it.
  static bool isAsyncCopyBuiltin(StringRef name) {
    return builtinClass(name) == BuiltinClass::AsyncCopy;
  }

  // vcMemcpyAsync(dst, src, nElems, pipe)
  //   dst    : __shared__ T[] slot to write (e.g. buf[0])
  //   src    : device pointer / SSBO index expression (e.g. in + offset)
  //   nElems : number of elements to copy (NOT bytes — element-level copy)
  //   pipe   : VcPipeline& sync token (ignored in the software lowering)
  // Lowers to a per-thread segment copy + barrier(), written directly to the
  // output stream at the call site (the prologue IS the statement — there is
  // no inline value). nElems must be divisible by the workgroup x size; a
  // remainder needs a tail handler the caller adds explicitly.
  void emitAsyncCopyCall(StringRef name, const std::vector<NodePtr> &args) {
    // vcPipeline* -> barrier() only, inline at the call site.
    if (name != "vcMemcpyAsync") {
      (*os) << "barrier() /*" << name.str() << "*/";
      return;
    }
    // vcMemcpyAsync: cooperative copy block written inline.
    if (args.size() < 3) { (*os) << "/*vcMemcpyAsync: missing args*/"; return; }
    // Capture dst / nElems as text (vote-style raw_string_ostream redirect).
    std::string dstStr, nStr;
    auto capture = [&](const ASTNode *n, std::string &out) {
      raw_string_ostream o(out);
      raw_ostream *saved = os;
      os = &o;
      emitExpr(n);
      os = saved;
    };
    capture(args[0].get(), dstStr);
    capture(args[2].get(), nStr);
    // The source is typically CUDA pointer arithmetic `in + offset`, which is
    // illegal as a GLSL SSBO array expression (`in_ + off`). Detect an Add
    // node and split it into base + offset so we can emit `base[off + i]`.
    // A bare indexable source (e.g. another shared array) is emitted as-is.
    std::string srcBaseStr, srcOffsetStr;
    bool srcIsAdd = false;
    const ASTNode *srcArg = args[1].get();
    if (srcArg && srcArg->getNodeType() == ASTNode::NodeKind::BinaryExpr) {
      auto *b = static_cast<const BinaryExpr *>(srcArg);
      if (b->op == BinaryOp::Add) {
        srcIsAdd = true;
        capture(b->lhs.get(), srcBaseStr);
        capture(b->rhs.get(), srcOffsetStr);
      }
    }
    if (!srcIsAdd) capture(srcArg, srcBaseStr);
    // Emit a block whose first line continues the current line (after the
    // statement's leading pad), so it sits correctly inside the enclosing
    // ExprStmt. Each thread copies a contiguous segment of length
    // nElems/blockDim.x.
    (*os) << "{ // vcMemcpyAsync cooperative copy\n";
    (*os) << "    int _vc_n = (" << nStr << ") / int(gl_WorkGroupSize.x);\n";
    (*os) << "    int _vc_base = int(gl_LocalInvocationIndex) * _vc_n;\n";
    (*os) << "    for (int _vc_i = 0; _vc_i < _vc_n; ++_vc_i) {\n";
    (*os) << "      " << dstStr << "[_vc_base + _vc_i] = " << srcBaseStr;
    if (srcIsAdd)
      (*os) << "[" << srcOffsetStr << " + _vc_base + _vc_i]";
    else
      (*os) << "[_vc_base + _vc_i]";
    (*os) << ";\n";
    (*os) << "    }\n";
    (*os) << "    barrier();\n";
    (*os) << "  }";
    // The enclosing ExprStmt appends `;` after this block, yielding `... }`
    // followed by an empty statement — harmless in GLSL.
  }

  // Pre-pass walker: visit a statement's expressions (with output discarded)
  // so vote calls push their reduction prologues into `preStmts`. Recurses
  // through nested statements so a vote buried in an `if` body is hoisted
  // correctly. Non-vote expression text is thrown away (os is nulls()).
  void scanVotePrologues(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanVotePrologues(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt: {
      auto *ds = static_cast<const DeclStmt *>(n);
      for (VarDecl *d : ds->decls)
        if (d && d->init) emitExpr(d->init.get());
      return;
    }
    case ASTNode::NodeKind::ExprStmt:
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitExpr(e);
      return;
    case ASTNode::NodeKind::ReturnStmt:
      emitExpr(static_cast<const ReturnStmt *>(n)->value.get());
      return;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      emitExpr(iff->cond.get());
      scanVotePrologues(iff->thenStmt.get());
      scanVotePrologues(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanVotePrologues(fs->init.get());
      emitExpr(fs->cond.get());
      emitExpr(fs->step.get());
      scanVotePrologues(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      emitExpr(ws->cond.get());
      scanVotePrologues(ws->body.get());
      return;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      scanVotePrologues(ds->body.get());
      emitExpr(ds->cond.get());
      return;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      emitExpr(sw->cond.get());
      scanVotePrologues(sw->body.get());
      return;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      emitExpr(cs->value.get());
      scanVotePrologues(cs->sub.get());
      return;
    }
    default:
      return;
    }
  }


  void emitBody() {
    // Each __global__ kernel is emitted as its own .comp unit (see emitAll),
    // so the entry function is always `main`. glslc then defaults to the
    // `main` entry point; the host loads each kernel's separate SPIR-V blob
    // with entryPoint="main" (kernels are distinguished by which blob is
    // loaded, not by the entry-point name). shaderc's -fentry-point flag is
    // broken on common distro builds (triggers a glslang built-in parse
    // error), so we avoid it and rely on `main`.
    // Rebuild localTypes for the kernel now that all __device__ helpers
    // (each of which repopulated it) have been emitted.
    localTypes.clear();
    pointerLocals.clear();
    refLocals.clear();
    localVarDecls.clear();
    for (const auto &p : kernel->params)
      if (p->type) localTypes[p->name] = p->type;
    if (kernel->body) {
      SmallVector<const VarDecl *, 8> shared;
      collectShared(kernel->body.get(), shared);
      for (const VarDecl *v : shared)
        if (v && v->type) {
          localTypes[v->name] = v->type;
          localVarDecls[v->name] = v;
        }
    }
    (*os) << "void main() {\n";
    if (kernel->body &&
        kernel->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(kernel->body.get());
      for (auto &s : cs->statements)
        emitStmt(s.get(), 1);
    }
    (*os) << "}\n";
  }

  void pad(unsigned n) { for (unsigned i = 0; i < n; ++i) (*os) << "  "; }

  // Emit a for-loop initializer clause without the trailing ';' / newline.
  void emitForInit(const ASTNode *n) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::DeclStmt) {
      auto *ds = static_cast<const DeclStmt *>(n);
      bool first = true;
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        if (!first) (*os) << ", ";
        // C multi-declarator syntax: the type appears once, then `a = 0, b = 3`.
        // Emitting the type per-declarator (`int a = 0, int b = 3`) is a GLSL
        // (and C) syntax error, so only the first declarator carries the type.
        if (first)
          (*os) << glslType(d->type) << " ";
        first = false;
        (*os) << glslName(d->name);
        for (int64_t dim : d->arrayDims)
          (*os) << "[" << dim << "]";
        if (d->init) { (*os) << " = "; emitExpr(d->init.get()); }
      }
      return;
    }
    if (n->getNodeType() == ASTNode::NodeKind::ExprStmt) {
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitExpr(e);
    }
  }

  void emitStmt(const ASTNode *n, unsigned indent) {
    if (!n) return;
    // Pre-pass: walk this statement's expressions with output discarded, so any
    // __syncthreads_count/and/or calls push their reduction prologue into
    // `preStmts`. We then emit that prologue (uniform control flow, before the
    // statement) so the inline value reference the normal emit produces is
    // already computed. The vote call's value is a bare `_vc_vote_result` ref,
    // so the prologue genuinely must precede it in source order.
    if (usesVote) {
      raw_ostream *saved = os;
      os = &nulls();
      emittingVoteRef = false; // pre-pass: vote calls push prologues
      scanVotePrologues(n);
      os = saved;
      if (!preStmts.empty()) { (*os) << preStmts; preStmts.clear(); }
      emittingVoteRef = true; // real emit: vote calls emit only the ref
    }
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt: {
      pad(indent); (*os) << "{\n";
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        emitStmt(s.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *ds = static_cast<const DeclStmt *>(n);
      auto *d = ds->decl;
      if (!d) break;
      // Register every declarator's type so subsequent expressions (and the
      // init of later declarators in the same statement) can resolve it.
      for (VarDecl *vd : ds->decls) {
        if (vd && vd->type) localTypes[vd->name] = vd->type;
        if (vd) localVarDecls[vd->name] = vd;
      }
      // Pre-register any pointer-typed local that models a tracked pattern
      // (ptr+int, &arr[i], bare alias). These emit NO GLSL decl — a plain
      // `int q = ...` would be a broken scalar — and their uses are rewritten
      // to `base[offset (+k)]` in emitExpr. A DeclStmt may consist entirely
      // of such locals, so we must skip the whole type/name emission when the
      // first declarator is registered.
      bool firstIsPtr = d && tryRegisterPointerLocal(d);
      // Reference locals are elided too: a by-value copy would detach the alias.
      bool firstIsRef = !firstIsPtr && d && tryRegisterRefLocal(d);
      if (d->isShared) break; // hoisted to a `shared` global
      if (firstIsPtr || firstIsRef) {
        // First declarator is elided. Emit any additional (non-pointer) ones
        // as fresh declarations — they can't reuse the elided type prefix.
        for (unsigned i = 1; i < ds->decls.size(); ++i) {
          VarDecl *vd = ds->decls[i];
          if (!vd || tryRegisterPointerLocal(vd) || tryRegisterRefLocal(vd))
            continue;
          pad(indent);
          if (vd->isConst) (*os) << "const ";
          (*os) << glslType(vd->type) << " " << glslName(vd->name);
          for (int64_t dim : vd->arrayDims)
            (*os) << "[" << dim << "]";
          emitVarInit(vd, indent);
          (*os) << ";\n";
          if (!deferredArrInit.empty()) {
            (*os) << deferredArrInit;
            deferredArrInit.clear();
          }
        }
        break;
      }
      pad(indent);
      if (d->isConst) (*os) << "const ";
      (*os) << glslType(d->type) << " " << glslName(d->name);
      for (int64_t dim : d->arrayDims)
        (*os) << "[" << dim << "]";
      emitVarInit(d, indent);
      // Additional declarators sharing this statement's type (`int a, b;`).
      // C/GLSL multi-declarator syntax: the type appears once before the first
      // declarator; later declarators are just names (re-emitting the type, as
      // in `int a, int b;`, is a syntax error). `const` is a type-qualifier
      // shared by the whole declaration, so it is not repeated either.
      for (unsigned i = 1; i < ds->decls.size(); ++i) {
        VarDecl *vd = ds->decls[i];
        // A reference declarator in tail position (`int a = 1, &r = a;`) is
        // elided like the leading case: it has no GLSL declaration of its own.
        if (vd && tryRegisterPointerLocal(vd)) continue;
        if (vd && tryRegisterRefLocal(vd)) continue;
        (*os) << ", " << glslName(vd->name);
        for (int64_t dim : vd->arrayDims)
          (*os) << "[" << dim << "]";
        emitVarInit(vd, indent);
      }
      (*os) << ";\n";
      // Multi-dimensional array initializers are lowered as element-wise
      // assignments after the declaration (glslc rejects C-style `{}` init for
      // multi-dim arrays), collected during emitVarInit.
      if (!deferredArrInit.empty()) {
        (*os) << deferredArrInit;
        deferredArrInit.clear();
      }
      break;
    }
    case ASTNode::NodeKind::ExprStmt: {
      pad(indent);
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitExpr(e);
      (*os) << ";\n";
      break;
    }
    case ASTNode::NodeKind::ReturnStmt:
      pad(indent); (*os) << "return";
      if (auto *r = static_cast<const ReturnStmt *>(n)->value.get()) {
        (*os) << " "; emitExpr(r);
      }
      (*os) << ";\n";
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      pad(indent); (*os) << "if ("; emitCondition(iff->cond.get()); (*os) << ") {\n";
      if (iff->thenStmt) emitStmt(iff->thenStmt.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      if (iff->elseStmt) {
        pad(indent); (*os) << "else {\n";
        emitStmt(iff->elseStmt.get(), indent + 1);
        pad(indent); (*os) << "}\n";
      }
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      pad(indent); (*os) << "for (";
      if (fs->init)
        emitForInit(fs->init.get()); // emits without trailing newline
      (*os) << "; ";
      if (fs->cond) emitCondition(fs->cond.get());
      (*os) << "; ";
      if (fs->step) emitExpr(fs->step.get());
      (*os) << ") {\n";
      if (fs->body) emitStmt(fs->body.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      pad(indent); (*os) << "while ("; emitCondition(ws->cond.get()); (*os) << ") {\n";
      if (ws->body) emitStmt(ws->body.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      pad(indent); (*os) << "do {\n";
      if (ds->body) emitStmt(ds->body.get(), indent + 1);
      pad(indent); (*os) << "} while ("; emitCondition(ds->cond.get()); (*os) << ");\n";
      break;
    }
    case ASTNode::NodeKind::BreakStmt:
      pad(indent); (*os) << "break;\n";
      break;
    case ASTNode::NodeKind::ContinueStmt:
      pad(indent); (*os) << "continue;\n";
      break;
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      pad(indent); (*os) << "switch (";
      emitExpr(sw->cond.get());
      (*os) << ") {\n";
      // Emit the case/default labels directly (no extra {} block: GLSL
      // forbids nesting case labels inside a nested compound).
      if (sw->body &&
          sw->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
        auto *cs = static_cast<const CompoundStmt *>(sw->body.get());
        for (auto &s : cs->statements) emitStmt(s.get(), indent + 1);
      } else if (sw->body) {
        emitStmt(sw->body.get(), indent + 1);
      }
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      pad(indent);
      if (cs->value) { (*os) << "case "; emitExpr(cs->value.get()); (*os) << ":\n"; }
      else (*os) << "default:\n";
      if (cs->sub) emitStmt(cs->sub.get(), indent + 1);
      break;
    }
    default:
      break;
    }
  }

  // Build the device mangled name for a `::`-scoped MemberAccessExpr chain:
  // `ns::func` -> "ns_func", `A::B::C` -> "A_B_C". Collects the chain parts
  // from the AST, joins them with "::", then reuses the shared mangleScopeName
  // so there is one mangling rule.
  static std::string mangleScopeChainGLSL(const MemberAccessExpr *ma) {
    if (!ma) return {};
    std::vector<std::string> parts;
    parts.push_back(ma->member.str());
    const ASTNode *cur = ma->base.get();
    while (cur) {
      if (cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
        auto *sub = static_cast<const MemberAccessExpr *>(cur);
        if (!sub->isScope) break;
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
    std::string chain;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) chain += "::";
      chain += parts[i];
    }
    return mangleScopeName(chain);
  }

  // For `obj.method(...)`, recover the mangled free-function name
  // `Class_method`. The Class name comes from the object's record type when the
  // base is a simple DeclRefExpr to a typed variable; otherwise we can't
  // recover it and emit a best-effort `method` (which won't resolve, but keeps
  // emission well-formed).
  std::string memberCallMethodName(const MemberAccessExpr *ma) const {
    StringRef method = ma->member;
    if (ma->base &&
        ma->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      StringRef baseName = static_cast<const DeclRefExpr *>(ma->base.get())->name;
      // Look up the base variable's type through the symbol context. The GLSL
      // emitter doesn't carry a type table, so recover the class name by
      // scanning funcDecls for a key of the form "<Class>_<method>" — if exactly
      // one such method exists, that's the callee.
      std::string suffix = "_" + method.str();
      std::string match;
      for (auto &kv : funcDecls) {
        if (kv.getKey().ends_with(suffix) && kv.getValue()->isMethod) {
          if (!match.empty()) { match.clear(); break; } // ambiguous
          match = kv.getKey().str();
        }
      }
      if (!match.empty()) return match;
      (void)baseName;
    }
    return method.str();
  }

  // Emit call arguments followed by default-argument completions, looked up by
  // the callee's mangled device name.
  // Detect a sub-array-to-pointer decay argument: a partial subscript of a
  // local multi-dim array (`b[i]` on `int b[2][3]`) passed where a pointer
  // parameter (`int *row`) is expected. The GLSL backend has no pointer type
  // and no way to view a sub-array as one, so this is unsupported here — the
  // MLIR backend inlines the callee instead. `resolveVar` maps a root
  // DeclRefExpr name to its VarDecl (so the same logic works during the emit
  // pass, against localVarDecls, and during the pre-scan, against a temporary
  // body-wide VarDecl table). Returns true if `arg` is such a partial
  // subscript leaving array dimensions unconsumed.
  bool isSubArrayDecayArg(
      const ASTNode *arg,
      llvm::function_ref<const VarDecl *(StringRef)> resolveVar) const {
    if (!arg || arg->getNodeType() != ASTNode::NodeKind::IndexExpr)
      return false;
    // Walk the IndexExpr chain to its root, counting subscripts consumed.
    const ASTNode *cur = arg;
    unsigned dimsConsumed = 0;
    while (cur && cur->getNodeType() == ASTNode::NodeKind::IndexExpr) {
      ++dimsConsumed;
      cur = static_cast<const IndexExpr *>(cur)->base.get();
    }
    if (!cur || cur->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    const auto *ref = static_cast<const DeclRefExpr *>(cur);
    const VarDecl *vd = resolveVar(ref->name);
    if (!vd || vd->arrayDims.empty()) return false;
    return dimsConsumed < vd->arrayDims.size();
  }

  void emitCallArgsWithDefaults(StringRef mangledName,
                                const std::vector<NodePtr> &args) {
    for (unsigned i = 0; i < args.size(); ++i) {
      if (i) (*os) << ", ";
      emitExpr(args[i].get());
    }
    auto fit = funcDecls.find(mangledName);
    if (fit != funcDecls.end()) {
      const FunctionDecl *calleeFn = fit->second;
      for (unsigned i = args.size(); i < calleeFn->params.size(); ++i) {
        if (calleeFn->params[i]->defaultVal) {
          if (i) (*os) << ", ";
          emitExpr(calleeFn->params[i]->defaultVal.get());
        }
      }
    }
  }

  // Best-effort inference of an expression's result type, for operand
  // promotion in binary expressions. The AST carries no result type on
  // expression nodes (Sema computes types transiently during checking), so we
  // reconstruct from literal structure, casts, and the localTypes table. Returns
  // null when the type cannot be determined — callers treat null as "unknown,
  // emit as-is" (no promotion), which is always safe (it just leaves the
  // original expression, which glslc will reject only if it was genuinely
  // ill-typed, matching today's behavior for unsupported mixes).
  const Type *exprType(const ASTNode *n) const {
    if (!n) return nullptr;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral: {
      auto *il = static_cast<const IntegerLiteral *>(n);
      return new BuiltinType(il->isLong ? BuiltinTypeKind::Int64
                                        : BuiltinTypeKind::Int32);
    }
    case ASTNode::NodeKind::FloatLiteral: {
      auto *fl = static_cast<const FloatLiteral *>(n);
      // A bare `1.5` literal is double; `1.5f` is float. There is no half
      // literal spelling in C (half values arise from __half casts/vars), so a
      // float literal is never Float16 here.
      return new BuiltinType(fl->isFloat32 ? BuiltinTypeKind::Float32
                                          : BuiltinTypeKind::Float64);
    }
    case ASTNode::NodeKind::BoolLiteral:
      return new BuiltinType(BuiltinTypeKind::Bool);
    case ASTNode::NodeKind::DeclRefExpr: {
      auto *d = static_cast<const DeclRefExpr *>(n);
      auto it = localTypes.find(d->name);
      if (it != localTypes.end()) return it->second;
      return nullptr;
    }
    case ASTNode::NodeKind::CStyleCastExpr:
      return static_cast<const CStyleCastExpr *>(n)->target;
    case ASTNode::NodeKind::UnaryExpr:
      // `!` yields int in C (0/1); other unary ops (Neg, Not, Deref, ++/--)
      // preserve or approximate the operand type. Deref would return the
      // pointee but pointer derefs rarely appear in mixed-width arithmetic, so
      // the operand type is a fine approximation there.
      if (static_cast<const UnaryExpr *>(n)->op == UnaryOp::LNot)
        return new BuiltinType(BuiltinTypeKind::Int32);
      return exprType(static_cast<const UnaryExpr *>(n)->operand.get());
    case ASTNode::NodeKind::MemberAccessExpr: {
      // `v.x` on a vector yields the element scalar type; `.xy`/`.xyz`/...
      // yield a smaller vector of the same element. Struct field access needs
      // record-layout info we don't keep here, so it falls through to null.
      auto *m = static_cast<const MemberAccessExpr *>(n);
      if (m->isScope || m->member.empty()) return nullptr;
      const Type *baseTy = exprType(m->base.get());
      if (!baseTy || baseTy->getKind() != TypeKind::Vector) return nullptr;
      auto *vty = static_cast<const VectorType *>(baseTy);
      if (m->member.size() == 1) return vty->elem;
      unsigned n = m->member.size();
      if (n >= 2 && n <= 4)
        return new VectorType(vty->elem, n);
      return nullptr;
    }
    case ASTNode::NodeKind::CallExpr: {
      // A constructor-style call whose callee names a builtin/vector type
      // (float(...), int(...), __half(...), float4(...)) yields that type.
      // Resolved user overloads would need the callee's return type, which we
      // don't track here; return null for those (rare in arithmetic).
      auto *c = static_cast<const CallExpr *>(n);
      if (!c->callee ||
          c->callee->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
        return nullptr;
      StringRef name = static_cast<const DeclRefExpr *>(c->callee.get())->name;
      return ctorResultType(name);
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      // Comparison/logical operators yield bool; arithmetic yields the
      // promoted operand type (mirrors C usual conversions via promoteTypes).
      if (b->op == BinaryOp::LAnd || b->op == BinaryOp::LOr ||
          b->op == BinaryOp::Eq || b->op == BinaryOp::NEq ||
          b->op == BinaryOp::Lt || b->op == BinaryOp::Gt ||
          b->op == BinaryOp::Le || b->op == BinaryOp::Ge)
        return new BuiltinType(BuiltinTypeKind::Bool);
      const Type *lt = exprType(b->lhs.get());
      const Type *rt = exprType(b->rhs.get());
      if (b->op == BinaryOp::Assign) return lt ? lt : rt;
      return promoteTypes(lt, rt);
    }
    default:
      return nullptr;
    }
  }

  // Map a CUDA scalar/vector ctor name to its result Type, or null.
  const Type *ctorResultType(StringRef name) const {
    // Scalar type names.
    if (name == "float") return new BuiltinType(BuiltinTypeKind::Float32);
    if (name == "double") return new BuiltinType(BuiltinTypeKind::Float64);
    if (name == "int") return new BuiltinType(BuiltinTypeKind::Int32);
    if (name == "uint") return new BuiltinType(BuiltinTypeKind::UInt32);
    if (name == "long") return new BuiltinType(BuiltinTypeKind::Int64);
    if (name == "ulong") return new BuiltinType(BuiltinTypeKind::UInt64);
    if (name == "bool") return new BuiltinType(BuiltinTypeKind::Bool);
    if (name == "half" || name == "__half")
      return new BuiltinType(BuiltinTypeKind::Float16);
    // Vector ctor names: float4, int3, __half2, ... reuse the same base table
    // the emitter uses to map them to GLSL constructors.
    struct Base { const char *prefix; BuiltinTypeKind kind; };
    static constexpr Base bases[] = {
        {"float", BuiltinTypeKind::Float32},
        {"int", BuiltinTypeKind::Int32},
        {"uint", BuiltinTypeKind::UInt32},
        {"double", BuiltinTypeKind::Float64},
        {"long", BuiltinTypeKind::Int64},
        {"ulong", BuiltinTypeKind::UInt64},
        {"bool", BuiltinTypeKind::Bool},
        {"half", BuiltinTypeKind::Float16},
        {"__half", BuiltinTypeKind::Float16},
    };
    for (const Base &b : bases) {
      StringRef p = b.prefix;
      if (name.size() > p.size() && name.starts_with(p)) {
        unsigned count = 0;
        if (name.substr(p.size()).getAsInteger(10, count) && count >= 2 &&
            count <= 4)
          return new VectorType(new BuiltinType(b.kind), count);
      }
    }
    return nullptr;
  }

  // Resolve typedefs to the underlying type (defensive; most device types are
  // already builtin).
  static const Type *resolveTypedef(const Type *t) {
    while (t && t->getKind() == TypeKind::Typedef)
      t = static_cast<const TypedefType *>(t)->decl->underlying;
    return t;
  }

  static bool isFloatType(const Type *t) {
    t = resolveTypedef(t);
    return t && t->getKind() == TypeKind::Builtin &&
           (static_cast<const BuiltinType *>(t)->builtin ==
                BuiltinTypeKind::Float16 ||
            static_cast<const BuiltinType *>(t)->builtin ==
                BuiltinTypeKind::Float32 ||
            static_cast<const BuiltinType *>(t)->builtin ==
                BuiltinTypeKind::Float64);
  }
  static bool isIntType(const Type *t) {
    t = resolveTypedef(t);
    if (!t || t->getKind() != TypeKind::Builtin) return false;
    switch (static_cast<const BuiltinType *>(t)->builtin) {
    case BuiltinTypeKind::Int32: case BuiltinTypeKind::UInt32:
    case BuiltinTypeKind::Int64: case BuiltinTypeKind::UInt64:
    case BuiltinTypeKind::Bool:
      return true;
    default:
      return false;
    }
  }

  // C usual arithmetic conversions restricted to the cases GLSL's explicit-
  // arithmetic-types extension rejects outright. GLSL implicitly promotes
  // int/float/double mixes the way C does (`int * double` compiles), but it
  // will NOT implicitly involve float16_t — `float16_t * int`, `float16_t *
  // float`, even `float16_t + float` are hard errors. So we only need to
  // synthesize a common type when one side is float16 and the other is a
  // different numeric type; in every other case we return null and emit the
  // operands as-is (letting GLSL's own promotion handle it, matching the
  // pre-promotion behavior). Returns null when no coercion is needed/known.
  //
  // When float16 meets:
  //   - int    -> common = float16 (CUDA: int converts to half)
  //   - float  -> common = float  (CUDA: half widens to float)
  //   - double -> common = double (CUDA: half widens to double)
  //   - float16-> common = float16 (no-op; emitExprCoerced skips matching)
  static const Type *promoteTypes(const Type *lt, const Type *rt) {
    lt = resolveTypedef(lt);
    rt = resolveTypedef(rt);
    if (!lt || !rt) return nullptr;
    // Vector operand: recurse on the element type; if a common element type
    // emerges, the common vector type keeps the LHS count (no implicit
    // broadcast). Only handle same-count vectors / scalar-vs-scalar here.
    if (lt->getKind() == TypeKind::Vector || rt->getKind() == TypeKind::Vector) {
      const VectorType *lv = lt->getKind() == TypeKind::Vector
                                 ? static_cast<const VectorType *>(lt) : nullptr;
      const VectorType *rv = rt->getKind() == TypeKind::Vector
                                 ? static_cast<const VectorType *>(rt) : nullptr;
      // Need at least one vector and matching counts when both are vectors.
      unsigned count = 0;
      if (lv && rv) { if (lv->count != rv->count) return nullptr; count = lv->count; }
      else count = lv ? lv->count : rv->count;
      const Type *le = lv ? lv->elem : lt;
      const Type *re = rv ? rv->elem : rt;
      const Type *ce = promoteTypes(le, re);
      if (!ce) return nullptr;
      return new VectorType(const_cast<Type *>(ce), count);
    }
    if (!isFloatType(lt) && !isFloatType(rt)) return nullptr;
    bool lh = isHalfTy(lt), rh = isHalfTy(rt);
    if (!lh && !rh) return nullptr; // float/double/int mix without float16
    // Exactly one or both sides are float16.
    if (lh && rh) return lt;                 // both float16, no coercion needed
    const Type *halfTy = lh ? lt : rt;
    const Type *otherTy = lh ? rt : lt;
    if (isIntType(otherTy)) return halfTy;   // int -> float16
    // float/double: widen half to the other float's width.
    return otherTy;
  }
  static bool isHalfTy(const Type *t) {
    t = resolveTypedef(t);
    return t && t->getKind() == TypeKind::Builtin &&
           static_cast<const BuiltinType *>(t)->builtin ==
               BuiltinTypeKind::Float16;
  }

  static bool isBoolTy(const Type *t) {
    t = resolveTypedef(t);
    return t && t->getKind() == TypeKind::Builtin &&
           static_cast<const BuiltinType *>(t)->builtin ==
               BuiltinTypeKind::Bool;
  }

  // Try to register a pointer-typed local (`int *q = ...`) into pointerLocals
  // so its uses can be rewritten to `base[offset (+k)]`. Returns true if the
  // local was registered (caller must then SKIP emitting any GLSL decl for
  // it — a plain `int q = ...` would be a broken scalar). Returns false for
  // non-pointer locals or pointer inits we don't model (those emit normally,
  // likely producing a glslc error, preserving prior behavior).
  bool tryRegisterPointerLocal(const VarDecl *d) {
    if (!d || !d->type || d->type->getKind() != TypeKind::Pointer || !d->init)
      return false;
    auto set = [&](const std::string &base, const std::string &off) {
      pointerLocals[d->name] = {base, off};
    };
    // Resolve the GLSL base name for a DeclRefExpr operand: an SSBO kernel
    // param (its glslName is the array name), another pointer local (inherit
    // its base+offset), or a local array name.
    auto baseNameOf = [&](const ASTNode *n, std::string &out) -> bool {
      if (!n || n->getNodeType() != ASTNode::NodeKind::DeclRefExpr) return false;
      StringRef nm = static_cast<const DeclRefExpr *>(n)->name;
      if (auto p = pointerLocals.find(nm); p != pointerLocals.end()) {
        // Inherit: q = p  =>  q uses p's base, offset 0 (alias of p's element
        // 0... but for `q = p` as a pointer alias we want the same base+off).
        out = p->second.base;
        return true;
      }
      out = glslName(nm);
      return true;
    };
    auto offsetStr = [&](const ASTNode *n) -> std::string {
      std::string s;
      raw_string_ostream o(s);
      raw_ostream *saved = os;
      os = &o;
      emitExpr(n);
      os = saved;
      o.flush();
      return s;
    };

    const ASTNode *init = d->init.get();
    // `int *q = base;` — bare alias of an SSBO param or local array.
    if (init->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      std::string base;
      if (baseNameOf(init, base)) { set(base, "0"); return true; }
      return false;
    }
    // `int *q = base + off;` — pointer arithmetic.
    if (init->getNodeType() == ASTNode::NodeKind::BinaryExpr) {
      auto *b = static_cast<const BinaryExpr *>(init);
      if (b->op == BinaryOp::Add) {
        std::string base;
        if (baseNameOf(b->lhs.get(), base)) {
          set(base, offsetStr(b->rhs.get()));
          return true;
        }
        if (baseNameOf(b->rhs.get(), base)) {
          set(base, offsetStr(b->lhs.get()));
          return true;
        }
      }
      return false;
    }
    // `int *q = &arr[i];` — address-of indexed element.
    if (init->getNodeType() == ASTNode::NodeKind::UnaryExpr &&
        static_cast<const UnaryExpr *>(init)->op == UnaryOp::AddrOf) {
      const ASTNode *operand = static_cast<const UnaryExpr *>(init)->operand.get();
      if (operand && operand->getNodeType() == ASTNode::NodeKind::IndexExpr) {
        auto *ie = static_cast<const IndexExpr *>(operand);
        std::string base;
        if (baseNameOf(ie->base.get(), base)) {
          set(base, offsetStr(ie->index.get()));
          return true;
        }
      }
      return false;
    }
    return false;
  }

  // Try to register a reference local (`int &r = <lvalue>;`) into refLocals.
  // Returns true if the local was registered (caller must then SKIP emitting any
  // GLSL decl for it — a by-value copy would silently detach the alias). The
  // rewrite target is the GLSL text of the bound lvalue (`x`, `out_[i]`, `a[0]`);
  // anything else is not an lvalue and falls back to a by-value declaration.
  bool tryRegisterRefLocal(const VarDecl *d) {
    if (!d || !d->type || !isReferenceType(d->type) || !d->init) return false;
    const ASTNode *init = d->init.get();
    // Strip an explicitly written `&` (`int &r = &x;`) — same intent.
    if (init->getNodeType() == ASTNode::NodeKind::UnaryExpr &&
        static_cast<const UnaryExpr *>(init)->op == UnaryOp::AddrOf)
      init = static_cast<const UnaryExpr *>(init)->operand.get();
    switch (init->getNodeType()) {
    case ASTNode::NodeKind::DeclRefExpr:
    case ASTNode::NodeKind::IndexExpr:
    case ASTNode::NodeKind::MemberAccessExpr:
      break;
    default:
      // `int &r = a + b;` is not an lvalue binding; emit a by-value copy
      // instead of a hard error rather than rejecting the program.
      return false;
    }
    // Emit the bound lvalue's GLSL text into a buffer (the same redirect trick
    // tryRegisterPointerLocal uses for its offsets). Since DeclRefExpr
    // emission consults refLocals, a chained alias (`int &s = r;`) inherits
    // the earlier substitution for free.
    std::string text;
    raw_string_ostream o(text);
    raw_ostream *saved = os;
    os = &o;
    emitExpr(init);
    os = saved;
    o.flush();
    if (text.empty()) return false;
    refLocals[d->name] = text;
    return true;
  }

  // Emit the initializer part of a declarator (` = <expr>`), OR, for a multi-
  // dimensional array with an InitListExpr initializer, defer element-wise
  // assignments into `deferredArrInit` (flushed by the caller after the
  // declaration statement). glslc rejects C-style `{}` init for multi-dim
  // arrays (`int b[2][3] = {{1,2,3},{4,5,6}}` is a hard error), so those are
  // lowered as `b[i][j] = v;` assignments. Single-dim array `{}` init and all
  // scalar inits use the normal `= expr` form, which glslc accepts.
  std::string deferredArrInit;
  void emitVarInit(const VarDecl *d, unsigned indent) {
    if (!d->init) return;
    if (d->arrayDims.size() > 1 &&
        d->init->getNodeType() == ASTNode::NodeKind::InitListExpr) {
      SmallVector<const ASTNode *, 16> flat;
      flattenInitList(static_cast<const InitListExpr *>(d->init.get()), flat);
      int64_t total = 1;
      for (int64_t dim : d->arrayDims) total *= dim;
      // Redirect emission into a string buffer so the element-wise assignments
      // land after the declaration statement (the caller flushes
      // deferredArrInit after `;`). pad()/emitExpr() both write through `os`.
      raw_string_ostream buf(deferredArrInit);
      raw_ostream *saved = os;
      os = &buf;
      for (size_t i = 0; i < flat.size() && (int64_t)i < total; ++i) {
        pad(indent);
        (*os) << glslName(d->name);
        // Row-major linear index -> per-dimension subscripts. The last
        // dimension varies fastest, so peel it off first; store subscripts
        // dimension-major (subs[0] is outermost) for `[subs[0]][subs[1]]...`.
        int64_t rem = (int64_t)i;
        SmallVector<int64_t, 4> subs(d->arrayDims.size());
        for (int d2 = (int)d->arrayDims.size() - 1; d2 >= 0; --d2) {
          subs[d2] = rem % d->arrayDims[d2];
          rem /= d->arrayDims[d2];
        }
        for (int64_t s : subs)
          (*os) << "[" << s << "]";
        (*os) << " = ";
        emitExpr(flat[i]);
        (*os) << ";\n";
      }
      os = saved;
      buf.flush();
      return;
    }
    (*os) << " = ";
    emitExpr(d->init.get());
  }

  // Recursively flatten a (possibly nested) InitListExpr into a flat list of
  // scalar/ctor-call elements. Only InitListExpr children are descended into;
  // any other element (IntegerLiteral, FloatLiteral, CallExpr, ...) is kept as
  // a leaf. This turns `{{1,2,3},{4,5,6}}` into `{1,2,3,4,5,6}` for GLSL's
  // multi-dim array initializer, while leaving `float4(1,2,3,4)` constructor
  // calls intact.
  void flattenInitList(const InitListExpr *il,
                       SmallVectorImpl<const ASTNode *> &out) const {
    for (const auto &e : il->elements) {
      if (!e) continue;
      if (e->getNodeType() == ASTNode::NodeKind::InitListExpr)
        flattenInitList(static_cast<const InitListExpr *>(e.get()), out);
      else
        out.push_back(e.get());
    }
  }

  // Emit a condition expression coerced to bool. C/CUDA accept any scalar as a
  // truth value (`if (n & 1)`, `while (n)`, `(a & 1) ? x : y`, `!a`); GLSL
  // requires a genuine bool, so a condition whose type is not provably bool is
  // wrapped in `bool(...)`. exprType is conservative (returns null for many
  // int-typed expressions like `a & 1`), so null is treated as "needs coercion"
  // — wrapping a real bool in `bool(...)` is a harmless no-op constructor.
  void emitCondition(const ASTNode *n) {
    if (!n) { emitExpr(n); return; }
    const Type *et = resolveTypedef(exprType(n));
    if (!isBoolTy(et)) {
      (*os) << "bool(";
      emitExpr(n);
      (*os) << ")";
      return;
    }
    emitExpr(n);
  }

  // Emit `n` wrapped in `glslType(target)(...)` unless its inferred type already
  // matches target (or is unknown — emit as-is, the caller took the risk).
  void emitExprCoerced(const ASTNode *n, const Type *target) {
    if (!n) { emitExpr(n); return; }
    const Type *et = resolveTypedef(exprType(n));
    const Type *tt = resolveTypedef(target);
    if (et && tt && sameScalarType(et, tt)) { emitExpr(n); return; }
    (*os) << glslType(target) << "(";
    emitExpr(n);
    (*os) << ")";
  }
  static bool sameScalarType(const Type *a, const Type *b) {
    if (a->getKind() != b->getKind()) return false;
    if (a->getKind() == TypeKind::Builtin)
      return static_cast<const BuiltinType *>(a)->builtin ==
             static_cast<const BuiltinType *>(b)->builtin;
    if (a->getKind() == TypeKind::Vector) {
      auto *va = static_cast<const VectorType *>(a);
      auto *vb = static_cast<const VectorType *>(b);
      return va->count == vb->count && sameScalarType(va->elem, vb->elem);
    }
    return false;
  }

  // Emit a binary expression, inserting constructor casts so both operands are
  // the common promoted type. GLSL's GL_EXT_shader_explicit_arithmetic_types
  // forbids implicit mixed-width arithmetic (`float16_t * int` is a hard
  // error), so we apply C's usual arithmetic conversions explicitly: the
  // operand whose inferred type differs from the common type is wrapped in
  // `commonType(...)`. Assignment uses the LHS type as the target.
  void emitBinary(const BinaryExpr *b) {
    (*os) << "(";
    if (b->op == BinaryOp::Assign) {
      // Coerce the RHS to the LHS type ONLY when the LHS is float16 and the RHS
      // is a different numeric type — GLSL accepts ordinary narrowing assigns
      // (`float x = 1.0;`), but `float16_t h = 2;` (int) / `= 1.0;` (double)
      // is a hard error under explicit arithmetic types. Wrap the RHS in a
      // float16_t(...) constructor in that case.
      emitExpr(b->lhs.get());
      (*os) << " " << binopStr(b->op) << " ";
      const Type *lt = resolveTypedef(exprType(b->lhs.get()));
      const Type *rt = resolveTypedef(exprType(b->rhs.get()));
      if (lt && isHalfTy(lt) && !(rt && isHalfTy(rt)) && rt &&
          (isFloatType(rt) || isIntType(rt) ||
           rt->getKind() == TypeKind::Vector))
        emitExprCoerced(b->rhs.get(), lt);
      else
        emitExpr(b->rhs.get());
    } else {
      const Type *lt = exprType(b->lhs.get());
      const Type *rt = exprType(b->rhs.get());
      const Type *common = promoteTypes(lt, rt);
      if (common) {
        emitExprCoerced(b->lhs.get(), common);
        (*os) << " " << binopStr(b->op) << " ";
        emitExprCoerced(b->rhs.get(), common);
      } else {
        emitExpr(b->lhs.get());
        (*os) << " " << binopStr(b->op) << " ";
        emitExpr(b->rhs.get());
      }
    }
    (*os) << ")";
  }

  void emitExpr(const ASTNode *n) {
    if (!n) { (*os) << "/*null*/"; return; }
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral:
      (*os) << static_cast<const IntegerLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::FloatLiteral:
      (*os) << static_cast<const FloatLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::BoolLiteral:
      (*os) << (static_cast<const BoolLiteral *>(n)->value ? "true" : "false");
      break;
    case ASTNode::NodeKind::SizeOfExpr: {
      // GLSL has no sizeof. Sema folded the byte count into `folded` during
      // analysis; emit it as a bare integer literal. If folding failed (0),
      // emit 0 so the expression is still well-formed.
      const auto *s = static_cast<const SizeOfExpr *>(n);
      (*os) << s->folded;
      break;
    }
    case ASTNode::NodeKind::CharLiteral:
      // GLSL has no char; lower as the int code point (C promotes char to int).
      (*os) << static_cast<const CharLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::StringLiteral: {
      // GLSL has no string type generally, but debugPrintfEXT (kernel printf)
      // takes a string-literal format string as its first argument. Re-emit the
      // decoded string as a GLSL string literal. The parser already decoded
      // escapes (\n -> newline byte), so re-escape control/quote/backslash so
      // the literal stays on one line and is valid GLSL.
      const std::string &s =
          static_cast<const vc::StringLiteral *>(n)->value;
      (*os) << "\"";
      for (unsigned char ch : s) {
        switch (ch) {
        case '\\': (*os) << "\\\\"; break;
        case '"':  (*os) << "\\\""; break;
        case '\n': (*os) << "\\n"; break;
        case '\r': (*os) << "\\r"; break;
        case '\t': (*os) << "\\t"; break;
        default:
          if (ch < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02x", ch);
            (*os) << buf;
          } else {
            (*os) << (char)ch;
          }
        }
      }
      (*os) << "\"";
      break;
    }
    case ASTNode::NodeKind::DeclRefExpr: {
      StringRef name = static_cast<const DeclRefExpr *>(n)->name;
      // CUDA warpSize -> Vulkan subgroup size (runtime value, adapts to
      // hardware; not a fixed 32 literal). gl_SubgroupSize is uint; wrap in
      // int() to match CUDA's int warpSize (and keep arithmetic on it int).
      if (name == "warpSize") { (*os) << "int(gl_SubgroupSize)"; break; }
      // `int &r = x;` — substitute the lvalue `r` was bound to. Textual
      // substitution is safe here because the bound expression is always an
      // lvalue (a name or an index, never a call or a literal), and GLSL has no
      // aliasing construct to express this any other way.
      if (auto rl = refLocals.find(name); rl != refLocals.end()) {
        (*os) << rl->second;
        break;
      }
      // Scalar params live in the push-constant block; qualify them.
      bool isScalar = false;
      for (StringRef s : scalarParams)
        if (s == name) { isScalar = true; break; }
      if (isScalar) (*os) << "pc.";
      (*os) << glslName(name);
      break;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      emitBinary(static_cast<const BinaryExpr *>(n));
      break;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      auto *u = static_cast<const UnaryExpr *>(n);
      // Postfix ++/-- place the operator after the operand; all others prefix.
      if (u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec) {
        emitExpr(u->operand.get());
        (*os) << unaryopStr(u->op);
      } else if (u->op == UnaryOp::LNot) {
        // C's `!` yields an int (0/1); GLSL's `!` yields bool and requires a
        // bool operand. Emit `int(!bool(operand))` so the result is usable as
        // both a scalar (`int x = !a;`) and a condition (emitCondition sees an
        // int and wraps `bool(...)` — redundant but glslc folds it).
        (*os) << "int(!bool(";
        emitExpr(u->operand.get());
        (*os) << "))";
      } else if (u->op == UnaryOp::Deref &&
                 u->operand &&
                 u->operand->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        // `*q` where q is a tracked pointer local -> base[off]. (GLSL has no
        // pointer type; a pointer local is a (base, offset) pair, so a deref
        // is a single-element array index.)
        StringRef bn = static_cast<const DeclRefExpr *>(u->operand.get())->name;
        if (auto p = pointerLocals.find(bn); p != pointerLocals.end()) {
          (*os) << p->second.base << "[" << p->second.offset << "]";
          break;
        }
        (*os) << unaryopStr(u->op);
        emitExpr(u->operand.get());
      } else {
        (*os) << unaryopStr(u->op);
        emitExpr(u->operand.get());
      }
      break;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      (*os) << "(";
      emitCondition(c->cond.get());
      (*os) << " ? ";
      emitExpr(c->thenExpr.get());
      (*os) << " : ";
      emitExpr(c->elseExpr.get());
      (*os) << ")";
      break;
    }
    case ASTNode::NodeKind::CommaExpr: {
      auto *c = static_cast<const CommaExpr *>(n);
      (*os) << "(";
      emitExpr(c->lhs.get());
      (*os) << ", ";
      emitExpr(c->rhs.get());
      (*os) << ")";
      break;
    }
    case ASTNode::NodeKind::CStyleCastExpr: {
      // GLSL functional/constructor cast: T(expr). Works for scalars and
      // vectors (vec4(x), int(x), float(x)).
      auto *c = static_cast<const CStyleCastExpr *>(n);
      (*os) << glslType(c->target) << "(";
      emitExpr(c->sub.get());
      (*os) << ")";
      break;
    }
    case ASTNode::NodeKind::InitListExpr: {
      // GLSL's glslc rejects nested initializer lists for multi-dimensional
      // arrays (`int b[2][3] = {{1,2,3},{4,5,6}}` -> "cannot convert parameter
      // 1 from const int to 3-element array"), but accepts the flattened form
      // (`{1,2,3,4,5,6}`). C permits both, so we flatten nested InitListExprs
      // into a single scalar list. We only descend into child InitListExpr
      // elements; non-InitListExpr elements (scalars, ctor calls like
      // float4(...)) are emitted verbatim, so a vector/struct array init that
      // uses constructor calls is not broken apart.
      auto *il = static_cast<const InitListExpr *>(n);
      SmallVector<const ASTNode *, 8> flat;
      flattenInitList(il, flat);
      (*os) << "{ ";
      for (unsigned i = 0; i < flat.size(); ++i) {
        if (i) (*os) << ", ";
        emitExpr(flat[i]);
      }
      (*os) << " }";
      break;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      // `q[k]` where q is a tracked pointer local -> base[off + k].
      if (ie->base &&
          ie->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        StringRef bn = static_cast<const DeclRefExpr *>(ie->base.get())->name;
        if (auto p = pointerLocals.find(bn); p != pointerLocals.end()) {
          (*os) << p->second.base << "[";
          if (p->second.offset == "0") {
            emitExpr(ie->index.get());
          } else {
            (*os) << "(" << p->second.offset << ") + ";
            emitExpr(ie->index.get());
          }
          (*os) << "]";
          break;
        }
      }
      emitExpr(ie->base.get());
      (*os) << "[";
      emitExpr(ie->index.get());
      (*os) << "]";
      break;
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      // CUDA built-ins -> GLSL equivalents.
      auto *m = static_cast<const MemberAccessExpr *>(n);
      if (m->base &&
          m->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *base = static_cast<const DeclRefExpr *>(m->base.get());
        StringRef b = base->name;
        // gl_*InvocationID/WorkGroup* are uvec3; CUDA indices are int, so
        // wrap in int(...) to match the C-like semantics.
        if (b == "threadIdx") { (*os) << "int(gl_LocalInvocationID." << m->member << ")"; return; }
        if (b == "blockIdx")  { (*os) << "int(gl_WorkGroupID."     << m->member << ")"; return; }
        if (b == "blockDim")  { (*os) << "int(gl_WorkGroupSize."   << m->member << ")"; return; }
        if (b == "gridDim")   { (*os) << "int(gl_NumWorkGroups."   << m->member << ")"; return; }
      }
      // Scope-resolution member access that is NOT a call: `Kind::A`,
      // `ns::CONST`, `Outer::Inner` (type). Lower to the mangled device name
      // (`Kind_A`, `ns_CONST`). A chain like `A::B::C` becomes `A_B_C`.
      if (m->isScope) {
        (*os) << glslName(mangleScopeChainGLSL(m));
        return;
      }
      emitExpr(m->base.get());
      (*os) << "." << glslName(m->member);
      break;
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *ref = static_cast<const DeclRefExpr *>(c->callee.get());
        if (ref->name == "__syncthreads") { (*os) << "barrier()"; return; }
        // CUDA memory fences. CUDA fences only order memory visibility, not
        // execution; they are safe inside divergent control flow (e.g. inside
        // `if (tid == 0)`), which real kernels rely on. GLSL's memoryBarrier*
        // primitives are the direct counterpart — they order memory without
        // synchronizing execution. We must NOT pair them with barrier(): a
        // GLSL barrier() is only legal in uniform control flow, and emitting
        // one here would push it into the divergent branch that fences are
        // typically used in, producing undefined behavior. So lower purely to
        // the matching memory barrier, no execution barrier.
        //   __threadfence_block : block-visible  -> groupMemoryBarrier
        //   __threadfence       : device-visible -> memoryBarrierBuffer (SSBO)
        if (ref->name == "__threadfence_block") {
          (*os) << "groupMemoryBarrier()";
          return;
        }
        if (ref->name == "__threadfence") {
          (*os) << "memoryBarrierBuffer()";
          return;
        }
        // CUDA warp intrinsics -> Vulkan subgroup ops. The CUDA mask argument
        // (active-lane bitmask) has no subgroup equivalent and is dropped.
        if (isWarpIntrinsicName(ref->name)) {
          emitWarpIntrinsic(ref->name, c->args);
          return;
        }
        // __syncthreads_count/and/or: block-wide vote barriers. Each takes a
        // predicate and returns count / logical-and / logical-or of all threads'
        // predicates across the block, synchronizing like __syncthreads. These
        // cannot be a single GLSL expression: the reduction needs every thread
        // to write its predicate to a shared slot, barrier, let thread 0 fold,
        // then barrier again before all read the result. So we hoist that
        // prologue into `preStmts` (flushed by emitStmt before the enclosing
        // statement) and emit a bare reference to the result cell here. The
        // call must sit in uniform control flow (all threads reach it), which
        // matches the requirement that the embedded barriers() be uniform.
        if (ref->name == "__syncthreads_count" ||
            ref->name == "__syncthreads_and" ||
            ref->name == "__syncthreads_or") {
          emitVoteCall(ref->name, c->args);
          return;
        }
        // CUDA atomics: rewrite the CUDA pointer/value model to GLSL's
        // reference model before falling through to generic call emission.
        if (isAtomicName(ref->name)) {
          emitAtomicCall(ref->name, c->args);
          return;
        }
        // VC async-copy approximation: vcMemcpyAsync lowers to a software
        // cooperative copy + barrier (hoisted into preStmts); vcPipeline* are
        // barrier() wrappers. No hardware DMA exists in Vulkan.
        if (isAsyncCopyBuiltin(ref->name)) {
          emitAsyncCopyCall(ref->name, c->args);
          return;
        }
        // User __device__ helper or CUDA/math builtin: lower the name and emit
        // a normal GLSL call expression. Complete omitted trailing defaulted
        // parameters from the callee's signature (default arguments).
        //
        // If Sema resolved this call to a specific overload (resolvedCallee),
        // emit the call under that overload's parameter-mangled device symbol so
        // distinct overloads (f(int)->f_i, f(float)->f_f) hit the right emitted
        // function. Builtins have no resolvedCallee and keep the bare lowered
        // name. For methods called by unqualified name inside the class body,
        // resolvedCallee is the Class_method decl and its mangled name carries
        // the Class_ prefix.
        std::string callName;
        const FunctionDecl *calleeFn = nullptr;
        if (c->resolvedCallee) {
          callName = deviceMangledName(c->resolvedCallee);
          calleeFn = c->resolvedCallee;
        } else {
          callName = lowerBuiltinCall(ref->name);
          auto fit = funcDecls.find(ref->name);
          if (fit != funcDecls.end()) calleeFn = fit->second;
        }
        (*os) << callName << "(";
        for (unsigned i = 0; i < c->args.size(); ++i) {
          if (i) (*os) << ", ";
          emitExpr(c->args[i].get());
        }
        // Append defaults for any trailing params the call omitted.
        if (calleeFn) {
          for (unsigned i = c->args.size(); i < calleeFn->params.size(); ++i) {
            if (calleeFn->params[i]->defaultVal) {
              if (i) (*os) << ", ";
              emitExpr(calleeFn->params[i]->defaultVal.get());
            }
          }
        }
        (*os) << ")";
        return;
      }
      // Callee is a scoped or member-access expression: ns::func(), Class::m(),
      // or obj.method(). Lower to the mangled free-function form.
      if (c->callee && c->callee->getNodeType() ==
                            ASTNode::NodeKind::MemberAccessExpr) {
        auto *ma = static_cast<const MemberAccessExpr *>(c->callee.get());
        if (ma->isScope) {
          // ns::func(args) -> ns_func(args). Class::method(args) -> Class_method(args)
          // (no implicit `this` for scope calls — the caller names the method
          // directly, e.g. a static-like call). When Sema resolved the callee,
          // emit under its parameter-mangled device symbol so overloads bind
          // correctly; otherwise fall back to the bare scope-chain mangle.
          std::string mangled = c->resolvedCallee
                                    ? deviceMangledName(c->resolvedCallee)
                                    : mangleScopeChainGLSL(ma);
          (*os) << mangled << "(";
          emitCallArgsWithDefaults(mangled, c->args);
          (*os) << ")";
          return;
        }
        // obj.method(args) -> Class_method(obj, args). The object expression
        // becomes the first argument (`this`). When Sema resolved the method
        // (resolvedCallee), emit under its parameter-mangled device symbol so
        // overloaded methods bind to the right definition; otherwise fall back
        // to the heuristic name recovery (single-method classes, builtins).
        std::string methodName;
        const FunctionDecl *calleeFn = nullptr;
        if (c->resolvedCallee) {
          methodName = deviceMangledName(c->resolvedCallee);
          calleeFn = c->resolvedCallee;
        } else {
          methodName = memberCallMethodName(ma);
        }
        (*os) << methodName << "(";
        emitExpr(ma->base.get());
        for (auto &a : c->args) {
          (*os) << ", ";
          emitExpr(a.get());
        }
        // Append defaults for omitted trailing params (explicit args only; the
        // implicit `this` is always present and never defaulted).
        if (calleeFn) {
          for (unsigned i = c->args.size(); i < calleeFn->params.size(); ++i) {
            if (calleeFn->params[i]->defaultVal) {
              (*os) << ", ";
              emitExpr(calleeFn->params[i]->defaultVal.get());
            }
          }
        }
        (*os) << ")";
        return;
      }
      // Callee is not a simple name reference — emit a placeholder.
      (*os) << "/*call*/";
      break;
    }
    default:
      (*os) << "/*?*/";
      break;
    }
  }

  const char *binopStr(BinaryOp op) {
    switch (op) {
    case BinaryOp::Add: return "+";
    case BinaryOp::Sub: return "-";
    case BinaryOp::Mul: return "*";
    case BinaryOp::Div: return "/";
    case BinaryOp::Mod: return "%";
    case BinaryOp::Assign: return "=";
    case BinaryOp::Eq: return "==";
    case BinaryOp::NEq: return "!=";
    case BinaryOp::Lt: return "<";
    case BinaryOp::Gt: return ">";
    case BinaryOp::Le: return "<=";
    case BinaryOp::Ge: return ">=";
    case BinaryOp::Shl: return "<<";
    case BinaryOp::Shr: return ">>";
    case BinaryOp::And: return "&";
    case BinaryOp::Or: return "|";
    case BinaryOp::Xor: return "^";
    case BinaryOp::LAnd: return "&&";
    case BinaryOp::LOr: return "||";
    default: return "?";
    }
  }

  const char *unaryopStr(UnaryOp op) {
    switch (op) {
    case UnaryOp::Neg: return "-";
    case UnaryOp::Not: return "~";
    case UnaryOp::LNot: return "!";
    case UnaryOp::Deref: return "*";
    case UnaryOp::AddrOf: return "&";
    case UnaryOp::PreInc:
    case UnaryOp::PostInc: return "++";
    case UnaryOp::PreDec:
    case UnaryOp::PostDec: return "--";
    default: return "?";
    }
  }
};

} // namespace

std::vector<glsl::GLSLModule> vc::glsl::translateASTToGLSLSources(const TranslationUnit &tu) {
  return GLSLEmitter(nulls()).emitAll(tu);
}

bool vc::glsl::translateASTToGLSL(const TranslationUnit &tu,
                                  raw_ostream &os) {
  auto mods = translateASTToGLSLSources(tu);
  if (mods.empty()) {
    os << "// no __global__ kernel found\n";
    return false;
  }
  // Single-kernel convenience path: emit the first kernel's source verbatim.
  os << mods.front().source;
  return true;
}
