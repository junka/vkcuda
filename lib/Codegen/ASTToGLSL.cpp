//===- ASTToGLSL.cpp - Emit GLSL compute shader from AST -----------------===//

#include "vc/Codegen/ASTToGLSL.h"

#include "vc/Frontend/AST.h"
#include "vc/Frontend/BuiltinRegistry.h"
#include "vc/Frontend/Mangle.h"

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
  // Hoisted statements to emit before the current enclosing statement. The
  // vote intrinsics push their reduction prologue here; emitStmt flushes and
  // clears this before emitting each statement node.
  std::string preStmts;
  // During emitStmt's real emit pass, vote calls emit only the value ref
  // (their prologue was already flushed by the pre-pass). During the pre-pass
  // this is false and vote calls push the prologue into preStmts.
  bool emittingVoteRef = false;

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
    preStmts.clear();
    emittingVoteRef = false;
    if (k->body) scanDims(k->body.get());
    scanDoubleUsage(k);
    // A printf in a __device__ helper also needs the extension. Walk every
    // device function body reachable from this kernel's translation unit.
    for (const auto &kv : funcDecls)
      if (kv.second->body) scanDims(kv.second->body.get());

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
      if (f->params[i]->isConst) (*os) << "const ";
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
        {"half", "f16vec"},
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
        first = false;
        (*os) << glslType(d->type) << " " << glslName(d->name);
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
      auto *d = static_cast<const DeclStmt *>(n)->decl;
      if (!d) break;
      if (d->isShared) break; // hoisted to a `shared` global
      pad(indent);
      if (d->isConst) (*os) << "const ";
      (*os) << glslType(d->type) << " " << glslName(d->name);
      for (int64_t dim : d->arrayDims)
        (*os) << "[" << dim << "]";
      if (d->init) { (*os) << " = "; emitExpr(d->init.get()); }
      // Additional declarators sharing this statement's type (`int a, b;`).
      // GLSL allows comma-separated declarations in the same statement.
      for (unsigned i = 1; i < static_cast<const DeclStmt *>(n)->decls.size();
           ++i) {
        VarDecl *vd = static_cast<const DeclStmt *>(n)->decls[i];
        (*os) << ", ";
        if (vd->isConst) (*os) << "const ";
        (*os) << glslType(vd->type) << " " << glslName(vd->name);
        for (int64_t dim : vd->arrayDims)
          (*os) << "[" << dim << "]";
        if (vd->init) { (*os) << " = "; emitExpr(vd->init.get()); }
      }
      (*os) << ";\n";
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
      pad(indent); (*os) << "if ("; emitExpr(iff->cond.get()); (*os) << ") {\n";
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
      if (fs->cond) emitExpr(fs->cond.get());
      (*os) << "; ";
      if (fs->step) emitExpr(fs->step.get());
      (*os) << ") {\n";
      if (fs->body) emitStmt(fs->body.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      pad(indent); (*os) << "while ("; emitExpr(ws->cond.get()); (*os) << ") {\n";
      if (ws->body) emitStmt(ws->body.get(), indent + 1);
      pad(indent); (*os) << "}\n";
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      pad(indent); (*os) << "do {\n";
      if (ds->body) emitStmt(ds->body.get(), indent + 1);
      pad(indent); (*os) << "} while ("; emitExpr(ds->cond.get()); (*os) << ");\n";
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
      // Scalar params live in the push-constant block; qualify them.
      bool isScalar = false;
      for (StringRef s : scalarParams)
        if (s == name) { isScalar = true; break; }
      if (isScalar) (*os) << "pc.";
      (*os) << glslName(name);
      break;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      (*os) << "(";
      emitExpr(b->lhs.get());
      (*os) << " " << binopStr(b->op) << " ";
      emitExpr(b->rhs.get());
      (*os) << ")";
      break;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      auto *u = static_cast<const UnaryExpr *>(n);
      // Postfix ++/-- place the operator after the operand; all others prefix.
      if (u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec) {
        emitExpr(u->operand.get());
        (*os) << unaryopStr(u->op);
      } else {
        (*os) << unaryopStr(u->op);
        emitExpr(u->operand.get());
      }
      break;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      (*os) << "(";
      emitExpr(c->cond.get());
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
      auto *il = static_cast<const InitListExpr *>(n);
      (*os) << "{ ";
      for (unsigned i = 0; i < il->elements.size(); ++i) {
        if (i) (*os) << ", ";
        emitExpr(il->elements[i].get());
      }
      (*os) << " }";
      break;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
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
        (*os) << lowerBuiltinCall(ref->name) << "(";
        for (unsigned i = 0; i < c->args.size(); ++i) {
          if (i) (*os) << ", ";
          emitExpr(c->args[i].get());
        }
        // Append defaults for any trailing params the call omitted.
        auto fit = funcDecls.find(ref->name);
        if (fit != funcDecls.end()) {
          const FunctionDecl *calleeFn = fit->second;
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
          // directly, e.g. a static-like call).
          std::string mangled = mangleScopeChainGLSL(ma);
          (*os) << mangled << "(";
          emitCallArgsWithDefaults(mangled, c->args);
          (*os) << ")";
          return;
        }
        // obj.method(args) -> Class_method(obj, args). The object expression
        // becomes the first argument (`this`). The Class name is recovered from
        // the object's type when possible; otherwise we look for a method
        // registered under "<base>_<member>".
        std::string methodName = memberCallMethodName(ma);
        (*os) << methodName << "(";
        emitExpr(ma->base.get());
        for (auto &a : c->args) {
          (*os) << ", ";
          emitExpr(a.get());
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
