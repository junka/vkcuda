//===- ASTToGLSL.cpp - Emit GLSL compute shader from AST -----------------===//

#include "vc/Codegen/ASTToGLSL.h"

#include "vc/Frontend/AST.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>

using namespace vc;
using namespace llvm;

namespace {

class GLSLEmitter {
  raw_ostream &os;
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

public:
  GLSLEmitter(raw_ostream &o) : os(o) {}

  bool emit(const TranslationUnit &tu) {
    // Find the first __global__ function.
    const FunctionDecl *fn = nullptr;
    for (auto &d : tu.decls) {
      if (d->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
        auto *f = static_cast<const FunctionDecl *>(d.get());
        if (f->deviceAttr == DeviceAttr::Global) { fn = f; break; }
      }
    }
    if (!fn) {
      os << "// no __global__ kernel found\n";
      return false;
    }
    kernel = fn;
    params.assign(fn->params.begin(), fn->params.end());
    if (fn->body) scanDims(fn->body.get());

    emitHeader();
    emitStructDecls(tu);
    emitBindings();
    emitSharedDecls();
    emitDeviceFunctions(tu);
    emitBody();
    return true;
  }

private:
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
    os << "#version 460 core\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types : enable\n";
    os << "// generated by vc (GLSL backend)\n";
    if (usesSubgroup) {
      // CUDA warp intrinsics lower to Vulkan subgroup ops, which require
      // SPIR-V 1.3 (vulkan1.1). The driver detects this marker comment and
      // bumps glslc's target env accordingly. The KHR subgroup extensions
      // are split by capability; enable the subsets we actually use.
      os << "// vc:needs-spv1.3\n";
      os << "#extension GL_KHR_shader_subgroup_basic : enable\n";
      os << "#extension GL_KHR_shader_subgroup_ballot : enable\n";
      os << "#extension GL_KHR_shader_subgroup_shuffle : enable\n";
      os << "#extension GL_KHR_shader_subgroup_shuffle_relative : enable\n";
      os << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n";
      os << "#extension GL_KHR_shader_subgroup_vote : enable\n";
    }
    // Workgroup size is a specialization constant so the host can choose the
    // block dimensions at launch time (1D or 2D) without recompiling. Constant
    // IDs: 0 -> local_size_x, 1 -> local_size_y, 2 -> local_size_z. Unused
    // axes default to 1.
    os << "layout(local_size_x_id = 0";
    if (useY) os << ", local_size_y_id = 1";
    if (useZ) os << ", local_size_z_id = 2";
    os << ") in;\n\n";
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
      os << "layout(set = 0, binding = " << bindIdx << ") buffer B" << bindIdx
         << " {\n  " << ty << " " << p->name << "[];\n};\n\n";
      ++bindIdx;
    }
    if (!scalars.empty()) {
      os << "layout(push_constant) uniform PC {\n";
      for (const ParamDecl *p : scalars)
        os << "  " << glslType(p->type) << " " << p->name << ";\n";
      os << "} pc;\n\n";
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
      os << "shared " << ty << " " << v->name;
      // Emit explicit array dimensions if present (e.g. As[16][16]). A declared
      // but unsized dimension (extern __shared__ T s[], parsed as arrayDims={0})
      // defaults to the workgroup x size. A plain scalar (__shared__ int s;)
      // has no arrayDims and must NOT get a trailing [].
      if (!v->arrayDims.empty()) {
        for (int64_t d : v->arrayDims) {
          if (d > 0) os << "[" << d << "]";
          else os << "[gl_WorkGroupSize.x]"; // extern __shared__ T s[]
        }
      }
      if (v->init) { os << " = "; emitExpr(v->init.get()); }
      os << ";\n";
    }
    if (!sharedVars.empty()) os << "\n";
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
  // helper is declared before use (CUDA source is typically top-down).
  void emitDeviceFunctions(const TranslationUnit &tu) {
    for (auto &d : tu.decls) {
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr != DeviceAttr::Device) continue;
      emitFunction(f);
    }
  }

  // Emit every top-level `struct Name { ... };` as a GLSL struct definition,
  // before any function body that uses the type. GLSL requires types be
  // declared before use, so all structs are emitted up front regardless of
  // source order (mutual references via pointers are uncommon in kernels).
  void emitStructDecls(const TranslationUnit &tu) {
    for (auto &d : tu.decls) {
      if (d->getNodeType() != ASTNode::NodeKind::StructDecl) continue;
      auto *sd = static_cast<const StructDecl *>(d.get());
      os << "struct " << sd->name << " {\n";
      for (const FieldDecl *fd : sd->fields) {
        os << "  " << glslType(fd->type) << " " << fd->name;
        for (int64_t dim : fd->arrayDims)
          os << "[" << dim << "]";
        os << ";\n";
      }
      os << "};\n\n";
    }
  }

  // Emit one function signature + body (used for __device__ helpers).
  void emitFunction(const FunctionDecl *f) {
    os << glslType(f->returnType) << " " << f->name << "(";
    for (unsigned i = 0; i < f->params.size(); ++i) {
      if (i) os << ", ";
      os << glslType(f->params[i]->type) << " " << f->params[i]->name;
    }
    os << ") {\n";
    if (f->body &&
        f->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(f->body.get());
      for (auto &s : cs->statements)
        emitStmt(s.get(), 1);
    }
    os << "}\n\n";
  }

  // Map a CUDA/math builtin name to its GLSL equivalent. Returns the name
  // unchanged if no mapping is known (lets user helpers and GLSL builtins
  // pass through verbatim). CUDA single-precision math intrinsics (__sinf,
  // __expf, ...) lower to GLSL's float math builtins.
  StringRef lowerBuiltinCall(StringRef name) {
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
    return name == "atomicAdd" || name == "atomicSub" || name == "atomicExch" ||
           name == "atomicMin" || name == "atomicMax" || name == "atomicInc" ||
           name == "atomicDec" || name == "atomicCAS" || name == "atomicAnd" ||
           name == "atomicOr" || name == "atomicXor";
  }

  // --- CUDA warp -> Vulkan subgroup lowering --------------------------------
  // CUDA's warp model (fixed 32 lanes) maps to Vulkan subgroups (variable
  // size). warpSize becomes gl_SubgroupSize so kernels adapt to the hardware.
  // The __shfl_*/__ballot_sync/etc intrinsics map to subgroup* functions;
  // CUDA's mask argument (the active-lane bitmask) has no subgroup equivalent
  // and is dropped — subgroup ops act on the currently-active invocations,
  // which matches the common `mask = 0xffffffff` usage.
  static bool isWarpIntrinsicName(StringRef name) {
    return name == "__syncwarp" || name == "__ballot_sync" ||
           name == "__anySync" || name == "__allSync" || name == "__activemask" ||
           name == "__shfl_sync" || name == "__shfl_up_sync" ||
           name == "__shfl_down_sync" || name == "__shfl_xor_sync";
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
      os << "[0]";
      return;
    }
    emitExpr(target);
  }

  void emitAtomicCall(StringRef name, const std::vector<NodePtr> &args) {
    if (args.empty()) { os << lowerAtomicName(name) << "()"; return; }
    // atomicInc(a) / atomicDec(a) -> atomicAdd(a, 1) / atomicSub(a, 1).
    if (name == "atomicInc" || name == "atomicDec") {
      os << (name == "atomicInc" ? "atomicAdd" : "atomicSub") << "(";
      emitAtomicTarget(args[0].get());
      os << ", 1)";
      return;
    }
    os << lowerAtomicName(name) << "(";
    emitAtomicTarget(args[0].get());
    for (unsigned i = 1; i < args.size(); ++i) {
      os << ", ";
      emitExpr(args[i].get());
    }
    os << ")";
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
      os << "subgroupBarrier()";
      return;
    }
    if (name == "__activemask") {
      os << "subgroupBallot(true).x";
      return;
    }
    if (name == "__ballot_sync") {
      // args[0]=mask (dropped), args[1]=predicate
      os << "subgroupBallot(bool(";
      emitExpr(args[1].get());
      os << ")).x";
      return;
    }
    if (name == "__anySync") {
      os << "subgroupAny(bool(";
      emitExpr(args[1].get());
      os << "))";
      return;
    }
    if (name == "__allSync") {
      os << "subgroupAll(bool(";
      emitExpr(args[1].get());
      os << "))";
      return;
    }
    // __shfl* family: args[0]=mask (dropped), args[1]=value, args[2]=index/delta,
    // optional args[3]=width (dropped).
    StringRef glslFn = "subgroupShuffle";
    if (name == "__shfl_up_sync") glslFn = "subgroupShuffleUp";
    else if (name == "__shfl_down_sync") glslFn = "subgroupShuffleDown";
    else if (name == "__shfl_xor_sync") glslFn = "subgroupShuffleXor";
    os << glslFn << "(";
    emitExpr(args[1].get());
    os << ", ";
    emitExpr(args[2].get());
    os << ")";
  }


  void emitBody() {
    os << "void main() {\n";
    if (kernel->body &&
        kernel->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(kernel->body.get());
      for (auto &s : cs->statements)
        emitStmt(s.get(), 1);
    }
    os << "}\n";
  }

  void pad(unsigned n) { for (unsigned i = 0; i < n; ++i) os << "  "; }

  // Emit a for-loop initializer clause without the trailing ';' / newline.
  void emitForInit(const ASTNode *n) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::DeclStmt) {
      auto *ds = static_cast<const DeclStmt *>(n);
      bool first = true;
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        if (!first) os << ", ";
        first = false;
        os << glslType(d->type) << " " << d->name;
        for (int64_t dim : d->arrayDims)
          os << "[" << dim << "]";
        if (d->init) { os << " = "; emitExpr(d->init.get()); }
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
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt: {
      pad(indent); os << "{\n";
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        emitStmt(s.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *d = static_cast<const DeclStmt *>(n)->decl;
      if (!d) break;
      if (d->isShared) break; // hoisted to a `shared` global
      pad(indent);
      os << glslType(d->type) << " " << d->name;
      for (int64_t dim : d->arrayDims)
        os << "[" << dim << "]";
      if (d->init) { os << " = "; emitExpr(d->init.get()); }
      // Additional declarators sharing this statement's type (`int a, b;`).
      // GLSL allows comma-separated declarations in the same statement.
      for (unsigned i = 1; i < static_cast<const DeclStmt *>(n)->decls.size();
           ++i) {
        VarDecl *vd = static_cast<const DeclStmt *>(n)->decls[i];
        os << ", " << glslType(vd->type) << " " << vd->name;
        for (int64_t dim : vd->arrayDims)
          os << "[" << dim << "]";
        if (vd->init) { os << " = "; emitExpr(vd->init.get()); }
      }
      os << ";\n";
      break;
    }
    case ASTNode::NodeKind::ExprStmt: {
      pad(indent);
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitExpr(e);
      os << ";\n";
      break;
    }
    case ASTNode::NodeKind::ReturnStmt:
      pad(indent); os << "return";
      if (auto *r = static_cast<const ReturnStmt *>(n)->value.get()) {
        os << " "; emitExpr(r);
      }
      os << ";\n";
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      pad(indent); os << "if ("; emitExpr(iff->cond.get()); os << ") {\n";
      if (iff->thenStmt) emitStmt(iff->thenStmt.get(), indent + 1);
      pad(indent); os << "}\n";
      if (iff->elseStmt) {
        pad(indent); os << "else {\n";
        emitStmt(iff->elseStmt.get(), indent + 1);
        pad(indent); os << "}\n";
      }
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      pad(indent); os << "for (";
      if (fs->init)
        emitForInit(fs->init.get()); // emits without trailing newline
      os << "; ";
      if (fs->cond) emitExpr(fs->cond.get());
      os << "; ";
      if (fs->step) emitExpr(fs->step.get());
      os << ") {\n";
      if (fs->body) emitStmt(fs->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      pad(indent); os << "while ("; emitExpr(ws->cond.get()); os << ") {\n";
      if (ws->body) emitStmt(ws->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      pad(indent); os << "do {\n";
      if (ds->body) emitStmt(ds->body.get(), indent + 1);
      pad(indent); os << "} while ("; emitExpr(ds->cond.get()); os << ");\n";
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
      emitExpr(sw->cond.get());
      os << ") {\n";
      // Emit the case/default labels directly (no extra {} block: GLSL
      // forbids nesting case labels inside a nested compound).
      if (sw->body &&
          sw->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
        auto *cs = static_cast<const CompoundStmt *>(sw->body.get());
        for (auto &s : cs->statements) emitStmt(s.get(), indent + 1);
      } else if (sw->body) {
        emitStmt(sw->body.get(), indent + 1);
      }
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      pad(indent);
      if (cs->value) { os << "case "; emitExpr(cs->value.get()); os << ":\n"; }
      else os << "default:\n";
      if (cs->sub) emitStmt(cs->sub.get(), indent + 1);
      break;
    }
    default:
      break;
    }
  }

  void emitExpr(const ASTNode *n) {
    if (!n) { os << "/*null*/"; return; }
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral:
      os << static_cast<const IntegerLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::FloatLiteral:
      os << static_cast<const FloatLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::CharLiteral:
      // GLSL has no char; lower as the int code point (C promotes char to int).
      os << static_cast<const CharLiteral *>(n)->value;
      break;
    case ASTNode::NodeKind::StringLiteral:
      // GLSL has no string type; emit a placeholder comment + zero so the
      // expression is still well-formed if it ever appears in a value context.
      os << "/*string*/0";
      break;
    case ASTNode::NodeKind::DeclRefExpr: {
      StringRef name = static_cast<const DeclRefExpr *>(n)->name;
      // CUDA warpSize -> Vulkan subgroup size (runtime value, adapts to
      // hardware; not a fixed 32 literal). gl_SubgroupSize is uint; wrap in
      // int() to match CUDA's int warpSize (and keep arithmetic on it int).
      if (name == "warpSize") { os << "int(gl_SubgroupSize)"; break; }
      // Scalar params live in the push-constant block; qualify them.
      bool isScalar = false;
      for (StringRef s : scalarParams)
        if (s == name) { isScalar = true; break; }
      if (isScalar) os << "pc.";
      os << name;
      break;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      os << "(";
      emitExpr(b->lhs.get());
      os << " " << binopStr(b->op) << " ";
      emitExpr(b->rhs.get());
      os << ")";
      break;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      auto *u = static_cast<const UnaryExpr *>(n);
      // Postfix ++/-- place the operator after the operand; all others prefix.
      if (u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec) {
        emitExpr(u->operand.get());
        os << unaryopStr(u->op);
      } else {
        os << unaryopStr(u->op);
        emitExpr(u->operand.get());
      }
      break;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      os << "(";
      emitExpr(c->cond.get());
      os << " ? ";
      emitExpr(c->thenExpr.get());
      os << " : ";
      emitExpr(c->elseExpr.get());
      os << ")";
      break;
    }
    case ASTNode::NodeKind::CStyleCastExpr: {
      // GLSL functional/constructor cast: T(expr). Works for scalars and
      // vectors (vec4(x), int(x), float(x)).
      auto *c = static_cast<const CStyleCastExpr *>(n);
      os << glslType(c->target) << "(";
      emitExpr(c->sub.get());
      os << ")";
      break;
    }
    case ASTNode::NodeKind::InitListExpr: {
      auto *il = static_cast<const InitListExpr *>(n);
      os << "{ ";
      for (unsigned i = 0; i < il->elements.size(); ++i) {
        if (i) os << ", ";
        emitExpr(il->elements[i].get());
      }
      os << " }";
      break;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      emitExpr(ie->base.get());
      os << "[";
      emitExpr(ie->index.get());
      os << "]";
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
        if (b == "threadIdx") { os << "int(gl_LocalInvocationID." << m->member << ")"; return; }
        if (b == "blockIdx")  { os << "int(gl_WorkGroupID."     << m->member << ")"; return; }
        if (b == "blockDim")  { os << "int(gl_WorkGroupSize."   << m->member << ")"; return; }
        if (b == "gridDim")   { os << "int(gl_NumWorkGroups."   << m->member << ")"; return; }
      }
      emitExpr(m->base.get());
      os << "." << m->member;
      break;
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *ref = static_cast<const DeclRefExpr *>(c->callee.get());
        if (ref->name == "__syncthreads") { os << "barrier()"; return; }
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
          os << "groupMemoryBarrier()";
          return;
        }
        if (ref->name == "__threadfence") {
          os << "memoryBarrierBuffer()";
          return;
        }
        // CUDA warp intrinsics -> Vulkan subgroup ops. The CUDA mask argument
        // (active-lane bitmask) has no subgroup equivalent and is dropped.
        if (isWarpIntrinsicName(ref->name)) {
          emitWarpIntrinsic(ref->name, c->args);
          return;
        }
        // __syncthreads_count/and/or are voting barriers: they take a predicate
        // and return the count / logical-and / logical-or of all threads'
        // predicates across the block. GLSL's barrier() returns nothing and a
        // correct lowering needs a shared-array reduction lifted into statement
        // scope (it cannot be expressed as a single GLSL expression). VC does
        // not yet implement the statement-lifting pass this requires, so emit a
        // clear marker rather than silently wrong code.
        if (ref->name == "__syncthreads_count" ||
            ref->name == "__syncthreads_and" ||
            ref->name == "__syncthreads_or") {
          os << "/*VC_UNSUPPORTED:" << ref->name
             << " needs block-wide vote reduction*/";
          return;
        }
        // CUDA atomics: rewrite the CUDA pointer/value model to GLSL's
        // reference model before falling through to generic call emission.
        if (isAtomicName(ref->name)) {
          emitAtomicCall(ref->name, c->args);
          return;
        }
        // User __device__ helper or CUDA/math builtin: lower the name and emit
        // a normal GLSL call expression.
        os << lowerBuiltinCall(ref->name) << "(";
        for (unsigned i = 0; i < c->args.size(); ++i) {
          if (i) os << ", ";
          emitExpr(c->args[i].get());
        }
        os << ")";
        return;
      }
      // Callee is not a simple name reference — emit a placeholder.
      os << "/*call*/";
      break;
    }
    default:
      os << "/*?*/";
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

bool vc::glsl::translateASTToGLSL(const TranslationUnit &tu,
                                  raw_ostream &os) {
  return GLSLEmitter(os).emit(tu);
}
