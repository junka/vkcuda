//===- ASTToMLIR.cpp - Lower AST to VC-dialect MLIR -----------------------===//
//
// Walks the AST and emits MLIR. Device (__global__) functions become
// `func.func` entries wrapped by `vc.kernel`; thread/block indexing and
// barriers map to vc.* ops. Statements/expressions lower to the scf/arith/
// memref dialect "core" the VC dialect sits on top of:
//
//   if/else/conditional  -> scf.if / arith.select
//   for / while          -> scf.for / scf.while
//   arithmetic           -> arith.addi/... (+f for floats)
//   comparison           -> arith.cmpi / arith.cmpf
//   && ||                -> arith.andi / arith.ori (on i1)
//   casts                -> arith.sitofp / fptosi / ... / index_cast
//
// Integer literals and thread/block indices keep a language-level "int" form;
// index type appears only where MLIR structurally requires it (memref
// indexing, scf.for trip counts, block args), bridged with arith.index_cast.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"

#include "vc/Dialect/VC/Ops.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/BuiltinRegistry.h"
#include "vc/Frontend/Mangle.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

using namespace vc;
using namespace mlir;

namespace {

class ASTToMLIRImpl {
  MLIRContext &ctx;
  OpBuilder builder;
  ModuleOp module;
  // name -> func::FuncOp
  // NOTE: keys are std::string (not StringRef) because mangled device names
  // (`ns_func`) are computed strings that outlive no AST node — a StringRef
  // key would dangle once buildFunction returns. Top-level bare names are also
  // stored as std::string for uniformity.
  std::map<std::string, func::FuncOp> funcTable;
  // name -> FunctionDecl (for completing default arguments at call sites that
  // omit trailing defaulted parameters, mirroring the GLSL backend).
  std::map<std::string, const FunctionDecl *> funcDecls;
  // name -> Value (block arg / local memref / alloca)
  llvm::StringMap<Value> locals;
  // name -> the AST-level Type of the variable/param. Needed to resolve struct
  // field access: `result[i].f` requires knowing `result` is a `Result*` so the
  // field's byte offset can be looked up in recordLayouts. Scalars/pointers to
  // builtin types are not consulted, so a missing entry is non-fatal for them.
  llvm::StringMap<const vc::Type *> localTypes;
  // By-value struct parameters (features2's `sumcomp(Vec4 v)`) are scalarized
  // into N field-typed block args (a memref parameter would carry SSBO-pointer
  // semantics and fail to legalize as a by-value value). At the callee entry,
  // those N scalars are stored into a fresh local memref<Nxi32> slot bound to
  // the param name, so the body's `v.c[k]` field access reuses the same
  // structFieldAddr/local-struct path as a local `Vec4 v;` decl. This map
  // records, per param name, the [startArg, count) of the scalarized block args
  // and the field element types, used to materialize the slot at entry.
  struct StructParamSlots {
    unsigned startArg = 0;
    SmallVector<mlir::Type, 4> fieldTys;
  };
  llvm::StringMap<StructParamSlots> structParams;
  // name -> __constant__ global VarDecl, materialized lazily per kernel.
  llvm::StringMap<const VarDecl *> constGlobals;
  // name -> enum constant value. Unscoped enums are flattened at the top level
  // (and inside namespaces, mangled as `ns_NAME`): each EnumDecl's constants
  // are registered here so a bare `TILE`/`ADD`/`ns::CONST` DeclRefExpr folds to
  // an integer literal. GLSL emits `const int NAME=val;` and lets glslc resolve
  // the name; the spirv module has no such decl, so we fold at codegen time.
  llvm::StringMap<int64_t> enumConstants;
  // Persisted namespace-prefix strings so the recursive StringRefs in
  // visitTopLevel stay valid for the lifetime of the walk.
  std::vector<std::unique_ptr<std::string>> nsPrefixStore;
  // Entry block of the function being emitted (for hoisting const slots).
  Block *entryBlock = nullptr;
  // Set when any CUDA warp/subgroup intrinsic (warpSize, __shfl_*, __ballot_sync,
  // __anySync, __allSync, __syncwarp, __activemask) is lowered. The subgroup
  // spirv ops require SPIR-V 1.3 + GroupNonUniform* capabilities; VCToGPU reads
  // the `vc.uses_subgroup` module attr this drives to bump the target env.
  bool usesSubgroup = false;
  //--- Struct layout (for `Result*`-style SSBO field access) -------------//
  // The MLIR backend does not emit struct definitions (spirv.module needs no
  // record type declaration for raw-offset field access). Instead, each
  // top-level StructDecl is recorded as a layout: per-field byte offset (std430
  // packing) + total stride. A `Struct*` kernel arg is scalarized to a
  // memref<?xi8> SSBO; `result[i].field` lowers to a byte-offset store/load at
  // i*stride + fieldOffset. This is enough for the common "struct-as-output-
  // buffer" pattern (vectors.vc's `Result*`); it does not model struct value
  // passing, local structs, or nested records.
  struct FieldLayout {
    std::string name;
    mlir::Type type;
    uint32_t offset;
    std::vector<int64_t> arrayDims; // trailing array dims (e.g. v[2] -> {2})
  };
  struct StructLayout {
    std::vector<FieldLayout> fields;
    uint32_t stride; // sizeof, rounded up to the struct's max field alignment
  };
  llvm::StringMap<StructLayout> recordLayouts;
  // name -> StructDecl, so a method defined before its class body (class.vc's
  // `Accumulator::add` precedes `Class Accumulator { ... }`) can still force the
  // class layout before reading it. Populated as visitTopLevel walks tu.decls.
  llvm::StringMap<const StructDecl *> structDecls;
  const StructDecl *lookupStructDecl(StringRef name) const {
    auto it = structDecls.find(name);
    return it == structDecls.end() ? nullptr : it->second;
  }
  // Return type of the function currently being emitted (None for void).
  mlir::Type currentRetTy;
  // Set when any diagnostic error is emitted, so the driver can fail.
  bool hadError = false;

  //--- Loop break/continue state -----------------------------------------//
  // scf has no goto, so break/continue are modeled by threading two i1 flags
  // through the loop body as memref slots: `breakReq` (a break was requested)
  // and `contReq` (a continue was requested). Every statement in the body is
  // guarded by `if (!breakReq && !contReq)` so code after a break/continue is
  // skipped. The flags are reset to false at the top of each iteration; the
  // loop's while-condition becomes `cond && !breakReq`.
  unsigned loopDepth = 0;
  // Per-active-loop flag slots, indexed by loopDepth-1.
  SmallVector<Value, 4> breakFlags;
  SmallVector<Value, 4> continueFlags;
  // Insertion point to resume at when emitting the step/body guard (set by
  // emitGuardedStmt for the current innermost loop).
  // True while emitting statements that should be suppressed after break/
  // continue; when set, statements emit under an scf.if guard.
  bool guarding = false;
  // Depth of nested structured regions (scf.if then/else, loop bodies). The
  // value-return yield chain can only be entered at function top level
  // (depth 0): inside a region it would have to yield up through the enclosing
  // scf.if, which the current lowering does not model.
  unsigned structuredDepth = 0;

public:
  ASTToMLIRImpl(MLIRContext &c)
      : ctx(c), builder(&c) {
    ctx.getOrLoadDialect<vc::VCDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<memref::MemRefDialect>();
    ctx.getOrLoadDialect<scf::SCFDialect>();
    ctx.getOrLoadDialect<spirv::SPIRVDialect>();
    ctx.getOrLoadDialect<gpu::GPUDialect>();
  }

  ModuleOp translate(const TranslationUnit &tu) {
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module.getBody());
    // First pass: register + lay out every top-level (and nested-in-namespace)
    // StructDecl/class so a method defined before its class body (class.vc's
    // `Accumulator::add` precedes `Class Accumulator { ... }`) can resolve the
    // class layout when buildFunction synthesizes its `_this` parameter.
    for (auto &d : tu.decls)
      preRegisterStructs(d.get(), StringRef());
    for (auto &d : tu.decls) {
      // Pre-register __constant__ globals so device code can read them.
      if (d->getNodeType() == ASTNode::NodeKind::VarDecl) {
        auto *v = static_cast<const VarDecl *>(d.get());
        if (v->isConstant) constGlobals[v->name] = v;
      }
      visitTopLevel(d.get());
    }
    return module;
  }

  // Walk a decl tree collecting StructDecls into structDecls + recordLayouts.
  // Mirrors visitTopLevel's namespace recursion but only registers records.
  void preRegisterStructs(const ASTNode *n, StringRef nsPrefix) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::StructDecl) {
      auto *sd = static_cast<const StructDecl *>(n);
      structDecls[sd->name] = sd;
      recordStructLayout(sd);
      return;
    }
    if (n->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
      auto *ns = static_cast<const NamespaceDecl *>(n);
      std::string prefix =
          nsPrefix.empty() ? ns->name.str() : nsPrefix.str() + "::" + ns->name.str();
      nsPrefixStore.push_back(std::make_unique<std::string>(prefix));
      for (auto &d : ns->decls) preRegisterStructs(d.get(), *nsPrefixStore.back());
    }
  }

  bool failed() const { return hadError; }

  // True if device code referenced a warp/subgroup intrinsic (warpSize,
  // __shfl_*, __ballot_sync, __anySync, __allSync, __syncwarp, __activemask),
  // so the driver can bump the SPIR-V target env to 1.3 + subgroup caps.
  bool usedSubgroup() const { return usesSubgroup; }

private:
  Location loc(const ASTNode *n) {
    return FileLineColLoc::get(&ctx, "<vc>", n ? n->getLoc().line : 0,
                               n ? n->getLoc().col : 0);
  }

  // Report an unsupported/unsupported AST construct at `n` and mark the
  // translation as failed. Returns an empty Value so callers can write
  // `return error(n, "...");` in expression position.
  Value error(const ASTNode *n, const llvm::Twine &msg) {
    hadError = true;
    emitError(loc(n), msg);
    return Value();
  }

  static const char *nodeKindName(ASTNode::NodeKind k) {
    switch (k) {
    case ASTNode::NodeKind::IntegerLiteral: return "integer literal";
    case ASTNode::NodeKind::FloatLiteral: return "float literal";
    case ASTNode::NodeKind::BoolLiteral: return "bool literal";
    case ASTNode::NodeKind::CharLiteral: return "char literal";
    case ASTNode::NodeKind::StringLiteral: return "string literal";
    case ASTNode::NodeKind::DeclRefExpr: return "decl ref";
    case ASTNode::NodeKind::IndexExpr: return "index expr";
    case ASTNode::NodeKind::MemberAccessExpr: return "member access";
    case ASTNode::NodeKind::CallExpr: return "call";
    case ASTNode::NodeKind::UnaryExpr: return "unary expr";
    case ASTNode::NodeKind::BinaryExpr: return "binary expr";
    case ASTNode::NodeKind::ConditionalExpr: return "conditional expr";
    case ASTNode::NodeKind::CStyleCastExpr: return "cast";
    case ASTNode::NodeKind::InitListExpr: return "init list";
    case ASTNode::NodeKind::SizeOfExpr: return "sizeof";
    case ASTNode::NodeKind::LaunchExpr: return "launch";
    default: return "node";
    }
  }

  // Dispatch a top-level (or namespace-nested) declaration. `nsPrefix` is the
  // mangled device namespace prefix (empty at top level; `ns`/`outer_inner`
  // inside namespaces) — used to mangle enum constants and (via stampNamespace
  // on FunctionDecl) device function names so `ns::f`/`ns::CONST` resolve.
  void visitTopLevel(const ASTNode *n, StringRef nsPrefix = StringRef()) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::FunctionDecl) {
      auto *fn = static_cast<const FunctionDecl *>(n);
      // Only device code reaches the shader. Host functions (main, helpers)
      // are emitted by the ASTToHost backend for the host binary instead.
      if (fn->deviceAttr == DeviceAttr::Global ||
          fn->deviceAttr == DeviceAttr::Device)
        buildFunction(fn);
      return;
    }
    // KernelDecl is handled when its FunctionDecl is built (we emit a
    // vc.kernel wrapper there).
    if (n->getNodeType() == ASTNode::NodeKind::KernelDecl) return;
    // Struct definitions are recorded as byte layouts (recordLayouts) so a
    // `Struct*` kernel arg can be scalarized to memref<?xi8> and field access
    // lowered to byte-offset stores/loads. No IR is emitted for the struct
    // itself — the spirv.module needs no record type declaration for raw-offset
    // access. (See cvtType's PointerType<RecordType> handling and the
    // MemberAccessExpr lvalue path.)
    if (n->getNodeType() == ASTNode::NodeKind::StructDecl) {
      auto *sd = static_cast<const StructDecl *>(n);
      structDecls[sd->name] = sd;
      recordStructLayout(sd);
      return;
    }
    // Typedef aliases are resolved at use sites via cvtType (TypedefType branch);
    // no IR is emitted for the declaration itself (mirrors GLSL, which has no
    // typedef — emit resolves to the underlying type spelling).
    if (n->getNodeType() == ASTNode::NodeKind::TypedefDecl)
      return;
    // Enum: register each constant into enumConstants so bare `ADD`/`TILE`
    // DeclRefExprs fold to integer literals at use sites. Inside a namespace
    // the key is mangled `ns_NAME` (mirrors Sema's mangleScoped). No IR is
    // emitted — the spirv module has no enum/const-int decl, so GLSL's
    // `const int NAME=val;` approach has no analogue here.
    if (n->getNodeType() == ASTNode::NodeKind::EnumDecl) {
      auto *e = static_cast<const EnumDecl *>(n);
      for (const auto &c : e->constants) {
        if (nsPrefix.empty())
          enumConstants[c.name] = c.value;
        else
          enumConstants[nsPrefix.str() + "_" + c.name.str()] = c.value;
      }
      return;
    }
    // Namespace: recurse into the body with an extended prefix. Namespaces are
    // transparent on the device side (GLSL flattens them too); each member is
    // visited as if top-level, with names mangled `ns_member`. Nested
    // namespaces chain: `outer::inner::f` -> `outer_inner_f`.
    if (n->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
      auto *ns = static_cast<const NamespaceDecl *>(n);
      std::string inner = nsPrefix.empty()
                              ? ns->name.str()
                              : (nsPrefix.str() + "_" + ns->name.str());
      // Persist the prefix string so the recursive StringRef is stable.
      nsPrefixStore.push_back(std::make_unique<std::string>(std::move(inner)));
      for (auto &d : ns->decls)
        visitTopLevel(d.get(), *nsPrefixStore.back());
      return;
    }
    // __constant__ globals (VarDecl with isConstant) are pre-registered into
    // constGlobals by translate() and read at use sites; no IR is emitted for
    // the declaration here. Non-constant top-level VarDecls (host globals) are
    // not device-visible and are ignored.
    if (n->getNodeType() == ASTNode::NodeKind::VarDecl)
      return;
    // Top-level constructs the MLIR backend does not yet model. These would
    // otherwise vanish silently; surface them so the user knows the shader
    // is missing the construct.
    error(n, std::string("MLIR backend does not support top-level ") +
                nodeKindName(n->getNodeType()) +
                "; construct dropped from device code");
  }

  // Compute a std430-style byte layout for a struct: each field is placed at
  // its natural alignment (rounding the running offset up), and the struct
  // stride is the total size rounded up to the max field alignment. f32/i32
  // align to 4, f64/i64 to 8. Array fields (e.g. `float v[2]`) use the element
  // type's alignment times the element count. This mirrors how the GLSL
  // backend lays out `struct Result { float f; int i; int u; double d; int64_t l; }`
  // under std430 (f@0, i@4, u@8, d@16, l@24, stride=32) so the host's
  // sizeof(Result) matches.
  void recordStructLayout(const StructDecl *sd) {
    if (recordLayouts.count(sd->name)) return;
    StructLayout layout;
    uint32_t off = 0;
    uint32_t maxAlign = 1;
    for (const FieldDecl *f : sd->fields) {
      mlir::Type fty = cvtType(f->type);
      if (!fty) {
        // Unsupported field type (e.g. a nested struct): skip but keep the
        // struct resolvable; field access to it will error at use site.
        continue;
      }
      uint32_t align = fieldAlignment(fty);
      uint32_t size = fieldSize(fty, f->arrayDims);
      off = (off + align - 1) & ~(align - 1);
      layout.fields.push_back({f->name.str(), fty, off, f->arrayDims});
      off += size;
      if (align > maxAlign) maxAlign = align;
    }
    layout.stride = (off + maxAlign - 1) & ~(maxAlign - 1);
    if (layout.stride == 0) layout.stride = maxAlign;
    recordLayouts[sd->name] = std::move(layout);
  }

  // Alignment (bytes) of an MLIR element type under std430.
  uint32_t fieldAlignment(mlir::Type ty) {
    if (auto vty = ty.dyn_cast<mlir::VectorType>())
      ty = vty.getElementType();
    if (ty.isIntOrFloat())
      return ty.getIntOrFloatBitWidth() / 8;
    return 4;
  }
  // Size (bytes) of a field, accounting for trailing array dims.
  uint32_t fieldSize(mlir::Type ty, const std::vector<int64_t> &arrayDims) {
    if (auto vty = ty.dyn_cast<mlir::VectorType>())
      ty = vty.getElementType();
    uint32_t elem = ty.isIntOrFloat() ? (ty.getIntOrFloatBitWidth() / 8) : 4;
    uint32_t total = elem;
    for (int64_t d : arrayDims) total *= (d > 0 ? (uint32_t)d : 1);
    return total;
  }

  mlir::Type cvtType(const vc::Type *t) {
    if (!t) return mlir::Type();
    // typedef alias: resolve to the underlying type recursively. Sema keeps the
    // TypedefType on params/locals (e.g. `weight_t wgt`); without this, buildFunction
    // would get a null arg type and crash. Mirrors Sema's resolveTypedefs.
    if (isa<vc::TypedefType>(t))
      return cvtType(cast<vc::TypedefType>(t)->decl->underlying);
    if (isa<BuiltinType>(t)) {
      switch (cast<BuiltinType>(t)->builtin) {
      case BuiltinTypeKind::Void: return builder.getNoneType();
      // Bool lowers to i32, not i1: SPIR-V has no storage for 1-bit values, so
      // a `memref<i1, Function>` slot (a bool local) fails to legalize its
      // memref.store through GPUToSPIRV. Representing bool as i32 everywhere
      // (slot, parameter, literal) keeps storage legal; truthiness is recovered
      // via toI1 (cmpi != 0) at branch/logical-op points, and i1 comparison
      // results are widened back to i32 on store via castValue (extsi).
      case BuiltinTypeKind::Bool: return builder.getI32Type();
      case BuiltinTypeKind::Int32: case BuiltinTypeKind::UInt32:
        return builder.getI32Type();
      case BuiltinTypeKind::Int64: case BuiltinTypeKind::UInt64:
        return builder.getI64Type();
      case BuiltinTypeKind::Float16: return builder.getF16Type();
      case BuiltinTypeKind::Float32: return builder.getF32Type();
      case BuiltinTypeKind::Float64: return builder.getF64Type();
      }
    }
    // CUDA-style vector types (float4, int3, ...): a vector lives only as an
    // SSA register value, never as a memref element (GPUToSPIRV cannot legalize
    // memref.load/store on memref<?xvector<...>>). A vector *local* becomes a
    // 0-d memref<vector<...>> Function-storage slot — that shape DOES legalize
    // (the existing scalar local path uses the same 0-d memref form). A vector
    // *pointer* (float4*) is scalarized to memref<?xELEM> in the PointerType
    // branch below, so vectors never appear as SSBO element types.
    if (isa<vc::VectorType>(t)) {
      auto *v = cast<vc::VectorType>(t);
      mlir::Type elem = cvtType(v->elem);
      if (!elem) return mlir::Type();
      return mlir::VectorType::get({v->count}, elem);
    }
    // A bare struct value (local variable, or a non-pointer parameter): a local
    // `Accumulator acc;` / `Vec4 v;` / `VcPipeline pipe;` becomes a flat
    // memref<Nxi32, Function> slot — N = stride/4 i32 slots — reusing the same
    // byte-offset->slot math (structFieldAddr/loadStructField/storeStructField)
    // as the Pointer<RecordType> SSBO path below, only with no index*stride
    // term (one struct, not an array of them). This mirrors how the GLSL backend
    // treats a local struct as an ordinary variable. By-value struct *function
    // parameters* are NOT handled here (memref parameters carry SSBO-pointer
    // semantics); buildFunction scalarizes them into N field-typed args. The
    // three struct-value demos (features2/class/async_copy) all use 4-byte
    // fields, so i32 granularity covers them directly with no lo/hi splitting.
    if (isa<RecordType>(t)) {
      auto *sd = cast<RecordType>(t)->decl;
      auto it = recordLayouts.find(sd->name);
      if (it == recordLayouts.end()) return mlir::Type();
      int64_t slots = ((int64_t)it->second.stride + 3) / 4; // i32 slots
      if (slots < 1) slots = 1;
      return MemRefType::get({slots}, builder.getI32Type());
    }
    if (isa<PointerType>(t)) {
      // pointer-to-T  ->  memref<?xT> in the StorageBuffer (global device
      // memory) storage class; MemRefToSPIRV requires a SPIR-V storage class
      // attribute (not a numeric memory space) on every memref it lowers.
      // The 1-D dynamic memref gets a static strided<[1], offset: 0> layout,
      // which getVulkanElementPtr needs to emit an element pointer (it refuses
      // dynamic strides/offsets).
      //
      // A pointer-to-vector (float4*) is scalarized: the memref element type is
      // the vector's *element* type (f32), not the vector itself. memref<
      // ?xvector<4xf32>> would fail to legalize its loads/stores through
      // GPUToSPIRV; memref<?xf32> with manual 4-wide CompositeConstruct/
      // CompositeExtract at access sites is the legal path. (Full vector-
      // pointer dereference — `float4* p; p[i]` — is not yet implemented; the
      // current demos only dereference struct pointers and scalar pointers.)
      //
      // A pointer-to-struct (Result*) is scalarized to memref<?xi32>: the
      // struct has no MLIR type (we don't emit a spirv struct), and field
      // access is byte-offset stores/loads keyed on recordLayouts. i32 (4-byte)
      // granularity covers f32/i32 fields directly; i64/f64 fields are split
      // into lo/hi i32 stores via shift+mask (see storeStructField/loadStructField).
      // The runtime binds the whole device buffer as one raw SSBO, so any
      // element width works as long as the shader's std430 field offsets match
      // the host C++ struct layout (they do — both follow the same rules).
      const vc::Type *pointeeTy = cast<PointerType>(t)->pointee;
      mlir::Type pointee;
      if (isa<RecordType>(pointeeTy)) {
        // Struct pointer: use i32 as the granular element type (4-byte units).
        // recordLayouts holds byte offsets; access sites divide by 4 to get the
        // i32 slot index. i64/f64 fields span two slots and are split.
        pointee = builder.getI32Type();
      } else {
        pointee = cvtType(pointeeTy);
        if (auto vty = pointee.dyn_cast<mlir::VectorType>()) {
          pointee = vty.getElementType();
        }
      }
      auto layout = mlir::StridedLayoutAttr::get(
          &ctx, /*offset=*/0, ArrayRef<int64_t>{/*stride=*/1});
      return MemRefType::get(
          ArrayRef<int64_t>{ShapedType::kDynamic}, pointee, layout,
          spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::StorageBuffer));
    }
    return mlir::Type();
  }

  // Collect the flat scalar MLIR types of a struct's fields, expanding array
  // fields element-by-element (e.g. `struct Vec4 { float c[4]; }` -> [f32,f32,
  // f32,f32]). Used to scalarize a by-value struct function parameter: each
  // element becomes its own block arg, sidestepping the SSBO-pointer semantics
  // a memref parameter would carry. Returns empty if the struct isn't laid out
  // or has an unsupported field (the caller then leaves the param unexpanded).
  SmallVector<mlir::Type, 4> structFieldScalarTypes(const RecordType *rec) {
    SmallVector<mlir::Type, 4> out;
    if (!rec || !rec->decl) return out;
    auto it = recordLayouts.find(rec->decl->name);
    if (it == recordLayouts.end()) return out;
    for (const FieldLayout &f : it->second.fields) {
      if (f.arrayDims.empty()) {
        out.push_back(f.type);
      } else {
        int64_t n = 1;
        for (int64_t d : f.arrayDims) n *= (d > 0 ? d : 1);
        for (int64_t i = 0; i < n; ++i) out.push_back(f.type);
      }
    }
    return out;
  }

  // At a call site, a by-value struct argument (`sumcomp(v)` where `v` is a
  // local `Vec4`) must be expanded into N scalar field values matching the
  // callee's scalarized signature. Reads each field's i32 slot from the local
  // struct memref and bitcasts back to the field type (mirroring loadStructField
  // for the <=4-byte case). Returns true and appends to `out` on success; false
  // if `arg` isn't a local struct value (the caller falls back to a plain arg).
  bool expandStructArg(ASTNode *arg, SmallVectorImpl<Value> &out, Location l) {
    if (!arg || arg->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *ref = static_cast<DeclRefExpr *>(arg);
    auto tit = localTypes.find(ref->name);
    if (tit == localTypes.end()) return false;
    const vc::Type *t = tit->second;
    if (!t || !isa<RecordType>(t)) return false;
    auto *rec = cast<RecordType>(t);
    auto layIt = recordLayouts.find(rec->decl->name);
    if (layIt == recordLayouts.end()) return false;
    auto sit = locals.find(ref->name);
    if (sit == locals.end()) return false;
    Value mem = sit->second;
    if (!mem.getType().isa<MemRefType>()) return false;
    const StructLayout &layout = layIt->second;
    for (const FieldLayout &fld : layout.fields) {
      int64_t n = 1;
      for (int64_t d : fld.arrayDims) n *= (d > 0 ? d : 1);
      for (int64_t e = 0; e < n; ++e) {
        Value off = builder.create<arith::ConstantOp>(
            l, builder.getIndexType(),
            builder.getIndexAttr((int64_t)(fld.offset / 4 + e)));
        Value slotIdx = off;
        Value raw = builder.create<memref::LoadOp>(l, builder.getI32Type(), mem,
                                                    ValueRange{slotIdx});
        mlir::Type fty = fld.type;
        if (fty.isF32())
          out.push_back(builder.create<arith::BitcastOp>(l, fty, raw));
        else if (fty.isF16())
          out.push_back(builder.create<arith::BitcastOp>(l, fty, raw));
        else
          out.push_back(castValue(raw, fty, l));
      }
    }
    return true;
  }

  void buildFunction(const FunctionDecl *fn) {
    if (!fn) return;
    // Mangled device symbol (ns_func, or Class_method for a class method); the
    // host launch resolves `ns::func` / `obj.method` to the same key.
    std::string symName = deviceMangledName(fn);

    // A class method is never emitted as a func.func here: SPIR-V cannot pass
    // the Function-storage struct memref `_this` as a by-ref parameter through
    // func.call (FuncToSPIRV + GPUToSPIRV leave an unrealized_conversion_cast
    // they can't legalize for memref args). Instead, method call sites INLINE
    // the body with `_this` bound to the caller's object (see the non-scope
    // MemberAccessExpr callee path in visitExpr). We still register the decl in
    // funcDecls so the call site can find + inline it.
    if (fn->isMethod && !fn->className.empty()) {
      funcDecls[symName] = fn;
      return;
    }

    // A class method takes a leading `_this` parameter: the class record as a
    // flat memref<Nxi32> slot (NOT by-value scalars — methods mutate `this->f`
    // and the caller's object must see the writes, so the slot is passed by
    // reference). The memref arg is a Function-storage slot the caller allocas;
    // it is NOT an SSBO pointer (it carries no StorageBuffer storage class),
    // so func.call passing + ConvertFuncToSPIRV legalize it as a plain pointer
    // to a Function variable. Mirrors GLSL's `inout Class _this`.
    bool isMethod = fn->isMethod && !fn->className.empty();
    mlir::Type thisTy;
    const RecordType *thisRec = nullptr;
    if (isMethod) {
      // Ensure the class layout is recorded before reading it (the class
      // StructDecl may appear later in tu.decls than this method, as in
      // class.vc where `Accumulator::add` precedes the `Class Accumulator`
      // body — recordStructLayout is idempotent, so this is safe regardless).
      if (auto *sd = lookupStructDecl(fn->className))
        recordStructLayout(sd);
      auto *sd = new StructDecl(fn->getLoc(), fn->className);
      thisRec = new RecordType(sd);
      // Build the _this memref<Nxi32> with an explicit Function storage class
      // so it matches the caller's alloca slot type exactly — FuncToSPIRV then
      // lowers the parameter to a spirv.ptr<array<Nxi32>, Function> and the
      // call's argument type agrees. A bare memref<Nxi32> (no storage class)
      // is rejected by FuncToSPIRV's function-signature conversion, which
      // drops the function definition and leaves the call dangling.
      mlir::Type baseTy = cvtType(thisRec);
      if (auto mty = baseTy.dyn_cast<MemRefType>()) {
        thisTy = MemRefType::get(
            mty.getShape(), mty.getElementType(), MemRefLayoutAttrInterface(),
            spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
      } else {
        thisTy = baseTy;
      }
    }

    // Function signature. By-value struct parameters are scalarized into N
    // field-typed args (see structParams / structFieldScalarTypes); a memref
    // parameter would carry SSBO-pointer semantics and fail to legalize as a
    // by-value value. Non-struct params keep their cvtType result.
    SmallVector<mlir::Type> argTypes;
    if (isMethod && thisTy)
      argTypes.push_back(thisTy);
    structParams.clear();
    for (auto *p : fn->params) {
      if (p->type && isa<RecordType>(p->type)) {
        auto scalars = structFieldScalarTypes(cast<RecordType>(p->type));
        if (!scalars.empty()) {
          StructParamSlots sp;
          sp.startArg = argTypes.size();
          sp.fieldTys = scalars;
          for (mlir::Type st : scalars) argTypes.push_back(st);
          structParams[p->name] = std::move(sp);
          continue;
        }
      }
      mlir::Type pt = cvtType(p->type);
      if (!pt) {
        // Unsupported param type — skip cleanly rather than feed a null type to
        // getFunctionType (which would crash FuncOp::create).
        error(p, "unsupported parameter type in device function");
        continue;
      }
      argTypes.push_back(pt);
    }
    mlir::Type retTy = cvtType(fn->returnType);
    // A void function has an empty result list (SPIR-V entry points must not
    // declare a `none` result); the LLVM-style none type is an interior
    // convenience only.
    TypeRange results = retTy;
    if (retTy && retTy.isa<NoneType>())
      results = TypeRange{};
    FunctionType fty = builder.getFunctionType(argTypes, results);

    auto f = func::FuncOp::create(loc(fn), symName, fty);
    if (fn->deviceAttr == DeviceAttr::Global) {
      // Mark as a kernel entry point.
      f->setAttr("vc.kernel", builder.getUnitAttr());
    }
    module.push_back(f);
    funcTable[symName] = f;
    funcDecls[symName] = fn;
    currentRetTy = retTy;

    if (!fn->body) return;

    // Function body
    Block *entry = f.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    // locals/entryBlock are per-function state (binding args below).
    entryBlock = entry;
    locals.clear();
    localTypes.clear();

    // Bind the synthesized `_this` (methods): the leading block arg is the
    // caller's memref<Nxi32> slot for the object; register it as a bare
    // RecordType local so `this->sum` / `_this.sum` resolve through the local-
    // struct field path.
    unsigned argIdx = 0;
    if (isMethod && thisTy) {
      locals["_this"] = entry->getArgument(argIdx);
      localTypes["_this"] = thisRec;
      ++argIdx;
    }

    // Bind parameters to block args. A by-value struct param (in structParams)
    // is materialized as a fresh local memref<Nxi32> alloca at the entry, with
    // each scalarized block arg stored into the corresponding i32 slot (bitcast
    // f32->i32 to preserve the field's bit pattern, mirroring storeStructField).
    // The body then treats `v` exactly like a local `Vec4 v;` decl.
    for (unsigned i = 0; i < fn->params.size(); ++i) {
      const ParamDecl *p = fn->params[i];
      auto spit = structParams.find(p->name);
      if (spit != structParams.end()) {
        const StructParamSlots &sp = spit->second;
        // Alloca the struct's flat i32 slot at the entry.
        if (!isa<RecordType>(p->type)) continue;
        auto *rec = cast<RecordType>(p->type);
        auto layIt = recordLayouts.find(rec->decl->name);
        if (layIt == recordLayouts.end()) continue;
        int64_t slots = ((int64_t)layIt->second.stride + 3) / 4;
        if (slots < 1) slots = 1;
        MemRefType slotTy = MemRefType::get(
            {slots}, builder.getI32Type(), MemRefLayoutAttrInterface(),
            spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
        Value slot = builder.create<memref::AllocaOp>(loc(p), slotTy);
        locals[p->name] = slot;
        localTypes[p->name] = p->type;
        // Store each scalarized field arg into its slot.
        auto lit = recordLayouts.find(rec->decl->name);
        const StructLayout &layout = lit->second;
        unsigned scalarIdx = 0;
        for (const FieldLayout &fld : layout.fields) {
          int64_t n = 1;
          for (int64_t d : fld.arrayDims) n *= (d > 0 ? d : 1);
          for (int64_t e = 0; e < n; ++e) {
            if (scalarIdx >= sp.fieldTys.size()) break;
            Value arg = entry->getArgument(sp.startArg + scalarIdx);
            Value one = builder.create<arith::ConstantOp>(
                loc(p), builder.getIndexType(),
                builder.getIndexAttr((int64_t)(fld.offset / 4 + e)));
            Value slotIdx = one;
            Value v = arg;
            mlir::Type fty = fld.type;
            if (fty.isF32() || fty.isF16())
              v = builder.create<arith::BitcastOp>(loc(p),
                                                    builder.getI32Type(), v);
            else
              v = castValue(v, builder.getI32Type(), loc(p));
            builder.create<memref::StoreOp>(loc(p), v, slot,
                                            ValueRange{slotIdx});
            ++scalarIdx;
          }
        }
        argIdx = sp.startArg + sp.fieldTys.size();
        continue;
      }
      if (argIdx < entry->getNumArguments())
        locals[p->name] = entry->getArgument(argIdx);
      localTypes[p->name] = p->type;
      ++argIdx;
    }

    // Route the body through emitStatements so trailing bare-return guards
    // (`if (c) return;`) are wrapped in inverted scf.if instead of emitting
    // func.return inside a structured region (invalid for SPIR-V).
    if (fn->body) visitStmt(fn->body.get());

    // Ensure a void return has a terminator.
    if (builder.getBlock()->empty() ||
        !isa<func::ReturnOp>(builder.getBlock()->back()))
      builder.create<func::ReturnOp>(loc(fn));

    // For kernels, emit a vc.kernel wrapper referencing this function.
    if (fn->deviceAttr == DeviceAttr::Global) {
      builder.setInsertionPointToStart(module.getBody());
      auto symRef = SymbolRefAttr::get(&ctx, symName, {});
      builder.create<vc::KernelOp>(loc(fn), symRef);
    }
  }

  //===--------------------------------------------------------------------//
  // Value plumbing helpers
  //===--------------------------------------------------------------------//

  // A slot (local var) yields its memref address; load it when a scalar is
  // wanted. Index expressions are cast to index for memref indexing. A
  // spirv.ptr (a __shared__ scalar slot) loads via spirv.Load.
  Value loadValue(Value v, Location l) {
    if (!v) return v;
    if (auto mr = v.getType().dyn_cast<MemRefType>())
      return builder.create<memref::LoadOp>(l, v, ValueRange{});
    if (auto ptr = v.getType().dyn_cast<spirv::PointerType>())
      return builder.create<spirv::LoadOp>(l, ptr.getPointeeType(), v,
                                          /*memory_access=*/spirv::MemoryAccessAttr(),
                                          /*alignment=*/IntegerAttr());
    return v;
  }

  // Cast an arbitrary scalar to i1 ("truthiness"): compare != 0.
  Value toI1(Value v, Location l) {
    v = loadValue(v, l);
    if (!v) return v;
    mlir::Type ty = v.getType();
    if (ty.isInteger(1)) return v;
    Value zero = builder.create<arith::ConstantOp>(l, ty,
                                                   builder.getZeroAttr(ty));
    if (ty.isIntOrIndex())
      return builder.create<arith::CmpIOp>(l, arith::CmpIPredicate::ne, v, zero);
    return builder.create<arith::CmpFOp>(l, arith::CmpFPredicate::UNE, v, zero);
  }

  // Resolve an lvalue (a local-var slot or an array element) to a memref plus
  // its index list for load/store. Function params (block args) are scalars
  // and are not writable slots, so they fail here. Multi-dimensional indexing
  // `a[i][j]` is a chain of IndexExprs; we descend to the base memref and
  // collect every subscript (innermost last), so a memref<16x8xf32> gets two
  // indices in [i, j] order.
  //
  // __shared__ slots live behind a spirv.ptr (Workgroup): a scalar shared slot
  // is a spirv.ptr<T, Workgroup> (no indices, loaded/stored directly), and a
  // shared array is spirv.ptr<array<...>, Workgroup> indexed with
  // spirv.AccessChain. To keep the caller uniform, a shared scalar is returned
  // as `mem` = the pointer with empty indices (callers detect spirv.ptr and use
  // spirv.Load/Store), and a shared array element is returned as `mem` = the
  // AccessChain'd element pointer (spirv.ptr<elem, Workgroup>) with empty
  // indices — again a plain spirv.Load/Store target.
  bool lvalueAddress(ASTNode *n, Value &mem, SmallVectorImpl<Value> &indices) {
    if (!n) return false;
    if (n->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
      auto *ref = static_cast<DeclRefExpr *>(n);
      auto it = locals.find(ref->name);
      if (it == locals.end()) return false;
      mem = it->second;
      // A 0-d memref slot (local scalar) or a spirv.ptr scalar shared slot.
      return mem.getType().isa<MemRefType>() ||
             mem.getType().isa<spirv::PointerType>();
    }
    if (n->getNodeType() == ASTNode::NodeKind::IndexExpr) {
      SmallVector<IndexExpr *, 4> chain;
      ASTNode *cur = n;
      while (cur && cur->getNodeType() == ASTNode::NodeKind::IndexExpr) {
        auto *ie = static_cast<IndexExpr *>(cur);
        chain.push_back(ie);
        cur = ie->base.get();
      }
      // Struct array-field element: `pts[i].v[k]` = IndexExpr(MemberAccessExpr(
      // pts[i], "v"), k). The member `v` is an array field; its slot address is
      // fieldOffset + elemIdx*stride + k*elemSize, all in bytes, then /4 for the
      // i32 slot. Resolve via structFieldAddr with extraByteOff = k*elemSize.
      // (Only single-subscript array fields occur in the demos; multi-dim would
      // need the extra offset accumulated across the chain.)
      if (cur && cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr &&
          chain.size() == 1) {
        auto *ma = static_cast<MemberAccessExpr *>(cur);
        ASTNode *kNode = chain[0]->index.get();
        Value kv = loadValue(visitExpr(kNode), loc(kNode));
        if (kv) {
          kv = castValue(kv, builder.getI32Type(), loc(kNode));
          // Element size: resolve the field type to its scalar width. The field
          // is an array (float v[2] -> scalar f32 + arrayDims {2}); element size
          // = scalar width.
          Value fieldMem, slotIdx;
          mlir::Type fieldTy;
          std::vector<int64_t> adims;
          if (structFieldAddr(ma, loc(cur), Value(), fieldMem, slotIdx,
                              fieldTy, &adims) &&
              !adims.empty()) {
            unsigned elemBytes =
                fieldTy.isIntOrFloat() ? (fieldTy.getIntOrFloatBitWidth() / 8) : 4;
            Value elemSize = builder.create<arith::ConstantOp>(
                loc(cur), builder.getI32Type(),
                builder.getI32IntegerAttr((int32_t)elemBytes));
            Value extra = builder.create<arith::MulIOp>(loc(cur), kv, elemSize);
            // Re-resolve with the extra offset to get the element's slot.
            if (structFieldAddr(ma, loc(cur), extra, fieldMem, slotIdx,
                                fieldTy)) {
              mem = fieldMem;
              indices.clear();
              indices.push_back(slotIdx);
              return true;
            }
          }
        }
      }
      // The chain root must resolve to an addressable memory object. A subscript
      // of a non-array value (e.g. indexing a loaded scalar) is rejected.
      Value base = visitExpr(cur);
      if (!base) return false;
      // Shared array: spirv.ptr<array<...>, Workgroup>. Build a spirv.AccessChain
      // over the indices (outermost-first) and return the element pointer.
      if (base.getType().isa<spirv::PointerType>()) {
        SmallVector<Value> idxVals;
        for (auto *ie : llvm::reverse(chain)) {
          Value idx = loadValue(visitExpr(ie->index.get()),
                                loc(ie->index.get()));
          if (!idx) return false;
          // spirv.AccessChain indices are signless integers.
          if (idx.getType().isIndex())
            idx = builder.create<arith::IndexCastOp>(
                loc(ie->index.get()), builder.getI32Type(), idx);
          else if (!idx.getType().isSignlessInteger())
            idx = builder.create<arith::TruncIOp>(
                loc(ie->index.get()), builder.getI32Type(), idx);
          idxVals.push_back(idx);
        }
        // Result type: peel exactly one array dimension per provided index
        // (NOT all dimensions). A full chain (e.g. `voteArr[i]` on a 1-D
        // array) yields a scalar element ptr; a partial chain (e.g. `buf[slot]`
        // on a `T[2][64]` shared array) yields a row ptr `ptr<array<64xf32>>`,
        // which a further AccessChain / cooperative-copy loop can index.
        mlir::Type pointee =
            base.getType().cast<spirv::PointerType>().getPointeeType();
        mlir::Type elemTy = pointee;
        for (size_t i = 0, e = chain.size(); i < e; ++i) {
          auto arr = elemTy.dyn_cast<spirv::ArrayType>();
          if (!arr) break;
          elemTy = arr.getElementType();
        }
        spirv::PointerType elemPtr = spirv::PointerType::get(
            elemTy, base.getType().cast<spirv::PointerType>().getStorageClass());
        mem = builder.create<spirv::AccessChainOp>(loc(n), elemPtr, base,
                                                    idxVals);
        // Element pointer: callers use spirv.Load/Store on it directly.
        indices.clear();
        return true;
      }
      // memref array (kernel arg or local alloca).
      if (!base.getType().isa<MemRefType>()) return false;
      mem = base;
      for (auto *ie : llvm::reverse(chain)) {
        Value idx = loadValue(visitExpr(ie->index.get()), loc(ie->index.get()));
        if (!idx) return false;
        if (!idx.getType().isIndex())
          idx = builder.create<arith::IndexCastOp>(loc(ie->index.get()),
                                                   builder.getIndexType(), idx);
        indices.push_back(idx);
      }
      return true;
    }
    if (n->getNodeType() == ASTNode::NodeKind::UnaryExpr) {
      auto *u = static_cast<UnaryExpr *>(n);
      if (u->op != UnaryOp::Deref) return false;
      Value base = visitExpr(u->operand.get());
      if (!base) return false;
      if (auto ptr = base.getType().dyn_cast<spirv::PointerType>()) {
        (void)ptr;
        mem = base;
        indices.clear();
        return true;
      }
      if (auto mr = base.getType().dyn_cast<MemRefType>()) {
        mem = base;
        indices.clear();
        if (mr.getRank() == 0) return true;
        if (mr.getRank() == 1) {
          indices.push_back(builder.create<arith::ConstantIndexOp>(loc(n), 0));
          return true;
        }
      }
      return false;
    }
    return false;
  }

  // Read the current value of an lvalue node (slot or array element).
  Value loadLValue(ASTNode *n, Location l) {
    Value mem;
    SmallVector<Value> indices;
    if (!lvalueAddress(n, mem, indices)) return Value();
    // A spirv.ptr (shared scalar slot or shared array element via AccessChain)
    // loads via spirv.Load.
    if (auto ptr = mem.getType().dyn_cast<spirv::PointerType>())
      return builder.create<spirv::LoadOp>(l, ptr.getPointeeType(), mem,
                                          /*memory_access=*/spirv::MemoryAccessAttr(),
                                          /*alignment=*/IntegerAttr());
    return builder.create<memref::LoadOp>(l, mem, indices);
  }

  // Read `ptr[idx]` where `ptr` is a vector pointer (`float4*`, `int3*`, ...).
  // Such a pointer lowers to a scalarized memref<?xELEM> (ELEM = f32/i32/...,
  // NOT vector<NxELEM>), because memref<?xvector<...>> does not legalize its
  // loads/stores through GPUToSPIRV. So element `idx` of the vector array
  // spans N consecutive scalar slots [idx*N .. idx*N+N-1]; load each and
  // CompositeConstruct them back into a vector<NxELEM>. Returns null if `base`
  // is not a vector-pointer lvalue (caller falls back to the generic path).
  Value loadVectorPointerElement(IndexExpr *ie, Location l) {
    if (!ie->base ||
        ie->base->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return Value();
    auto *ref = static_cast<DeclRefExpr *>(ie->base.get());
    auto tit = localTypes.find(ref->name);
    if (tit == localTypes.end() || !tit->second) return Value();
    auto *ptrTy = dyn_cast<vc::PointerType>(tit->second);
    if (!ptrTy) return Value();
    auto *vecTy = dyn_cast<vc::VectorType>(ptrTy->pointee);
    if (!vecTy) return Value();
    // The memref the pointer lowered to.
    auto lit = locals.find(ref->name);
    if (lit == locals.end()) return Value();
    Value mem = lit->second;
    auto mty = mem.getType().dyn_cast<MemRefType>();
    if (!mty) return Value();
    unsigned n = vecTy->count;
    if (n == 0) return Value();

    // Base index `idx` (as i32 for the slot arithmetic), scaled by N.
    Value idx = loadValue(visitExpr(ie->index.get()), loc(ie->index.get()));
    if (!idx) return Value();
    idx = castValue(idx, builder.getI32Type(), loc(ie->index.get()));
    Value nConst = builder.create<arith::ConstantOp>(
        l, builder.getI32Type(), builder.getI32IntegerAttr((int32_t)n));
    Value baseSlot = builder.create<arith::MulIOp>(l, idx, nConst);

    mlir::Type elemTy = cvtType(vecTy->elem);
    mlir::VectorType vecMTy = mlir::VectorType::get({(int64_t)n}, elemTy);
    SmallVector<Value, 4> parts;
    for (unsigned k = 0; k < n; ++k) {
      Value off = builder.create<arith::ConstantOp>(
          l, builder.getI32Type(), builder.getI32IntegerAttr((int32_t)k));
      Value slot = builder.create<arith::AddIOp>(l, baseSlot, off);
      Value slotIdx = builder.create<arith::IndexCastOp>(
          l, builder.getIndexType(), slot);
      Value part = builder.create<memref::LoadOp>(l, mem, ValueRange{slotIdx});
      if (part.getType() != elemTy)
        part = castValue(part, elemTy, l);
      parts.push_back(part);
    }
    return builder.create<spirv::CompositeConstructOp>(l, vecMTy, parts);
  }

  // Store `val` into `ptr[idx]` where `ptr` is a vector pointer (`float4*`).
  // Mirror of loadVectorPointerElement: the pointer is scalarized to
  // memref<?xELEM>, so element `idx` spans N scalar slots [idx*N .. idx*N+N-1];
  // extract each component from the vector `val` and store it to its slot.
  // Returns true if handled (caller returns); false if `base` is not a
  // vector-pointer lvalue (caller falls back to the generic store path).
  bool storeVectorPointerElement(IndexExpr *ie, Value val, Location l) {
    if (!val || !ie->base ||
        ie->base->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *ref = static_cast<DeclRefExpr *>(ie->base.get());
    auto tit = localTypes.find(ref->name);
    if (tit == localTypes.end() || !tit->second) return false;
    auto *ptrTy = dyn_cast<vc::PointerType>(tit->second);
    if (!ptrTy) return false;
    auto *vecTy = dyn_cast<vc::VectorType>(ptrTy->pointee);
    if (!vecTy) return false;
    auto lit = locals.find(ref->name);
    if (lit == locals.end()) return false;
    Value mem = lit->second;
    if (!mem.getType().isa<MemRefType>()) return false;
    unsigned n = vecTy->count;
    if (n == 0) return false;

    mlir::Type elemTy = cvtType(vecTy->elem);
    mlir::VectorType vecMTy = mlir::VectorType::get({(int64_t)n}, elemTy);
    // Coerce val to the expected vector type (e.g. int4 stored via a float4
    // value is not expected, but scalar->vector broadcast or width mismatch is).
    if (val.getType() != vecMTy)
      val = castValue(val, vecMTy, l);

    Value idx = loadValue(visitExpr(ie->index.get()), loc(ie->index.get()));
    if (!idx) return false;
    idx = castValue(idx, builder.getI32Type(), loc(ie->index.get()));
    Value nConst = builder.create<arith::ConstantOp>(
        l, builder.getI32Type(), builder.getI32IntegerAttr((int32_t)n));
    Value baseSlot = builder.create<arith::MulIOp>(l, idx, nConst);
    for (unsigned k = 0; k < n; ++k) {
      Value off = builder.create<arith::ConstantOp>(
          l, builder.getI32Type(), builder.getI32IntegerAttr((int32_t)k));
      Value slot = builder.create<arith::AddIOp>(l, baseSlot, off);
      Value slotIdx = builder.create<arith::IndexCastOp>(
          l, builder.getIndexType(), slot);
      Value part = builder.create<spirv::CompositeExtractOp>(
          l, elemTy, val, builder.getI32ArrayAttr({(int32_t)k}));
      if (part.getType() != elemTy)
        part = castValue(part, elemTy, l);
      builder.create<memref::StoreOp>(l, part, mem, ValueRange{slotIdx});
    }
    return true;
  }

  // Store into a memref slot, bridging index<->int / widening ints.
  void storeTo(Value mem, ArrayRef<Value> indices, Value v, Location l) {
    if (!v) return;
    v = loadValue(v, l);
    // A spirv.ptr (shared scalar slot or shared array element) stores via
    // spirv.Store. The stored value must match the pointee element type.
    if (mem.getType().isa<spirv::PointerType>()) {
      mlir::Type elem = mem.getType().cast<spirv::PointerType>().getPointeeType();
      if (v.getType() != elem)
        v = castValue(v, elem, l);
      builder.create<spirv::StoreOp>(l, mem, v,
                                     /*memory_access=*/spirv::MemoryAccessAttr(),
                                     /*alignment=*/IntegerAttr());
      return;
    }
    mlir::Type elem = mem.getType().cast<MemRefType>().getElementType();
    if (v.getType() != elem) {
      if (v.getType().isIndex() && elem.isSignlessInteger())
        v = builder.create<arith::IndexCastOp>(l, elem, v);
      else if (elem.isIndex() && v.getType().isSignlessInteger())
        v = builder.create<arith::IndexCastOp>(l, elem, v);
      else if (v.getType().isSignlessInteger() && elem.isSignlessInteger() &&
               v.getType().getIntOrFloatBitWidth() <
                   elem.getIntOrFloatBitWidth())
        v = builder.create<arith::ExtSIOp>(l, elem, v);
      else if (v.getType().isa<mlir::FloatType>() &&
               elem.isa<mlir::FloatType>())
        // Float width reconciliation: a bare `0.0` literal (f64) stored into a
        // `float` local (f32) narrows via TruncFOp; an `f`-suffixed literal
        // stored into a double widens via ExtFOp. castValue handles both.
        v = castValue(v, elem, l);
    }
    // Struct slot store: the memref is the flat i32 view of a struct (SSBO or
    // local value), and `v` is the field's native type (e.g. `v.c[0] = 1.0f`
    // stores an f32 into an i32 slot). This is a BITCAST (preserve bits), not a
    // numeric conversion — the host reads the same bit pattern back as the
    // field's C type, mirroring storeStructField. Covers f32/f16 -> i32. A bare
    // `1.0` literal (f64) stored to a `float` field narrows to f32 first, then
    // bitcasts to i32 (a `double` array field spanning two i32 slots is still
    // unsupported — the lo/hi split only lives in storeStructField for scalar
    // struct fields, not array elements).
    if (v.getType() != elem && elem.isInteger(32)) {
      if (v.getType().isF64())
        v = builder.create<arith::TruncFOp>(l, builder.getF32Type(), v);
      if (v.getType().isF32() || v.getType().isF16())
        v = builder.create<arith::BitcastOp>(l, elem, v);
    }
    if (indices.empty())
      builder.create<memref::StoreOp>(l, v, mem);
    else
      builder.create<memref::StoreOp>(l, v, mem, indices);
  }

  // Store into a local-var slot (0-d memref), bridging index <-> int if needed.
  void storeValue(Value addr, Value v, Location l) {
    storeTo(addr, {}, v, l);
  }

  // Materialize a __constant__ global (scalar or array) as a slot at the
  // start of the current function's entry block, initialized from the
  // compile-time initializer. This is a functional (not physical) mapping:
  // per-kernel Function-storage stack storage instead of a shared shader
  // global. Implemented lazily on first reference; each kernel gets its own
  // slot, and locals[] memoizes it per function.
  Value materializeConstGlobal(const VarDecl *v, const ASTNode *useSite) {
    auto hit = locals.find(v->name);
    if (hit != locals.end()) return hit->second;
    Location l = loc(useSite);
    mlir::Type elem = cvtType(v->type);
    int64_t sz = (!v->arrayDims.empty() && v->arrayDims[0] > 0)
                     ? v->arrayDims[0]
                     : 0;
    MemRefType ty;
    if (sz > 0)
      ty = MemRefType::get(
          ArrayRef<int64_t>{sz}, elem, MemRefLayoutAttrInterface(),
          spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
    else
      ty = MemRefType::get(
          ArrayRef<int64_t>{}, elem, MemRefLayoutAttrInterface(),
          spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));

    // Hoist alloc + init into the entry block so the slot dominates every use.
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(entryBlock);
    Value addr = builder.create<memref::AllocaOp>(l, ty);
    if (v->init) {
      if (v->init->getNodeType() == ASTNode::NodeKind::InitListExpr) {
        auto *il = static_cast<InitListExpr *>(v->init.get());
        for (size_t i = 0; i < il->elements.size() && sz > 0; ++i) {
          SmallVector<Value, 1> idx;
          idx.push_back(builder.create<arith::ConstantOp>(
              l, builder.getIndexType(), builder.getIndexAttr(i)));
          storeTo(addr, idx, visitExpr(il->elements[i].get()), l);
        }
      } else {
        storeTo(addr, {}, visitExpr(v->init.get()), l);
      }
    }
    builder.restoreInsertionPoint(saved);
    locals[v->name] = addr;
    return addr;
  }

  // Create (once, by symbol name) a module-scope SPIR-V GlobalVariable in the
  // Workgroup storage class for a __shared__ variable, and return a
  // spirv.mlir.addressof of it at the current insertion point.
  //
  // Why spirv.GlobalVariable and not memref.global: stock MLIR's GPUToSPIRV /
  // MemRefToSPIRV passes do NOT legalize memref.global (a module-scope
  // memref.global with Workgroup storage is left as an unlegalizable op and
  // fails `failed to legalize operation 'memref.global'`). The working idiom is
  // a spirv.GlobalVariable + spirv.mlir.addressof, with spirv.Load / spirv.Store
  // / spirv.AccessChain on the resulting !spirv.ptr<..., Workgroup>. So shared
  // memory is emitted directly in the spirv dialect; the VCToGPU stage hoists
  // the GlobalVariable into the gpu.module (memref globals are hoisted the same
  // way) so it survives Stage 1 cleanup and is carried into the spirv.module by
  // GPUToSPIRV.
  //
  // The symbol is namespaced (`__vc_shared_<name>`) to avoid clashing with user
  // symbols; the same global is reused across kernels that declare a __shared__
  // var of the same name and shape.
  // Emit (or look up) a module-scope Workgroup-storage global for a
  // __shared__ decl. `shape` holds the array dims; any dim == 0 means
  // "extern __shared__ T s[]" — an unsized (runtime-sized) dimension. MLIR's
  // spirv::ArrayType only accepts a constant `unsigned` count, so we cannot
  // express a spec-constant-sized array from IR. Instead we emit a placeholder
  // size of 1 for each unsized dim (so the IR type-checks) AND stamp the global
  // with a recognizable symbol name `__vc_dynshared_<name>`. A post-serialize
  // binary patch in vc.cpp (patchDynamicSharedArrays) then locates each such
  // global's OpTypeArray and rewrites its length operand to reference the
  // `__vc_wg_x` spec constant (spec_id 0 = block x), mirroring how the GLSL
  // backend sizes `extern __shared__` to `gl_WorkGroupSize.x`.
  Value getOrCreateSharedGlobal(llvm::StringRef name, ArrayRef<int64_t> shape,
                                mlir::Type elemTy, Location l,
                                bool dynamic = false) {
    std::string sym = dynamic ? ("__vc_dynshared_") + name.str()
                              : ("__vc_shared_") + name.str();
    // The pointee type: the element for a scalar, or a spirv.array wrapping the
    // element for a (multi-dimensional) shared array. spirv.array is row-major
    // and nested for multi-dim (`float s[16][8]` -> array<16 x array<8 x f32>>).
    // An unsized dim (0, only valid for `extern __shared__`) becomes a
    // placeholder 1 in IR — the real size is patched into the serialized
    // SPIR-V binary afterward.
    mlir::Type pointee = elemTy;
    if (!shape.empty()) {
      for (auto dim : llvm::reverse(shape))
        pointee = spirv::ArrayType::get(pointee, dim > 0 ? (unsigned)dim : 1u);
    }
    spirv::PointerType ptrTy =
        spirv::PointerType::get(pointee, spirv::StorageClass::Workgroup);

    if (!module.lookupSymbol(sym)) {
      auto saved = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(module.getBody());
      builder.create<spirv::GlobalVariableOp>(l, TypeAttr::get(ptrTy),
                                              builder.getStringAttr(sym),
                                              /*initializer=*/FlatSymbolRefAttr());
      builder.restoreInsertionPoint(saved);
    }
    return builder.create<spirv::AddressOfOp>(l, ptrTy, sym);
  }

  // Materialize the subgroup size (CUDA `warpSize`) as an i32 SSA value.
  // Emitted as gpu.subgroup_size (returns index); GPUToSPIRV's
  // SingleDimLaunchConfigConversion lowers it to a spirv GlobalVariable
  // decorated `BuiltIn SubgroupSize` + a Load, creating the builtin global the
  // same way it creates WorkgroupId/LocalInvocationId (so the built_in
  // decoration lands in the op's Properties struct, which the serializer
  // requires — a hand-built {builtin = ...} dict attr is rejected). The result
  // is index-cast to i32 (VC ints are 32-bit). Flips usesSubgroup so VCToGPU
  // bumps the target env to 1.3 + subgroup caps.
  Value getSubgroupSize(Location l) {
    usesSubgroup = true;
    Value sz = builder.create<gpu::SubgroupSizeOp>(l, builder.getIndexType());
    return builder.create<arith::IndexCastOp>(l, builder.getI32Type(), sz);
  }

  // Put two operands on a common type for a binop: float stays float, and if
  // one side is index the other is promoted to index (indexing arithmetic).
  std::pair<Value, Value> commonize(Value l, Value r, Location lc) {
    l = loadValue(l, lc);
    r = loadValue(r, lc);
    if (l.getType() == r.getType()) return {l, r};
    // Mixed int/float: promote the integer side to the float type so a float
    // op (e.g. `TILE * scale` where TILE is int, scale is float) lowers to
    // arith.mulf instead of crashing arith.muli on an f32 operand. The float
    // width wins: a double operand (bare `1.5` literal, or a `double` local)
    // promotes an int partner to f64; an f32 partner promotes to f32.
    auto promoteIntToFloat = [&](Value &intSide, Value &fltSide) {
      mlir::Type ft = fltSide.getType();
      if (intSide.getType().isIndex())
        intSide = builder.create<arith::IndexCastOp>(lc, builder.getI32Type(),
                                                     intSide);
      intSide = builder.create<arith::SIToFPOp>(lc, ft, intSide);
    };
    if (l.getType().isa<mlir::FloatType>() && r.getType().isIntOrIndex()) {
      promoteIntToFloat(r, l);
      return {l, r};
    }
    if (r.getType().isa<mlir::FloatType>() && l.getType().isIntOrIndex()) {
      promoteIntToFloat(l, r);
      return {l, r};
    }
    // Mixed float widths (f64 vs f32, f64 vs f16, f32 vs f16): widen the
    // narrower to the wider (C usual arithmetic conversions). Truncation never
    // happens here — both sides end up the wider type, and a narrowing store
    // back to a narrower local happens later via castValue at the store site.
    if (l.getType().isa<mlir::FloatType>() &&
        r.getType().isa<mlir::FloatType>()) {
      unsigned lw = l.getType().getIntOrFloatBitWidth();
      unsigned rw = r.getType().getIntOrFloatBitWidth();
      if (lw > rw)
        r = builder.create<arith::ExtFOp>(lc, l.getType(), r);
      else if (rw > lw)
        l = builder.create<arith::ExtFOp>(lc, r.getType(), l);
      return {l, r};
    }
    if (l.getType().isIndex() && r.getType().isIntOrIndex()) {
      if (!r.getType().isIndex())
        r = builder.create<arith::IndexCastOp>(lc, builder.getIndexType(), r);
    } else if (r.getType().isIndex() && l.getType().isIntOrIndex()) {
      if (!l.getType().isIndex())
        l = builder.create<arith::IndexCastOp>(lc, builder.getIndexType(), l);
    } else if (l.getType().isSignlessInteger() &&
               r.getType().isSignlessInteger()) {
      unsigned lw = l.getType().getIntOrFloatBitWidth();
      unsigned rw = r.getType().getIntOrFloatBitWidth();
      if (lw != rw) {
        unsigned w = std::max(lw, rw);
        auto t = builder.getIntegerType(w);
        if (lw < w) l = builder.create<arith::ExtSIOp>(lc, t, l);
        if (rw < w) r = builder.create<arith::ExtSIOp>(lc, t, r);
      }
    }
    return {l, r};
  }

  Value castValue(Value v, mlir::Type to, Location lc) {
    v = loadValue(v, lc);
    mlir::Type from = v.getType();
    if (from == to) return v;
    if (from.isIndex() && to.isSignlessInteger())
      return builder.create<arith::IndexCastOp>(lc, to, v);
    if (from.isSignlessInteger() && to.isIndex())
      return builder.create<arith::IndexCastOp>(lc, to, v);
    if (from.isSignlessInteger() && to.isSignlessInteger()) {
      if (from.getIntOrFloatBitWidth() == to.getIntOrFloatBitWidth())
        return builder.create<arith::BitcastOp>(lc, to, v);
      if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth())
        return builder.create<arith::ExtSIOp>(lc, to, v);
      return builder.create<arith::TruncIOp>(lc, to, v);
    }
    if (from.isIntOrIndex() && to.isF32())
      return builder.create<arith::SIToFPOp>(lc, to, v);
    if (from.isIntOrIndex() && to.isF64())
      return builder.create<arith::SIToFPOp>(lc, to, v);
    if (from.isIntOrIndex() && to.isF16())
      return builder.create<arith::SIToFPOp>(lc, to, v);
    if (from.isF32() && to.isSignlessInteger())
      return builder.create<arith::FPToSIOp>(lc, to, v);
    if (from.isF64() && to.isSignlessInteger())
      return builder.create<arith::FPToSIOp>(lc, to, v);
    if (from.isF16() && to.isSignlessInteger())
      return builder.create<arith::FPToSIOp>(lc, to, v);
    if (from.isF32() && to.isF64())
      return builder.create<arith::ExtFOp>(lc, to, v);
    if (from.isF64() && to.isF32())
      return builder.create<arith::TruncFOp>(lc, to, v);
    // f16 ↔ wider floats: widen with ExtFOp, narrow with TruncFOp.
    if (from.isF16() && to.isF32())
      return builder.create<arith::ExtFOp>(lc, to, v);
    if (from.isF16() && to.isF64())
      return builder.create<arith::ExtFOp>(lc, to, v);
    if (from.isF32() && to.isF16())
      return builder.create<arith::TruncFOp>(lc, to, v);
    if (from.isF64() && to.isF16())
      return builder.create<arith::TruncFOp>(lc, to, v);
    return v;
  }

  //===--------------------------------------------------------------------//
  // Statements
  //===--------------------------------------------------------------------//

  // An `if (c) ... return;` with no else whose then-block ends in a return
  // (a bare `return;` or a `{ stmts; return; }`). Structured CFG (scf.if) has
  // no goto, so a return inside an scf.if is illegal; emitStatements lowers
  // these by wrap-around: the then-block (with its side effects) runs under
  // the original condition, and everything after the if runs under !c (the
  // fall-through path), which preserves C semantics — `if (c) { side; return;
  // } rest` is equivalent to `if (c) { side; } else { rest; }` since `rest`
  // only ever runs when c is false.
  static bool endsWithReturn(const ASTNode *t) {
    if (!t) return false;
    if (t->getNodeType() == ASTNode::NodeKind::ReturnStmt) return true;
    if (t->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto &ss = static_cast<const CompoundStmt *>(t)->statements;
      if (ss.empty()) return false;
      return ss.back()->getNodeType() == ASTNode::NodeKind::ReturnStmt;
    }
    return false;
  }
  // If `t` is a bare `return value;` (or a `{ ...; return value; }` block),
  // return the value expression; null for a bare `return;` (void). Used to
  // distinguish value-returning early returns (which need a yield chain) from
  // void early returns (which use the inverted-wrap-around).
  static ASTNode *trailingReturnValue(const ASTNode *t) {
    if (!t) return nullptr;
    if (t->getNodeType() == ASTNode::NodeKind::ReturnStmt)
      return static_cast<const ReturnStmt *>(t)->value.get();
    if (t->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto &ss = static_cast<const CompoundStmt *>(t)->statements;
      if (ss.empty()) return nullptr;
      if (ss.back()->getNodeType() == ASTNode::NodeKind::ReturnStmt)
        return static_cast<const ReturnStmt *>(ss.back().get())->value.get();
    }
    return nullptr;
  }
  static bool isReturnIf(const ASTNode *n) {
    if (!n || n->getNodeType() != ASTNode::NodeKind::IfStmt) return false;
    auto *iff = static_cast<const IfStmt *>(n);
    if (iff->elseStmt) return false;
    return endsWithReturn(iff->thenStmt.get());
  }
  // A value-returning early-return if: `if (c) ... return v;` (no else) where
  // the trailing return carries a value. Only meaningful inside functions that
  // return a value; these need a yield chain, not the void wrap-around.
  static bool isValueReturnIf(const ASTNode *n) {
    if (!isReturnIf(n)) return false;
    auto *iff = static_cast<const IfStmt *>(n);
    return trailingReturnValue(iff->thenStmt.get()) != nullptr;
  }

  // Emit stmts[from..end). Structured CFG ops (scf.if) have no goto, so an
  // early `return` cannot live inside a region. A bare
  // `if (c) return;` guard instead inverts the condition and wraps the
  // REMAINING statements in `scf.if(!c) { ... }`; the function's tail return
  // covers the fall-through path. A trailing guard (last statement) is
  // dropped — both its paths end at the tail return anyway.
  //
  // Value-returning early returns (`if (c) return v;`) cannot use the void
  // wrap-around (the value would be lost); when the enclosing function returns
  // a value they are lowered as a yield chain: the remaining statements become
  // the else branch of an scf.if that yields the return value on the taken
  // path and the fall-through value (recursively) on the other.
  void emitStatements(const std::vector<NodePtr> &stmts, size_t from) {
    while (from < stmts.size()) {
      ASTNode *s = stmts[from].get();
      // Value-returning early return: needs a yield chain, not the void
      // wrap-around. Only at function top level (not inside an scf.if region
      // or loop body) and only in functions that return a value.
      if (isValueReturnIf(s) && structuredDepth == 0 && currentRetTy &&
          !currentRetTy.isa<NoneType>()) {
        emitValueReturnChain(stmts, from);
        return;
      }
      // A value-returning switch (`switch(c){case..: return v; ...}`) at the
      // top level of a value-returning function: lower the whole switch to a
      // single yielding scf.if chain whose result is the function return, so
      // case-body returns become yields (a spirv.ReturnValue cannot live
      // inside an scf.if region in a value-returning function).
      if (s->getNodeType() == ASTNode::NodeKind::SwitchStmt &&
          structuredDepth == 0 && currentRetTy &&
          !currentRetTy.isa<NoneType>() &&
          isValueReturnSwitch(static_cast<SwitchStmt *>(s))) {
        emitValueReturnSwitch(static_cast<SwitchStmt *>(s));
        return;
      }
      if (isReturnIf(s)) {
        auto *iff = static_cast<IfStmt *>(s);
        Location l = loc(iff);
        Value cond = toI1(visitExpr(iff->cond.get()), l);
        if (!cond) return;
        // `if (c) { side; return; } rest`  ==>  `if (c) { side; } else { rest; }`
        // (rest only runs when c is false, matching the early return). A
        // trailing guard with no rest is just `if (c) { side; }`.
        bool hasRest = from + 1 < stmts.size();
        auto saved = builder.saveInsertionPoint();
        auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, cond, hasRest);
        // Then: the if's then-block minus its trailing return (the return is
        // implicit — the block falls through to the scf.if's end, and rest is
        // in the else so it is skipped, exactly as a return would).
        ++structuredDepth;
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        emitThenWithoutTrailingReturn(iff->thenStmt.get());
        if (hasRest) {
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          emitStatements(stmts, from + 1);
        }
        --structuredDepth;
        builder.restoreInsertionPoint(saved);
        return; // remainder was emitted inside the guard
      }
      // Inside a loop body, guard each statement against an earlier
      // break/continue in the same iteration so it is skipped.
      if (loopDepth > 0 && guarding)
        emitGuarded(s);
      else
        visitStmt(s);
      ++from;
    }
  }

  // Lower a run of statements starting at a value-returning early-return if
  // (`if (c) ... return v; [more...] return w;`) into a single scf.if that
  // yields the function's result: the then branch yields the early return's
  // value (after its side effects), the else branch yields the fall-through
  // result (recursively — more early returns nest as further scf.if yields, and
  // a trailing `return w;` yields w directly). The result becomes the
  // function's func.return. `if (c1) return v1; if (c2) return v2; return v3;`
  // lowers to:
  //   %r = scf.if %c1 -> T { ...; yield %v1 } else {
  //     %r2 = scf.if %c2 -> T { ...; yield %v2 } else { yield %v3 }
  //     yield %r2
  //   }
  //   return %r
  void emitValueReturnChain(const std::vector<NodePtr> &stmts, size_t from) {
    ASTNode *s = stmts[from].get();
    auto *iff = static_cast<IfStmt *>(s);
    Location l = loc(iff);
    Value cond = toI1(visitExpr(iff->cond.get()), l);
    if (!cond) return;
    mlir::Type retTy = currentRetTy;

    auto saved = builder.saveInsertionPoint();
    auto ifOp = builder.create<scf::IfOp>(l, TypeRange{retTy}, cond,
                                         /*withElse=*/true);

    // Then branch: side effects then yield the early return value.
    ++structuredDepth;
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    emitThenWithoutTrailingReturn(iff->thenStmt.get());
    ASTNode *retValNode = trailingReturnValue(iff->thenStmt.get());
    Value retVal = visitExpr(retValNode);
    retVal = loadValue(retVal, loc(retValNode));
    retVal = castValue(retVal, retTy, loc(retValNode));
    builder.create<scf::YieldOp>(l, ValueRange{retVal});

    // Else branch: the fall-through statements, recursively.
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    emitFallThroughToValue(stmts, from + 1, l, retTy);
    --structuredDepth;

    builder.restoreInsertionPoint(saved);
    // The scf.if result is the function's return value.
    builder.create<func::ReturnOp>(l, ValueRange{ifOp.getResult(0)});
  }

  // Emit the fall-through path after an early return as the else branch of a
  // yield chain: if the next statement is itself a value-returning early
  // return, recurse into another scf.if yield; otherwise emit statements
  // normally and the trailing `return v;` yields v. A missing trailing return
  // (fell off the end of a value-returning function) yields a zero of retTy.
  void emitFallThroughToValue(const std::vector<NodePtr> &stmts, size_t from,
                              Location l, mlir::Type retTy) {
    if (from >= stmts.size()) {
      Value zero = builder.create<arith::ConstantOp>(l, retTy,
                                                    builder.getZeroAttr(retTy));
      builder.create<scf::YieldOp>(l, ValueRange{zero});
      return;
    }
    ASTNode *s = stmts[from].get();
    if (isValueReturnIf(s)) {
      auto *iff = static_cast<IfStmt *>(s);
      Value cond = toI1(visitExpr(iff->cond.get()), l);
      if (!cond) {
        Value zero = builder.create<arith::ConstantOp>(
            l, retTy, builder.getZeroAttr(retTy));
        builder.create<scf::YieldOp>(l, ValueRange{zero});
        return;
      }
      auto ifOp = builder.create<scf::IfOp>(l, TypeRange{retTy}, cond, true);
      // The inner scf.if lives in the current region; after filling its two
      // sub-regions, resume in the current region to yield its result.
      auto outerSaved = builder.saveInsertionPoint();
      ++structuredDepth;
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      emitThenWithoutTrailingReturn(iff->thenStmt.get());
      ASTNode *retValNode = trailingReturnValue(iff->thenStmt.get());
      Value retVal = visitExpr(retValNode);
      retVal = loadValue(retVal, loc(retValNode));
      retVal = castValue(retVal, retTy, loc(retValNode));
      builder.create<scf::YieldOp>(l, ValueRange{retVal});
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      emitFallThroughToValue(stmts, from + 1, l, retTy);
      --structuredDepth;
      builder.restoreInsertionPoint(outerSaved);
      builder.create<scf::YieldOp>(l, ValueRange{ifOp.getResult(0)});
      return;
    }
    // No more early returns: emit statements normally; a trailing ReturnStmt
    // yields its value, otherwise (fell off end) yield zero.
    while (from < stmts.size()) {
      ASTNode *cur = stmts[from].get();
      if (cur->getNodeType() == ASTNode::NodeKind::ReturnStmt) {
        auto *r = static_cast<ReturnStmt *>(cur);
        Value v = r->value ? visitExpr(r->value.get()) : Value();
        if (v) {
          v = loadValue(v, loc(r->value.get()));
          v = castValue(v, retTy, loc(r->value.get()));
        } else {
          v = builder.create<arith::ConstantOp>(l, retTy,
                                                builder.getZeroAttr(retTy));
        }
        builder.create<scf::YieldOp>(l, ValueRange{v});
        return;
      }
      visitStmt(cur);
      ++from;
    }
    Value zero = builder.create<arith::ConstantOp>(l, retTy,
                                                  builder.getZeroAttr(retTy));
    builder.create<scf::YieldOp>(l, ValueRange{zero});
  }

  //===-- switch lowering ------------------------------------------------===//
  //
  // `switch (cond) { case V: stmts; break; ...; default: stmts; }` lowers to
  // a chain of scf.if (the spirv dialect has a Switch op, but our target's
  // GPUToSPIRV path + the small/dynamic case sets in the demos make an scf.if
  // chain simpler). C switch body is a flat statement list where Case/Default
  // are labels and Break ends a case's run — first group the body statements
  // into per-case buckets, then emit one scf.if per case (cond == val -> that
  // bucket), with `default` as the final else tail. No fallthrough is modeled
  // (every case in the demos ends with break or return); a case without an
  // explicit terminator would fall through to the next bucket's statements,
  // which this lowering does NOT do (recorded limitation).

  struct CaseBucket {
    Value val;            // empty => default
    bool isDefault = false;
    std::vector<ASTNode *> stmts;
  };

  // Group a switch body's flat statement list into per-case buckets. A case's
  // body is its `CaseStmt::sub` (the statement following the label) plus any
  // subsequent sibling statements up to the next Case/Default/Break (C
  // fallthrough into the next label's statements is NOT modeled — every case
  // in the demos ends with break or return). Returns the buckets in source
  // order; sets `defOut` to the default bucket index (-1 if none). Statements
  // before the first label are emitted inline into the current region (rare
  // dead code).
  SmallVector<CaseBucket, 8>
  collectSwitchBuckets(const ASTNode *body, Location l, int &defOut) {
    defOut = -1;
    SmallVector<CaseBucket, 8> buckets;
    const std::vector<NodePtr> *bodyStmts = nullptr;
    if (body && body->getNodeType() == ASTNode::NodeKind::CompoundStmt)
      bodyStmts = &static_cast<const CompoundStmt *>(body)->statements;
    CaseBucket *cur = nullptr;
    if (bodyStmts) {
      for (const auto &s : *bodyStmts) {
        if (s->getNodeType() == ASTNode::NodeKind::CaseStmt) {
          auto *cs = static_cast<CaseStmt *>(s.get());
          CaseBucket b;
          if (!cs->value) { b.isDefault = true; }
          else b.val = loadValue(visitExpr(cs->value.get()), l);
          // The case body starts with the label's `sub`: a single statement or
          // a CompoundStmt (unwrap the latter into individual statements).
          if (cs->sub) {
            if (cs->sub->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
              for (const auto &sub :
                   static_cast<CompoundStmt *>(cs->sub.get())->statements)
                b.stmts.push_back(sub.get());
            } else {
              b.stmts.push_back(cs->sub.get());
            }
          }
          buckets.push_back(std::move(b));
          cur = &buckets.back();
          if (cur->isDefault) defOut = buckets.size() - 1;
        } else if (s->getNodeType() == ASTNode::NodeKind::BreakStmt) {
          // Break ends the current case's statement run (no fallthrough).
          cur = nullptr;
        } else {
          // A sibling statement belonging to the current case (after its `sub`).
          // If no case is active (statement before the first label), emit it
          // inline before the switch chain (rare; switch-in-C allows dead code
          // before the first case).
          if (cur) cur->stmts.push_back(s.get());
          else visitStmt(s.get());
        }
      }
    }
    return buckets;
  }

  // A switch is a value-returning construct when its default case (or, with
  // no default, the fall-through off-the-end) yields a value — i.e. every
  // reachable case ends in `return v;`. We require a default that returns a
  // value (the common `switch { ...; default: return 0; }` shape) so the
  // yield chain always has a value on every path.
  bool isValueReturnSwitch(SwitchStmt *sw) {
    if (!sw->body ||
        sw->body->getNodeType() != ASTNode::NodeKind::CompoundStmt)
      return false;
    int defIdx = -1;
    Location l = loc(sw);
    // Evaluate case values into a throwaway builder position is not needed for
    // the predicate; but collectSwitchBuckets calls visitExpr for case values.
    // To avoid emitting IR during a predicate query, do a lightweight AST walk.
    const auto &ss = static_cast<CompoundStmt *>(sw->body.get())->statements;
    bool anyDefault = false;
    bool defaultReturnsValue = false;
    for (const auto &s : ss) {
      if (s->getNodeType() != ASTNode::NodeKind::CaseStmt) continue;
      auto *cs = static_cast<CaseStmt *>(s.get());
      if (!cs->value) {
        anyDefault = true;
        defaultReturnsValue = trailingReturnValue(cs->sub.get()) != nullptr;
      }
    }
    (void)defIdx; (void)l;
    return anyDefault && defaultReturnsValue;
  }

  // Void/general switch emission: each case body is emitted with visitStmt
  // (case-body `return` becomes func.ReturnOp, legal in void functions; in a
  // value-returning function this path is NOT taken — emitValueReturnSwitch
  // is used instead). break inside a case body is a no-op (each case lives in
  // its own scf.if then-region, so no fallthrough); loopDepth + a breakFlag
  // are pushed so a `break` reached inside a case body does not trip the
  // "break outside loop" error.
  void emitSwitch(SwitchStmt *sw) {
    Location l = loc(sw);
    Value cond = loadValue(visitExpr(sw->cond.get()), l);
    if (!cond) return;
    int defIdx = -1;
    auto buckets = collectSwitchBuckets(sw->body.get(), l, defIdx);

    Value brkFlag = makeFlagSlot(l);
    breakFlags.push_back(brkFlag);
    ++loopDepth;
    ++structuredDepth;

    std::function<void(unsigned)> emitChain = [&](unsigned i) {
      if (i < buckets.size()) {
        CaseBucket &b = buckets[i];
        // The default case is the tail of the else chain: emit its body here
        // (this position is reachable only when no earlier case matched), then
        // stop. (It must not recurse, since no case follows the default in a
        // well-formed switch — and recursing would drop the default body, the
        // bug that left `default: s=300;` unemitted and s stuck at 0.)
        if (b.isDefault) {
          for (ASTNode *st : b.stmts) visitStmt(st);
          return;
        }
        Value eq = builder.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::eq, cond, b.val);
        bool hasElse = (i + 1 < buckets.size()) || defIdx >= 0;
        auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, eq, hasElse);
        auto saved = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        for (ASTNode *st : b.stmts) visitStmt(st);
        if (hasElse) {
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          if (i + 1 < buckets.size()) emitChain(i + 1);
          else if (defIdx >= 0)
            for (ASTNode *st : buckets[defIdx].stmts) visitStmt(st);
        }
        builder.restoreInsertionPoint(saved);
      }
    };
    // If the first bucket is the default (switch starting with default:),
    // emit it unconditionally then the rest — rare; handled by treating the
    // default as the innermost else only when it's not the leading bucket.
    if (buckets.size() > 0 && buckets[0].isDefault) {
      // Leading default: run its statements, then a chain over the rest.
      for (ASTNode *st : buckets[0].stmts) visitStmt(st);
      // Re-emit remaining cases as a chain with no default tail.
      int savedDef = defIdx;
      defIdx = -1;
      // Build a sub-chain starting at 1 by offsetting; simplest is to recurse
      // with a slice via a lambda that skips index 0.
      std::function<void(unsigned)> emitRest = [&](unsigned i) {
        if (i < buckets.size()) {
          CaseBucket &b = buckets[i];
          if (b.isDefault) { emitRest(i + 1); return; }
          Value eq = builder.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::eq, cond, b.val);
          bool hasElse = (i + 1 < buckets.size());
          auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, eq, hasElse);
          auto saved = builder.saveInsertionPoint();
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          for (ASTNode *st : b.stmts) visitStmt(st);
          if (hasElse) {
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            emitRest(i + 1);
          }
          builder.restoreInsertionPoint(saved);
        }
      };
      emitRest(1);
      (void)savedDef;
    } else {
      emitChain(0);
    }

    --structuredDepth;
    --loopDepth;
    breakFlags.pop_back();
  }

  // Value-returning switch: lower the whole switch to a single scf.if chain
  // that yields the function's result type. Each case's body is emitted
  // without its trailing `return v;`, then yields v; the default is the
  // innermost else yield. The chain's result is returned by func.ReturnOp.
  void emitValueReturnSwitch(SwitchStmt *sw) {
    Location l = loc(sw);
    Value cond = loadValue(visitExpr(sw->cond.get()), l);
    if (!cond) return;
    mlir::Type retTy = currentRetTy;
    int defIdx = -1;
    // collectSwitchBuckets emits case-value IR into the current region (fine —
    // those constants live before the scf.if). Case bodies are NOT emitted by
    // collect; they're handled per-bucket below.
    auto buckets = collectSwitchBuckets(sw->body.get(), l, defIdx);

    // Helper: emit a bucket's body (statements 0..n-2, then the last minus its
    // trailing return) and return the value its trailing `return v;` carries,
    // cast to retTy. Used for both case and default buckets.
    auto emitBucketBodyAndValue = [&](CaseBucket &b) -> Value {
      for (size_t k = 0; k + 1 < b.stmts.size(); ++k)
        visitStmt(b.stmts[k]);
      if (!b.stmts.empty())
        emitThenWithoutTrailingReturn(b.stmts.back());
      ASTNode *retNode = b.stmts.empty() ? nullptr
                          : trailingReturnValue(b.stmts.back());
      Value rv = retNode ? visitExpr(retNode) : Value();
      if (rv) { rv = loadValue(rv, loc(retNode)); rv = castValue(rv, retTy, loc(retNode)); }
      else rv = builder.create<arith::ConstantOp>(l, retTy,
                                                  builder.getZeroAttr(retTy));
      return rv;
    };

    ++structuredDepth;
    // Build the case list (non-default buckets in source order).
    SmallVector<unsigned, 8> caseIdx;
    for (unsigned i = 0; i < buckets.size(); ++i)
      if (!buckets[i].isDefault) caseIdx.push_back(i);

    // emitChain(i): emit, in the CURRENT insertion region, the value of the
    // switch from case i onward — either an scf.if (cond==val -> yield caseVal,
    // else -> recurse) whose result is returned, or (at the tail) the default
    // bucket's body + value (or zero). The caller yields the returned value in
    // its own region.
    std::function<Value(unsigned)> emitChain = [&](unsigned i) -> Value {
      if (i < caseIdx.size()) {
        CaseBucket &b = buckets[caseIdx[i]];
        Value eq = builder.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::eq, cond, b.val);
        auto ifOp = builder.create<scf::IfOp>(l, TypeRange{retTy}, eq,
                                              /*withElse=*/true);
        auto saved = builder.saveInsertionPoint();
        // Then: case body, yield its return value (in the then region).
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value thenVal = emitBucketBodyAndValue(b);
        builder.create<scf::YieldOp>(l, ValueRange{thenVal});
        // Else: recurse — emits the rest in the else region and returns its
        // value, which we yield here in the else region before restoring.
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        Value elseVal = emitChain(i + 1);
        builder.create<scf::YieldOp>(l, ValueRange{elseVal});
        builder.restoreInsertionPoint(saved);
        return ifOp.getResult(0);
      }
      // Innermost tail (emitted in the parent's else region): the default
      // bucket's body + value, or zero if no default.
      if (defIdx >= 0) {
        CaseBucket &d = buckets[defIdx];
        return emitBucketBodyAndValue(d);
      }
      return builder.create<arith::ConstantOp>(l, retTy,
                                               builder.getZeroAttr(retTy));
    };
    Value res = emitChain(0);
    --structuredDepth;
    if (res)
      builder.create<func::ReturnOp>(l, ValueRange{res});
  }

  // Emit an if-then-block (a bare ReturnStmt or a CompoundStmt) skipping its
  // trailing ReturnStmt: the return is modeled by the wrap-around in
  // emitStatements, not by an actual (illegal) return inside the scf.if.
  void emitThenWithoutTrailingReturn(ASTNode *thenBlock) {
    if (!thenBlock) return;
    if (thenBlock->getNodeType() == ASTNode::NodeKind::ReturnStmt) return;
    if (thenBlock->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto &ss = static_cast<CompoundStmt *>(thenBlock)->statements;
      // Emit all but the last if it is the trailing return.
      size_t upto = ss.size();
      if (upto > 0 &&
          ss[upto - 1]->getNodeType() == ASTNode::NodeKind::ReturnStmt)
        --upto;
      for (size_t i = 0; i < upto; ++i)
        visitStmt(ss[i].get());
      return;
    }
    visitStmt(thenBlock);
  }

  void visitStmt(ASTNode *n) {
    if (!n) return;
    Location l = loc(n);
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      emitStatements(static_cast<CompoundStmt *>(n)->statements, 0);
      break;
    case ASTNode::NodeKind::ReturnStmt: {
      auto *r = static_cast<ReturnStmt *>(n);
      Value v = r->value ? visitExpr(r->value.get()) : Value();
      if (v) {
        v = loadValue(v, loc(r->value.get()));
        // Bridge index-typed expressions to the function's result type.
        if (currentRetTy && !currentRetTy.isa<NoneType>())
          v = castValue(v, currentRetTy, l);
      }
      builder.create<func::ReturnOp>(l, v ? ValueRange(v) : ValueRange());
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *d = static_cast<DeclStmt *>(n)->decl;
      if (!d) break;
      // Locals become stack allocations tagged with the SPIR-V Function
      // storage class; MemRefToSPIRV refuses to lower allocas whose memory
      // space is not exactly #spirv.storage_class<Function>. __shared__
      // maps to the Workgroup storage class (block-visible shared memory):
      // SPIR-V models workgroup memory as module-scope Variables, not per-
      // thread allocas, so __shared__ decls become memref.global symbols
      // fetched via memref.get_global in the body.
      mlir::Type ty = cvtType(d->type);
      if (!ty) { error(d, "unsupported local type"); break; }
      // Build the memref shape from arrayDims. A scalar decl (no arrayDims)
      // uses a 0-d memref as a mutable slot; `float a[16]` -> memref<16xf32>;
      // `float a[16][8]` -> memref<16x8xf32>. An unsized dimension
      // (extern __shared__ T s[], arrayDims={0}) emits a placeholder size of
      // 1 here and is sized to the workgroup-x spec constant by a post-
      // serialize binary patch (patchDynamicSharedArrays in vc.cpp).
      //
      // A bare struct-value decl (`Accumulator acc;`) returns a 1-D
      // memref<Nxi32> from cvtType (the struct's flat i32 slot layout); its
      // arrayDims are empty, so we adopt the cvtType result's own shape rather
      // than building a 0-d slot. Struct *arrays* (`Accumulator acc[2];`) prepend
      // the array dims onto that shape.
      SmallVector<int64_t, 4> shape;
      bool hasUnsizedDim = false;
      for (size_t di = 0; di < d->arrayDims.size(); ++di) {
        int64_t dim = d->arrayDims[di];
        if (dim <= 0) {
          // extern __shared__ T s[]: unsized (runtime-sized) dimension. We
          // emit a placeholder 1 here and let the post-serialize binary patch
          // (patchDynamicSharedArrays in vc.cpp) rewrite the array's length
          // to the __vc_wg_x spec constant. Only the trailing dim may be
          // unsized (the CUDA form `extern __shared__ T s[]`); a non-trailing
          // zero is still rejected as malformed.
          if (di + 1 != d->arrayDims.size()) {
            error(d, "only the trailing dimension of an extern __shared__ "
                     "array may be unsized");
          }
          hasUnsizedDim = true;
          shape.push_back(0); // placeholder; rewritten by the binary patch
        } else {
          shape.push_back(dim);
        }
      }
      // If the element type is itself a struct memref (cvtType returned a 1-D
      // memref<Nxi32> for a bare RecordType), splice its leading dimension in
      // front of our array dims so a plain `acc;` is memref<Nxi32> (not a 0-d
      // slot of memrefs) and `acc[2];` is memref<2xNxi32>.
      mlir::Type elemTy = ty;
      if (auto mty = ty.dyn_cast<MemRefType>()) {
        if (d->type && isa<RecordType>(d->type)) {
          // Adopt the struct's own i32-slot shape, then any outer array dims.
          SmallVector<int64_t, 4> combined;
          for (int64_t d2 : mty.getShape()) combined.push_back(d2);
          for (int64_t d2 : shape) combined.push_back(d2);
          shape = combined;
          elemTy = mty.getElementType();
        }
      }
      if (d->isShared) {
        // Module-scope global in Workgroup storage, fetched per use.
        Value addr = getOrCreateSharedGlobal(d->name, shape, elemTy, l,
                                             /*dynamic=*/hasUnsizedDim);
        locals[d->name] = addr;
        localTypes[d->name] = d->type;
        // __shared__ decls may not have a non-constant initializer in CUDA
        // (no host-visible init); ignore any initializer for the global.
        break;
      }
      MemRefType slotTy = MemRefType::get(
          shape, elemTy, MemRefLayoutAttrInterface(),
          spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
      // Hoist the allocation to the function entry block: a memref.alloca
      // emitted inside an scf region (loops/if) may not survive the
      // structured-CFG -> SPIR-V legalization cleanly.
      OpBuilder::InsertPoint savedIp;
      bool hoisted = false;
      if (entryBlock && builder.getInsertionBlock() != entryBlock) {
        savedIp = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(entryBlock);
        hoisted = true;
      }
      Value addr = builder.create<memref::AllocaOp>(l, slotTy);
      if (hoisted) builder.restoreInsertionPoint(savedIp);
      locals[d->name] = addr;
      localTypes[d->name] = d->type;
      if (d->init) {
        if (d->init->getNodeType() == ASTNode::NodeKind::InitListExpr) {
          auto *il = static_cast<InitListExpr *>(d->init.get());
          int64_t total = 1;
          for (int64_t dim : shape) total *= dim;
          for (size_t i = 0; i < il->elements.size() && (int64_t)i < total; ++i) {
            SmallVector<Value, 1> idx;
            idx.push_back(builder.create<arith::ConstantOp>(
                l, builder.getIndexType(), builder.getIndexAttr(i)));
            storeTo(addr, idx, visitExpr(il->elements[i].get()), l);
          }
        } else {
          storeValue(addr, visitExpr(d->init.get()), loc(d));
        }
      }
      break;
    }
    case ASTNode::NodeKind::ExprStmt:
      if (auto *e = static_cast<ExprStmt *>(n)->expr.get())
        (void)visitExpr(e);
      break;
    case ASTNode::NodeKind::IfStmt:
      emitIf(static_cast<IfStmt *>(n));
      break;
    case ASTNode::NodeKind::ForStmt:
      emitFor(static_cast<ForStmt *>(n));
      break;
    case ASTNode::NodeKind::WhileStmt:
      emitWhile(static_cast<WhileStmt *>(n));
      break;
    case ASTNode::NodeKind::DoStmt:
      emitDo(static_cast<DoStmt *>(n));
      break;
    case ASTNode::NodeKind::BreakStmt:
      // Handled inside loop bodies via loop-carried exit flags when a loop is
      // active; a bare break outside a loop is unreachable in valid C.
      if (loopDepth == 0)
        error(n, "'break' outside of a loop is not supported");
      else
        emitBreak(n);
      break;
    case ASTNode::NodeKind::ContinueStmt:
      if (loopDepth == 0)
        error(n, "'continue' outside of a loop is not supported");
      else
        emitContinue(n);
      break;
    case ASTNode::NodeKind::SwitchStmt:
      emitSwitch(static_cast<SwitchStmt *>(n));
      break;
    case ASTNode::NodeKind::CaseStmt:
      // A CaseStmt reached outside a SwitchStmt body is unreachable in valid C
      // (the SwitchStmt handler walks the body directly). No-op rather than
      // crash if it somehow appears standalone.
      break;
    default:
      error(n, std::string("MLIR backend cannot lower ") +
                  nodeKindName(n->getNodeType()) + " statement; dropped");
      break;
    }
  }

  void emitIf(const IfStmt *s) {
    Location l = loc(s);
    Value cond = toI1(visitExpr(s->cond.get()), l);
    if (!cond) return;
    bool hasElse = bool(s->elseStmt);
    auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, cond, hasElse);

    auto saved = builder.saveInsertionPoint();
    ++structuredDepth;
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (s->thenStmt) visitStmt(s->thenStmt.get());
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (hasElse) visitStmt(s->elseStmt.get());
    --structuredDepth;
    builder.restoreInsertionPoint(saved);
  }

  void emitFor(const ForStmt *s) {
    // for (init; cond; step) lowers to
    //   init;
    //   scf.while { cond && !breakReq } do { body; step }
    // which preserves C evaluation semantics exactly (cond/step may depend on
    // locals through memref slots). A tighter scf.for lowering could be
    // added later once cond is a canonical `i < bound` form.
    //
    // NOTE: unlike scf.if, scf.while's default builder only adds empty
    // regions -- we must create the entry block ourselves via createBlock
    // (each call also makes that block the insertion point).
    Location l = loc(s);
    if (s->init) visitStmt(s->init.get());
    Value brkFlag = makeFlagSlot(l);
    Value cntFlag = makeFlagSlot(l);
    breakFlags.push_back(brkFlag);
    continueFlags.push_back(cntFlag);
    ++loopDepth;
    ++structuredDepth;
    auto whileOp = builder.create<scf::WhileOp>(l, TypeRange{}, ValueRange{});

    auto saved = builder.saveInsertionPoint();
    builder.createBlock(&whileOp.getBefore());
    // Reset the continue flag at the top of each iteration (a continue only
    // suppresses the rest of ONE iteration); break persists to exit.
    storeFlag(cntFlag, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(false)), l);
    Value cond = s->cond ? toI1(visitExpr(s->cond.get()), l)
                         : builder.create<arith::ConstantOp>(
                               l, builder.getBoolAttr(true));
    if (!cond) { builder.restoreInsertionPoint(saved); return; }
    Value notBrk = builder.create<arith::XOrIOp>(
        l, loadFlag(brkFlag, l),
        builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)));
    Value keep = builder.create<arith::AndIOp>(l, cond, notBrk);
    builder.create<scf::ConditionOp>(l, keep, ValueRange{});
    builder.createBlock(&whileOp.getAfter());
    bool prevGuard = guarding;
    guarding = true;
    if (s->body) visitStmt(s->body.get());
    // The step runs unless a break was requested (continue only skips to the
    // step, then the next iteration's condition).
    guarding = prevGuard;
    if (s->step) {
      Value notBrk2 = builder.create<arith::XOrIOp>(
          l, loadFlag(brkFlag, l),
          builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)));
      auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, notBrk2, false);
      auto s2 = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      (void)visitExpr(s->step.get());
      builder.restoreInsertionPoint(s2);
    }
    builder.create<scf::YieldOp>(l);
    builder.restoreInsertionPoint(saved);
    breakFlags.pop_back();
    continueFlags.pop_back();
    --loopDepth;
    --structuredDepth;
  }

  void emitWhile(const WhileStmt *s) {
    Location l = loc(s);
    Value brkFlag = makeFlagSlot(l);
    Value cntFlag = makeFlagSlot(l);
    breakFlags.push_back(brkFlag);
    continueFlags.push_back(cntFlag);
    ++loopDepth;
    ++structuredDepth;
    auto whileOp = builder.create<scf::WhileOp>(l, TypeRange{}, ValueRange{});

    auto saved = builder.saveInsertionPoint();
    builder.createBlock(&whileOp.getBefore());
    storeFlag(cntFlag, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(false)), l);
    Value cond = toI1(visitExpr(s->cond.get()), l);
    if (!cond) { builder.restoreInsertionPoint(saved); return; }
    Value notBrk = builder.create<arith::XOrIOp>(
        l, loadFlag(brkFlag, l),
        builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)));
    Value keep = builder.create<arith::AndIOp>(l, cond, notBrk);
    builder.create<scf::ConditionOp>(l, keep, ValueRange{});
    builder.createBlock(&whileOp.getAfter());
    bool prevGuard = guarding;
    guarding = true;
    if (s->body) visitStmt(s->body.get());
    guarding = prevGuard;
    builder.create<scf::YieldOp>(l);
    builder.restoreInsertionPoint(saved);
    breakFlags.pop_back();
    continueFlags.pop_back();
    --loopDepth;
    --structuredDepth;
  }

  void emitDo(const DoStmt *s) {
    // do { body } while (cond) runs the body once, then repeats while cond.
    // A first-iteration flag makes the scf.while condition region start true.
    // i32 (not i1) storage: SPIR-V has no 1-bit store (see makeFlagSlot).
    Location l = loc(s);
    auto flagTy =
        MemRefType::get(ArrayRef<int64_t>{}, builder.getI32Type(),
                        MemRefLayoutAttrInterface(),
                        spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
    Value flag = builder.create<memref::AllocaOp>(l, flagTy);
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getI32IntegerAttr(1)),
        flag);
    Value brkFlag = makeFlagSlot(l);
    Value cntFlag = makeFlagSlot(l);
    breakFlags.push_back(brkFlag);
    continueFlags.push_back(cntFlag);
    ++loopDepth;
    ++structuredDepth;
    auto whileOp = builder.create<scf::WhileOp>(l, TypeRange{}, ValueRange{});

    auto saved = builder.saveInsertionPoint();
    builder.createBlock(&whileOp.getBefore());
    storeFlag(cntFlag, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(false)), l);
    Value firstRaw = builder.create<memref::LoadOp>(l, flag, ValueRange{});
    Value first = toI1(firstRaw, l);
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getI32IntegerAttr(0)),
        flag);
    Value cond = s->cond ? toI1(visitExpr(s->cond.get()), l) : Value();
    if (!cond) { builder.restoreInsertionPoint(saved); return; }
    Value notBrk = builder.create<arith::XOrIOp>(
        l, loadFlag(brkFlag, l),
        builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)));
    Value pass = builder.create<arith::OrIOp>(l, first, cond);
    pass = builder.create<arith::AndIOp>(l, pass, notBrk);
    builder.create<scf::ConditionOp>(l, pass, ValueRange{});
    builder.createBlock(&whileOp.getAfter());
    bool prevGuard = guarding;
    guarding = true;
    if (s->body) visitStmt(s->body.get());
    guarding = prevGuard;
    builder.create<scf::YieldOp>(l);
    builder.restoreInsertionPoint(saved);
    breakFlags.pop_back();
    continueFlags.pop_back();
    --loopDepth;
    --structuredDepth;
  }

  //--- break/continue ------------------------------------------------------//

  // A break/continue flag slot. SPIR-V has no 1-bit storage (a memref<i1>
  // store fails to legalize through GPUToSPIRV — see cvtType's Bool→i32 note),
  // so the slot is an i32 holding 0/1. loadFlag bridges back to i1 so the
  // existing boolean logic at the call sites (XOrIOp/AndIOp/OrIOp on i1) is
  // unchanged.
  Value makeFlagSlot(const Location &l) {
    auto flagTy =
        MemRefType::get(ArrayRef<int64_t>{}, builder.getI32Type(),
                        MemRefLayoutAttrInterface(),
                        spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
    Value slot = builder.create<memref::AllocaOp>(l, flagTy);
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getI32IntegerAttr(0)),
        slot);
    return slot;
  }

  Value loadFlag(Value slot, const Location &l) {
    Value raw = builder.create<memref::LoadOp>(l, slot, ValueRange{});
    return toI1(raw, l);
  }

  void storeFlag(Value slot, Value v, const Location &l) {
    // v is an i1 (truthiness); store as i32 0/1 for SPIR-V storage.
    Value asI1 = toI1(v, l);
    Value asI32 = builder.create<arith::ExtUIOp>(
        l, builder.getI32Type(), asI1);
    builder.create<memref::StoreOp>(l, asI32, slot);
  }

  // A break/continue sets the innermost loop's flag true. Subsequent guarded
  // statements in this iteration are skipped (see emitGuarded).
  void emitBreak(const ASTNode *n) {
    Location l = loc(n);
    Value slot = breakFlags.back();
    storeFlag(slot, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)),
              l);
  }
  void emitContinue(const ASTNode *n) {
    Location l = loc(n);
    Value slot = continueFlags.back();
    storeFlag(slot, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)),
              l);
  }

  // Whether the innermost loop has a break/continue pending this iteration.
  Value loopPending(const Location &l) {
    Value brk = loadFlag(breakFlags.back(), l);
    Value cnt = loadFlag(continueFlags.back(), l);
    Value any = builder.create<arith::OrIOp>(l, brk, cnt);
    Value notAny = builder.create<arith::XOrIOp>(
        l, any, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)));
    return notAny; // true = keep executing this iteration
  }

  // Emit a statement, guarded by the current loop's break/continue flags so
  // that code after a break/continue is skipped within the same iteration.
  void emitGuarded(ASTNode *n) {
    if (!n) return;
    if (loopDepth == 0 || !guarding) { visitStmt(n); return; }
    Location l = loc(n);
    Value keep = loopPending(l);
    auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, keep, /*else=*/false);
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    visitStmt(n);
    builder.restoreInsertionPoint(saved);
  }

  //===--------------------------------------------------------------------//
  // Expressions
  //===--------------------------------------------------------------------//

  Value visitExpr(ASTNode *n) {
    if (!n) return Value();
    Location l = loc(n);
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral:
      return builder.create<arith::ConstantOp>(
          l, builder.getI32IntegerAttr(
                 static_cast<IntegerLiteral *>(n)->value));
    case ASTNode::NodeKind::FloatLiteral: {
      auto *fl = static_cast<FloatLiteral *>(n);
      double v = fl->value;
      // A bare `1.5` literal is double (f64) by C rules; `1.5f` is float (f32).
      // The parser sets isFloat32 only for the `f`/`F` suffix. Emitting a bare
      // literal as f32 would make `double x = 1.5;` / `(double)i + 1.5` mix
      // f64 and f32 operands and fail arith.addf type checking.
      mlir::Type fltTy = fl->isFloat32 ? builder.getF32Type()
                                       : builder.getF64Type();
      return builder.create<arith::ConstantOp>(
          l, fltTy, builder.getFloatAttr(fltTy, v));
    }
    case ASTNode::NodeKind::CharLiteral:
      return builder.create<arith::ConstantOp>(
          l, builder.getI32IntegerAttr(static_cast<CharLiteral *>(n)->value));
    case ASTNode::NodeKind::BoolLiteral:
      // true/false -> i32 1/0 (bool is represented as i32 throughout; see
      // cvtType). Never i1, which has no SPIR-V storage.
      return builder.create<arith::ConstantOp>(
          l, builder.getI32IntegerAttr(
                 static_cast<BoolLiteral *>(n)->value ? 1 : 0));
    case ASTNode::NodeKind::DeclRefExpr: {
      auto name = static_cast<DeclRefExpr *>(n)->name;
      auto it = locals.find(name);
      if (it != locals.end())
        return it->second; // a memref slot or block arg
      // A __constant__ global (scalar or array) referenced from device code.
      auto cg = constGlobals.find(name);
      if (cg != constGlobals.end())
        return materializeConstGlobal(cg->second, n);
      // Unscoped enum constant — fold to its integer literal. Sema registers
      // these (top-level or mangled ns_NAME); the spirv module has no `const
      // int NAME=val;` decl, so we materialize the value at the use site.
      auto ec = enumConstants.find(name);
      if (ec != enumConstants.end())
        return builder.create<arith::ConstantOp>(
            l, builder.getI32IntegerAttr(static_cast<int32_t>(ec->second)));
      // warpSize — the runtime subgroup size. Lowered to the spirv SubgroupSize
      // builtin (a module-scope Input global of i32), not a constant: the
      // device's subgroup size is adaptive (32 on typical GPUs, but not
      // guaranteed), mirroring the GLSL backend's `int(gl_SubgroupSize)`.
      if (name == "warpSize")
        return getSubgroupSize(l);
      return error(n, std::string("use of undeclared identifier '") +
                          name.str() + "' in device code");
    }
    case ASTNode::NodeKind::UnaryExpr:
      return emitUnary(static_cast<UnaryExpr *>(n));
    case ASTNode::NodeKind::BinaryExpr:
      return emitBinary(static_cast<BinaryExpr *>(n));
    case ASTNode::NodeKind::ConditionalExpr:
      return emitConditional(static_cast<ConditionalExpr *>(n));
    case ASTNode::NodeKind::CStyleCastExpr: {
      auto *c = static_cast<CStyleCastExpr *>(n);
      mlir::Type to = cvtType(c->target);
      if (!to) return error(n, "cast to unsupported type");
      return castValue(visitExpr(c->sub.get()), to, l);
    }
    case ASTNode::NodeKind::CommaExpr: {
      // `(a, b)`: evaluate a for its side effects, discard it, yield b.
      auto *ce = static_cast<CommaExpr *>(n);
      (void)visitExpr(ce->lhs.get());
      return visitExpr(ce->rhs.get());
    }
    case ASTNode::NodeKind::SizeOfExpr: {
      // Sema folds the byte count into `folded` during type checking
      // (sizeOfType: half=2, int/float=4, long/double/ptr=8, vector=elem*count,
      // record=sum of field sizes). GLSL emits `const int NAME=val;` and lets
      // glslc resolve bare names; the MLIR/spirv module has no such decl, so
      // fold to a literal here. Sizes are small and always fit i32.
      auto *s = static_cast<SizeOfExpr *>(n);
      int64_t v = s->folded;
      if (v == 0) {
        // Sema could not fold (warned at the Sema pass). Fall back to 0 so the
        // shader still compiles rather than crashing codegen.
        v = 0;
      }
      return builder.create<arith::ConstantOp>(l, builder.getI32IntegerAttr(
                                                       static_cast<int32_t>(v)));
    }
    case ASTNode::NodeKind::IndexExpr: {
      // base[idx...] where base is a memref arg / slot or a shared spirv.ptr.
      // Reuses lvalueAddress so multi-dimensional `a[i][j]` collects every
      // subscript into one load, and so shared-array indexing (spirv.AccessChain
      // + spirv.Load) is handled uniformly.
      auto *ie = static_cast<IndexExpr *>(n);
      // Struct array-field element read: `pts[i].v[k]` where `v` is a vector
      // field. The i32 slot holds the element's bit pattern; bitcast back to
      // the field's scalar element type (f32/i32/...). lvalueAddress resolves
      // the slot; we just retype the loaded value.
      if (ie->base && ie->base->getNodeType() ==
                          ASTNode::NodeKind::MemberAccessExpr) {
        Value mem;
        SmallVector<Value, 4> idxs;
        if (lvalueAddress(ie, mem, idxs) && mem.getType().isa<MemRefType>() &&
            idxs.size() == 1) {
          // Confirm this was the struct-array-field path by checking the base
          // member resolves to a vector field (structFieldAddr-only path sets
          // a single index; the generic memref path also can, so guard by
          // re-checking the member is a struct array field).
          auto *ma = static_cast<MemberAccessExpr *>(ie->base.get());
          Value fm, si;
          mlir::Type fty;
          std::vector<int64_t> adims;
          if (structFieldAddr(ma, l, Value(), fm, si, fty, &adims) &&
              !adims.empty()) {
            mlir::Type elemTy = fty;
            Value raw = builder.create<memref::LoadOp>(l, mem, idxs);
            if (elemTy.isF32())
              return builder.create<arith::BitcastOp>(l, builder.getF32Type(),
                                                      raw);
            if (elemTy.isF16())
              return builder.create<arith::BitcastOp>(l, builder.getF16Type(),
                                                      raw);
            if (elemTy.isF64()) {
              // 8-byte element: load lo+hi and combine (rare for array fields).
              Value one = builder.create<arith::ConstantOp>(
                  l, builder.getIndexType(), builder.getIndexAttr(1));
              Value slotHi = builder.create<arith::AddIOp>(l, idxs[0], one);
              Value hi = builder.create<memref::LoadOp>(
                  l, mem, ValueRange{slotHi});
              Value lo64 = builder.create<arith::ExtUIOp>(
                  l, builder.getI64Type(), raw);
              Value hi64 = builder.create<arith::ExtUIOp>(
                  l, builder.getI64Type(), hi);
              Value sh = builder.create<arith::ConstantOp>(
                  l, builder.getI64Type(), builder.getI64IntegerAttr(32));
              Value hiUp = builder.create<arith::ShLIOp>(l, hi64, sh);
              Value asI64 = builder.create<arith::OrIOp>(l, lo64, hiUp);
              return builder.create<arith::BitcastOp>(l, builder.getF64Type(),
                                                      asI64);
            }
            return castValue(raw, elemTy, l);
          }
        }
      }
      // Vector-pointer element read: `float4* p; p[i]` loads N consecutive
      // scalar slots (the pointer is scalarized to memref<?xELEM>) and
      // CompositeConstructs them into a vector<NxELEM>. See
      // loadVectorPointerElement for why the memref can't be <?xvector<...>>.
      if (ie->base && ie->base->getNodeType() ==
                          ASTNode::NodeKind::DeclRefExpr) {
        if (Value v = loadVectorPointerElement(ie, l))
          return v;
      }
      Value loaded = loadLValue(ie, l);
      if (!loaded)
        return error(ie, "subscript of non-array value");
      return loaded;
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      // threadIdx.x / blockIdx.x / blockDim.x / gridDim.x
      auto *m = static_cast<MemberAccessExpr *>(n);
      if (m->base &&
          m->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *base = static_cast<DeclRefExpr *>(m->base.get());
        vc::Dim dim = vc::Dim::x;
        if (m->member == "y") dim = vc::Dim::y;
        else if (m->member == "z") dim = vc::Dim::z;
        Value idxV;
        if (base->name == "threadIdx")
          idxV = builder.create<vc::ThreadIdOp>(l, builder.getIndexType(), dim);
        else if (base->name == "blockIdx")
          idxV = builder.create<vc::BlockIdOp>(l, builder.getIndexType(), dim);
        else if (base->name == "blockDim")
          idxV = builder.create<vc::BlockDimOp>(l, builder.getIndexType(), dim);
        else if (base->name == "gridDim")
          idxV = builder.create<vc::GridDimOp>(l, builder.getIndexType(), dim);
        if (idxV)
          // Language-level ints are 32-bit; keep index only where needed.
          return builder.create<arith::IndexCastOp>(
              l, builder.getI32Type(), idxV);
      }
      // Struct field read: `pts[i].x` / `result[i].field` where the base is a
      // pointer-to-struct SSBO. Symmetric to storeStructField's offset math:
      // byteOff = elemIdx*stride + fieldOffset; slot = byteOff/4; load i32 (or
      // lo+hi for 8-byte) and bitcast back to the field type. Try this BEFORE
      // the vector-swizzle path (a struct field of vector type would otherwise
      // be misread; the demos only read scalar fields this way).
      if (Value fv = loadStructField(m, l))
        return fv;
      // Vector swizzle read: base is a vector value, member is one of
      // x/y/z/w (single component -> scalar extract) or a multi-char swizzle
      // like xy/xyz/xz (-> vector shuffle selecting a subset). The GLSL backend
      // emits these verbatim (GLSL native); here we lower to spirv ops since
      // the MLIR/spirv dialect has no swizzle sugar.
      Value base = visitExpr(m->base.get());
      if (base) base = loadValue(base, l);
      if (base && base.getType().isa<mlir::VectorType>()) {
        auto vty = base.getType().cast<mlir::VectorType>();
        unsigned srcN = vty.getNumElements();
        // Map swizzle chars to element indices.
        auto charToIdx = [](char c) -> int {
          switch (c) {
          case 'x': case 'r': case 's': return 0;
          case 'y': case 'g': case 't': return 1;
          case 'z': case 'b': case 'p': return 2;
          case 'w': case 'a': case 'q': return 3;
          default: return -1;
          }
        };
        llvm::StringRef sw = m->member;
        SmallVector<int32_t, 4> idxs;
        bool ok = true;
        for (char c : sw) {
          int i = charToIdx(c);
          if (i < 0 || (unsigned)i >= srcN) { ok = false; break; }
          idxs.push_back(i);
        }
        if (ok && !idxs.empty()) {
          if (idxs.size() == 1) {
            // Single component -> CompositeExtract to a scalar.
            return builder.create<spirv::CompositeExtractOp>(
                l, vty.getElementType(), base,
                builder.getI32ArrayAttr(idxs));
          }
          // Multi-component -> VectorShuffle selecting `idxs` from base,based
          // (same vector as both inputs).
          mlir::VectorType resTy =
              mlir::VectorType::get({(int64_t)idxs.size()}, vty.getElementType());
          return builder.create<spirv::VectorShuffleOp>(
              l, resTy, base, base, builder.getI32ArrayAttr(idxs));
        }
      }
      return error(n, "MLIR backend does not support member access "
                      "other than threadIdx/blockIdx/blockDim/gridDim "
                      "and vector swizzles");
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<CallExpr *>(n);
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        auto *ref = static_cast<DeclRefExpr *>(c->callee.get());
        if (ref->name == "__syncthreads") {
          builder.create<vc::BarrierOp>(l);
          return Value();
        }
        // Kernel-internal printf is supported only in the GLSL backend (it
        // lowers to GL_EXT_debug_printf -> NonSemantic.DebugPrintf, captured by
        // the validation layer). The MLIR SPIR-V path does not emit arbitrary
        // NonSemantic ExtInst sets, so emit a clear diagnostic instead of
        // silently mis-lowering. Use the GLSL backend (vcc) for kernel printf.
        if (ref->name == "printf") {
          return error(n, "kernel printf is not supported in the MLIR backend; "
                          "use the GLSL backend (vcc) or build with -emit=glsl");
        }
        // Vector constructors: float4(...) / int3(...) / make_float4(...) /
        // uint4(...) / double2(...) / long4(...) / float3(...) / float2(...).
        // CUDA's make_<vec> drops the "make_" prefix to yield the vec name.
        // These lower to spirv.CompositeConstruct over the (casted) scalar
        // args; the result is a vector<NxELEM> SSA value (never stored in a
        // memref — see cvtType's VectorType note). Dispatch before the
        // builtin/funcTable lookup so a user __device__ helper named e.g.
        // `int4` would not be shadowed (no such helpers exist in practice).
        llvm::StringRef ctorName = ref->name;
        if (ctorName.starts_with("make_"))
          ctorName = ctorName.substr(5);
        mlir::Type vecElemTy;
        unsigned vecCount = 0;
        if (vectorCtorInfo(ctorName, vecElemTy, vecCount)) {
          if (c->args.empty())
            return error(n, "vector constructor needs at least one argument");
          SmallVector<Value> parts;
          for (auto &a : c->args) {
            Value av = loadValue(visitExpr(a.get()), loc(a.get()));
            if (!av)
              return error(a.get(), "could not evaluate vector ctor argument");
            av = castValue(av, vecElemTy, loc(a.get()));
            parts.push_back(av);
          }
          // Single-arg broadcast: float4(1.0) -> CompositeConstruct of the same
          // value N times.
          if (parts.size() == 1 && vecCount > 1)
            parts.assign(vecCount, parts[0]);
          if (parts.size() != vecCount)
            return error(n, "vector constructor argument count mismatch");
          mlir::VectorType vty = mlir::VectorType::get({vecCount}, vecElemTy);
          return builder.create<spirv::CompositeConstructOp>(l, vty, parts);
        }
        // CUDA builtins (math intrinsics, atomics, fences, votes). These are
        // NOT __device__ helpers (funcTable miss), so dispatch before the
        // funcTable lookup. `matched` is set true if `name` is a recognized
        // builtin (even a void-returning one); the result Value is null for
        // void builtins.
        bool matched = false;
        if (auto bv = emitBuiltinCall(ref->name, c->args, l, matched))
          return bv;
        if (matched) return Value();
        // When Sema resolved this call to a specific overload (resolvedCallee),
        // look up the callee by its parameter-mangled device symbol so distinct
        // overloads (f(int)->f_i, f(float)->f_f) hit the right funcTable entry.
        // Builtins have no resolvedCallee and never reach here (matched above).
        std::string calleeSym = c->resolvedCallee
                                    ? deviceMangledName(c->resolvedCallee)
                                    : ref->name.str();
        const FunctionDecl *calleeFn = c->resolvedCallee;
        auto fit = funcTable.find(calleeSym);
        if (fit != funcTable.end()) {
          SmallVector<Value> args;
          for (auto &a : c->args) {
            // A by-value struct argument is scalarized into N field scalars
            // matching the callee's scalarized signature (see buildFunction).
            if (expandStructArg(a.get(), args, loc(a.get())))
              continue;
            Value av = visitExpr(a.get());
            if (!av) return error(a.get(), "could not evaluate call argument");
            args.push_back(loadValue(av, loc(a.get())));
          }
          // Complete trailing defaulted parameters the call omits, mirroring
          // the GLSL backend: append each default expression from the callee
          // signature until the argument count matches the parameter count.
          if (!calleeFn) {
            auto dit = funcDecls.find(calleeSym);
            if (dit != funcDecls.end()) calleeFn = dit->second;
          }
          if (calleeFn) {
            for (unsigned i = c->args.size(); i < calleeFn->params.size(); ++i) {
              if (!calleeFn->params[i]->defaultVal) break;
              Value dv = visitExpr(calleeFn->params[i]->defaultVal.get());
              if (!dv) return error(calleeFn->params[i]->defaultVal.get(),
                                   "could not evaluate default argument");
              args.push_back(loadValue(dv, loc(calleeFn->params[i]->defaultVal.get())));
            }
          }
          auto call = builder.create<func::CallOp>(l, fit->second, args);
          FunctionType fty = fit->second.getFunctionType();
          if (fty.getNumResults() == 0) return Value();
          return call.getResult(0);
        }
      }
      // Scoped call `ns::func(args)` (MemberAccessExpr callee with isScope):
      // mangle to `ns_func` and look up in funcTable, mirroring the host's
      // launchHandleName and buildFunction's deviceMangledName. The scope chain is
      // left-nested: MemberAccessExpr(base=MemberAccessExpr(...), member=f) for
      // `outer::inner::f`, flattened to `outer_inner_f`.
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
        auto *ma = static_cast<MemberAccessExpr *>(c->callee.get());
        if (ma->isScope) {
          // Walk the scope chain to collect [outer, inner, ..., func].
          std::vector<std::string> parts;
          parts.push_back(ma->member.str());
          const ASTNode *cur = ma->base.get();
          while (cur) {
            if (cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
              auto *sub = static_cast<const MemberAccessExpr *>(cur);
              parts.push_back(sub->member.str());
              cur = sub->base.get();
            } else if (cur->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
              parts.push_back(static_cast<const DeclRefExpr *>(cur)->name.str());
              break;
            } else
              break;
          }
          std::reverse(parts.begin(), parts.end());
          std::string chain;
          for (auto &p : parts) {
            if (!chain.empty()) chain += "::";
            chain += p;
          }
          // Prefer the Sema-resolved callee's parameter-mangled symbol so
          // overloaded scoped functions bind to the right funcTable entry;
          // fall back to the bare scope-chain mangle when unresolved.
          std::string mangled = c->resolvedCallee
                                    ? deviceMangledName(c->resolvedCallee)
                                    : mangleScopeName(chain);
          const FunctionDecl *calleeFn = c->resolvedCallee;
          auto fit = funcTable.find(mangled);
          if (fit != funcTable.end()) {
            SmallVector<Value> args;
            for (auto &a : c->args) {
              Value av = visitExpr(a.get());
              if (!av) return error(a.get(), "could not evaluate call argument");
              args.push_back(loadValue(av, loc(a.get())));
            }
            if (!calleeFn) {
              auto dit = funcDecls.find(mangled);
              if (dit != funcDecls.end()) calleeFn = dit->second;
            }
            if (calleeFn) {
              for (unsigned i = c->args.size(); i < calleeFn->params.size(); ++i) {
                if (!calleeFn->params[i]->defaultVal) break;
                Value dv = visitExpr(calleeFn->params[i]->defaultVal.get());
                if (!dv) return error(calleeFn->params[i]->defaultVal.get(),
                                     "could not evaluate default argument");
                args.push_back(loadValue(dv, loc(calleeFn->params[i]->defaultVal.get())));
              }
            }
            auto call = builder.create<func::CallOp>(l, fit->second, args);
            FunctionType fty = fit->second.getFunctionType();
            if (fty.getNumResults() == 0) return Value();
            return call.getResult(0);
          }
        } else {
          // Object method call `obj.method(args)` (non-scope MemberAccessExpr
          // callee). The method is `Class_method` taking `this` as its leading
          // parameter (see buildFunction / deviceMangledName). SPIR-V's
          // func.call cannot pass a Function-storage struct memref as a by-ref
          // parameter (FuncToSPIRV + GPUToSPIRV leave an
          // unrealized_conversion_cast they can't legalize for memref args), so
          // the MLIR backend INLINES the method body at the call site with
          // `_this` bound to the caller's object slot — a method that mutates
          // `this->field` then writes directly to the caller's object, matching
          // C++ reference semantics. (The method is still emitted as a
          // func.func for completeness, but never func.called.) This mirrors
          // how the GLSL backend's `inout Class _this` achieves by-ref.
          std::string suffix = "_" + ma->member.str();
          // Prefer the Sema-resolved method (resolvedCallee) so overloaded
          // methods bind correctly; fall back to the funcDecls suffix scan for
          // single-method classes when Sema didn't resolve (shouldn't happen
          // for device code, but keeps the path defensive).
          const FunctionDecl *methodFn = c->resolvedCallee;
          if (!methodFn) {
            for (auto &kv : funcDecls) {
              if (kv.first.ends_with(suffix) && kv.second->isMethod) {
                if (methodFn) { methodFn = nullptr; break; } // ambiguous
                methodFn = kv.second;
              }
            }
          }
          if (methodFn && methodFn->body) {
            // Resolve the object's memref slot + record type.
            Value base = visitExpr(ma->base.get());
            if (!base)
              return error(ma->base.get(),
                           "could not evaluate method-call object");
            // Recover the object's record type from localTypes (base must be a
            // DeclRefExpr to a local struct value).
            const RecordType *objRec = nullptr;
            if (ma->base &&
                ma->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
              auto *ref = static_cast<DeclRefExpr *>(ma->base.get());
              auto tit = localTypes.find(ref->name);
              if (tit != localTypes.end() && tit->second &&
                  isa<RecordType>(tit->second))
                objRec = cast<RecordType>(tit->second);
            }
            if (!objRec)
              return error(ma->base.get(),
                           "method call on a non-struct object is unsupported");

            // Evaluate the explicit arguments (and complete defaulted ones)
            // into values BEFORE swapping in the method's parameter bindings,
            // so argument expressions still see the caller's scope.
            SmallVector<Value> argVals;
            SmallVector<const ParamDecl *> boundParams;
            for (auto &a : c->args) {
              if (expandStructArg(a.get(), argVals, loc(a.get())))
                continue;
              Value av = visitExpr(a.get());
              if (!av) return error(a.get(), "could not evaluate call argument");
              argVals.push_back(loadValue(av, loc(a.get())));
            }
            unsigned ai = 0;
            for (auto *p : methodFn->params) {
              if (ai < argVals.size()) {
                boundParams.push_back(p);
                ++ai;
              } else if (p->defaultVal) {
                Value dv = visitExpr(p->defaultVal.get());
                if (!dv) return error(p->defaultVal.get(),
                                     "could not evaluate default argument");
                argVals.push_back(loadValue(dv, loc(p->defaultVal.get())));
                boundParams.push_back(p);
              } else
                break;
            }

            // Save the caller's per-function symbol state; the inlined body
            // gets a fresh scope with `_this` + the method's params bound.
            llvm::StringMap<Value> savedLocals = locals;
            llvm::StringMap<const vc::Type *> savedLocalTypes = localTypes;
            auto savedRetTy = currentRetTy;
            unsigned savedLoopDepth = loopDepth;
            unsigned savedStructuredDepth = structuredDepth;
            mlir::Type methodRetTy = cvtType(methodFn->returnType);
            currentRetTy = methodRetTy;

            // Bind `_this` to the caller's object slot.
            locals.clear();
            localTypes.clear();
            locals["_this"] = base;
            localTypes["_this"] = objRec;
            // Bind the method's parameters to fresh slots holding the arg
            // values (by value; the demos' methods don't mutate their params).
            for (unsigned i = 0; i < boundParams.size() && i < argVals.size();
                 ++i) {
              const ParamDecl *p = boundParams[i];
              mlir::Type pt = cvtType(p->type);
              if (!pt) continue;
              MemRefType slotTy = MemRefType::get(
                  {}, pt, MemRefLayoutAttrInterface(),
                  spirv::StorageClassAttr::get(&ctx,
                                                spirv::StorageClass::Function));
              Value slot = builder.create<memref::AllocaOp>(loc(p), slotTy);
              locals[p->name] = slot;
              localTypes[p->name] = p->type;
              storeValue(slot, argVals[i], loc(p));
            }

            if (methodFn->body &&
                methodFn->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
              auto *cs = static_cast<CompoundStmt *>(methodFn->body.get());
              for (auto &s : cs->statements) visitStmt(s.get());
            } else if (methodFn->body)
              visitStmt(methodFn->body.get());

            // Restore the caller's scope.
            locals = savedLocals;
            localTypes = savedLocalTypes;
            currentRetTy = savedRetTy;
            loopDepth = savedLoopDepth;
            structuredDepth = savedStructuredDepth;
            // A void method yields nothing; a value-returning method would need
            // a result binding, but the demos' methods are all void.
            return Value();
          }
        }
      }
      return error(n, "MLIR backend does not support this call "
                      "(unknown callee or builtin); call dropped");
    }
    default:
      return error(n, std::string("MLIR backend cannot lower ") +
                          nodeKindName(n->getNodeType()));
    }
  }

  //===--------------------------------------------------------------------//
  // CUDA builtin lowering (math intrinsics, atomics, fences, votes)
  //===--------------------------------------------------------------------//

  // Recognize a CUDA-style vector constructor name (float4, int3, uint2,
  // double2, long4, ...) and return its element MLIR type + component count.
  // Mirrors Parser::makeVectorType's base table. Returns false for non-vector
  // names. Used by the CallExpr handler to lower `float4(a,b,c,d)` and
  // `make_float4(...)` to spirv.CompositeConstruct.
  bool vectorCtorInfo(llvm::StringRef name, mlir::Type &elemTy,
                      unsigned &count) {
    struct Base { const char *prefix; BuiltinTypeKind kind; };
    static constexpr Base bases[] = {
        {"float", BuiltinTypeKind::Float32},
        {"int", BuiltinTypeKind::Int32},
        {"uint", BuiltinTypeKind::UInt32},
        {"double", BuiltinTypeKind::Float64},
        {"bool", BuiltinTypeKind::Bool},
        {"long", BuiltinTypeKind::Int64},
        {"ulong", BuiltinTypeKind::UInt64},
        {"half", BuiltinTypeKind::Float16},
    };
    for (const Base &b : bases) {
      llvm::StringRef p = b.prefix;
      if (name.size() == p.size() + 1 && name.starts_with(p)) {
        char d = name.back();
        if (d >= '2' && d <= '4') {
          elemTy = cvtType(new BuiltinType(b.kind));
          count = d - '0';
          return true;
        }
      }
    }
    return false;
  }

  // Dispatch a CUDA builtin call. Sets `matched=true` if `name` is a
  // recognized builtin (so the caller can return without falling through to
  // funcTable / the unknown-callee error). Returns the result Value, or null
  // for void builtins (fences, __syncthreads_* are value-returning though).
  Value emitBuiltinCall(llvm::StringRef name,
                        const std::vector<NodePtr> &args, Location l,
                        bool &matched) {
    matched = true;
    if (auto v = emitMathBuiltin(name, args, l)) return v;
    if (auto v = emitAtomicBuiltin(name, args, l)) return v;
    if (emitFenceBuiltin(name, l)) return Value();
    if (auto v = emitVoteBuiltin(name, args, l)) return v;
    // emitWarpBuiltin handles void __syncwarp (returns null with warpMatched
    // set), so dispatch it before the matched=false fallback.
    bool warpMatched = false;
    if (auto v = emitWarpBuiltin(name, args, l, warpMatched)) return v;
    if (warpMatched) return Value();
    // VC async-copy approximation (vcMemcpyAsync / vcPipeline*).
    bool asyncMatched = false;
    if (auto v = emitAsyncCopyBuiltin(name, args, l, asyncMatched)) return v;
    if (asyncMatched) return Value();
    matched = false;
    return Value();
  }

  // CUDA math intrinsics -> spirv.GL.* (GLSLstd450). Single-precision only:
  // __sinf/__cosf/.../sinf/cosf/.../sqrtf/fabsf/fminf/fmaxf/powf/floorf/ceilf/
  // expf/logf. f32 throughout; int args are promoted via sitofp. Returns null
  // (and leaves `matched` semantics to the caller) if `name` is not a math
  // builtin.
  Value emitMathBuiltin(llvm::StringRef name,
                        const std::vector<NodePtr> &args, Location l) {
    // Geometric builtins on float vectors: dot(a,b) and cross(a,b). The spirv
    // dialect has no GLDot/GLCross (MLIR 18's SPIRVGLOps only covers the
    // GLSLstd450 arithmetic set), so these are expanded with elementary
    // spirv.CompositeExtract + arith.mulf/addf/subf + spirv.CompositeConstruct.
    // dot: N-dim inner product -> scalar f32. cross: 3-dim cross product ->
    // vector<3xf32>. (CUDA __f variants are not stripped here; vectors.vc uses
    // the bare `dot`/`cross` spellings.)
    if (name == "dot" && args.size() == 2) {
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!a || !b) return Value();
      auto vty = a.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != b.getType()) return Value();
      mlir::Type elem = vty.getElementType();
      unsigned n = vty.getNumElements();
      if (n == 0) return Value();
      Value sum;
      for (unsigned i = 0; i < n; ++i) {
        Value ai = builder.create<spirv::CompositeExtractOp>(
            l, elem, a, builder.getI32ArrayAttr({(int32_t)i}));
        Value bi = builder.create<spirv::CompositeExtractOp>(
            l, elem, b, builder.getI32ArrayAttr({(int32_t)i}));
        Value prod = builder.create<arith::MulFOp>(l, ai, bi);
        sum = sum ? (Value)builder.create<arith::AddFOp>(l, sum, prod) : prod;
      }
      return sum;
    }
    if (name == "cross" && args.size() == 2) {
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!a || !b) return Value();
      auto vty = a.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != b.getType() || vty.getNumElements() != 3)
        return Value();
      mlir::Type elem = vty.getElementType();
      auto ext = [&](Value v, int i) {
        return builder.create<spirv::CompositeExtractOp>(l, elem, v,
            builder.getI32ArrayAttr({(int32_t)i}));
      };
      Value ax = ext(a, 0), ay = ext(a, 1), az = ext(a, 2);
      Value bx = ext(b, 0), by = ext(b, 1), bz = ext(b, 2);
      // c.x = a.y*b.z - a.z*b.y ; c.y = a.z*b.x - a.x*b.z ; c.z = a.x*b.y - a.y*b.x
      Value cx = builder.create<arith::SubFOp>(l,
          builder.create<arith::MulFOp>(l, ay, bz),
          builder.create<arith::MulFOp>(l, az, by));
      Value cy = builder.create<arith::SubFOp>(l,
          builder.create<arith::MulFOp>(l, az, bx),
          builder.create<arith::MulFOp>(l, ax, bz));
      Value cz = builder.create<arith::SubFOp>(l,
          builder.create<arith::MulFOp>(l, ax, by),
          builder.create<arith::MulFOp>(l, ay, bx));
      return builder.create<spirv::CompositeConstructOp>(l, vty,
          ValueRange{cx, cy, cz});
    }
    // Geometric builtins on float vectors. Like dot/cross above, the spirv
    // dialect (LLVM 18) has NO GLLength/GLNormalize/GLDistance/GLReflect/
    // GLRefract/GLFaceForward op, so these are expanded with elementary ops.
    // All are defined in terms of dot(v,v) and scalar arithmetic:
    //   length(v)       = sqrt(dot(v,v))
    //   normalize(v)    = v * rsqrt(dot(v,v))   (= v / length(v))
    //   distance(a,b)   = length(a-b)
    //   reflect(I,N)    = I - 2*dot(N,I)*N
    //   faceforward(N,I,Nref) = dot(Nref,I) < 0 ? N : -N
    //   refract(I,N,eta)= k = 1-eta^2*(1-dot(N,I)^2); eta*I - (eta*dot(N,I)+sqrt(k))*N, or 0 if k<0
    // `mod(x,y) = x - y*floor(x/y)` is the float-modulo (GLSL `mod`, always
    // non-negative result unlike fmod); operands may be scalar or vector.
    auto emitDot = [this, l](Value a, Value b) -> Value {
      auto vty = a.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != b.getType()) return Value();
      mlir::Type elem = vty.getElementType();
      unsigned n = vty.getNumElements();
      if (n == 0) return Value();
      Value sum;
      for (unsigned i = 0; i < n; ++i) {
        Value ai = builder.create<spirv::CompositeExtractOp>(
            l, elem, a, builder.getI32ArrayAttr({(int32_t)i}));
        Value bi = builder.create<spirv::CompositeExtractOp>(
            l, elem, b, builder.getI32ArrayAttr({(int32_t)i}));
        Value prod = builder.create<arith::MulFOp>(l, ai, bi);
        sum = sum ? (Value)builder.create<arith::AddFOp>(l, sum, prod) : prod;
      }
      return sum;
    };
    // length(v) -> sqrt(dot(v,v)). GLSL Length accepts any float vector.
    if (name == "length" && args.size() == 1) {
      Value v = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      if (!v) return Value();
      auto vty = v.getType().dyn_cast<mlir::VectorType>();
      if (!vty) return Value();  // scalar: length(x) = abs(x)
      Value d = emitDot(v, v);
      if (!d) return Value();
      return builder.create<spirv::GLSqrtOp>(l, d);
    }
    // normalize(v) -> v * rsqrt(dot(v,v)) = v / sqrt(dot(v,v)).
    if (name == "normalize" && args.size() == 1) {
      Value v = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      if (!v) return Value();
      auto vty = v.getType().dyn_cast<mlir::VectorType>();
      if (!vty) return Value();  // scalar: normalize(x) = x<0?-1:1 (sign)
      mlir::Type elem = vty.getElementType();
      Value d = emitDot(v, v);
      if (!d) return Value();
      Value len = builder.create<spirv::GLSqrtOp>(l, d);
      // Splat the scalar length across the vector for elementwise division.
      SmallVector<Value, 4> lenSplat(vty.getNumElements(), len);
      Value splat = builder.create<spirv::CompositeConstructOp>(l, vty, lenSplat);
      return builder.create<arith::DivFOp>(l, v, splat);
    }
    // distance(a,b) -> length(a-b).
    if (name == "distance" && args.size() == 2) {
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!a || !b) return Value();
      auto vty = a.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != b.getType()) return Value();
      Value diff = builder.create<arith::SubFOp>(l, a, b);
      Value d = emitDot(diff, diff);
      if (!d) return Value();
      return builder.create<spirv::GLSqrtOp>(l, d);
    }
    // reflect(I, N) -> I - 2*dot(N,I)*N.
    if (name == "reflect" && args.size() == 2) {
      Value I = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value N = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!I || !N) return Value();
      auto vty = I.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != N.getType()) return Value();
      mlir::Type elem = vty.getElementType();
      Value dNI = emitDot(N, I);
      if (!dNI) return Value();
      Value two = builder.create<arith::ConstantOp>(l, elem,
          builder.getFloatAttr(elem, 2.0));
      Value scale = builder.create<arith::MulFOp>(l, two, dNI);
      SmallVector<Value, 4> scaleSplat(vty.getNumElements(), scale);
      Value splat = builder.create<spirv::CompositeConstructOp>(l, vty, scaleSplat);
      Value NScaled = builder.create<arith::MulFOp>(l, N, splat);
      return builder.create<arith::SubFOp>(l, I, NScaled);
    }
    // faceforward(N, I, Nref) -> dot(Nref, I) < 0 ? N : -N.
    if (name == "faceforward" && args.size() == 3) {
      Value N = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value I = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      Value Nref = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
      if (!N || !I || !Nref) return Value();
      auto vty = N.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != I.getType() || vty != Nref.getType()) return Value();
      mlir::Type elem = vty.getElementType();
      Value d = emitDot(Nref, I);
      if (!d) return Value();
      Value zero = builder.create<arith::ConstantOp>(l, elem,
          builder.getFloatAttr(elem, 0.0));
      Value cond = builder.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT,
                                                 d, zero);
      Value negN = builder.create<arith::NegFOp>(l, N);
      // Use scf.if (yield) instead of select: arith.select / spirv.Select with
      // a scalar-i1 cond and vector result don't legalize under GPUToSPIRV
      // (spirv.Select is marked illegal for this shape pre-1.4). scf.if yields
      // the vector on each branch and lowers cleanly.
      auto ifOp = builder.create<scf::IfOp>(l, TypeRange{vty}, cond,
                                           /*withElse=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      builder.create<scf::YieldOp>(l, ValueRange{N});
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<scf::YieldOp>(l, ValueRange{negN});
      builder.setInsertionPointAfter(ifOp);
      return ifOp.getResult(0);
    }
    // refract(I, N, eta) -> k = 1 - eta^2*(1 - dot(N,I)^2);
    //   if k < 0: 0, else eta*I - (eta*dot(N,I) + sqrt(k))*N
    if (name == "refract" && args.size() == 3) {
      Value I = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value N = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      Value eta = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
      if (!I || !N || !eta) return Value();
      auto vty = I.getType().dyn_cast<mlir::VectorType>();
      if (!vty || vty != N.getType()) return Value();
      mlir::Type elem = vty.getElementType();
      Value dNI = emitDot(N, I);
      if (!dNI) return Value();
      Value one = builder.create<arith::ConstantOp>(l, elem,
          builder.getFloatAttr(elem, 1.0));
      Value dNI2 = builder.create<arith::MulFOp>(l, dNI, dNI);
      Value oneMinus = builder.create<arith::SubFOp>(l, one, dNI2);
      Value eta2 = builder.create<arith::MulFOp>(l, eta, eta);
      Value t = builder.create<arith::MulFOp>(l, eta2, oneMinus);
      Value k = builder.create<arith::SubFOp>(l, one, t);
      Value sqrtK = builder.create<spirv::GLSqrtOp>(l, k);
      Value etaDot = builder.create<arith::MulFOp>(l, eta, dNI);
      Value coeff = builder.create<arith::AddFOp>(l, etaDot, sqrtK);
      Value zero = builder.create<arith::ConstantOp>(l, elem,
          builder.getFloatAttr(elem, 0.0));
      Value cond = builder.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT,
                                                 k, zero);
      // eta*I
      SmallVector<Value, 4> etaVals(vty.getNumElements(), eta);
      Value etaSplat = builder.create<spirv::CompositeConstructOp>(l, vty, etaVals);
      Value etaI = builder.create<arith::MulFOp>(l, etaSplat, I);
      // coeff*N
      SmallVector<Value, 4> coeffVals(vty.getNumElements(), coeff);
      Value coeffSplat = builder.create<spirv::CompositeConstructOp>(l, vty, coeffVals);
      Value coeffN = builder.create<arith::MulFOp>(l, coeffSplat, N);
      Value refr = builder.create<arith::SubFOp>(l, etaI, coeffN);
      SmallVector<Value, 4> zeroVals(vty.getNumElements(), zero);
      Value zeroVec = builder.create<spirv::CompositeConstructOp>(l, vty, zeroVals);
      // scf.if instead of select (vector result + scalar cond doesn't legalize
      // via arith.select/spirv.Select; see faceforward above).
      auto ifOp = builder.create<scf::IfOp>(l, TypeRange{vty}, cond,
                                           /*withElse=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      builder.create<scf::YieldOp>(l, ValueRange{zeroVec});
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<scf::YieldOp>(l, ValueRange{refr});
      builder.setInsertionPointAfter(ifOp);
      return ifOp.getResult(0);
    }
    // mod(x, y) -> x - y*floor(x/y). GLSL mod (always non-negative); any float
    // width; operands scalar or vector (elementwise). Same shape as fmod below
    // but matched here before the `strip` so the bare `mod` spelling is caught.
    if (name == "mod" && args.size() == 2) {
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!a || !b) return Value();
      Value div = builder.create<arith::DivFOp>(l, a, b);
      Value fl = builder.create<spirv::GLFloorOp>(l, div);
      Value prod = builder.create<arith::MulFOp>(l, b, fl);
      return builder.create<arith::SubFOp>(l, a, prod);
    }
    // Normalize: strip a leading `__` and a trailing `f` to get the base
    // (e.g. __sinf -> sin, sqrtf -> sqrt, fabsf -> fabs). Only recognized math
    // bases are accepted so unrelated names fall through. NOTE: `isinf`/`isnan`
    // are NOT subject to the trailing-`f` strip (they would become `isin`/`isna`
    // and fall through); they are matched by their full spelling below.
    auto strip = [](llvm::StringRef n) -> llvm::StringRef {
      if (n.starts_with("__")) n = n.drop_front(2);
      // Don't strip the trailing `f` from isnan/isinf — those are the real
      // spellings, not `f`-suffixed float intrinsics.
      if (n == "isnan" || n == "isinf") return n;
      if (n.ends_with("f") && n.size() > 1) n = n.drop_back();
      return n;
    };
    llvm::StringRef base = strip(name);

    // Unary float math. Operates in the operand's NATIVE float width
    // (toNativeFloat preserves f16/f64; only int/index promotes to f32) so a
    // `double`/`__half` arg is lowered in its own precision. Sqrt/InverseSqrt/
    // FAbs/FSign/Floor/Ceil/Round accept any float width (SPIRV_Float); the
    // transcendental set below is f16/f32-only (SPIRV_Float16or32) so a double
    // operand is run through runTranscendental (truncate→f32 op→extend).
    struct UnaryMath { const char *name; };
    static constexpr llvm::StringRef unaryMath[] = {
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
        "fabs", "abs", "floor", "ceil", "round", "trunc", "sign",
        "isnan", "isinf",
        "fract", "degrees", "radians"};
    for (auto m : unaryMath) {
      if (base != m) continue;
      if (args.empty()) return Value();
      Value x = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      if (!x) return Value();
      mlir::Type origTy = x.getType();
      x = toNativeFloat(x, loc(args[0].get()));
      mlir::Type workTy = x.getType();
      // isnan/isinf: spirv.IsNan/IsInf return i1 (bool). Native width (any
      // SPIRV_Float). Result flows through the existing bool pipeline (i1 is
      // what comparisons produce too; stored/used as i32 via the usual path).
      if (base == "isnan") return builder.create<spirv::IsNanOp>(l, x);
      if (base == "isinf") return builder.create<spirv::IsInfOp>(l, x);
      // trunc (round toward zero): no spirv.GL opcode. Equivalent to
      // x >= 0 ? floor(x) : ceil(x) — preserves native float width.
      if (base == "trunc") {
        Value zero = builder.create<arith::ConstantOp>(l, workTy,
            builder.getFloatAttr(workTy, 0.0));
        Value ge0 = builder.create<arith::CmpFOp>(l,
            arith::CmpFPredicate::OGE, x, zero);
        Value fl = builder.create<spirv::GLFloorOp>(l, x);
        Value ce = builder.create<spirv::GLCeilOp>(l, x);
        return builder.create<arith::SelectOp>(l, workTy, ge0, fl, ce);
      }
      // Any-width ops (SPIRV_Float): emit directly on the native-width value.
      if (base == "sqrt") return builder.create<spirv::GLSqrtOp>(l, x);
      if (base == "inversesqrt") return builder.create<spirv::GLInverseSqrtOp>(l, x);
      if (base == "fabs" || base == "abs")
        return builder.create<spirv::GLFAbsOp>(l, x);
      if (base == "floor") return builder.create<spirv::GLFloorOp>(l, x);
      if (base == "ceil") return builder.create<spirv::GLCeilOp>(l, x);
      if (base == "round") return builder.create<spirv::GLRoundOp>(l, x);
      if (base == "sign") return builder.create<spirv::GLFSignOp>(l, x);
      // Transcendentals (SPIRV_Float16or32): no f64 opcode. For f64, run in f32
      // and extend back; for f16/f32, emit directly.
      auto run = [&](std::function<Value(Value)> emit) {
        return runTranscendental(std::move(emit), x, origTy, l);
      };
      if (base == "sin") return run([&](Value v) {
        return builder.create<spirv::GLSinOp>(l, v); });
      if (base == "cos") return run([&](Value v) {
        return builder.create<spirv::GLCosOp>(l, v); });
      if (base == "tan") return run([&](Value v) {
        return builder.create<spirv::GLTanOp>(l, v); });
      if (base == "asin") return run([&](Value v) {
        return builder.create<spirv::GLAsinOp>(l, v); });
      if (base == "acos") return run([&](Value v) {
        return builder.create<spirv::GLAcosOp>(l, v); });
      if (base == "atan") return run([&](Value v) {
        return builder.create<spirv::GLAtanOp>(l, v); });
      if (base == "sinh") return run([&](Value v) {
        return builder.create<spirv::GLSinhOp>(l, v); });
      if (base == "cosh") return run([&](Value v) {
        return builder.create<spirv::GLCoshOp>(l, v); });
      if (base == "tanh") return run([&](Value v) {
        return builder.create<spirv::GLTanhOp>(l, v); });
      if (base == "exp") return run([&](Value v) {
        return builder.create<spirv::GLExpOp>(l, v); });
      if (base == "log") return run([&](Value v) {
        return builder.create<spirv::GLLogOp>(l, v); });
      // exp2/log2: SPIRV has no GLExp2/GLLog2 opcode; approximate via
      // exp/log + the natural-log-of-2 constant. exp2(x) = 2^x = e^(x·ln2),
      // log2(x) = ln(x)/ln(2). Preserves width via runTranscendental.
      if (base == "exp2") return run([&](Value v) {
        Value ln2 = builder.create<arith::ConstantOp>(l, workTy,
            builder.getFloatAttr(workTy, 0.6931471805599453));
        Value scaled = builder.create<arith::MulFOp>(l, v, ln2);
        return builder.create<spirv::GLExpOp>(l, scaled); });
      if (base == "log2") return run([&](Value v) {
        Value ln = builder.create<spirv::GLLogOp>(l, v);
        Value ln2 = builder.create<arith::ConstantOp>(l, workTy,
            builder.getFloatAttr(workTy, 0.6931471805599453));
        return builder.create<arith::DivFOp>(l, ln, ln2); });
      // fract(x) = x - floor(x). No GLFract opcode; any float width via
      // GLFloorOp. (GLSL fract returns the fractional part in [0,1).)
      if (base == "fract") return run([&](Value v) {
        Value fl = builder.create<spirv::GLFloorOp>(l, v);
        return builder.create<arith::SubFOp>(l, v, fl); });
      // degrees(x) = x * (180/pi); radians(x) = x * (pi/180). No GL opcode.
      if (base == "degrees") return run([&](Value v) {
        Value k = builder.create<arith::ConstantOp>(l, workTy,
            builder.getFloatAttr(workTy, 57.29577951308232));
        return builder.create<arith::MulFOp>(l, v, k); });
      if (base == "radians") return run([&](Value v) {
        Value k = builder.create<arith::ConstantOp>(l, workTy,
            builder.getFloatAttr(workTy, 0.017453292519943295));
        return builder.create<arith::MulFOp>(l, v, k); });
    }
    // Binary float math: pow (f16/f32 only) and fmin/fmax/fmod (any width).
    // Operands coerce to the SAME width — the operand's native float width if
    // either is float, else f32 — so f64 min/max/fmod stay in f64.
    if (base == "pow" || base == "fmin" || base == "fmax" || base == "fmod") {
      if (args.size() < 2) return Value();
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!a || !b) return Value();
      // Pick the result width: prefer the wider float operand, else f32.
      mlir::Type aT = a.getType(), bT = b.getType();
      mlir::Type resTy = builder.getF32Type();
      if (aT.isF64() || bT.isF64()) resTy = builder.getF64Type();
      else if (aT.isF16() || bT.isF16()) resTy = builder.getF16Type();
      else if (aT.isF32() || bT.isF32()) resTy = builder.getF32Type();
      a = castValue(a, resTy, loc(args[0].get()));
      b = castValue(b, resTy, loc(args[1].get()));
      if (base == "pow") {
        // Pow is SPIRV_Float16or32: no f64 opcode. Truncate→f32→extend.
        return runTranscendental(
            [&](Value va) {
              Value vb = castValue(b, va.getType(), loc(args[1].get()));
              return builder.create<spirv::GLPowOp>(l, va, vb);
            }, a, resTy, l);
      }
      if (base == "fmin") return builder.create<spirv::GLFMinOp>(l, a, b);
      if (base == "fmax") return builder.create<spirv::GLFMaxOp>(l, a, b);
      // fmod -> a - b*floor(a/b)  (any float width via GLFloorOp)
      Value div = builder.create<arith::DivFOp>(l, a, b);
      Value fl = builder.create<spirv::GLFloorOp>(l, div);
      Value prod = builder.create<arith::MulFOp>(l, b, fl);
      return builder.create<arith::SubFOp>(l, a, prod);
    }
    // step(edge, x) = (x < edge) ? 0 : 1. No GL opcode; any float width.
    if (base == "step") {
      if (args.size() < 2) return Value();
      Value edge = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value x = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!edge || !x) return Value();
      mlir::Type eT = edge.getType(), xT = x.getType();
      mlir::Type resTy = builder.getF32Type();
      if (eT.isF64() || xT.isF64()) resTy = builder.getF64Type();
      else if (eT.isF16() || xT.isF16()) resTy = builder.getF16Type();
      else if (eT.isF32() || xT.isF32()) resTy = builder.getF32Type();
      edge = castValue(edge, resTy, loc(args[0].get()));
      x = castValue(x, resTy, loc(args[1].get()));
      Value cond = builder.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT,
                                                 x, edge);
      Value zero = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 0.0));
      Value one = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 1.0));
      return builder.create<arith::SelectOp>(l, resTy, cond, zero, one);
    }
    // Ternary: clamp(x, lo, hi) -> GLFClamp (any float width).
    if (base == "clamp") {
      if (args.size() < 3) return Value();
      Value x = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value lo = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      Value hi = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
      if (!x || !lo || !hi) return Value();
      mlir::Type xT = x.getType(), loT = lo.getType(), hiT = hi.getType();
      mlir::Type resTy = builder.getF32Type();
      if (xT.isF64() || loT.isF64() || hiT.isF64()) resTy = builder.getF64Type();
      else if (xT.isF16() || loT.isF16() || hiT.isF16()) resTy = builder.getF16Type();
      else if (xT.isF32() || loT.isF32() || hiT.isF32()) resTy = builder.getF32Type();
      x = castValue(x, resTy, loc(args[0].get()));
      lo = castValue(lo, resTy, loc(args[1].get()));
      hi = castValue(hi, resTy, loc(args[2].get()));
      return builder.create<spirv::GLFClampOp>(l, resTy, x, lo, hi);
    }
    // fma(a, b, c) -> a*b + c, single fused op (spirv.GL.Fma, any float width).
    // mix(x, y, a) -> x*(1-a) + y*a (spirv.GL.FMix, any float width). Both
    // preserve native precision: operands coerce to the widest float among them.
    if (base == "fma" || base == "mix") {
      if (args.size() < 3) return Value();
      Value a = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value b = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      Value c = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
      if (!a || !b || !c) return Value();
      mlir::Type aT = a.getType(), bT = b.getType(), cT = c.getType();
      mlir::Type resTy = builder.getF32Type();
      if (aT.isF64() || bT.isF64() || cT.isF64()) resTy = builder.getF64Type();
      else if (aT.isF16() || bT.isF16() || cT.isF16()) resTy = builder.getF16Type();
      else if (aT.isF32() || bT.isF32() || cT.isF32()) resTy = builder.getF32Type();
      a = castValue(a, resTy, loc(args[0].get()));
      b = castValue(b, resTy, loc(args[1].get()));
      c = castValue(c, resTy, loc(args[2].get()));
      if (base == "fma")
        return builder.create<spirv::GLFmaOp>(l, resTy, a, b, c);
      return builder.create<spirv::GLFMixOp>(l, resTy, a, b, c);
    }
    // smoothstep(e0, e1, x) = t*t*(3-2*t), t = clamp((x-e0)/(e1-e0), 0, 1).
    // No GL opcode; any float width. e0==e1 is left to produce NaN (matches
    // GLSL, which leaves the divide-by-zero behavior undefined).
    if (base == "smoothstep") {
      if (args.size() < 3) return Value();
      Value e0 = loadValue(visitExpr(args[0].get()), loc(args[0].get()));
      Value e1 = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      Value x = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
      if (!e0 || !e1 || !x) return Value();
      mlir::Type t0 = e0.getType(), t1 = e1.getType(), t2 = x.getType();
      mlir::Type resTy = builder.getF32Type();
      if (t0.isF64() || t1.isF64() || t2.isF64()) resTy = builder.getF64Type();
      else if (t0.isF16() || t1.isF16() || t2.isF16()) resTy = builder.getF16Type();
      else if (t0.isF32() || t1.isF32() || t2.isF32()) resTy = builder.getF32Type();
      e0 = castValue(e0, resTy, loc(args[0].get()));
      e1 = castValue(e1, resTy, loc(args[1].get()));
      x = castValue(x, resTy, loc(args[2].get()));
      Value zero = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 0.0));
      Value one = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 1.0));
      Value three = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 3.0));
      Value two = builder.create<arith::ConstantOp>(l, resTy,
          builder.getFloatAttr(resTy, 2.0));
      Value num = builder.create<arith::SubFOp>(l, x, e0);
      Value den = builder.create<arith::SubFOp>(l, e1, e0);
      Value t = builder.create<arith::DivFOp>(l, num, den);
      t = builder.create<spirv::GLFClampOp>(l, resTy, t, zero, one);
      Value twoT = builder.create<arith::MulFOp>(l, two, t);
      Value inner = builder.create<arith::SubFOp>(l, three, twoT);
      Value tT = builder.create<arith::MulFOp>(l, t, t);
      return builder.create<arith::MulFOp>(l, tT, inner);
    }
    return Value();
  }

  // Coerce a scalar value to f32 (sitofp from int/index; f32 unchanged).
  Value toF32(Value v, Location l) {
    if (!v) return v;
    v = loadValue(v, l);
    mlir::Type ty = v.getType();
    if (ty.isF32()) return v;
    if (ty.isIntOrIndex()) {
      if (ty.isIndex())
        v = builder.create<arith::IndexCastOp>(l, builder.getI32Type(), v);
      return builder.create<arith::SIToFPOp>(l, builder.getF32Type(), v);
    }
    return v;
  }

  // Width-preserving float coercion for math builtins. Integer/index operands
  // promote to f32 (CUDA's default); float operands KEEP their width so a
  // `double`/`__half` argument is lowered in its native precision instead of
  // being silently truncated to f32 by toF32. This is what lets
  // `double x = sqrt(d)` compute in f64 (Sqrt's SPIRV_Float constraint accepts
  // any width) — the old toF32 path forced f32 and lost 29 bits of precision.
  Value toNativeFloat(Value v, Location l) {
    if (!v) return v;
    v = loadValue(v, l);
    mlir::Type ty = v.getType();
    if (ty.isa<mlir::FloatType>()) return v; // f16/f32/f64 — keep
    return toF32(v, l);                      // int/index → f32
  }

  // The GLSLstd450 transcendental set (sin/cos/tan/asin/acos/atan/sinh/cosh/
  // tanh/exp/log/pow) is constrained to SPIRV_Float16or32 in the spirv dialect
  // — no f64 opcode exists. For a double operand we have no native f64 path, so
  // truncate to f32, run the op, and extend back to f64. This matches what a
  // GLSL driver does for the f64-limited GLSLstd450 entries and preserves the
  // operand's storage width at the call site (the result is still f64, just
  // computed with f32 transcendental precision — documented limitation).
  Value runTranscendental(
      std::function<Value(Value)> emit, Value x, mlir::Type origTy,
      Location l) {
    Value work = x;
    if (origTy.isF64())
      work = builder.create<arith::TruncFOp>(l, builder.getF32Type(), x);
    Value r = emit(work);
    if (origTy.isF64())
      r = builder.create<arith::ExtFOp>(l, origTy, r);
    return r;
  }

  // CUDA atomic builtins -> spirv.Atomic*. Returns null if `name` is not an
  // atomic builtin.
  //
  // Three first-argument shapes (mirroring the GLSL backend):
  //   atomicAdd(&shared, v)    — & on a __shared__ scalar: spirv.ptr<_,Workgroup>
  //   atomicAdd(&arr[i], v)    — & on a __shared__/SSBO array element: AccessChain
  //   atomicAdd(counter, v)    — bare SSBO param (T*): memref, indexed [0]
  // The pointer must be a spirv.ptr. __shared__ slots/elements already are;
  // an SSBO memref element is reached via spirv.AccessChain on the memref's
  // underlying pointer (the gpu.func arg lowers to a spirv.ptr under GPUToSPIRV,
  // but at this layer we emit spirv.AccessChain which GPUToSPIRV legalizes on
  // the memref-as-pointer). scope: Workgroup for __shared__, Device for SSBO.
  Value emitAtomicBuiltin(llvm::StringRef name,
                          const std::vector<NodePtr> &args, Location l) {
    if (!isAtomicName(name)) return Value();
    if (args.size() < 1) return Value();

    // atomicInc(a)/atomicDec(a) carry no explicit value; default to 1.
    bool isIncDec = (name == "atomicInc" || name == "atomicDec");

    // Resolve the pointer operand. Two storage shapes are supported:
    //   - __shared__ scalar/array element  -> spirv.ptr (spirv.Atomic* path)
    //   - SSBO kernel param / local memref -> memref + indices
    //     (memref.atomic_rmw path; GPUToSPIRV legalizes it to spirv.Atomic)
    AtomicPtr ptr = resolveAtomicPtr(args[0].get(), l);
    if (!ptr.valid()) {
      error(args[0].get(), "atomic target must be an addressable scalar or "
                           "array element (__shared__ or SSBO/global)");
      return Value();
    }

    // The value operand.
    Value val;
    if (isIncDec) {
      val = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                              builder.getI32IntegerAttr(1));
    } else {
      if (args.size() < 2) return Value();
      val = loadValue(visitExpr(args[1].get()), loc(args[1].get()));
      if (!val) return Value();
      val = toI32(val, loc(args[1].get()));
    }

    // atomicSub has no direct memref.atomic_rmw kind (no 'subi'); lower as
    // addi(-val). atomicDec likewise = addi(-1).
    if (name == "atomicSub" || name == "atomicDec") {
      Value neg = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                                    builder.getI32IntegerAttr(0));
      val = builder.create<arith::SubIOp>(l, neg, val);
      name = "atomicAdd";
    }
    // atomicXor has no memref.atomic_rmw kind; on SSBO/global emit a marked
    // 'addi' (same trick as atomicExch below) and let rewriteMarkedAtomics
    // turn the spirv.AtomicIAdd into a spirv.AtomicXor post-conversion.
    bool wantXor = (name == "atomicXor");

    if (ptr.isMemref) {
      // memref.atomic_rmw path (SSBO / global).
      // Kinds available: addi, andi, ori, maxs, maxu, mins, minu, assign.
      // xori/subi are absent (sub lowered to addi(-v) above).
      // atomicXor & atomicExch: memref.atomic_rmw has no 'xori'/'assign'
      // lowering in MLIR 18's GPUToSPIRV. Emit each as a marked 'addi' (which
      // DOES legalize) and let the post-conversion rewrite pass
      // (rewriteMarkedAtomics in LoweringPasses) turn the resulting
      // spirv.AtomicIAdd into a spirv.AtomicXor / spirv.AtomicExchange. The
      // marker survives GPUToSPIRV verbatim.
      if (wantXor || name == "atomicExch") {
        SmallVector<Value> idx = ptr.memrefIndices;
        if (idx.empty())
          idx.push_back(builder.create<arith::ConstantOp>(
              l, builder.getIndexType(), builder.getIndexAttr(0)));
        for (Value &i : idx)
          if (!i.getType().isIndex())
            i = builder.create<arith::IndexCastOp>(l, builder.getIndexType(),
                                                    i);
        auto rmw = builder.create<memref::AtomicRMWOp>(
            l, ptr.elemTy, arith::AtomicRMWKind::addi, val, ptr.memrefBase,
            idx);
        rmw->setAttr("vc.atomic_kind",
                     builder.getStringAttr(wantXor ? "xor" : "exch"));
        return rmw;
      }
      using RMW = arith::AtomicRMWKind;
      RMW kind;
      if (name == "atomicAdd" || name == "atomicInc") kind = RMW::addi;
      else if (name == "atomicMin") kind = RMW::mins;
      else if (name == "atomicMax") kind = RMW::maxs;
      else if (name == "atomicAnd") kind = RMW::andi;
      else if (name == "atomicOr")  kind = RMW::ori;
      else if (name == "atomicExch") kind = RMW::assign;
      else {
        error(args[0].get(), "this atomic op on SSBO/global is not supported "
                             "in the MLIR backend");
        return Value();
      }
      // Bare SSBO param with no index (CUDA `atomicAdd(counter, v)`) means
      // counter[0]; supply a 0 index.
      SmallVector<Value> idx = ptr.memrefIndices;
      if (idx.empty())
        idx.push_back(builder.create<arith::ConstantOp>(
            l, builder.getIndexType(), builder.getIndexAttr(0)));
      // memref.atomic_rmw requires index-typed indices.
      for (Value &i : idx)
        if (!i.getType().isIndex())
          i = builder.create<arith::IndexCastOp>(l, builder.getIndexType(), i);
      return builder.create<memref::AtomicRMWOp>(l, ptr.elemTy, kind, val,
                                                 ptr.memrefBase, idx);
    }

    // spirv.ptr path (__shared__). GPUToSPIRV carries these ops through.
    spirv::Scope scope = ptr.isShared ? spirv::Scope::Workgroup
                                      : spirv::Scope::Device;
    auto semantics = spirv::MemorySemantics::None;

    if (name == "atomicAdd" || name == "atomicInc")
      return builder.create<spirv::AtomicIAddOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                 scope, semantics, val);
    if (name == "atomicSub" || name == "atomicDec")
      return builder.create<spirv::AtomicISubOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                 scope, semantics, val);
    if (name == "atomicMin")
      return builder.create<spirv::AtomicSMinOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                 scope, semantics, val);
    if (name == "atomicMax")
      return builder.create<spirv::AtomicSMaxOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                 scope, semantics, val);
    if (name == "atomicAnd")
      return builder.create<spirv::AtomicAndOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                scope, semantics, val);
    if (name == "atomicOr")
      return builder.create<spirv::AtomicOrOp>(l, ptr.elemTy, ptr.spirvAddr,
                                               scope, semantics, val);
    if (name == "atomicXor")
      return builder.create<spirv::AtomicXorOp>(l, ptr.elemTy, ptr.spirvAddr,
                                                scope, semantics, val);
    if (name == "atomicExch")
      return builder.create<spirv::AtomicExchangeOp>(l, ptr.elemTy,
                                                     ptr.spirvAddr, scope,
                                                     semantics, val);
    if (name == "atomicCAS") {
      // atomicCAS(ptr, expected, desired) -> spirv.AtomicCompareExchange.
      if (args.size() < 3) return Value();
      Value expected = toI32(loadValue(visitExpr(args[1].get()),
                                       loc(args[1].get())),
                             loc(args[1].get()));
      Value desired = toI32(loadValue(visitExpr(args[2].get()),
                                      loc(args[2].get())),
                            loc(args[2].get()));
      return builder.create<spirv::AtomicCompareExchangeOp>(
          l, ptr.elemTy, ptr.spirvAddr, scope, semantics, semantics, desired,
          expected);
    }
    return Value();
  }

  // Resolved atomic target. Exactly one of (spirvAddr) / (memrefBase) is set.
  struct AtomicPtr {
    Value spirvAddr;             // spirv.ptr to the element (__shared__)
    mlir::Type elemTy;           // the element type (i32)
    bool isShared = false;       // Workgroup scope vs Device (spirv path)

    Value memrefBase;            // memref value (SSBO param / alloca)
    SmallVector<Value> memrefIndices; // element indices (empty = [0])
    bool isMemref = false;

    bool valid() const { return spirvAddr || isMemref; }
  };

  static bool isAtomicName(llvm::StringRef name) {
    return builtinClass(name) == BuiltinClass::Atomic;
  }

  // Resolve a CUDA atomic pointer argument (`&x`, `&arr[i]`, or bare `ptr`)
  // to either a spirv.ptr (__shared__) or a memref + indices (SSBO/global).
  AtomicPtr resolveAtomicPtr(ASTNode *arg, Location l) {
    AtomicPtr out;
    out.elemTy = builder.getI32Type();
    // Strip a leading `&` (UnaryExpr AddrOf): atomic takes the address of the
    // lvalue, not its loaded value.
    ASTNode *inner = arg;
    if (arg && arg->getNodeType() == ASTNode::NodeKind::UnaryExpr) {
      auto *u = static_cast<UnaryExpr *>(arg);
      if (u->op == UnaryOp::AddrOf) inner = u->operand.get();
    }
    Value mem;
    SmallVector<Value> indices;
    if (!lvalueAddress(inner, mem, indices)) return out;

    // __shared__ scalar/array element: mem is a spirv.ptr to the element.
    if (auto ptr = mem.getType().dyn_cast<spirv::PointerType>()) {
      out.spirvAddr = mem;
      out.elemTy = ptr.getPointeeType();
      // The pointee may itself be a scalar (shared scalar) or already an
      // element pointer from spirv.AccessChain (shared array element).
      mlir::Type pt = ptr.getPointeeType();
      while (auto arr = pt.dyn_cast<spirv::ArrayType>())
        pt = arr.getElementType();
      out.elemTy = pt;
      out.isShared = (ptr.getStorageClass() == spirv::StorageClass::Workgroup);
      return out;
    }
    // memref (SSBO kernel param or local alloca). memref.atomic_rmw takes the
    // memref + element indices directly; GPUToSPIRV legalizes it to a
    // spirv.AccessChain + spirv.Atomic on the StorageBuffer pointer.
    if (auto mr = mem.getType().dyn_cast<MemRefType>()) {
      out.memrefBase = mem;
      out.memrefIndices = indices;
      out.elemTy = mr.getElementType();
      out.isMemref = true;
      return out;
    }
    return out;
  }

  // Coerce a scalar value to i32 (index-cast / trunc / zext as needed).
  Value toI32(Value v, Location l) {
    if (!v) return v;
    v = loadValue(v, l);
    mlir::Type ty = v.getType();
    if (ty.isInteger(32)) return v;
    if (ty.isIndex())
      return builder.create<arith::IndexCastOp>(l, builder.getI32Type(), v);
    if (ty.isIntOrIndex() && ty.getIntOrFloatBitWidth() < 32)
      return builder.create<arith::ExtUIOp>(l, builder.getI32Type(), v);
    if (ty.isIntOrIndex())
      return builder.create<arith::TruncIOp>(l, builder.getI32Type(), v);
    return v;
  }

  // __threadfence / __threadfence_block -> spirv.MemoryBarrier. Returns true
  // if `name` was a recognized fence builtin.
  //
  // __threadfence       -> scope Device,    semantics AcquireRelease|UniformMemory
  // __threadfence_block -> scope Workgroup,  semantics AcquireRelease|WorkgroupMemory
  //
  // spirv.MemoryBarrier is a memory fence ONLY (it orders memory operations
  // but is not an execution barrier). CUDA __threadfence is likewise a memory
  // fence without execution synchronization, so no vc::BarrierOp here. (GLSL
  // backend lesson: memory note [[vc-threadfence-lowering-no-barrier]].)
  bool emitFenceBuiltin(llvm::StringRef name, Location l) {
    if (name == "__threadfence") {
      builder.create<spirv::MemoryBarrierOp>(
          l, spirv::Scope::Device,
          spirv::MemorySemantics::AcquireRelease |
              spirv::MemorySemantics::UniformMemory);
      return true;
    }
    if (name == "__threadfence_block") {
      builder.create<spirv::MemoryBarrierOp>(
          l, spirv::Scope::Workgroup,
          spirv::MemorySemantics::AcquireRelease |
              spirv::MemorySemantics::WorkgroupMemory);
      return true;
    }
    return false;
  }

  // __syncthreads_count/and/or(pred) -> block-wide vote via a shared-array
  // reduction. CUDA's block-wide votes are NOT warp ops; they reduce a
  // per-thread predicate across the whole block, so each lane writes its
  // booleanized predicate to a __shared__ slot, two barriers bracket a
  // thread-0 fold, and every lane reads the broadcast result. (Mirrors the
  // GLSL backend's reduction; memory note [[vc-vote-barriers-statement-hoisting]].)
  //
  //   count(pred) = #lanes with pred != 0
  //   and(pred)   = (count == blockDim.x) ? 1 : 0
  //   or(pred)    = (count != 0) ? 1 : 0
  //
  // The call sites in vote.vc are in uniform control flow (kernel top level),
  // so the two barriers are reached by every lane and do not diverge.
  Value emitVoteBuiltin(llvm::StringRef name,
                        const std::vector<NodePtr> &args, Location l) {
    if (name != "__syncthreads_count" && name != "__syncthreads_and" &&
        name != "__syncthreads_or")
      return Value();
    if (args.size() < 1) return Value();

    // Per-lane predicate, booleanized to i1 then widened to i32 (SPIR-V has no
    // 1-bit storage; memory note [[vc-mlir-expr-basics-bool-comma-promotion]]).
    Value pred = toI1(visitExpr(args[0].get()), loc(args[0].get()));
    if (!pred) return Value();
    Value predI32 = builder.create<arith::ExtUIOp>(l, builder.getI32Type(), pred);

    // threadIdx.x and blockDim.x as i32.
    Value tidIdx = builder.create<vc::ThreadIdOp>(l, builder.getIndexType(),
                                                  vc::Dim::x);
    Value tid = builder.create<arith::IndexCastOp>(l, builder.getI32Type(),
                                                   tidIdx);
    Value bdimIdx = builder.create<vc::BlockDimOp>(l, builder.getIndexType(),
                                                   vc::Dim::x);
    Value bdim = builder.create<arith::IndexCastOp>(l, builder.getI32Type(),
                                                    bdimIdx);

    // __shared__ int voteArr[1024] (CUDA blockDim cap) and __shared__ int voteRes.
    Value voteArr = getOrCreateSharedGlobal("voteArr", {1024},
                                            builder.getI32Type(), l);
    Value voteRes = getOrCreateSharedGlobal("voteResult", {},
                                            builder.getI32Type(), l);

    // voteArr[tid] = predI32  (spirv.AccessChain over the shared array).
    spirv::PointerType arrPtrTy = voteArr.getType().cast<spirv::PointerType>();
    spirv::PointerType elemPtrTy = spirv::PointerType::get(
        builder.getI32Type(), arrPtrTy.getStorageClass());
    Value slot = builder.create<spirv::AccessChainOp>(l, elemPtrTy, voteArr,
                                                      ValueRange{tid});
    builder.create<spirv::StoreOp>(l, slot, predI32,
                                   spirv::MemoryAccessAttr(), IntegerAttr());

    // Barrier 1: every lane has written its slot before thread 0 folds.
    builder.create<vc::BarrierOp>(l);

    // Thread 0 folds voteArr[0..blockDim) into voteRes. Use scf.while:
    //   %acc = 0; %j = 0;
    //   while (j < bdim) { acc += voteArr[j]; j += 1; }
    // The reduction runs inside `if (tid == 0)` so only one lane folds; the
    // barriers outside the if keep control flow uniform.
    Value zero = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                                   builder.getI32IntegerAttr(0));
    Value isT0 = builder.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq,
                                               tid, zero);

    auto saved = builder.saveInsertionPoint();
    auto ifOp = builder.create<scf::IfOp>(l, TypeRange{}, isT0,
                                          /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // scf.while carrying (j, acc). The before-block args are the loop-carried
    // values (seeded by the WhileOp operands); the after-block args are the
    // values yielded by ConditionOp. After building both regions, the builder
    // is left inside the after-block; repoint it to just after the WhileOp so
    // the fold below stays inside the scf.if's then-block.
    auto whileOp = builder.create<scf::WhileOp>(
        l, TypeRange{builder.getI32Type(), builder.getI32Type()},
        ValueRange{zero, zero});
    // before-region: condition = j < bdim; yield (j, acc) to the after-block.
    // The before-block's args are the loop-carried values (matching the
    // WhileOp's result types); createBlock seeds an empty block, so add them.
    Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
    beforeBlock->addArgument(builder.getI32Type(), l);
    beforeBlock->addArgument(builder.getI32Type(), l);
    {
      Value j = beforeBlock->getArgument(0);
      Value acc = beforeBlock->getArgument(1);
      Value cond = builder.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt,
                                                 j, bdim);
      builder.create<scf::ConditionOp>(l, cond, ValueRange{j, acc});
    }
    // after-region: load voteArr[j], acc += v, j += 1; yield (j+1, acc+v).
    Block *afterBlock = builder.createBlock(&whileOp.getAfter());
    afterBlock->addArgument(builder.getI32Type(), l);
    afterBlock->addArgument(builder.getI32Type(), l);
    {
      Value j = afterBlock->getArgument(0);
      Value acc = afterBlock->getArgument(1);
      Value jslot = builder.create<spirv::AccessChainOp>(l, elemPtrTy,
                                                          voteArr, ValueRange{j});
      Value v = builder.create<spirv::LoadOp>(l, builder.getI32Type(), jslot,
                                              spirv::MemoryAccessAttr(),
                                              IntegerAttr());
      Value newAcc = builder.create<arith::AddIOp>(l, acc, v);
      Value one = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                                    builder.getI32IntegerAttr(1));
      Value newJ = builder.create<arith::AddIOp>(l, j, one);
      builder.create<scf::YieldOp>(l, ValueRange{newJ, newAcc});
    }
    // Continue emitting inside the then-block, right after the WhileOp.
    builder.setInsertionPointAfter(whileOp);

    Value count = whileOp.getResult(1);
    // Compute the vote result from the count.
    Value res;
    if (name == "__syncthreads_count") {
      res = count;
    } else if (name == "__syncthreads_and") {
      Value all = builder.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq,
                                                count, bdim);
      res = builder.create<arith::ExtUIOp>(l, builder.getI32Type(), all);
    } else { // __syncthreads_or
      Value any = builder.create<arith::CmpIOp>(l, arith::CmpIPredicate::ne,
                                                count, zero);
      res = builder.create<arith::ExtUIOp>(l, builder.getI32Type(), any);
    }
    builder.create<spirv::StoreOp>(l, voteRes, res,
                                   spirv::MemoryAccessAttr(), IntegerAttr());

    builder.restoreInsertionPoint(saved);

    // Barrier 2: thread 0's write to voteRes is visible to all lanes.
    builder.create<vc::BarrierOp>(l);

    // Every lane reads the broadcast result.
    return builder.create<spirv::LoadOp>(l, builder.getI32Type(), voteRes,
                                         spirv::MemoryAccessAttr(),
                                         IntegerAttr());
  }

  // Lower a CUDA warp intrinsic to its spirv subgroup counterpart. CUDA warp
  // intrinsics take a leading `mask` argument (active-lane bitmask); subgroups
  // have no such concept (ops apply to active invocations), so the mask is
  // dropped. Layout per intrinsic (mirrors the GLSL backend's emitWarpIntrinsic
  // in ASTToGLSL.cpp:801):
  //   __syncwarp(mask?)               -> spirv.ControlBarrier Subgroup (void)
  //   __activemask()                  -> GroupNonUniformBallot(true).x  -> i32
  //   __ballot_sync(mask, pred)       -> GroupNonUniformBallot(pred).x  -> i32
  //   __anySync(mask, pred)           -> GroupNonUniformLogicalOr Reduce(pred) -> i32
  //   __allSync(mask, pred)           -> GroupNonUniformLogicalAnd Reduce(pred) -> i32
  //   __shfl_sync(mask, v, lane)      -> GroupNonUniformShuffle(v, lane)
  //   __shfl_up_sync(mask, v, d)      -> GroupNonUniformShuffleUp(v, d)
  //   __shfl_down_sync(mask, v, d)    -> GroupNonUniformShuffleDown(v, d)
  //   __shfl_xor_sync(mask, v, lm)    -> GroupNonUniformShuffleXor(v, lm)
  // __ballot_sync/__activemask return a 32-bit bitmask in CUDA; GroupNonUniform
  // Ballot returns vector<4xi32>, so CompositeExtract component 0 yields the
  // low 32 bits (correct for subgroup<=32). Vote results are i1; VC bool is
  // i32, so ExtUI back to i32. `warpMatched` is set true for any recognized
  // intrinsic (including void __syncwarp, which returns a null Value). Returns
  // null (and leaves warpMatched false) if `name` is not a warp intrinsic.
  Value emitWarpBuiltin(llvm::StringRef name,
                        const std::vector<NodePtr> &args, Location l,
                        bool &warpMatched) {
    warpMatched = false;
    auto i1toi32 = [&](Value v) {
      return builder.create<arith::ExtUIOp>(l, builder.getI32Type(), v);
    };
    // ballot: args[0]=mask (dropped), args[1]=predicate (or true for
    // __activemask). Returns the low 32 bits of the uvec4 ballot.
    auto ballotLow = [&](Value predI1) -> Value {
      mlir::VectorType v4i32 =
          mlir::VectorType::get({4}, builder.getI32Type());
      Value v = builder.create<spirv::GroupNonUniformBallotOp>(
          l, v4i32, spirv::Scope::Subgroup, predI1);
      return builder.create<spirv::CompositeExtractOp>(
          l, builder.getI32Type(), v, builder.getI32ArrayAttr({0}));
    };

    if (name == "__syncwarp") {
      warpMatched = true;
      usesSubgroup = true;
      // subgroupBarrier = ControlBarrier(Subgroup, Subgroup, None): an
      // execution+memory sync scoped to the subgroup. MemorySemantics None is
      // fine for a pure execution barrier (CUDA __syncwarp is a warp sync).
      builder.create<spirv::ControlBarrierOp>(
          l, spirv::Scope::Subgroup, spirv::Scope::Subgroup,
          spirv::MemorySemantics::None);
      return Value();
    }
    if (name == "__activemask") {
      warpMatched = true;
      usesSubgroup = true;
      Value t = builder.create<spirv::ConstantOp>(
          l, builder.getI1Type(), builder.getBoolAttr(true));
      return ballotLow(t);
    }
    if (name == "__ballot_sync") {
      warpMatched = true;
      usesSubgroup = true;
      // args[0]=mask (dropped), args[1]=predicate
      if (args.size() < 2)
        return error(nullptr, "__ballot_sync needs (mask, predicate)");
      Value pred = toI1(visitExpr(args[1].get()), l);
      return ballotLow(pred);
    }
    if (name == "__anySync") {
      warpMatched = true;
      usesSubgroup = true;
      if (args.size() < 2)
        return error(nullptr, "__anySync needs (mask, predicate)");
      Value pred = toI1(visitExpr(args[1].get()), l);
      Value r = builder.create<spirv::GroupNonUniformLogicalOrOp>(
          l, builder.getI1Type(), spirv::Scope::Subgroup,
          spirv::GroupOperation::Reduce, pred, Value());
      return i1toi32(r);
    }
    if (name == "__allSync") {
      warpMatched = true;
      usesSubgroup = true;
      if (args.size() < 2)
        return error(nullptr, "__allSync needs (mask, predicate)");
      Value pred = toI1(visitExpr(args[1].get()), l);
      Value r = builder.create<spirv::GroupNonUniformLogicalAndOp>(
          l, builder.getI1Type(), spirv::Scope::Subgroup,
          spirv::GroupOperation::Reduce, pred, Value());
      return i1toi32(r);
    }
    // __shfl* family: args[0]=mask (dropped), args[1]=value, args[2]=index/
    // delta, optional args[3]=width (dropped). The value keeps its type; the
    // index/delta is i32 (CUDA int). GroupNonUniformShuffle* are pure and
    // result-typed == value type.
    if (name == "__shfl_sync" || name == "__shfl_up_sync" ||
        name == "__shfl_down_sync" || name == "__shfl_xor_sync") {
      warpMatched = true;
      usesSubgroup = true;
      if (args.size() < 3)
        return error(nullptr, "warp shuffle needs (mask, value, index)");
      Value val = loadValue(visitExpr(args[1].get()), l);
      Value idx = loadValue(visitExpr(args[2].get()), l);
      // Shuffle ops require an i32 index (not index type).
      if (idx.getType().isIndex())
        idx = builder.create<arith::IndexCastOp>(l, builder.getI32Type(), idx);
      if (name == "__shfl_sync")
        return builder.create<spirv::GroupNonUniformShuffleOp>(
            l, val.getType(), spirv::Scope::Subgroup, val, idx);
      if (name == "__shfl_up_sync")
        return builder.create<spirv::GroupNonUniformShuffleUpOp>(
            l, val.getType(), spirv::Scope::Subgroup, val, idx);
      if (name == "__shfl_down_sync")
        return builder.create<spirv::GroupNonUniformShuffleDownOp>(
            l, val.getType(), spirv::Scope::Subgroup, val, idx);
      return builder.create<spirv::GroupNonUniformShuffleXorOp>(
          l, val.getType(), spirv::Scope::Subgroup, val, idx);
    }
    return Value();
  }

  // VC async-copy approximation (mirrors GLSL emitAsyncCopyCall in
  // ASTToGLSL.cpp:923). Vulkan/SPIR-V has no TMA hardware, so vcMemcpyAsync is
  // a SOFTWARE cooperative copy: every workgroup thread copies a contiguous
  // segment of length nElems/blockDim.x, followed by a barrier. The pipeline
  // sync token (VcPipeline) carries no real state — vcPipeline* are plain
  // barrier() wrappers. All these builtins are void.
  //
  //   vcPipelineProducerCommit/ConsumerWait/ConsumerCommit(pipe) -> BarrierOp
  //   vcMemcpyAsync(dst, src, nElems, pipe):
  //     dst = shared array element (buf[slot]) -> spirv.ptr Workgroup
  //     src = device pointer expr, typically `in + offset` (BinOp::Add); split
  //           into base (memref<?xf32> kernel arg) + offset (i32). A bare
  //           indexable source uses offset 0.
  //     n = nElems / blockDim.x;  base = threadIdx.x * n
  //     scf.for i in 0..n: dst[base+i] = src[offset + base + i]
  //     BarrierOp
  Value emitAsyncCopyBuiltin(llvm::StringRef name,
                             const std::vector<NodePtr> &args, Location l,
                             bool &matched) {
    matched = false;
    if (name == "vcPipelineProducerCommit" ||
        name == "vcPipelineConsumerWait" ||
        name == "vcPipelineConsumerCommit") {
      matched = true;
      builder.create<vc::BarrierOp>(l);
      return Value();
    }
    if (name != "vcMemcpyAsync")
      return Value();
    matched = true;
    if (args.size() < 3) {
      error(args.empty() ? nullptr : args[0].get(),
            "vcMemcpyAsync needs (dst, src, nElems[, pipe])");
      return Value();
    }

    // dst: shared array element like buf[slot] -> spirv.ptr (Workgroup).
    Value dstMem;
    SmallVector<Value> dstIdx;
    if (!lvalueAddress(args[0].get(), dstMem, dstIdx) ||
        !dstMem.getType().isa<spirv::PointerType>()) {
      error(args[0].get(),
            "vcMemcpyAsync dst must be a __shared__ array element");
      return Value();
    }
    spirv::PointerType dstPtrTy =
        dstMem.getType().cast<spirv::PointerType>();
    spirv::StorageClass dstSC = dstPtrTy.getStorageClass();
    // The dst lvalue may be `buf[slot]` — a row pointer `ptr<array<64xf32>>`,
    // not yet a scalar element. Drill down to the innermost scalar element type;
    // the loop's per-element AccessChain walks the remaining array dimensions.
    mlir::Type dstElemTy = dstPtrTy.getPointeeType();
    while (auto arr = dstElemTy.dyn_cast<spirv::ArrayType>())
      dstElemTy = arr.getElementType();
    spirv::PointerType dstElemPtrTy =
        spirv::PointerType::get(dstElemTy, dstSC);

    // src: split `in + offset` into base (memref<?xf32>) + offset (i32).
    ASTNode *srcArg = args[1].get();
    Value srcBase;
    Value off = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                                  builder.getI32IntegerAttr(0));
    if (srcArg && srcArg->getNodeType() == ASTNode::NodeKind::BinaryExpr) {
      auto *b = static_cast<BinaryExpr *>(srcArg);
      if (b->op == BinaryOp::Add) {
        srcBase = visitExpr(b->lhs.get());
        Value ov = loadValue(visitExpr(b->rhs.get()), loc(b->rhs.get()));
        if (ov)
          off = castValue(ov, builder.getI32Type(), loc(b->rhs.get()));
      }
    }
    if (!srcBase)
      srcBase = visitExpr(srcArg);
    if (!srcBase || !srcBase.getType().isa<MemRefType>()) {
      error(args[1].get(),
            "vcMemcpyAsync src must be an indexable device pointer");
      return Value();
    }
    MemRefType srcMTy = srcBase.getType().cast<MemRefType>();
    mlir::Type srcElemTy = srcMTy.getElementType();

    // nElems as i32; blockDim / tid as i32.
    Value nElems = loadValue(visitExpr(args[2].get()), loc(args[2].get()));
    if (nElems)
      nElems = castValue(nElems, builder.getI32Type(), loc(args[2].get()));
    if (!nElems) return Value();

    Value bdimIdx = builder.create<vc::BlockDimOp>(l, builder.getIndexType(),
                                                   vc::Dim::x);
    Value bdim = builder.create<arith::IndexCastOp>(l, builder.getI32Type(),
                                                    bdimIdx);
    Value tidIdx = builder.create<vc::ThreadIdOp>(l, builder.getIndexType(),
                                                  vc::Dim::x);
    Value tid = builder.create<arith::IndexCastOp>(l, builder.getI32Type(),
                                                   tidIdx);

    // n = nElems / blockDim.x ;  base = tid * n
    Value n = builder.create<arith::DivSIOp>(l, nElems, bdim);
    Value base = builder.create<arith::MulIOp>(l, tid, n);

    // scf.for i = 0..n step 1: dst[base+i] = src[offset + base + i].
    Value zero = builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                                   builder.getI32IntegerAttr(0));
    Value lb = builder.create<arith::ConstantOp>(l, builder.getIndexType(),
                                                 builder.getIndexAttr(0));
    Value ubIdx = builder.create<arith::IndexCastOp>(l, builder.getIndexType(),
                                                     n);
    Value step = builder.create<arith::ConstantOp>(l, builder.getIndexType(),
                                                   builder.getIndexAttr(1));
    auto saved = builder.saveInsertionPoint();
    // scf::ForOp's builder creates the body block pre-terminated with an
    // scf.yield (empty iterArgs); set the insertion point to the block start
    // so emitted ops land before that terminator.
    auto forOp = builder.create<scf::ForOp>(l, lb, ubIdx, step, ValueRange{});
    builder.setInsertionPointToStart(forOp.getBody());
    {
      Value i = builder.create<arith::IndexCastOp>(l, builder.getI32Type(),
                                                   forOp.getInductionVar());
      // dst index (i32) -> spirv.AccessChain over the shared array ptr.
      Value dstOff = builder.create<arith::AddIOp>(l, base, i);
      Value dstAddr = builder.create<spirv::AccessChainOp>(
          l, dstElemPtrTy, dstMem, ValueRange{dstOff});
      // src index (index) -> memref.load.
      Value srcIdxI32 = builder.create<arith::AddIOp>(l, off, dstOff);
      Value srcIdx = builder.create<arith::IndexCastOp>(l,
                                                        builder.getIndexType(),
                                                        srcIdxI32);
      Value val = builder.create<memref::LoadOp>(l, srcBase, ValueRange{srcIdx});
      // Store with the shared element type (coerce if src elem differs, e.g.
      // matching bit widths across address spaces).
      if (val.getType() != dstElemTy)
        val = castValue(val, dstElemTy, l);
      builder.create<spirv::StoreOp>(l, dstAddr, val,
                                     spirv::MemoryAccessAttr(),
                                     IntegerAttr());
    }
    builder.restoreInsertionPoint(saved);

    // Trailing barrier: every lane's copy is visible before consumption.
    builder.create<vc::BarrierOp>(l);
    return Value();
  }

  Value emitUnary(const UnaryExpr *u) {
    Location l = loc(u);
    if (u->op == UnaryOp::AddrOf) {
      Value mem;
      SmallVector<Value> indices;
      if (!lvalueAddress(u->operand.get(), mem, indices))
        return error(u, "& requires an addressable lvalue");
      if (!indices.empty())
        return error(u, "address of indexed memref element is unsupported");
      return mem;
    }
    if (u->op == UnaryOp::Deref) {
      Value mem;
      SmallVector<Value> indices;
      if (!lvalueAddress(const_cast<UnaryExpr *>(u), mem, indices))
        return error(u, "* requires a pointer or pointer-like parameter");
      if (auto ptr = mem.getType().dyn_cast<spirv::PointerType>())
        return builder.create<spirv::LoadOp>(l, ptr.getPointeeType(), mem,
                                            spirv::MemoryAccessAttr(),
                                            IntegerAttr());
      return builder.create<memref::LoadOp>(l, mem, indices);
    }
    Value v = visitExpr(u->operand.get());
    switch (u->op) {
    case UnaryOp::Neg: {
      // -x : 0 - x
      Value loaded = loadValue(v, l);
      mlir::Type ty = loaded.getType();
      if (ty.isF32())
        return builder.create<arith::NegFOp>(l, loaded);
      Value zero = builder.create<arith::ConstantOp>(l, ty,
                                                     builder.getZeroAttr(ty));
      return builder.create<arith::SubIOp>(l, zero, loaded);
    }
    case UnaryOp::LNot:
      return builder.create<arith::XOrIOp>(
          l, toI1(v, l), builder.create<arith::ConstantOp>(
                             l, builder.getBoolAttr(true)));
    case UnaryOp::Not: {
      Value loaded = loadValue(v, l);
      mlir::Type ty = loaded.getType();
      auto minusOne = builder.create<arith::ConstantOp>(
          l, ty, builder.getIntegerAttr(ty, -1));
      return builder.create<arith::XOrIOp>(l, loaded, minusOne);
    }
    case UnaryOp::PreInc:
    case UnaryOp::PreDec:
    case UnaryOp::PostInc:
    case UnaryOp::PostDec: {
      // Operand must be a slot (assignment-like side effect).
      Value addr = v;
      if (!addr || !addr.getType().isa<MemRefType>())
        return error(u, "++/-- requires an lvalue slot");
      mlir::Type elem = addr.getType().cast<MemRefType>().getElementType();
      Value cur = builder.create<memref::LoadOp>(l, addr, ValueRange{});
      if (!elem.isIntOrIndex() && !elem.isF32())
        return error(u, "++/-- on non-scalar element type");
      Value one = elem.isF32()
                      ? builder.create<arith::ConstantOp>(
                            l, builder.getFloatAttr(elem, 1.0))
                      : builder.create<arith::ConstantOp>(
                            l, builder.getIntegerAttr(elem, 1));
      bool isInc = u->op == UnaryOp::PreInc || u->op == UnaryOp::PostInc;
      Value nxt = elem.isF32()
                      ? (Value)builder.create<arith::AddFOp>(l, cur, one)
                      : (Value)builder.create<arith::AddIOp>(l, cur, one);
      if (!isInc)
        nxt = elem.isF32() ? (Value)builder.create<arith::SubFOp>(l, cur, one)
                           : (Value)builder.create<arith::SubIOp>(l, cur, one);
      builder.create<memref::StoreOp>(l, nxt, addr);
      bool isPost = u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec;
      return isPost ? cur : nxt;
    }
    default:
      return v;
    }
  }

  // Resolve the (memref, slotIdx, fieldType) for a struct field access, shared
  // by loadStructField and the array-field lvalue path. Returns true if `m` is
  // a `ptr->struct.field` access resolvable to an SSBO slot. `extraByteOff` is
  // added to the field offset (used for array-field element indexing:
  // `pts[i].v[k]` adds k*elemSize). On success, `mem` is the SSBO memref,
  // `slotIdx` is the i32-granular slot index, and `fieldTy` is the field's
  // MLIR type.
  bool structFieldAddr(const MemberAccessExpr *m, Location l,
                       Value extraByteOff, Value &mem, Value &slotIdx,
                       mlir::Type &fieldTy,
                       std::vector<int64_t> *arrayDimsOut = nullptr) {
    if (!m->base) return false;
    ASTNode *base = m->base.get();
    ASTNode *indexNode = nullptr;
    if (base->getNodeType() == ASTNode::NodeKind::IndexExpr) {
      auto *ie = static_cast<IndexExpr *>(base);
      indexNode = ie->index.get();
      base = ie->base.get();
    }
    if (!base || base->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *ref = static_cast<DeclRefExpr *>(base);
    auto tit = localTypes.find(ref->name);
    if (tit == localTypes.end()) return false;
    const vc::Type *t = tit->second;
    if (!t) return false;
    // Two struct-access shapes share this path:
    //  - SSBO: `Result* r; r[i].f` — t is PointerType<RecordType>, the base
    //    memref is the memref<?xi32> SSBO view, and the element index multiplies
    //    the struct stride into the byte offset.
    //  - Local struct value: `Accumulator acc; acc.sum` / `v.c[k]` — t is a bare
    //    RecordType, the base memref is the local memref<Nxi32, Function> slot,
    //    and there is no index*stride term (a single struct, base is a plain
    //    DeclRefExpr so indexNode stays null).
    const RecordType *rec = nullptr;
    bool isLocalStruct = false;
    if (isa<PointerType>(t)) {
      const vc::Type *pointee = cast<PointerType>(t)->pointee;
      if (!pointee || !isa<RecordType>(pointee)) return false;
      rec = cast<RecordType>(pointee);
    } else if (isa<RecordType>(t)) {
      rec = cast<RecordType>(t);
      isLocalStruct = true;
    } else {
      return false;
    }
    auto lit = recordLayouts.find(rec->decl->name);
    if (lit == recordLayouts.end()) return false;
    const StructLayout &layout = lit->second;
    const FieldLayout *fld = nullptr;
    for (const auto &f : layout.fields)
      if (f.name == m->member) { fld = &f; break; }
    if (!fld) return false;
    if (arrayDimsOut) *arrayDimsOut = fld->arrayDims;

    auto sit = locals.find(ref->name);
    if (sit == locals.end()) return false;
    mem = sit->second;
    if (!mem.getType().isa<MemRefType>()) return false;
    fieldTy = fld->type;

    // byteOff = fieldOffset + (elemIdx*stride for SSBO array) + extraByteOff
    Value byteOff = builder.create<arith::ConstantOp>(
        l, builder.getI32Type(),
        builder.getI32IntegerAttr((int32_t)fld->offset));
    // The index*stride term applies only to SSBO struct arrays (`r[i]`); a
    // local struct value has no outer element index.
    if (indexNode && !isLocalStruct) {
      Value idx = loadValue(visitExpr(indexNode), loc(indexNode));
      if (!idx) return false;
      idx = castValue(idx, builder.getI32Type(), l);
      Value strideConst = builder.create<arith::ConstantOp>(
          l, builder.getI32Type(),
          builder.getI32IntegerAttr((int32_t)layout.stride));
      Value elemOff = builder.create<arith::MulIOp>(l, idx, strideConst);
      byteOff = builder.create<arith::AddIOp>(l, byteOff, elemOff);
    }
    if (extraByteOff)
      byteOff = builder.create<arith::AddIOp>(l, byteOff, extraByteOff);
    slotIdx = builder.create<arith::ShRSIOp>(
        l, byteOff,
        builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                          builder.getI32IntegerAttr(2)));
    slotIdx = castValue(slotIdx, builder.getIndexType(), l);
    return true;
  }

  // Load a scalar struct field `pts[i].x` / `result[i].f`. Returns the field
  // value in its native type (bitcast from the i32 slot's bit pattern, mirroring
  // storeStructField). Returns null if `m` is not a struct-pointer field access
  // or the field is an array (array fields are read element-wise via the
  // IndexExpr lvalue path, not as a whole).
  Value loadStructField(const MemberAccessExpr *m, Location l) {
    if (!m->base) return Value();
    // Reject array fields: `pts[i].v` (v is float[2]) is not a scalar load; it
    // is only meaningful when indexed (`pts[i].v[0]`), handled by the IndexExpr
    // lvalue path. Array-ness lives in FieldDecl::arrayDims (the field's MLIR
    // type is the scalar element, not a vector).
    Value mem, slotIdx;
    mlir::Type fieldTy;
    std::vector<int64_t> adims;
    if (!structFieldAddr(m, l, Value(), mem, slotIdx, fieldTy, &adims))
      return Value();
    if (!adims.empty())
      return Value(); // array field read as a whole — not supported here.
    unsigned bytes =
        fieldTy.isIntOrFloat() ? (fieldTy.getIntOrFloatBitWidth() / 8) : 4;
    if (bytes <= 4) {
      // f32/i32/bool: load the i32 slot and bitcast back to the field type.
      Value raw = builder.create<memref::LoadOp>(
          l, builder.getI32Type(), mem, ValueRange{slotIdx});
      if (fieldTy.isF32())
        return builder.create<arith::BitcastOp>(l, builder.getF32Type(), raw);
      if (fieldTy.isF16())
        return builder.create<arith::BitcastOp>(l, builder.getF16Type(), raw);
      return castValue(raw, fieldTy, l);
    }
    // i64/f64: load lo+hi i32 slots, combine into i64, bitcast back.
    Value lo = builder.create<memref::LoadOp>(
        l, builder.getI32Type(), mem, ValueRange{slotIdx});
    Value one = builder.create<arith::ConstantOp>(
        l, builder.getIndexType(), builder.getIndexAttr(1));
    Value slotHi = builder.create<arith::AddIOp>(l, slotIdx, one);
    Value hi = builder.create<memref::LoadOp>(
        l, builder.getI32Type(), mem, ValueRange{slotHi});
    Value lo64 = builder.create<arith::ExtUIOp>(l, builder.getI64Type(), lo);
    Value hi64 = builder.create<arith::ExtUIOp>(l, builder.getI64Type(), hi);
    Value sh = builder.create<arith::ConstantOp>(
        l, builder.getI64Type(), builder.getI64IntegerAttr(32));
    Value hiUp = builder.create<arith::ShLIOp>(l, hi64, sh);
    Value asI64 = builder.create<arith::OrIOp>(l, lo64, hiUp);
    if (fieldTy.isF64())
      return builder.create<arith::BitcastOp>(l, builder.getF64Type(), asI64);
    return castValue(asI64, fieldTy, l);
  }

  // Store `rhs` into struct field `m` (e.g. `result[i].f = rhs`). Returns true
  // if handled (the LHS was a struct-pointer field access), false to fall
  // through to the generic lvalue path. The base of `m` is either a DeclRefExpr
  // (a `Result*` param/local) or an IndexExpr (`result[i]`, indexing the SSBO
  // view memref<?xi32>). The struct name is recovered from localTypes[basevar],
  // the field's byte offset from recordLayouts, and the store is an i32-slot
  // memref.store (f32/i32) or a lo/hi pair (i64/f64 via shift+mask).
  bool storeStructField(const MemberAccessExpr *m, Value rhs, Location l) {
    if (!m->base) return false;
    // Resolve the base variable name and optional element index.
    ASTNode *base = m->base.get();
    NodePtr indexExpr; // (unused; we read index from IndexExpr directly)
    ASTNode *indexNode = nullptr;
    if (base->getNodeType() == ASTNode::NodeKind::IndexExpr) {
      auto *ie = static_cast<IndexExpr *>(base);
      indexNode = ie->index.get();
      base = ie->base.get();
    }
    if (!base || base->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *ref = static_cast<DeclRefExpr *>(base);
    auto tit = localTypes.find(ref->name);
    if (tit == localTypes.end()) return false;
    const vc::Type *t = tit->second;
    if (!t) return false;
    // SSBO struct pointer (`Result* r; r[i].f = ...`) or a bare local struct
    // value (`acc.sum = ...`); see structFieldAddr for the shape distinction.
    const RecordType *rec = nullptr;
    bool isLocalStruct = false;
    if (isa<PointerType>(t)) {
      const vc::Type *pointee = cast<PointerType>(t)->pointee;
      if (!pointee || !isa<RecordType>(pointee)) return false;
      rec = cast<RecordType>(pointee);
    } else if (isa<RecordType>(t)) {
      rec = cast<RecordType>(t);
      isLocalStruct = true;
    } else {
      return false;
    }
    auto lit = recordLayouts.find(rec->decl->name);
    if (lit == recordLayouts.end()) return false;
    const StructLayout &layout = lit->second;
    // Find the field.
    const FieldLayout *fld = nullptr;
    for (const auto &f : layout.fields)
      if (f.name == m->member) { fld = &f; break; }
    if (!fld) return false;

    // Base memref — the SSBO memref<?xi32> view, or a local struct's
    // memref<Nxi32, Function> slot.
    auto sit = locals.find(ref->name);
    if (sit == locals.end()) return false;
    Value mem = sit->second;
    if (!mem.getType().isa<MemRefType>()) return false;

    // Byte offset = elementIndex * structStride + fieldOffset. The element term
    // is SSBO-only (local struct has no outer index).
    Value byteOff = builder.create<arith::ConstantOp>(
        l, builder.getI32Type(),
        builder.getI32IntegerAttr((int32_t)fld->offset));
    if (indexNode && !isLocalStruct) {
      Value idx = loadValue(visitExpr(indexNode), loc(indexNode));
      if (!idx) return false;
      idx = castValue(idx, builder.getI32Type(), l);
      Value strideConst = builder.create<arith::ConstantOp>(
          l, builder.getI32Type(), builder.getI32IntegerAttr((int32_t)layout.stride));
      Value elemOff = builder.create<arith::MulIOp>(l, idx, strideConst);
      byteOff = builder.create<arith::AddIOp>(l, byteOff, elemOff);
    }
    // i32 slot index = byteOff / 4.
    Value slotIdx = builder.create<arith::ShRSIOp>(
        l, byteOff,
        builder.create<arith::ConstantOp>(l, builder.getI32Type(),
                                          builder.getI32IntegerAttr(2)));
    slotIdx = castValue(slotIdx, builder.getIndexType(), l);

    rhs = loadValue(rhs, l);
    mlir::Type fty = fld->type;
    unsigned bytes = fty.isIntOrFloat() ? (fty.getIntOrFloatBitWidth() / 8) : 4;
    if (bytes <= 4) {
      // f32/i32/bool: store the value's bit pattern into a single i32 slot.
      // This is a BITCAST (reinterpret bits), not a numeric conversion — the
      // host reads the same bits back as the field's C type (float f reads the
      // f32 bit pattern written here). arith.bitcast f32->i32 preserves bits.
      Value v = rhs;
      if (v.getType().isF32() || v.getType().isF16())
        v = builder.create<arith::BitcastOp>(l, builder.getI32Type(), v);
      else
        v = castValue(rhs, builder.getI32Type(), l);
      builder.create<memref::StoreOp>(l, v, mem, ValueRange{slotIdx});
    } else {
      // i64/f64: store the bit pattern split into lo/hi i32 across two slots.
      // Again bitcast (not numeric convert) so the host reads the raw f64/i64
      // bits. arith.bitcast f64->i64, then shift+mask into two i32 halves.
      Value asI64 = rhs;
      if (rhs.getType().isF64())
        asI64 = builder.create<arith::BitcastOp>(l, builder.getI64Type(), rhs);
      else
        asI64 = castValue(rhs, builder.getI64Type(), l);
      Value mask = builder.create<arith::ConstantOp>(
          l, builder.getI64Type(),
          builder.getI64IntegerAttr(0xFFFFFFFFll));
      Value lo64 = builder.create<arith::AndIOp>(l, asI64, mask);
      Value hi64 = builder.create<arith::ShRUIOp>(
          l, asI64,
          builder.create<arith::ConstantOp>(l, builder.getI64Type(),
                                            builder.getI64IntegerAttr(32)));
      Value lo = builder.create<arith::TruncIOp>(l, builder.getI32Type(), lo64);
      Value hi = builder.create<arith::TruncIOp>(l, builder.getI32Type(), hi64);
      Value one = builder.create<arith::ConstantOp>(
          l, builder.getIndexType(), builder.getIndexAttr(1));
      Value slotHi = builder.create<arith::AddIOp>(l, slotIdx, one);
      builder.create<memref::StoreOp>(l, lo, mem, ValueRange{slotIdx});
      builder.create<memref::StoreOp>(l, hi, mem, ValueRange{slotHi});
    }
    return true;
  }

  Value emitBinary(const BinaryExpr *b) {
    Location l = loc(b);
    if (b->op == BinaryOp::Assign) {
      // Vector swizzle write: `sw.xy = rhs`, `sw.xz = float2(...)` where `sw`
      // is a vector local slot and the LHS member is a multi-char swizzle.
      // SPIR-V has no swizzle-store: lower as load-current + CompositeInsert
      // each written component + store-back. (Single-component write like
      // `sw.x = v` also works through this path.)
      if (b->lhs && b->lhs->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
        auto *m = static_cast<MemberAccessExpr *>(b->lhs.get());
        if (m->base &&
            m->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
          auto *base = static_cast<DeclRefExpr *>(m->base.get());
          auto slotIt = locals.find(base->name);
          if (slotIt != locals.end() &&
              slotIt->second.getType().isa<MemRefType>() &&
              slotIt->second.getType()
                  .cast<MemRefType>()
                  .getElementType()
                  .isa<mlir::VectorType>()) {
            Value rhs = visitExpr(b->rhs.get());
            if (!rhs)
              return error(b->rhs.get(),
                           "could not evaluate swizzle-assignment RHS");
            rhs = loadValue(rhs, l);
            Value slot = slotIt->second;
            mlir::VectorType slotVty = slot.getType()
                                           .cast<MemRefType>()
                                           .getElementType()
                                           .cast<mlir::VectorType>();
            Value cur = builder.create<memref::LoadOp>(l, slot, ValueRange{});
            // Resolve which components of `cur` the swizzle names.
            auto charToIdx = [](char c) -> int {
              switch (c) {
              case 'x': case 'r': case 's': return 0;
              case 'y': case 'g': case 't': return 1;
              case 'z': case 'b': case 'p': return 2;
              case 'w': case 'a': case 'q': return 3;
              default: return -1;
              }
            };
            llvm::StringRef sw = m->member;
            SmallVector<int32_t, 4> dstIdxs;
            bool ok = !sw.empty();
            for (char c : sw) {
              int i = charToIdx(c);
              if (i < 0 || (unsigned)i >= slotVty.getNumElements()) {
                ok = false;
                break;
              }
              dstIdxs.push_back(i);
            }
            if (!ok)
              return error(b->lhs.get(), "invalid vector swizzle on LHS");
            // Pull each source component from rhs. rhs may be a vector of the
            // swizzle width, or a scalar broadcast to all written components.
            Value updated = cur;
            for (size_t i = 0; i < dstIdxs.size(); ++i) {
              Value part;
              if (rhs.getType().isa<mlir::VectorType>()) {
                part = builder.create<spirv::CompositeExtractOp>(
                    l, rhs.getType().cast<mlir::VectorType>().getElementType(),
                    rhs, builder.getI32ArrayAttr({(int32_t)i}));
              } else {
                part = castValue(rhs, slotVty.getElementType(), l);
              }
              updated = builder.create<spirv::CompositeInsertOp>(
                  l, slotVty, part, updated,
                  builder.getI32ArrayAttr({dstIdxs[i]}));
            }
            builder.create<memref::StoreOp>(l, updated, slot);
            return rhs;
          }
        }
      }
      Value rhs = visitExpr(b->rhs.get());
      if (!rhs) return error(b->rhs.get(), "could not evaluate assignment RHS");
      // Struct field write: `result[i].field = rhs` (or `result.field` for a
      // scalar struct ptr). `result` is a memref<?xi32> SSBO view of the struct
      // buffer; the field's byte offset (from recordLayouts) becomes an i32
      // slot index (offset/4), plus i*struct_stride/4 for the element. i64/f64
      // fields are split into lo/hi i32 stores. See cvtType's PointerType<
      // RecordType> note.
      if (b->lhs &&
          b->lhs->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
        auto *m = static_cast<MemberAccessExpr *>(b->lhs.get());
        if (storeStructField(m, rhs, l)) return rhs;
      }
      // Vector-pointer element write: `float4* p; p[i] = vec` scatters the
      // vector's components into N consecutive scalar slots. See
      // storeVectorPointerElement (mirror of the read path).
      if (b->lhs &&
          b->lhs->getNodeType() == ASTNode::NodeKind::IndexExpr) {
        auto *ie = static_cast<IndexExpr *>(b->lhs.get());
        if (storeVectorPointerElement(ie, rhs, l)) return rhs;
      }
      Value mem;
      SmallVector<Value> indices;
      if (lvalueAddress(b->lhs.get(), mem, indices))
        storeTo(mem, indices, rhs, l);
      else
        error(b->lhs.get(), "assignment LHS is not an assignable lvalue");
      return rhs;
    }
    Value lhs = visitExpr(b->lhs.get());
    Value rhs = visitExpr(b->rhs.get());
    if (!lhs) return error(b->lhs.get(), "could not evaluate binary LHS");
    if (!rhs) return error(b->rhs.get(), "could not evaluate binary RHS");

    switch (b->op) {
    case BinaryOp::LAnd:
      return builder.create<arith::AndIOp>(l, toI1(lhs, loc(b->lhs.get())),
                                           toI1(rhs, loc(b->rhs.get())));
    case BinaryOp::LOr:
      return builder.create<arith::OrIOp>(l, toI1(lhs, loc(b->lhs.get())),
                                          toI1(rhs, loc(b->rhs.get())));
    default:
      break;
    }

    std::tie(lhs, rhs) = commonize(lhs, rhs, l);
    // Float vs int dispatch keys off the (scalar) element type, so the same
    // path handles vec+vec and scalar+scalar: arith.addf on vector<4xf32>
    // legalizes to spirv.FAdd via convert-arith-to-spirv, and arith.addi on
    // vector<4xi32> to spirv.IAdd.
    mlir::Type elemTy = lhs.getType();
    if (auto vty = elemTy.dyn_cast<mlir::VectorType>())
      elemTy = vty.getElementType();
    bool isFloat = elemTy.isF32() || elemTy.isF64() || elemTy.isF16();
    switch (b->op) {
    case BinaryOp::Add:
      return isFloat ? (Value)builder.create<arith::AddFOp>(l, lhs, rhs)
                     : (Value)builder.create<arith::AddIOp>(l, lhs, rhs);
    case BinaryOp::Sub:
      return isFloat ? (Value)builder.create<arith::SubFOp>(l, lhs, rhs)
                     : (Value)builder.create<arith::SubIOp>(l, lhs, rhs);
    case BinaryOp::Mul:
      return isFloat ? (Value)builder.create<arith::MulFOp>(l, lhs, rhs)
                     : (Value)builder.create<arith::MulIOp>(l, lhs, rhs);
    case BinaryOp::Div:
      return isFloat ? (Value)builder.create<arith::DivFOp>(l, lhs, rhs)
                     : (Value)builder.create<arith::DivSIOp>(l, lhs, rhs);
    case BinaryOp::Mod:
      return isFloat ? (Value)builder.create<arith::RemFOp>(l, lhs, rhs)
                     : (Value)builder.create<arith::RemSIOp>(l, lhs, rhs);
    case BinaryOp::Shl:
      return builder.create<arith::ShLIOp>(l, lhs, rhs);
    case BinaryOp::Shr:
      return builder.create<arith::ShRSIOp>(l, lhs, rhs);
    case BinaryOp::And:
      return builder.create<arith::AndIOp>(l, lhs, rhs);
    case BinaryOp::Or:
      return builder.create<arith::OrIOp>(l, lhs, rhs);
    case BinaryOp::Xor:
      return builder.create<arith::XOrIOp>(l, lhs, rhs);
    case BinaryOp::Eq: case BinaryOp::NEq: case BinaryOp::Lt:
    case BinaryOp::Gt: case BinaryOp::Le: case BinaryOp::Ge: {
      if (isFloat) {
        arith::CmpFPredicate p = arith::CmpFPredicate::OEQ;
        switch (b->op) {
        case BinaryOp::Eq: p = arith::CmpFPredicate::OEQ; break;
        case BinaryOp::NEq: p = arith::CmpFPredicate::ONE; break;
        case BinaryOp::Lt: p = arith::CmpFPredicate::OLT; break;
        case BinaryOp::Gt: p = arith::CmpFPredicate::OGT; break;
        case BinaryOp::Le: p = arith::CmpFPredicate::OLE; break;
        case BinaryOp::Ge: p = arith::CmpFPredicate::OGE; break;
        default: break;
        }
        return builder.create<arith::CmpFOp>(l, p, lhs, rhs);
      }
      arith::CmpIPredicate p = arith::CmpIPredicate::eq;
      switch (b->op) {
      case BinaryOp::Eq: p = arith::CmpIPredicate::eq; break;
      case BinaryOp::NEq: p = arith::CmpIPredicate::ne; break;
      case BinaryOp::Lt: p = arith::CmpIPredicate::slt; break;
      case BinaryOp::Gt: p = arith::CmpIPredicate::sgt; break;
      case BinaryOp::Le: p = arith::CmpIPredicate::sle; break;
      case BinaryOp::Ge: p = arith::CmpIPredicate::sge; break;
      default: break;
      }
      return builder.create<arith::CmpIOp>(l, p, lhs, rhs);
    }
    default:
      return error(b, "MLIR backend does not support this binary operator");
    }
  }

  Value emitConditional(const ConditionalExpr *c) {
    Location l = loc(c);
    Value cond = toI1(visitExpr(c->cond.get()), loc(c->cond.get()));
    Value tv = loadValue(visitExpr(c->thenExpr.get()), loc(c->thenExpr.get()));
    Value fv = loadValue(visitExpr(c->elseExpr.get()), loc(c->elseExpr.get()));
    if (!cond) return error(c->cond.get(), "could not evaluate ?: condition");
    if (!tv) return error(c->thenExpr.get(), "could not evaluate ?: true arm");
    if (!fv) return error(c->elseExpr.get(), "could not evaluate ?: false arm");
    if (tv.getType() != fv.getType()) {
      // Fold to the wider integer type when they differ.
      mlir::Type tt = tv.getType(), ft = fv.getType();
      if (tt.isSignlessInteger() && ft.isSignlessInteger()) {
        unsigned w = std::max(tt.getIntOrFloatBitWidth(),
                              ft.getIntOrFloatBitWidth());
        auto t = builder.getIntegerType(w);
        if (tv.getType().getIntOrFloatBitWidth() < w)
          tv = builder.create<arith::ExtSIOp>(l, t, tv);
        if (fv.getType().getIntOrFloatBitWidth() < w)
          fv = builder.create<arith::ExtSIOp>(l, t, fv);
      }
    }
    return builder.create<arith::SelectOp>(l, tv.getType(), cond, tv, fv);
  }
};

} // namespace

OwningOpRef<ModuleOp> vc::codegen::translateASTToMLIR(const TranslationUnit &tu,
                                                      MLIRContext &ctx) {
  ASTToMLIRImpl impl(ctx);
  ModuleOp module = impl.translate(tu);
  if (impl.failed()) {
    // A diagnostic was emitted for an unsupported construct; signal failure
    // so the driver does not report success on a partial shader.
    module.erase();
    return OwningOpRef<ModuleOp>();
  }
  // Flag subgroup usage so VCToGPU can bump the SPIR-V target env to 1.3 and
  // advertise the GroupNonUniform* capabilities the lowered warp ops need.
  if (impl.usedSubgroup())
    module->setAttr("vc.uses_subgroup", UnitAttr::get(&ctx));
  return OwningOpRef<ModuleOp>(module);
}
