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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  llvm::DenseMap<llvm::StringRef, func::FuncOp> funcTable;
  // name -> FunctionDecl (for completing default arguments at call sites that
  // omit trailing defaulted parameters, mirroring the GLSL backend).
  llvm::DenseMap<llvm::StringRef, const FunctionDecl *> funcDecls;
  // name -> Value (block arg / local memref / alloca)
  llvm::StringMap<Value> locals;
  // name -> __constant__ global VarDecl, materialized lazily per kernel.
  llvm::StringMap<const VarDecl *> constGlobals;
  // Entry block of the function being emitted (for hoisting const slots).
  Block *entryBlock = nullptr;
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
  }

  ModuleOp translate(const TranslationUnit &tu) {
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module.getBody());
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

  bool failed() const { return hadError; }

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

  void visitTopLevel(const ASTNode *n) {
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
    // Top-level constructs the MLIR backend does not yet model. These would
    // otherwise vanish silently; surface them so the user knows the shader
    // is missing the construct.
    if (n->getNodeType() == ASTNode::NodeKind::StructDecl ||
        n->getNodeType() == ASTNode::NodeKind::NamespaceDecl ||
        n->getNodeType() == ASTNode::NodeKind::EnumDecl ||
        n->getNodeType() == ASTNode::NodeKind::TypedefDecl)
      error(n, std::string("MLIR backend does not support top-level ") +
                  nodeKindName(n->getNodeType()) +
                  "; construct dropped from device code");
  }

  mlir::Type cvtType(const vc::Type *t) {
    if (!t) return mlir::Type();
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
    if (isa<PointerType>(t)) {
      // pointer-to-T  ->  memref<?xT> in the StorageBuffer (global device
      // memory) storage class; MemRefToSPIRV requires a SPIR-V storage class
      // attribute (not a numeric memory space) on every memref it lowers.
      // The 1-D dynamic memref gets a static strided<[1], offset: 0> layout,
      // which getVulkanElementPtr needs to emit an element pointer (it refuses
      // dynamic strides/offsets).
      mlir::Type pointee = cvtType(cast<PointerType>(t)->pointee);
      auto layout = mlir::StridedLayoutAttr::get(
          &ctx, /*offset=*/0, ArrayRef<int64_t>{/*stride=*/1});
      return MemRefType::get(
          ArrayRef<int64_t>{ShapedType::kDynamic}, pointee, layout,
          spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::StorageBuffer));
    }
    return mlir::Type();
  }

  void buildFunction(const FunctionDecl *fn) {
    if (!fn) return;
    // Function signature
    SmallVector<mlir::Type> argTypes;
    for (auto *p : fn->params)
      argTypes.push_back(cvtType(p->type));
    mlir::Type retTy = cvtType(fn->returnType);
    // A void function has an empty result list (SPIR-V entry points must not
    // declare a `none` result); the LLVM-style none type is an interior
    // convenience only.
    TypeRange results = retTy;
    if (retTy && retTy.isa<NoneType>())
      results = TypeRange{};
    FunctionType fty = builder.getFunctionType(argTypes, results);

    auto f = func::FuncOp::create(loc(fn), fn->name, fty);
    if (fn->deviceAttr == DeviceAttr::Global) {
      // Mark as a kernel entry point.
      f->setAttr("vc.kernel", builder.getUnitAttr());
    }
    module.push_back(f);
    funcTable[fn->name] = f;
    funcDecls[fn->name] = fn;
    currentRetTy = retTy;

    if (!fn->body) return;

    // Function body
    Block *entry = f.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    // locals/entryBlock are per-function state (binding args below).
    entryBlock = entry;
    locals.clear();

    // Bind parameters to block args.
    for (unsigned i = 0; i < fn->params.size(); ++i)
      locals[fn->params[i]->name] = entry->getArgument(i);

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
      auto symRef = SymbolRefAttr::get(&ctx, fn->name, {});
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
        // Element type of the deepest array.
        mlir::Type pointee =
            base.getType().cast<spirv::PointerType>().getPointeeType();
        mlir::Type elemTy = pointee;
        while (auto arr = elemTy.dyn_cast<spirv::ArrayType>())
          elemTy = arr.getElementType();
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
  Value getOrCreateSharedGlobal(llvm::StringRef name, ArrayRef<int64_t> shape,
                                mlir::Type elemTy, Location l) {
    std::string sym = ("__vc_shared_") + name.str();
    // The pointee type: the element for a scalar, or a spirv.array wrapping the
    // element for a (multi-dimensional) shared array. spirv.array is row-major
    // and nested for multi-dim (`float s[16][8]` -> array<16 x array<8 x f32>>).
    mlir::Type pointee = elemTy;
    if (!shape.empty()) {
      for (auto dim : llvm::reverse(shape))
        pointee = spirv::ArrayType::get(pointee, dim);
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

  // Put two operands on a common type for a binop: float stays float, and if
  // one side is index the other is promoted to index (indexing arithmetic).
  std::pair<Value, Value> commonize(Value l, Value r, Location lc) {
    l = loadValue(l, lc);
    r = loadValue(r, lc);
    if (l.getType() == r.getType()) return {l, r};
    // Mixed int/float: promote the integer side to the float type so a float
    // op (e.g. `TILE * scale` where TILE is int, scale is float) lowers to
    // arith.mulf instead of crashing arith.muli on an f32 operand.
    if (l.getType().isF32() && r.getType().isIntOrIndex()) {
      if (r.getType().isIndex())
        r = builder.create<arith::IndexCastOp>(lc, builder.getI32Type(), r);
      return {l, builder.create<arith::SIToFPOp>(lc, builder.getF32Type(), r)};
    }
    if (r.getType().isF32() && l.getType().isIntOrIndex()) {
      if (l.getType().isIndex())
        l = builder.create<arith::IndexCastOp>(lc, builder.getI32Type(), l);
      return {builder.create<arith::SIToFPOp>(lc, builder.getF32Type(), l), r};
    }
    if (l.getType().isF32() || r.getType().isF32()) return {l, r};
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
    if (from.isF32() && to.isSignlessInteger())
      return builder.create<arith::FPToSIOp>(lc, to, v);
    if (from.isF64() && to.isSignlessInteger())
      return builder.create<arith::FPToSIOp>(lc, to, v);
    if (from.isF32() && to.isF64())
      return builder.create<arith::ExtFOp>(lc, to, v);
    if (from.isF64() && to.isF32())
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
      // (extern __shared__ T s[], arrayDims={0}) is not supported here.
      SmallVector<int64_t, 4> shape;
      for (int64_t dim : d->arrayDims) {
        if (dim <= 0) {
          error(d, "unsized __shared__ array (extern __shared__ T s[]) is not "
                   "supported by the MLIR backend; give it an explicit size");
          shape.push_back(1); // keep going so the slot has a valid type
        } else {
          shape.push_back(dim);
        }
      }
      if (d->isShared) {
        // Module-scope global in Workgroup storage, fetched per use.
        Value addr = getOrCreateSharedGlobal(d->name, shape, ty, l);
        locals[d->name] = addr;
        // __shared__ decls may not have a non-constant initializer in CUDA
        // (no host-visible init); ignore any initializer for the global.
        break;
      }
      MemRefType slotTy = MemRefType::get(
          shape, ty, MemRefLayoutAttrInterface(),
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
    case ASTNode::NodeKind::CaseStmt:
      // TODO: switch lowers to a chain of scf.if once needed.
      error(n, std::string("MLIR backend does not support ") +
                  nodeKindName(n->getNodeType()) + "; statement dropped");
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
    Location l = loc(s);
    auto flagTy =
        MemRefType::get(ArrayRef<int64_t>{}, builder.getI1Type(),
                        MemRefLayoutAttrInterface(),
                        spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
    Value flag = builder.create<memref::AllocaOp>(l, flagTy);
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(true)),
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
    Value first = builder.create<memref::LoadOp>(l, flag, ValueRange{});
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(false)),
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

  Value makeFlagSlot(const Location &l) {
    auto flagTy =
        MemRefType::get(ArrayRef<int64_t>{}, builder.getI1Type(),
                        MemRefLayoutAttrInterface(),
                        spirv::StorageClassAttr::get(&ctx, spirv::StorageClass::Function));
    Value slot = builder.create<memref::AllocaOp>(l, flagTy);
    builder.create<memref::StoreOp>(
        l, builder.create<arith::ConstantOp>(l, builder.getBoolAttr(false)), slot);
    return slot;
  }

  Value loadFlag(Value slot, const Location &l) {
    return builder.create<memref::LoadOp>(l, slot, ValueRange{});
  }

  void storeFlag(Value slot, Value v, const Location &l) {
    builder.create<memref::StoreOp>(l, v, slot);
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
      double v = static_cast<FloatLiteral *>(n)->value;
      return builder.create<arith::ConstantOp>(
          l, builder.getF32Type(), builder.getFloatAttr(builder.getF32Type(), v));
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
    case ASTNode::NodeKind::IndexExpr: {
      // base[idx...] where base is a memref arg / slot or a shared spirv.ptr.
      // Reuses lvalueAddress so multi-dimensional `a[i][j]` collects every
      // subscript into one load, and so shared-array indexing (spirv.AccessChain
      // + spirv.Load) is handled uniformly.
      auto *ie = static_cast<IndexExpr *>(n);
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
      return error(n, "MLIR backend does not support member access "
                      "other than threadIdx/blockIdx/blockDim/gridDim");
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
        auto fit = funcTable.find(ref->name);
        if (fit != funcTable.end()) {
          SmallVector<Value> args;
          for (auto &a : c->args) {
            Value av = visitExpr(a.get());
            if (!av) return error(a.get(), "could not evaluate call argument");
            args.push_back(loadValue(av, loc(a.get())));
          }
          // Complete trailing defaulted parameters the call omits, mirroring
          // the GLSL backend: append each default expression from the callee
          // signature until the argument count matches the parameter count.
          auto dit = funcDecls.find(ref->name);
          if (dit != funcDecls.end()) {
            const FunctionDecl *calleeFn = dit->second;
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
      return error(n, "MLIR backend does not support this call "
                      "(unknown callee or builtin); call dropped");
    }
    default:
      return error(n, std::string("MLIR backend cannot lower ") +
                          nodeKindName(n->getNodeType()));
    }
  }

  Value emitUnary(const UnaryExpr *u) {
    Location l = loc(u);
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
      // Deref / AddrOf: degenerate to the operand value (TODO: real lvalue
      // semantics when struct/array members need them).
      return v;
    }
  }

  Value emitBinary(const BinaryExpr *b) {
    Location l = loc(b);
    if (b->op == BinaryOp::Assign) {
      Value rhs = visitExpr(b->rhs.get());
      if (!rhs) return error(b->rhs.get(), "could not evaluate assignment RHS");
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
    bool isFloat = lhs.getType().isF32();
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
  return OwningOpRef<ModuleOp>(module);
}