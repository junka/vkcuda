//===- ASTToMLIR.cpp - Lower AST to VC-dialect MLIR -----------------------===//
//
// Walks the AST and emits MLIR. Device (__global__) functions become
// `func.func` entries wrapped by `vc.kernel`; thread/block indexing and
// barriers map to vc.* ops. This scaffold covers the vector-add subset
// and is the place to extend as the language grows.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"

#include "vc/Dialect/VC/Ops.h"
#include "vc/Frontend/AST.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

public:
  ASTToMLIRImpl(MLIRContext &c)
      : ctx(c), builder(&c) {
    ctx.getOrLoadDialect<vc::VCDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<memref::MemRefDialect>();
  }

  ModuleOp translate(const TranslationUnit &tu) {
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module.getBody());
    for (auto &d : tu.decls)
      visitTopLevel(d.get());
    return module;
  }

private:
  Location loc(const ASTNode *n) {
    return FileLineColLoc::get(&ctx, "<vc>", n ? n->getLoc().line : 0,
                               n ? n->getLoc().col : 0);
  }

  void visitTopLevel(const ASTNode *n) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::FunctionDecl)
      buildFunction(static_cast<const FunctionDecl *>(n));
    // KernelDecl is handled when its FunctionDecl is built (we emit a
    // vc.kernel wrapper there).
  }

  Type cvtType(const Type *t) {
    if (!t) return Type();
    if (isa<BuiltinType>(t)) {
      switch (cast<BuiltinType>(t)->builtin) {
      case BuiltinTypeKind::Void: return builder.getType<VoidType>();
      case BuiltinTypeKind::Bool: return builder.getI1Type();
      case BuiltinTypeKind::Int32: case BuiltinTypeKind::UInt32:
        return builder.getI32Type();
      case BuiltinTypeKind::Int64: case BuiltinTypeKind::UInt64:
        return builder.getI64Type();
      case BuiltinTypeKind::Float32: return builder.getF32Type();
      case BuiltinTypeKind::Float64: return builder.getF64Type();
      }
    }
    if (isa<PointerType>(t)) {
      // pointer-to-T  ->  memref<?xT> (device/global address space = 1)
      Type pointee = cvtType(cast<PointerType>(t)->pointee);
      return MemRefType::get({ShapedType::kDynamic}, pointee, {}, 1);
    }
    return Type();
  }

  void buildFunction(const FunctionDecl *fn) {
    if (!fn) return;
    // Function signature
    SmallVector<Type> argTypes;
    for (auto *p : fn->params)
      argTypes.push_back(cvtType(p->type));
    Type retTy = cvtType(fn->returnType);
    FunctionType fty = builder.getFunctionType(argTypes, retTy);

    auto f = func::FuncOp::create(loc(fn), fn->name, fty);
    if (fn->deviceAttr == DeviceAttr::Global) {
      // Mark as a kernel entry point.
      f->setAttr("vc.kernel", builder.getUnitAttr());
    }
    module.push_back(f);
    funcTable[fn->name] = f;

    if (!fn->body) return;

    // Function body
    Block *entry = f.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Bind parameters to block args.
    for (unsigned i = 0; i < fn->params.size(); ++i)
      locals[fn->params[i]->name] = entry->getArgument(i);

    if (auto *cs = dyn_cast<CompoundStmt>(fn->body.get()))
      for (auto &s : cs->statements)
        visitStmt(s.get());

    // Ensure a void return has a terminator.
    if (builder.getBlock()->empty() ||
        !isa<func::ReturnOp>(builder.getBlock()->back()))
      builder.create<func::ReturnOp>(loc(fn));

    // For kernels, emit a vc.kernel wrapper referencing this function.
    if (fn->deviceAttr == DeviceAttr::Global) {
      builder.setInsertionPointToStart(module.getBody());
      auto symRef = SymbolRefAttr::get(ctx, fn->name);
      builder.create<vc::KernelOp>(loc(fn), symRef);
    }
  }

  // name -> Value (block arg / local memref / alloca)
  llvm::StringMap<Value> locals;

  void visitStmt(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<CompoundStmt *>(n)->statements)
        visitStmt(s.get());
      break;
    case ASTNode::NodeKind::ReturnStmt: {
      auto *r = static_cast<ReturnStmt *>(n);
      Value v = r->value ? visitExpr(r->value.get()) : Value();
      builder.create<func::ReturnOp>(loc(n), v ? ValueRange(v) : ValueRange());
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *d = static_cast<DeclStmt *>(n)->decl;
      if (!d) break;
      // Locals become stack allocations; __shared__ would map to workgroup
      // memory space (TODO: use memref.alloca with memory space 3).
      Type ty = cvtType(d->type);
      // For a scalar decl we use a 0-d memref as a mutable slot.
      MemRefType slotTy = MemRefType::get({}, ty);
      Value addr = builder.create<memref::AllocaOp>(loc(d), slotTy);
      locals[d->name] = addr;
      if (d->init) {
        Value v = visitExpr(d->init.get());
        builder.create<memref::StoreOp>(loc(d), v, addr);
      }
      break;
    }
    case ASTNode::NodeKind::ExprStmt:
      if (auto *e = static_cast<ExprStmt *>(n)->expr.get())
        (void)visitExpr(e);
      break;
    default:
      break;
    }
  }

  Value visitExpr(const ASTNode *n) {
    if (!n) return Value();
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral:
      return builder.create<arith::ConstantIndexOp>(
          loc(n), static_cast<IntegerLiteral *>(n)->value);
    case ASTNode::NodeKind::FloatLiteral: {
      double v = static_cast<FloatLiteral *>(n)->value;
      return builder.create<arith::ConstantOp>(
          loc(n), builder.getF32Type(),
          builder.getFloatAttr(builder.getF32Type(), v));
    }
    case ASTNode::NodeKind::DeclRefExpr: {
      auto name = static_cast<DeclRefExpr *>(n)->name;
      auto it = locals.find(name);
      if (it != locals.end())
        return it->second; // a memref slot or block arg
      return Value();
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<BinaryExpr *>(n);
      if (b->op == BinaryOp::Assign) {
        // Store rhs into lhs address (lhs must be a memref slot / indexable).
        Value lhsAddr = visitExpr(b->lhs.get());
        Value rhs = visitExpr(b->rhs.get());
        builder.create<memref::StoreOp>(loc(n), rhs, lhsAddr);
        return rhs;
      }
      Value l = visitExpr(b->lhs.get());
      Value r = visitExpr(b->rhs.get());
      // If operands are memref slots, load them first.
      if (auto mr = l.getType().dyn_cast<MemRefType>())
        l = builder.create<memref::LoadOp>(loc(n), l, ValueRange{});
      if (auto mr = r.getType().dyn_cast<MemRefType>())
        r = builder.create<memref::LoadOp>(loc(n), r, ValueRange{});
      switch (b->op) {
      case BinaryOp::Add:
        if (l.getType().isF32())
          return builder.create<arith::AddFOp>(loc(n), l, r);
        return builder.create<arith::AddIOp>(loc(n), l, r);
      case BinaryOp::Sub:
        if (l.getType().isF32())
          return builder.create<arith::SubFOp>(loc(n), l, r);
        return builder.create<arith::SubIOp>(loc(n), l, r);
      case BinaryOp::Mul:
        if (l.getType().isF32())
          return builder.create<arith::MulFOp>(loc(n), l, r);
        return builder.create<arith::MulIOp>(loc(n), l, r);
      case BinaryOp::Div:
        if (l.getType().isF32())
          return builder.create<arith::DivFOp>(loc(n), l, r);
        return builder.create<arith::DivSIOp>(loc(n), l, r);
      default:
        return Value();
      }
    }
    case ASTNode::NodeKind::IndexExpr: {
      // base[idx] where base is a memref arg.
      auto *ie = static_cast<IndexExpr *>(n);
      Value base = visitExpr(ie->base.get());
      Value idx = visitExpr(ie->index.get());
      if (base.getType().isa<MemRefType>())
        return builder.create<memref::LoadOp>(loc(n), base, ValueRange{idx});
      return Value();
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      // threadIdx.x / blockIdx.x / blockDim.x / gridDim.x
      auto *m = static_cast<MemberAccessExpr *>(n);
      if (auto *base = dyn_cast<DeclRefExpr>(m->base.get())) {
        vc::Dim dim = vc::Dim::x;
        if (m->member == "y") dim = vc::Dim::y;
        else if (m->member == "z") dim = vc::Dim::z;
        if (base->name == "threadIdx")
          return builder.create<vc::ThreadIdOp>(loc(n), dim);
        if (base->name == "blockIdx")
          return builder.create<vc::BlockIdOp>(loc(n), dim);
        if (base->name == "blockDim")
          return builder.create<vc::BlockDimOp>(loc(n), dim);
        if (base->name == "gridDim")
          return builder.create<vc::GridDimOp>(loc(n), dim);
      }
      return Value();
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<CallExpr *>(n);
      if (auto *ref = dyn_cast<DeclRefExpr>(c->callee.get())) {
        if (ref->name == "__syncthreads")
          return builder.create<vc::BarrierOp>(loc(n));
        // TODO: resolve user/device functions.
      }
      return Value();
    }
    default:
      return Value();
    }
  }
};

} // namespace

OwningOpRef<ModuleOp> vc::codegen::translateASTToMLIR(const TranslationUnit &tu,
                                                      MLIRContext &ctx) {
  ASTToMLIRImpl impl(ctx);
  return impl.translate(tu);
}
