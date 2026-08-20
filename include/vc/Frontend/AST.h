//===- AST.h - AST node definitions for the VC language -------------------===//
//
// AST node hierarchy for VC. Covers the subset needed for a CUDA-like
// vector-add kernel and its host launch site; designed to grow.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_AST_H
#define VC_FRONTEND_AST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/SMLoc.h"

#include <memory>
#include <string>
#include <vector>

namespace vc {

using llvm::SMLoc;
using llvm::StringRef;

//===----------------------------------------------------------------------===//
// Source location / diagnostics
//===----------------------------------------------------------------------===//

struct SourceLocation {
  SMLoc loc;
  unsigned line = 0;
  unsigned col = 0;
};

enum class DiagnosticKind { Error, Warning, Note };

struct Diagnostic {
  DiagnosticKind kind = DiagnosticKind::Error;
  SourceLocation where;
  std::string message;
};

//===----------------------------------------------------------------------===//
// Type kinds
//===----------------------------------------------------------------------===//

enum class BuiltinTypeKind {
  Void,
  Bool,
  Int32,
  UInt32,
  Int64,
  UInt64,
  Float32,
  Float64,
};

enum class TypeKind { Builtin, Pointer };

class Type {
  TypeKind kind;

public:
  Type(TypeKind k) : kind(k) {}
  TypeKind getKind() const { return kind; }
  virtual ~Type() = default;
};

class BuiltinType : public Type {
public:
  BuiltinTypeKind builtin;

  BuiltinType(BuiltinTypeKind b)
      : Type(TypeKind::Builtin), builtin(b) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Builtin; }
};

class PointerType : public Type {
public:
  Type *pointee;

  PointerType(Type *p) : Type(TypeKind::Pointer), pointee(p) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Pointer; }
};

//===----------------------------------------------------------------------===//
// AST node base
//===----------------------------------------------------------------------===//

class ASTNode {
  SourceLocation loc;

public:
  ASTNode(SourceLocation l) : loc(l) {}
  virtual ~ASTNode() = default;
  SourceLocation getLoc() const { return loc; }

  enum class NodeKind {
    // Decls
    TranslationUnit,
    KernelDecl,
    FunctionDecl,
    ParamDecl,
    VarDecl,
    // Statements
    CompoundStmt,
    ReturnStmt,
    DeclStmt,
    ExprStmt,
    IfStmt,
    ForStmt,
    WhileStmt,
    // Expressions
    BinaryExpr,
    UnaryExpr,
    CallExpr,
    DeclRefExpr,
    IntegerLiteral,
    FloatLiteral,
    IndexExpr,        // a[i]
    MemberAccessExpr, // threadIdx.x
    LaunchExpr,       // kernel<<<grid,block>>>(args)
  };

  virtual NodeKind getNodeType() const = 0;
};

using NodePtr = std::unique_ptr<ASTNode>;

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

class TranslationUnit : public ASTNode {
public:
  std::vector<NodePtr> decls;
  std::vector<Diagnostic> diagnostics;

  TranslationUnit(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::TranslationUnit; }

  llvm::iterator_range<std::vector<NodePtr>::iterator> getDecls() {
    return {decls.begin(), decls.end()};
  }
};

// CUDA __global__/__device__/__host__ attributes on a function.
enum class DeviceAttr { None, Global, Device, Host };

class ParamDecl : public ASTNode {
public:
  Type *type;
  StringRef name;
  ParamDecl(SourceLocation l, Type *t, StringRef n)
      : ASTNode(l), type(t), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::ParamDecl; }
};

class FunctionDecl : public ASTNode {
public:
  Type *returnType = nullptr;
  StringRef name;
  std::vector<ParamDecl *> params;
  NodePtr body; // CompoundStmt, may be null
  DeviceAttr deviceAttr = DeviceAttr::None;

  FunctionDecl(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::FunctionDecl; }
};

class KernelDecl : public ASTNode {
public:
  FunctionDecl *func = nullptr; // the underlying __global__ function

  KernelDecl(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::KernelDecl; }
};

class VarDecl : public ASTNode {
public:
  Type *type;
  StringRef name;
  NodePtr init; // optional initializer expr
  bool isShared = false; // CUDA __shared__
  // Trailing array dimensions, e.g. "float a[16][8]" -> {16,8}. Empty for
  // a scalar. A runtime-sized pointer param leaves this empty.
  std::vector<int64_t> arrayDims;

  VarDecl(SourceLocation l, Type *t, StringRef n)
      : ASTNode(l), type(t), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::VarDecl; }
};

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

class CompoundStmt : public ASTNode {
public:
  std::vector<NodePtr> statements;
  CompoundStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::CompoundStmt; }
};

class ReturnStmt : public ASTNode {
public:
  NodePtr value;
  ReturnStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::ReturnStmt; }
};

class DeclStmt : public ASTNode {
public:
  VarDecl *decl;
  DeclStmt(SourceLocation l, VarDecl *d) : ASTNode(l), decl(d) {}
  NodeKind getNodeType() const override { return NodeKind::DeclStmt; }
};

class ExprStmt : public ASTNode {
public:
  NodePtr expr;
  ExprStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::ExprStmt; }
};

class IfStmt : public ASTNode {
public:
  NodePtr cond;
  NodePtr thenStmt;
  NodePtr elseStmt; // optional
  IfStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::IfStmt; }
};

// C-style for(init; cond; step) body.  init/step may be null.
class ForStmt : public ASTNode {
public:
  NodePtr init; // DeclStmt or ExprStmt, optional
  NodePtr cond; // ExprStmt's expr, optional
  NodePtr step; // Expr, optional
  NodePtr body; // CompoundStmt or single stmt
  ForStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::ForStmt; }
};

class WhileStmt : public ASTNode {
public:
  NodePtr cond;
  NodePtr body; // CompoundStmt or single stmt
  WhileStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::WhileStmt; }
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

enum class BinaryOp { Add, Sub, Mul, Div, Assign, Eq, NEq, Lt, Gt, Le, Ge,
                      And, Or, LAnd, LOr };

class BinaryExpr : public ASTNode {
public:
  BinaryOp op;
  NodePtr lhs, rhs;
  BinaryExpr(SourceLocation l, BinaryOp o, NodePtr a, NodePtr b)
      : ASTNode(l), op(o), lhs(std::move(a)), rhs(std::move(b)) {}
  NodeKind getNodeType() const override { return NodeKind::BinaryExpr; }
};

enum class UnaryOp { Neg, Not, LNot, Deref, AddrOf };

class UnaryExpr : public ASTNode {
public:
  UnaryOp op;
  NodePtr operand;
  UnaryExpr(SourceLocation l, UnaryOp o, NodePtr e)
      : ASTNode(l), op(o), operand(std::move(e)) {}
  NodeKind getNodeType() const override { return NodeKind::UnaryExpr; }
};

class CallExpr : public ASTNode {
public:
  NodePtr callee;     // DeclRefExpr
  std::vector<NodePtr> args;
  CallExpr(SourceLocation l, NodePtr c) : ASTNode(l), callee(std::move(c)) {}
  NodeKind getNodeType() const override { return NodeKind::CallExpr; }
};

class DeclRefExpr : public ASTNode {
public:
  StringRef name;
  DeclRefExpr(SourceLocation l, StringRef n) : ASTNode(l), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::DeclRefExpr; }
};

class IntegerLiteral : public ASTNode {
public:
  int64_t value;
  IntegerLiteral(SourceLocation l, int64_t v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::IntegerLiteral; }
};

class FloatLiteral : public ASTNode {
public:
  double value;
  FloatLiteral(SourceLocation l, double v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::FloatLiteral; }
};

class IndexExpr : public ASTNode {
public:
  NodePtr base;
  NodePtr index;
  IndexExpr(SourceLocation l, NodePtr b, NodePtr i)
      : ASTNode(l), base(std::move(b)), index(std::move(i)) {}
  NodeKind getNodeType() const override { return NodeKind::IndexExpr; }
};

// e.g. threadIdx.x  ->  base="threadIdx", member="x"
class MemberAccessExpr : public ASTNode {
public:
  NodePtr base;
  StringRef member;
  MemberAccessExpr(SourceLocation l, NodePtr b, StringRef m)
      : ASTNode(l), base(std::move(b)), member(m) {}
  NodeKind getNodeType() const override { return NodeKind::MemberAccessExpr; }
};

// kernel<<<grid, block>>>(args...)   — CUDA launch syntax
class LaunchExpr : public ASTNode {
public:
  NodePtr callee;          // DeclRefExpr naming the kernel
  NodePtr gridDim;         // grid size (int or dim3-like)
  NodePtr blockDim;        // block size
  std::vector<NodePtr> args;
  LaunchExpr(SourceLocation l, NodePtr c, NodePtr g, NodePtr b)
      : ASTNode(l), callee(std::move(c)), gridDim(std::move(g)),
        blockDim(std::move(b)) {}
  NodeKind getNodeType() const override { return NodeKind::LaunchExpr; }
};

} // namespace vc

#endif // VC_FRONTEND_AST_H
