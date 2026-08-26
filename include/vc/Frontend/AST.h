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
  // True when this diagnostic was already rendered to the user (currently the
  // Parser prints errors inline via SourceMgr as it hits them). Sema reports
  // whatever does not carry this flag, so nothing is printed twice.
  bool rendered = false;
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
  Float16,
  Float32,
  Float64,
};

enum class TypeKind { Builtin, Pointer, Reference, Vector, Record, Typedef };

// Forward declarations: RecordType/TypedefType point at these decls, which are
// defined as ASTNodes further down. A pointer is all that's needed here.
class StructDecl;
class TypedefDecl;

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

// A C++ lvalue reference `T&`. Host-only (device kernels don't use references);
// the host backend emits `T&`, the GLSL backend never sees one. Used for host
// helper functions like `reference(const Point &p, float w)`.
class ReferenceType : public Type {
public:
  Type *pointee;

  ReferenceType(Type *p) : Type(TypeKind::Reference), pointee(p) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Reference; }
};

// A vector of `count` elements of type `elem` (e.g. float4 -> elem=Float32,
// count=4). CUDA/GPUSims names like `float4`/`int3` lower to this. GLSL has
// native vec3/ivec4/etc. support and native swizzles (`.xyz`).
class VectorType : public Type {
public:
  Type *elem;
  unsigned count;

  VectorType(Type *e, unsigned c) : Type(TypeKind::Vector), elem(e), count(c) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Vector; }
};

// A user-defined struct type. Refers to its StructDecl by pointer so the type
// can name the struct before/while it's being defined. GLSL emits the struct
// definition as `struct Name { fields };` before main().
class RecordType : public Type {
public:
  StructDecl *decl;

  RecordType(StructDecl *d) : Type(TypeKind::Record), decl(d) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Record; }
};

// A typedef alias. `typedef <underlying> <name>;` — usages of `name` produce a
// TypedefType that lowers to its underlying type's GLSL spelling. GLSL has no
// typedef, so emit resolves to the underlying type.
class TypedefType : public Type {
public:
  TypedefDecl *decl;

  TypedefType(TypedefDecl *d) : Type(TypeKind::Typedef), decl(d) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::Typedef; }
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
    StructDecl,
    FieldDecl,
    TypedefDecl,
    EnumDecl,
    // Statements
    CompoundStmt,
    ReturnStmt,
    DeclStmt,
    ExprStmt,
    IfStmt,
    ForStmt,
    WhileStmt,
    DoStmt,
    BreakStmt,
    ContinueStmt,
    SwitchStmt,
    CaseStmt, // value null => default
    // Expressions
    BinaryExpr,
    UnaryExpr,
    ConditionalExpr, // cond ? then : else
    CStyleCastExpr,  // (T)expr or T(expr) functional cast
    InitListExpr,    // { a, b, c }
    CallExpr,
    DeclRefExpr,
    IntegerLiteral,
    FloatLiteral,
    CharLiteral,   // 'A' -> 65, decoded
    StringLiteral, // "..." decoded
    IndexExpr,        // a[i]
    MemberAccessExpr, // threadIdx.x
    LaunchExpr,       // kernel<<<grid,block>>>(args)
  };

  virtual NodeKind getNodeType() const = 0;
};

using NodePtr = std::unique_ptr<ASTNode>;

/// Deep-clone an expression node. Used by the parser to expand compound
/// assignment (`a += b` -> `a = a + b`) and `++`/`--` into plain AST, so no
/// new node kinds are needed for those. Covers the expression subset that
/// can appear as an lvalue/rvalue in such contexts; returns null for
/// statements/decls. Defined in ASTHelpers.cpp.
NodePtr cloneExpr(const ASTNode *n);

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

class TranslationUnit : public ASTNode {
public:
  std::vector<NodePtr> decls;
  std::vector<Diagnostic> diagnostics;
  // Preprocessor directive lines (`#include ...`, `#define ...`) captured
  // verbatim from source. The host C++ backend emits these at the top of the
  // generated .cpp; the device (GLSL) backend ignores them.
  std::vector<std::string> hostPpLines;

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
  bool isConstant = false; // CUDA __constant__ (device-resident read-only global)
  // Trailing array dimensions, e.g. "float a[16][8]" -> {16,8}. Empty for
  // a scalar. A runtime-sized pointer param leaves this empty.
  std::vector<int64_t> arrayDims;

  VarDecl(SourceLocation l, Type *t, StringRef n)
      : ASTNode(l), type(t), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::VarDecl; }
};

// A named field within a struct. Owned by its StructDecl (raw pointers; the
// StructDecl owns and deletes them).
class FieldDecl : public ASTNode {
public:
  Type *type;
  StringRef name;
  // Optional trailing array dims for a field, e.g. `float v[4]`.
  std::vector<int64_t> arrayDims;

  FieldDecl(SourceLocation l, Type *t, StringRef n)
      : ASTNode(l), type(t), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::FieldDecl; }
};

// `struct Name { Type field; ... };` — defines a RecordType. Fields are owned
// by the decl (deleted in the destructor). GLSL lowers to `struct Name {...}`.
class StructDecl : public ASTNode {
public:
  StringRef name;
  std::vector<FieldDecl *> fields;

  StructDecl(SourceLocation l, StringRef n) : ASTNode(l), name(n) {}
  ~StructDecl() override {
    for (auto *f : fields) delete f;
  }
  NodeKind getNodeType() const override { return NodeKind::StructDecl; }
};

// `enum [Name] { A, B = 5, C };` — defines integer constants. GLSL has no enum,
// so the backend emits each constant as `const int NAME = <value>;`. Values are
// computed at parse time (default 0, auto-increment, explicit `= const-expr`
// resets). VC supports unscoped enums only (`enum class`'s `E::A` access needs
// a scope operator, not yet implemented).
class EnumDecl : public ASTNode {
public:
  StringRef name; // may be empty for an anonymous enum
  struct Constant {
    StringRef name;
    int64_t value;
  };
  std::vector<Constant> constants;

  EnumDecl(SourceLocation l, StringRef n) : ASTNode(l), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::EnumDecl; }
};

// `typedef <underlying> <name>;` — defines a TypedefType alias. GLSL has no
// typedef, so emit resolves to the underlying type.
class TypedefDecl : public ASTNode {
public:
  StringRef name;
  Type *underlying;

  TypedefDecl(SourceLocation l, StringRef n, Type *u)
      : ASTNode(l), name(n), underlying(u) {}
  NodeKind getNodeType() const override { return NodeKind::TypedefDecl; }
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
  // The first (often only) declarator. Kept for single-decl compatibility;
  // `decls` holds all declarators when a statement declares several names
  // sharing one type (`int a, b, c;`). VarDecls are heap-allocated and not
  // owned by the DeclStmt (matching the rest of the AST's VarDecl handling).
  VarDecl *decl = nullptr;
  std::vector<VarDecl *> decls;
  DeclStmt(SourceLocation l, VarDecl *d) : ASTNode(l), decl(d) {
    if (d) decls.push_back(d);
  }
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

// C: `do body while (cond);`  — body executes at least once, then repeats
// while cond is true.
class DoStmt : public ASTNode {
public:
  NodePtr body;
  NodePtr cond;
  DoStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::DoStmt; }
};

class BreakStmt : public ASTNode {
public:
  BreakStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::BreakStmt; }
};

class ContinueStmt : public ASTNode {
public:
  ContinueStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::ContinueStmt; }
};

// C: `switch (cond) { case X: ...; default: ...; }`. body is a CompoundStmt
// whose statements are CaseStmt (and possibly other stmts).
class SwitchStmt : public ASTNode {
public:
  NodePtr cond;
  NodePtr body;
  SwitchStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::SwitchStmt; }
};

// One case label: `case <value>: <sub>` or `default: <sub>` (value == null).
class CaseStmt : public ASTNode {
public:
  NodePtr value; // null for default
  NodePtr sub;   // statement following the label (may be a CompoundStmt)
  CaseStmt(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::CaseStmt; }
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

enum class BinaryOp { Add, Sub, Mul, Div, Mod, Assign, Eq, NEq, Lt, Gt, Le, Ge,
                      Shl, Shr, And, Or, Xor, LAnd, LOr };

class BinaryExpr : public ASTNode {
public:
  BinaryOp op;
  NodePtr lhs, rhs;
  BinaryExpr(SourceLocation l, BinaryOp o, NodePtr a, NodePtr b)
      : ASTNode(l), op(o), lhs(std::move(a)), rhs(std::move(b)) {}
  NodeKind getNodeType() const override { return NodeKind::BinaryExpr; }
};

enum class UnaryOp {
  Neg, Not, LNot, Deref, AddrOf,
  // ++ / -- as real unary nodes (not parser-expanded). GLSL models the
  // pre/post old-vs-new value semantics natively, so passing these through is
  // more correct than rewriting to `a = a + 1`.
  PreInc, PostInc, PreDec, PostDec,
};

class UnaryExpr : public ASTNode {
public:
  UnaryOp op;
  NodePtr operand;
  UnaryExpr(SourceLocation l, UnaryOp o, NodePtr e)
      : ASTNode(l), op(o), operand(std::move(e)) {}
  NodeKind getNodeType() const override { return NodeKind::UnaryExpr; }
};

// C ternary:  cond ? thenExpr : elseExpr   (right-associative)
class ConditionalExpr : public ASTNode {
public:
  NodePtr cond, thenExpr, elseExpr;
  ConditionalExpr(SourceLocation l, NodePtr c, NodePtr t, NodePtr e)
      : ASTNode(l), cond(std::move(c)), thenExpr(std::move(t)),
        elseExpr(std::move(e)) {}
  NodeKind getNodeType() const override { return NodeKind::ConditionalExpr; }
};

// C-style cast `(T)expr` or functional cast `T(expr)`. Emits as the GLSL
// constructor form `T(expr)` (valid for scalars and vectors alike).
class CStyleCastExpr : public ASTNode {
public:
  Type *target;
  NodePtr sub;
  CStyleCastExpr(SourceLocation l, Type *t, NodePtr s)
      : ASTNode(l), target(t), sub(std::move(s)) {}
  NodeKind getNodeType() const override { return NodeKind::CStyleCastExpr; }
};

// Brace-enclosed initializer list: { a, b, c }. Appears as a VarDecl
// initializer. Emits as GLSL `{ a, b, c }` (valid for vector/array init).
class InitListExpr : public ASTNode {
public:
  std::vector<NodePtr> elements;
  InitListExpr(SourceLocation l) : ASTNode(l) {}
  NodeKind getNodeType() const override { return NodeKind::InitListExpr; }
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

// A character literal 'A' / '\n', decoded to its integer value. GLSL has no
// char type; it lowers as an int constant (matching C's promotion).
class CharLiteral : public ASTNode {
public:
  int64_t value;
  CharLiteral(SourceLocation l, int64_t v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::CharLiteral; }
};

// A string literal "..." with quotes stripped and escapes decoded. VC kernels
// have no string type, so this is primarily so source carrying strings parses
// cleanly; the backend emits a placeholder comment.
class StringLiteral : public ASTNode {
public:
  std::string value;
  StringLiteral(SourceLocation l, std::string v) : ASTNode(l), value(std::move(v)) {}
  NodeKind getNodeType() const override { return NodeKind::StringLiteral; }
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
// `isScope` distinguishes C++ scope resolution `::` (e.g. VCMemcpyKind::HostToDevice)
// from member access `.`. Both parse into this node; the host backend emits `::` when set.
class MemberAccessExpr : public ASTNode {
public:
  NodePtr base;
  StringRef member;
  bool isScope = false; // true if the source used `::` rather than `.`
  MemberAccessExpr(SourceLocation l, NodePtr b, StringRef m)
      : ASTNode(l), base(std::move(b)), member(m) {}
  NodeKind getNodeType() const override { return NodeKind::MemberAccessExpr; }
};

// kernel<<<grid, block>>>(args...)   — CUDA launch syntax
// grid/block may be `dim3(a,b)` (parsed as a CallExpr); the parser splits such
// a call into the X component (gridDim/blockDim) and Y (gridDimY/blockDimY),
// null Y meaning 1D. `stream` captures the optional 4th launch argument
// (<<<g,b,shmem,stream>>>); null means the default stream.
class LaunchExpr : public ASTNode {
public:
  NodePtr callee;          // DeclRefExpr naming the kernel
  NodePtr gridDim;         // grid size X (int or dim3.x)
  NodePtr blockDim;        // block size X
  NodePtr gridDimY;        // grid size Y (null = 1D)
  NodePtr blockDimY;       // block size Y (null = 1D)
  NodePtr stream;          // optional stream handle (null = default stream)
  std::vector<NodePtr> args;
  LaunchExpr(SourceLocation l, NodePtr c, NodePtr g, NodePtr b,
             NodePtr gy = nullptr, NodePtr by = nullptr, NodePtr s = nullptr)
      : ASTNode(l), callee(std::move(c)), gridDim(std::move(g)),
        blockDim(std::move(b)), gridDimY(std::move(gy)),
        blockDimY(std::move(by)), stream(std::move(s)) {}
  NodeKind getNodeType() const override { return NodeKind::LaunchExpr; }
};

} // namespace vc

#endif // VC_FRONTEND_AST_H
