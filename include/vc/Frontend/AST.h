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

enum class TypeKind { Builtin, Pointer, Reference, Vector, Record, Typedef, WmmaFragment };

// Forward declarations: RecordType/TypedefType point at these decls, which are
// defined as ASTNodes further down. A pointer is all that's needed here.
class StructDecl;
class TypedefDecl;

// CUDA wmma::fragment — a tensor-core tile. The dims (M,N,K) come from the source
// `wmma::fragment<use, M, N, K, T, layout>` template args and match a shape the
// Vulkan target actually supports (e.g. 8x8x16 fp16-input/fp32-accumulate). A/B
// carry their Layout at the type level (CUDA semantics); the accumulator has
// Layout::None and takes a runtime layout_t at load/store instead.
enum class WmmaUse { MatrixA, MatrixB, Accumulator };
enum class WmmaPrecision { F16, F32 };
enum class WmmaLayout { RowMajor, ColMajor, None };

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

// A wmma::fragment tile type (tensor-core register tile). Carries the use
// (matrix_a/matrix_b/accumulator), element precision, compile-time layout (A/B
// only), and the (M,N,K) shape. Storage is opaque/warp-distributed: fragments
// are only initialized by load_matrix_sync / mma_sync and consumed by
// mma_sync / store_matrix_sync; direct element access is not supported.
class WmmaFragmentType : public Type {
public:
  WmmaUse use;
  WmmaPrecision prec;
  WmmaLayout layout;
  unsigned M, N, K;

  WmmaFragmentType(WmmaUse u, WmmaPrecision p, WmmaLayout l, unsigned m,
                   unsigned n, unsigned k)
      : Type(TypeKind::WmmaFragment), use(u), prec(p), layout(l), M(m), N(n), K(k) {}
  static bool classof(const Type *t) { return t->getKind() == TypeKind::WmmaFragment; }
  bool isAccumulator() const { return use == WmmaUse::Accumulator; }
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
    NamespaceDecl,
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
    CommaExpr,       // a, b  (evaluates a, result is b)
    CStyleCastExpr,  // (T)expr or T(expr) functional cast
    InitListExpr,    // { a, b, c }
    CallExpr,
    DeclRefExpr,
    IntegerLiteral,
    FloatLiteral,
    BoolLiteral,   // true/false, lowers to GLSL true/false
    CharLiteral,   // 'A' -> 65, decoded
    StringLiteral, // "..." decoded
    IndexExpr,        // a[i]
    MemberAccessExpr, // threadIdx.x
    SizeOfExpr,       // sizeof(T) or sizeof expr
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

// C storage class specifiers. `static`/`extern` are recognized by the lexer and
// recorded here so the host C++ backend can emit them verbatim. The device GLSL
// backend strips them (GLSL has no storage-class keywords; its globals are just
// shader globals). Function-local `static` is rejected by Sema inside device
// code (GLSL has no function-local static storage — `__shared__` is the device
// persistent-storage mechanism).
enum class StorageClass { None, Static, Extern };

class ParamDecl : public ASTNode {
public:
  Type *type;
  StringRef name;
  bool isConst = false; // C `const` qualifier on this parameter
  // Optional default-argument expression (`void f(int a, int b = 10)`). Null
  // when the parameter has no default. The GLSL/C++ definition omits `= val`
  // (defaults belong on declarations, and VC's helpers are definitions); call
  // sites are completed by the backends using these expressions.
  NodePtr defaultVal;
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
  StorageClass storageClass = StorageClass::None; // C `static`/`extern`
  // C++ class member function. `isMethod` marks this as a method of `className`;
  // on the device the GLSL backend lowers it to a free function
  // `Class_method(Class _this, args...)` (this as the first parameter, since
  // GLSL structs have no member functions). The host backend emits a real C++
  // method. `className` is empty for free functions.
  bool isMethod = false;
  StringRef className;
  // Enclosing namespace scope, dotted with `::` for nesting (e.g. `outer` or
  // `outer::inner`). Empty for top-level functions. The device backend mangles
  // `ns::func` to `ns_func` (and `outer::inner::func` to `outer_inner_func`);
  // the host backend emits the verbatim `namespace ns { ... }` block and keeps
  // the qualified spelling at call sites, so this field is device-only.
  StringRef nsName;
  // Whether the source defined a body (false => declaration only, e.g. an
  // out-of-line class method). Methods declared inside a class body may have a
  // null body; methods defined out-of-line (`void C::f() {...}`) have one.
  bool hasBody = false;

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
  bool isConst = false; // C `const` qualifier (cv-qualifier on a local/param)
  StorageClass storageClass = StorageClass::None; // C `static`/`extern`
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
// `isClass` marks a C++ `class` (vs `struct`): only the default access differs,
// and VC doesn't enforce access control, so the distinction matters only for
// the host backend (which emits `class Name { public: ... };`). `methods` are
// member functions declared inside the body; they are NOT owned here (they're
// owned by the TranslationUnit / NamespaceDecl decl list as FunctionDecls), so
// the destructor does not delete them. GLSL never emits methods on the struct.
class StructDecl : public ASTNode {
public:
  StringRef name;
  std::vector<FieldDecl *> fields;
  bool isClass = false;
  std::vector<FunctionDecl *> methods; // non-owning

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

// `namespace Name { decls... }` — a named scope. The body decls are owned here
// (they are moved out of the parser's decl stack). GLSL flattens namespaces:
// the device backend recurses into the body and emits each decl with a mangled
// name (`ns::f` -> `ns_f`) rather than emitting a namespace block (GLSL has no
// namespaces). The host backend emits `namespace Name { ... }` verbatim.
class NamespaceDecl : public ASTNode {
public:
  StringRef name;
  std::vector<NodePtr> decls;
  NamespaceDecl(SourceLocation l, StringRef n) : ASTNode(l), name(n) {}
  NodeKind getNodeType() const override { return NodeKind::NamespaceDecl; }
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

// C comma operator: `a, b` — evaluates a (for its side effects), then b; the
// result and type are b's. Lower than assignment in precedence, so it appears
// at statement-init/step and parenthesized contexts, not as a call/init-list
// element (the parser uses parseAssignment for those to keep `f(a, b)` as two
// args, not one CommaExpr).
class CommaExpr : public ASTNode {
public:
  NodePtr lhs, rhs;
  CommaExpr(SourceLocation l, NodePtr a, NodePtr b)
      : ASTNode(l), lhs(std::move(a)), rhs(std::move(b)) {}
  NodeKind getNodeType() const override { return NodeKind::CommaExpr; }
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
  // The callee FunctionDecl selected by Sema overload resolution. Null for
  // builtins (which have no FunctionDecl), unscoped calls Sema couldn't
  // resolve, or before Sema runs. Backends use this to emit the call against
  // the resolved overload's mangled device symbol rather than re-resolving.
  FunctionDecl *resolvedCallee = nullptr;
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
  // True if the source spelled a `long` suffix (`l`/`L`). The lexer consumes
  // integer suffixes without recording them, so this flag is the only record
  // that the literal was `100L` (long) rather than `100` (int). Host codegen
  // re-attaches the `L` so a spilled launch arg deduces the right width,
  // matching a long device parameter.
  bool isLong = false;
  IntegerLiteral(SourceLocation l, int64_t v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::IntegerLiteral; }
};

class FloatLiteral : public ASTNode {
public:
  double value;
  // True if the source spelled a single-precision suffix (`f`/`F`). The lexer
  // drops the suffix when building the token text, so this flag is the only
  // record that the literal was `2.5f` (float) rather than `2.5` (double).
  // Host codegen relies on it to re-attach the `f` suffix so a spilled launch
  // arg `auto x = (2.5f)` deduces float, matching a float device parameter.
  bool isFloat32 = false;
  FloatLiteral(SourceLocation l, double v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::FloatLiteral; }
};

// A boolean literal `true`/`false`. GLSL and C++ both spell these `true`/
// `false`; the node carries the value (1/0) so Sema can type it as Bool and
// the dumpers can render the source spelling.
class BoolLiteral : public ASTNode {
public:
  bool value;
  BoolLiteral(SourceLocation l, bool v) : ASTNode(l), value(v) {}
  NodeKind getNodeType() const override { return NodeKind::BoolLiteral; }
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

// `sizeof(T)` (isType=true, target set, sub null) or `sizeof expr` /
// `sizeof(expr)` (isType=false, sub set, target null). GLSL has no sizeof, so
// the GLSL backend folds it to a byte-count literal (Sema fills `folded` from
// the type/operand type); the host C++ backend emits `sizeof(...)` verbatim and
// lets g++ compute it. Byte counts are coarse (struct = sum of fields, no
// alignment padding) — sufficient for kernel sizing, not ABI-exact.
class SizeOfExpr : public ASTNode {
public:
  Type *target = nullptr;
  NodePtr sub;
  bool isType = false;
  int64_t folded = 0; // filled by Sema; >0 means "already computed"
  SizeOfExpr(SourceLocation l, Type *t, NodePtr s, bool isTy)
      : ASTNode(l), target(t), sub(std::move(s)), isType(isTy) {}
  NodeKind getNodeType() const override { return NodeKind::SizeOfExpr; }
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
