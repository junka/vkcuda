//===- Parser.h - Recursive-descent parser --------------------------------===//
//
// Parses VC token streams into AST. Covers the CUDA-like subset sufficient
// for a vector-add kernel plus its host launch site.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_PARSER_H
#define VC_FRONTEND_PARSER_H

#include "vc/Frontend/AST.h"
#include "vc/Frontend/Lexer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>

namespace vc {

class Parser {
  Lexer &lexer;
  llvm::SourceMgr &srcMgr;
  Token curTok;
  TranslationUnit &tu;
  // User-named types (struct/typedef names) seen so far at top level, so
  // parseType() can recognize them as types in declarations. Populated as
  // struct/typedef decls are parsed. Sema does the full validation later.
  llvm::StringMap<Type *> typeNames;

  // Set by parseBaseType when it consumes a leading `const` qualifier, so the
  // enclosing declarator (parseVarDecl/parseParam) can record isConst on the
  // VarDecl/ParamDecl. parseBaseType returns a bare Type* and can't carry the
  // flag itself. Read-and-clear after building each declarator.
  bool lastBaseWasConst = false;

  // Likewise for C storage-class specifiers `static`/`extern` consumed as a
  // leading prefix in parseBaseType. Read-and-clear after building each decl.
  StorageClass lastBaseStorage = StorageClass::None;

  // Object-like macros from `#define NAME <literal>` (minimal preprocessor).
  // Only literal replacements (int/float) are supported; the replacement text
  // is stored verbatim and re-parsed into an IntegerLiteral/FloatLiteral at the
  // identifier use site in parsePrimary. Identifier-to-identifier macros are
  // NOT supported (would need multiple passes).
  llvm::StringMap<std::string> defines;

  // `const int NAME = <literal>;` values seen so far in the current scope, so
  // array dimensions spelled with named constants (`T arr[M * K]`) can be
  // constant-folded by evalConstInt. Sema does full symbol resolution later,
  // but arrayDims are stamped at parse time and the host backend emits them
  // verbatim — a non-constant dim falls back to 0, which for a sized local
  // array produces `T arr[0]` and a stack smash at runtime. Only integer
  // literals (and integer-literal init expressions) are recorded: this mirrors
  // the `#define` path's "literal only" limitation and keeps the parse
  // single-pass. Block scoping is approximated (an inner redefinition shadows
  // the outer entry; entries are not popped on scope exit, which is fine
  // because a later same-named const just overwrites the value).
  llvm::StringMap<int64_t> constInts;

  // Stable storage for the mangled name strings synthesized for scoped types
  // (`ns::Class` -> "ns_Class"). The StructDecl::name StringRef of a scoped
  // RecordType points into these strings, so they must outlive the parse.
  std::vector<std::unique_ptr<std::string>> scopedTypeStrs;

  // Stable storage for composed nested-namespace scope strings (e.g.
  // "outer::inner") built by stampNamespace. FunctionDecl::nsName StringRefs
  // point into these, so they must outlive the parse.
  std::vector<std::unique_ptr<std::string>> scopedNameStrs;

public:
  Parser(Lexer &lex, llvm::SourceMgr &sm, TranslationUnit &unit)
      : lexer(lex), srcMgr(sm), tu(unit) {
    curTok = lexer.lex();
  }

  /// Parse the whole translation unit.
  bool parseTranslationUnit();

private:
  // Token helpers
  void advance() { curTok = lexer.lex(); }
  bool consume(TokKind k);
  bool expect(TokKind k, const char *what);
  Diagnostic error(const Token &at, std::string msg);

  // Top level
  bool parseTopLevelDecl();
  bool parseFunctionOrKernel();
  bool parseStructDecl();
  bool parseClassDecl();
  bool parseTypedefDecl();
  bool parseEnumDecl();
  bool parseConstantDecl();
  bool parseNamespaceDecl();
  // Set FunctionDecl::nsName on free functions in a namespace body (recursing
  // into nested namespaces, composing the scope as "outer::inner").
  void stampNamespace(const std::vector<NodePtr> &decls, llvm::StringRef prefix);
  // Parse decls that may appear nested inside a namespace/class body; collects
  // them into `out` (a namespace's owned decls). Returns false on unrecoverable
  // error.
  bool parseNestedDecls(std::vector<NodePtr> &out);
  // File-scope `static`/`extern` variable declaration (no device qualifier).
  bool parseGlobalVarDecl();

  // Parse a `#define NAME <literal>` directive's text into `defines`.
  void tryParseDefine(llvm::StringRef line);
  // Build an IntegerLiteral/FloatLiteral node from a macro replacement text,
  // or return nullptr if the text isn't a recognized literal (so identifier
  // macros are left as DeclRefExpr rather than mis-emitted).
  NodePtr makeLiteralFromText(llvm::StringRef text, SourceLocation loc);

  // Types / decls
  Type *parseType();
  /// Parse only the base type specifier (no trailing pointer `*`s). Used by
  /// declarations so each declarator can carry its own pointer levels in a
  /// shared-type comma list (`int *a, *b;`).
  Type *parseBaseType();
  /// Does the current token start a type? (builtin keyword, a vector name
  /// like float4, or a known struct/typedef name.) Used to disambiguate
  /// declaration vs expression statements.
  bool startsType(const Token &t);
  /// If `name` is a CUDA-style vector name (float4, int3, ...), build a
  /// VectorType; otherwise return null.
  static Type *makeVectorType(llvm::StringRef name);
  bool parseDeviceAttrs(DeviceAttr &out);
  ParamDecl *parseParam();
  VarDecl *parseVarDecl(Type *ty);
  /// Best-effort compile-time integer evaluation for array dimensions and
  /// similar constant contexts. Returns true and sets `out` for a literal or
  /// a foldable arithmetic/bitwise expression over literals; false otherwise.
  // Fold a constant integer from an expression node. Handles literals,
  // unary/binary integer ops, and named `const int` constants (see constInts)
  // so array dimensions like `T arr[M * K]` fold at parse time.
  bool evalConstInt(const ASTNode *e, int64_t &out);

  // Statements
  NodePtr parseStatement();
  NodePtr parseCompoundStmt();
  NodePtr parseReturnStmt();
  NodePtr parseForStmt();
  NodePtr parseWhileStmt();
  NodePtr parseDoStmt();
  NodePtr parseSwitchStmt();
  NodePtr parseDeclOrExprStmt();

  // Expressions (precedence climbing)
  NodePtr parseExpression();
  NodePtr parseAssignment();
  NodePtr parseConditional();
  NodePtr parseLogicalOr();
  NodePtr parseBitwiseOr();
  NodePtr parseBitwiseXor();
  NodePtr parseBitwiseAnd();
  NodePtr parseLogicalAnd();
  NodePtr parseEquality();
  NodePtr parseRelational();
  NodePtr parseShift();
  NodePtr parseAdditive();
  NodePtr parseMultiplicative();
  NodePtr parseUnary();
  NodePtr parsePostfix();
  NodePtr parsePrimary();

  // CUDA launch:  ident <<< g, b >>> ( args )
  NodePtr tryParseLaunch(NodePtr &callee);

  SourceLocation toSourceLoc(const Token &t) const {
    SourceLocation s;
    s.loc = llvm::SMLoc::getFromPointer(t.text.data());
    s.line = t.line;
    s.col = t.col;
    return s;
  }
};

} // namespace vc

#endif // VC_FRONTEND_PARSER_H
