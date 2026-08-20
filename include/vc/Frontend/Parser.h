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

#include "llvm/Support/SourceMgr.h"

namespace vc {

class Parser {
  Lexer &lexer;
  llvm::SourceMgr &srcMgr;
  Token curTok;
  TranslationUnit &tu;

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

  // Types / decls
  Type *parseType();
  bool parseDeviceAttrs(DeviceAttr &out);
  ParamDecl *parseParam();
  VarDecl *parseVarDecl(Type *ty);

  // Statements
  NodePtr parseStatement();
  NodePtr parseCompoundStmt();
  NodePtr parseReturnStmt();
  NodePtr parseForStmt();
  NodePtr parseWhileStmt();
  NodePtr parseDeclOrExprStmt();

  // Expressions (precedence climbing)
  NodePtr parseExpression();
  NodePtr parseAssignment();
  NodePtr parseLogicalOr();
  NodePtr parseLogicalAnd();
  NodePtr parseEquality();
  NodePtr parseRelational();
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
