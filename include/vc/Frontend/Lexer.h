//===- Lexer.h - Tokenizer for the VC language ----------------------------===//
//
// Hand-written lexer for a CUDA-like source. Produces a token stream the
// recursive-descent parser consumes directly.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_LEXER_H
#define VC_FRONTEND_LEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

#include <vector>

namespace vc {

enum class TokKind {
  eof,
  unknown,

  // A preprocessor directive line (`#include ...`, `#define ...`) captured
  // verbatim from line start to end of line. The parser treats it as opaque
  // text to pass through to the host C++ backend.
  hash_line,

  // Punctuation / operators
  l_paren, r_paren,
  l_brace, r_brace,
  l_square, r_square,
  comma, semi, colon, coloncolon, // : ::
  question, // ?
  star, plus, minus, slash, percent,
  assign,
  // compound assignment
  plus_equal, minus_equal, star_equal, slash_equal, percent_equal,
  lessless_equal, greatergreater_equal,
  amp_equal, pipe_equal, caret_equal,
  // increment / decrement
  plus_plus, minus_minus,
  eq, ne, lt, gt, le, ge,
  lessless, greatergreater, // << >>
  amp_amp, pipe_pipe, bang,
  amp, pipe, caret, tilde,
  dot,
  arrow, // ->
  // CUDA triple-chevron launch:  >>>
  launch_close, // >>>

  // Literals / identifiers
  identifier,
  int_literal,
  float_literal,
  // Char/string literals are tokenized for friendliness; the parser currently
  // does not lower them to AST nodes (VC kernels don't use them).
  char_literal,
  string_literal,

  // Keywords
  kw_void, kw_bool, kw_int, kw_uint, kw_long, kw_ulong, kw_float, kw_double,
  kw_half,
  kw_return, kw_if, kw_else, kw_for, kw_while, kw_do,
  kw_break, kw_continue,
  kw_switch, kw_case, kw_default,
  kw_const,
  kw_true, kw_false,
  kw_sizeof,
  // C storage classes (recognized so `static`/`extern` don't lex as identifiers
  // and can be recorded on VarDecl/FunctionDecl for host passthrough).
  kw_static, kw_extern,
  // CUDA attributes
  kw_global, kw_device, kw_host, kw_shared, kw_restrict, kw_constant,
  // CUDA builtins recognized as keywords
  kw_syncthreads,
  kw_dim3,
  kw_wmma, // namespace marker: wmma::fragment ... (loosely)
  // Type-definition keywords
  kw_struct, kw_typedef, kw_enum,
  // C++ organizational constructs. `namespace`/`class` introduce new decls;
  // `public`/`private`/`protected` are lexed+parsed but access control is NOT
  // enforced (GLSL has none; host C++ gets a `public:` segment). `this` is
  // recognized so member-function bodies lex cleanly (device lowers to a
  // synthesized first param; host passes through). `using` is lexed only.
  kw_namespace, kw_class, kw_public, kw_private, kw_protected, kw_this, kw_using,
};

struct Token {
  TokKind kind = TokKind::eof;
  llvm::StringRef text;
  unsigned line = 0;
  unsigned col = 0;

  bool is(TokKind k) const { return kind == k; }
  bool isOneOf(TokKind a, TokKind b) const { return is(a) || is(b); }
  template <typename... Ts>
  bool isOneOf(TokKind a, TokKind b, Ts... rest) const {
    return is(a) || isOneOf(b, rest...);
  }
};

class Lexer {
  llvm::SourceMgr &srcMgr;
  llvm::StringRef buffer;
  unsigned pos = 0;
  unsigned curLine = 1;
  unsigned curCol = 1;
  int bufferID;
  // A preprocessor line (`#...` at line start) captured during whitespace
  // skipping. lex() emits it as a hash_line token before the next real token.
  bool haveHashLine = false;
  llvm::StringRef hashLineText;
  unsigned hashLineLine = 0;
  unsigned hashLineCol = 0;

public:
  Lexer(llvm::SourceMgr &sm, int mainBufferID);

  /// Lex the next token.
  Token lex();
  /// Peek without consuming.
  Token peek();

  /// Opaque save/restore of the lexer position, for speculative parsing
  /// (e.g. distinguishing a C-style cast `(T)x` from a grouping `(x)`).
  struct Pos {
    unsigned pos = 0;
    unsigned line = 1;
    unsigned col = 1;
  };
  Pos savePos() const { return {pos, curLine, curCol}; }
  void restorePos(const Pos &p) { pos = p.pos; curLine = p.line; curCol = p.col; }

private:
  char curChar() const { return pos < buffer.size() ? buffer[pos] : '\0'; }
  char nextChar();
  void skipWhitespaceAndComments();

  Token makeToken(TokKind k, llvm::StringRef text, unsigned line,
                  unsigned col) {
    return Token{k, text, line, col};
  }

  Token lexIdentifier();
  Token lexNumber();
  Token lexCharLiteral();
  Token lexStringLiteral();
  TokKind classifyKeyword(llvm::StringRef ident);
};

} // namespace vc

#endif // VC_FRONTEND_LEXER_H
