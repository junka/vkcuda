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

  // Punctuation / operators
  l_paren, r_paren,
  l_brace, r_brace,
  l_square, r_square,
  comma, semi, colon,
  star, plus, minus, slash, percent,
  assign,
  eq, ne, lt, gt, le, ge,
  amp_amp, pipe_pipe, bang,
  amp, pipe, caret, tilde,
  dot,
  // CUDA triple-chevron launch:  >>>
  launch_close, // >>>

  // Literals / identifiers
  identifier,
  int_literal,
  float_literal,

  // Keywords
  kw_void, kw_bool, kw_int, kw_uint, kw_long, kw_ulong, kw_float, kw_double,
  kw_return, kw_if, kw_else, kw_for, kw_while, kw_const,
  // CUDA attributes
  kw_global, kw_device, kw_host, kw_shared, kw_restrict,
  // CUDA builtins recognized as keywords
  kw_syncthreads,
  kw_dim3,
  kw_wmma, // namespace marker: wmma::fragment ... (loosely)
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

public:
  Lexer(llvm::SourceMgr &sm, int mainBufferID);

  /// Lex the next token.
  Token lex();
  /// Peek without consuming.
  Token peek();

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
  TokKind classifyKeyword(llvm::StringRef ident);
};

} // namespace vc

#endif // VC_FRONTEND_LEXER_H
