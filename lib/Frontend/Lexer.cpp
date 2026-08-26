//===- Lexer.cpp - Tokenizer implementation -------------------------------===//

#include "vc/Frontend/Lexer.h"

#include "llvm/ADT/StringSwitch.h"

using namespace vc;
using namespace llvm;

Lexer::Lexer(SourceMgr &sm, int mainBufferID)
    : srcMgr(sm), bufferID(mainBufferID) {
  buffer = sm.getMemoryBuffer(mainBufferID)->getBuffer();
}

char Lexer::nextChar() {
  if (pos >= buffer.size())
    return '\0';
  char c = buffer[pos++];
  if (c == '\n') {
    ++curLine;
    curCol = 1;
  } else {
    ++curCol;
  }
  return c;
}

void Lexer::skipWhitespaceAndComments() {
  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      nextChar();
    } else if (c == '/' && pos + 1 < buffer.size() && buffer[pos + 1] == '/') {
      // line comment
      while (pos < buffer.size() && buffer[pos] != '\n')
        nextChar();
    } else if (c == '/' && pos + 1 < buffer.size() &&
               buffer[pos + 1] == '*') {
      // block comment
      nextChar();
      nextChar();
      while (pos + 1 < buffer.size() &&
             !(buffer[pos] == '*' && buffer[pos + 1] == '/'))
        nextChar();
      if (pos + 1 < buffer.size()) {
        nextChar();
        nextChar();
      }
    } else if (c == '#' && curCol == 1) {
      // Preprocessor directive at line start: capture the whole line (from the
      // '#') verbatim as a hash_line token. Line continuations and multi-line
      // directives are not handled; one physical line == one token.
      unsigned start = pos;
      unsigned startLine = curLine, startCol = curCol;
      while (pos < buffer.size() && buffer[pos] != '\n')
        nextChar();
      hashLineText = buffer.substr(start, pos - start);
      hashLineLine = startLine;
      hashLineCol = startCol;
      haveHashLine = true;
      // Consume the trailing newline (if any) so the next token starts clean.
      if (pos < buffer.size() && buffer[pos] == '\n')
        nextChar();
      // Stop here so lex() emits this one hash_line token. Without the break a
      // run of consecutive `#` lines would overwrite hashLineText and only the
      // last would survive (dropping e.g. `#include <cstdio>` / `<cmath>`).
      break;
    } else {
      break;
    }
  }
}

TokKind Lexer::classifyKeyword(StringRef ident) {
  return StringSwitch<TokKind>(ident)
      .Case("void", TokKind::kw_void)
      .Case("bool", TokKind::kw_bool)
      .Case("int", TokKind::kw_int)
      .Case("unsigned", TokKind::kw_uint)
      .Case("uint", TokKind::kw_uint)
      .Case("long", TokKind::kw_long)
      .Case("float", TokKind::kw_float)
      .Case("double", TokKind::kw_double)
      // CUDA half-precision: both the bare `half` (modern CUDA) and the
      // double-underscore `__half` (legacy CUDA header type) spell the same
      // scalar type. `half2`/`half3`/`half4` are NOT keywords — they lex as
      // identifiers and makeVectorType turns them into f16vecN.
      .Case("half", TokKind::kw_half)
      .Case("__half", TokKind::kw_half)
      .Case("return", TokKind::kw_return)
      .Case("if", TokKind::kw_if)
      .Case("else", TokKind::kw_else)
      .Case("for", TokKind::kw_for)
      .Case("while", TokKind::kw_while)
      .Case("do", TokKind::kw_do)
      .Case("break", TokKind::kw_break)
      .Case("continue", TokKind::kw_continue)
      .Case("switch", TokKind::kw_switch)
      .Case("case", TokKind::kw_case)
      .Case("default", TokKind::kw_default)
      .Case("const", TokKind::kw_const)
      .Case("__global__", TokKind::kw_global)
      .Case("__device__", TokKind::kw_device)
      .Case("__host__", TokKind::kw_host)
      .Case("__shared__", TokKind::kw_shared)
      .Case("__restrict__", TokKind::kw_restrict)
      // CUDA __constant__ variables: device-resident read-only globals. VC
      // lowers these to `const` GLSL/C++ globals with compile-time initializers
      // only (no cudaMemcpyToSymbol runtime path). Useful for lookup tables and
      // fixed coefficients that live on the device.
      .Case("__constant__", TokKind::kw_constant)
      .Case("__syncthreads", TokKind::kw_syncthreads)
      // `dim3` is intentionally NOT a keyword: leaving it as a plain identifier
      // lets `dim3(N,N)` parse as a CallExpr, which the launch translator then
      // splits into X/Y components for vcLaunchKernel2D. A `struct dim3` def is
      // injected in the host preamble so the expression also type-checks.
      .Case("wmma", TokKind::kw_wmma)
      .Case("struct", TokKind::kw_struct)
      .Case("typedef", TokKind::kw_typedef)
      .Case("enum", TokKind::kw_enum)
      .Default(TokKind::identifier);
}

Token Lexer::lexIdentifier() {
  unsigned startCol = curCol;
  unsigned startLine = curLine;
  unsigned start = pos;
  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      nextChar();
    else
      break;
  }
  StringRef text = buffer.substr(start, pos - start);
  return makeToken(classifyKeyword(text), text, startLine, startCol);
}

Token Lexer::lexNumber() {
  unsigned startCol = curCol;
  unsigned startLine = curLine;
  unsigned start = pos;
  bool isFloat = false;

  // Hexadecimal integer: 0x / 0X followed by hex digits.
  if (pos + 1 < buffer.size() && buffer[pos] == '0' &&
      (buffer[pos + 1] == 'x' || buffer[pos + 1] == 'X')) {
    nextChar(); // '0'
    nextChar(); // 'x'/'X'
    while (pos < buffer.size() &&
           std::isxdigit(static_cast<unsigned char>(buffer[pos])))
      nextChar();
    // consume optional integer suffix
    while (pos < buffer.size() &&
           (buffer[pos] == 'u' || buffer[pos] == 'U' ||
            buffer[pos] == 'l' || buffer[pos] == 'L'))
      nextChar();
    StringRef text = buffer.substr(start, pos - start);
    return makeToken(TokKind::int_literal, text, startLine, startCol);
  }

  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (std::isdigit(static_cast<unsigned char>(c))) {
      nextChar();
    } else if (c == '.' && !isFloat) {
      isFloat = true;
      nextChar();
    } else if (c == 'e' || c == 'E') {
      // Floating-point exponent: e/E, optional +/-, then digits. Allowed
      // both after a fractional part (2.5e2) and as a pure-int exponent
      // (1e-3); in the latter case this turns the token into a float.
      unsigned savePos = pos;
      unsigned saveLine = curLine, saveCol = curCol;
      nextChar(); // 'e'
      if (pos < buffer.size() &&
          (buffer[pos] == '+' || buffer[pos] == '-'))
        nextChar();
      if (pos < buffer.size() &&
          std::isdigit(static_cast<unsigned char>(buffer[pos]))) {
        isFloat = true;
        while (pos < buffer.size() &&
               std::isdigit(static_cast<unsigned char>(buffer[pos])))
          nextChar();
      } else {
        // Not actually an exponent (e.g. "1e" with no digits) — roll back and
        // stop here so the number ends before the 'e'.
        pos = savePos; curLine = saveLine; curCol = saveCol;
        break;
      }
    } else if (c == 'f' || c == 'F') {
      isFloat = true;
      nextChar();
      break;
    } else if (c == 'u' || c == 'U' || c == 'l' || c == 'L') {
      // type suffix — consumed but ignored for now
      nextChar();
    } else {
      break;
    }
  }
  StringRef text = buffer.substr(start, pos - start);
  return makeToken(isFloat ? TokKind::float_literal : TokKind::int_literal,
                   text, startLine, startCol);
}

Token Lexer::lex() {
  skipWhitespaceAndComments();

  // If whitespace skipping captured a preprocessor line, emit it before the
  // next real token. (A '#' line is consumed entirely by the skip loop, so
  // this always precedes the token that follows it on the next line.)
  if (haveHashLine) {
    haveHashLine = false;
    return makeToken(TokKind::hash_line, hashLineText, hashLineLine,
                     hashLineCol);
  }

  unsigned startLine = curLine;
  unsigned startCol = curCol;
  if (pos >= buffer.size())
    return makeToken(TokKind::eof, StringRef(), startLine, startCol);

  char c = buffer[pos];

  // CUDA launch close: ">>>"
  if (c == '>' && pos + 2 < buffer.size() && buffer[pos + 1] == '>' &&
      buffer[pos + 2] == '>') {
    StringRef text = buffer.substr(pos, 3);
    nextChar();
    nextChar();
    nextChar();
    return makeToken(TokKind::launch_close, text, startLine, startCol);
  }

  if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    return lexIdentifier();
  if (std::isdigit(static_cast<unsigned char>(c)))
    return lexNumber();
  if (c == '\'')
    return lexCharLiteral();
  if (c == '"')
    return lexStringLiteral();

  TokKind kind = TokKind::unknown;
  unsigned start = pos;
  switch (c) {
  case '(': kind = TokKind::l_paren; break;
  case ')': kind = TokKind::r_paren; break;
  case '{': kind = TokKind::l_brace; break;
  case '}': kind = TokKind::r_brace; break;
  case '[': kind = TokKind::l_square; break;
  case ']': kind = TokKind::r_square; break;
  case ',': kind = TokKind::comma; break;
  case ';': kind = TokKind::semi; break;
  case ':': kind = TokKind::colon; break;
  case '?': kind = TokKind::question; break;
  case '~': kind = TokKind::tilde; break;
  case '.':
    if (pos + 1 < buffer.size() && std::isdigit(static_cast<unsigned char>(buffer[pos + 1]))) {
      // ".5" style float literal: lexer hit '.' but the next char is a digit.
      // Back up and lex as a number so the fractional float parses.
      return lexNumber();
    }
    kind = TokKind::dot;
    break;
  case '+':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '+') {
      nextChar(); kind = TokKind::plus_plus;
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::plus_equal;
    } else {
      kind = TokKind::plus;
    }
    break;
  case '-':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '-') {
      nextChar(); kind = TokKind::minus_minus;
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::minus_equal;
    } else {
      kind = TokKind::minus;
    }
    break;
  case '*':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::star_equal;
    } else {
      kind = TokKind::star;
    }
    break;
  case '/':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::slash_equal;
    } else {
      kind = TokKind::slash;
    }
    break;
  case '%':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::percent_equal;
    } else {
      kind = TokKind::percent;
    }
    break;
  case '^':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::caret_equal;
    } else {
      kind = TokKind::caret;
    }
    break;
  case '!':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::ne;
    } else {
      kind = TokKind::bang;
    }
    break;
  case '=':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::eq;
    } else {
      kind = TokKind::assign;
    }
    break;
  case '<':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '<') {
      nextChar();
      if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
        nextChar(); kind = TokKind::lessless_equal;
      } else {
        kind = TokKind::lessless;
      }
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::le;
    } else {
      kind = TokKind::lt;
    }
    break;
  case '>':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '>') {
      nextChar();
      if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
        nextChar(); kind = TokKind::greatergreater_equal;
      } else {
        kind = TokKind::greatergreater;
      }
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::ge;
    } else {
      kind = TokKind::gt;
    }
    break;
  case '&':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '&') {
      nextChar(); kind = TokKind::amp_amp;
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::amp_equal;
    } else {
      kind = TokKind::amp;
    }
    break;
  case '|':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '|') {
      nextChar(); kind = TokKind::pipe_pipe;
    } else if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar(); kind = TokKind::pipe_equal;
    } else {
      kind = TokKind::pipe;
    }
    break;
  default:
    kind = TokKind::unknown;
    break;
  }

  // Consume the first char of the operator (the switch above only advanced
  // past the extra chars of multi-char operators).
  nextChar();
  StringRef text = buffer.substr(start, pos - start);
  return makeToken(kind, text, startLine, startCol);
}

Token Lexer::lexCharLiteral() {
  unsigned startLine = curLine;
  unsigned startCol = curCol;
  unsigned start = pos;
  nextChar(); // opening '
  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (c == '\\') { // escape: skip next char
      nextChar();
      if (pos < buffer.size()) nextChar();
      continue;
    }
    if (c == '\'') { nextChar(); break; }
    if (c == '\n') break; // unterminated
    nextChar();
  }
  StringRef text = buffer.substr(start, pos - start);
  return makeToken(TokKind::char_literal, text, startLine, startCol);
}

Token Lexer::lexStringLiteral() {
  unsigned startLine = curLine;
  unsigned startCol = curCol;
  unsigned start = pos;
  nextChar(); // opening "
  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (c == '\\') { // escape: skip next char
      nextChar();
      if (pos < buffer.size()) nextChar();
      continue;
    }
    if (c == '"') { nextChar(); break; }
    if (c == '\n') break; // unterminated
    nextChar();
  }
  StringRef text = buffer.substr(start, pos - start);
  return makeToken(TokKind::string_literal, text, startLine, startCol);
}

Token Lexer::peek() {
  // Cheap lookahead: snapshot state, lex, restore.
  unsigned savedPos = pos;
  unsigned savedLine = curLine;
  unsigned savedCol = curCol;
  bool savedHaveHash = haveHashLine;
  llvm::StringRef savedHashText = hashLineText;
  unsigned savedHashLine = hashLineLine, savedHashCol = hashLineCol;
  Token t = lex();
  pos = savedPos;
  curLine = savedLine;
  curCol = savedCol;
  haveHashLine = savedHaveHash;
  hashLineText = savedHashText;
  hashLineLine = savedHashLine;
  hashLineCol = savedHashCol;
  return t;
}
