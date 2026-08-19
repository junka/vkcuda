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
      .Case("return", TokKind::kw_return)
      .Case("if", TokKind::kw_if)
      .Case("else", TokKind::kw_else)
      .Case("for", TokKind::kw_for)
      .Case("while", TokKind::kw_while)
      .Case("const", TokKind::kw_const)
      .Case("__global__", TokKind::kw_global)
      .Case("__device__", TokKind::kw_device)
      .Case("__host__", TokKind::kw_host)
      .Case("__shared__", TokKind::kw_shared)
      .Case("__restrict__", TokKind::kw_restrict)
      .Case("__syncthreads", TokKind::kw_syncthreads)
      .Case("dim3", TokKind::kw_dim3)
      .Case("wmma", TokKind::kw_wmma)
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
  while (pos < buffer.size()) {
    char c = buffer[pos];
    if (std::isdigit(static_cast<unsigned char>(c))) {
      nextChar();
    } else if (c == '.' && !isFloat) {
      isFloat = true;
      nextChar();
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

  TokKind kind = TokKind::unknown;
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
  case '+': kind = TokKind::plus; break;
  case '-': kind = TokKind::minus; break;
  case '*': kind = TokKind::star; break;
  case '/': kind = TokKind::slash; break;
  case '%': kind = TokKind::percent; break;
  case '.': kind = TokKind::dot; break;
  case '~': kind = TokKind::tilde; break;
  case '^': kind = TokKind::caret; break;
  case '!':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar();
      kind = TokKind::ne;
    } else {
      kind = TokKind::bang;
    }
    break;
  case '=':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar();
      kind = TokKind::eq;
    } else {
      kind = TokKind::assign;
    }
    break;
  case '<':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar();
      kind = TokKind::le;
    } else {
      kind = TokKind::lt;
    }
    break;
  case '>':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '=') {
      nextChar();
      kind = TokKind::ge;
    } else {
      kind = TokKind::gt;
    }
    break;
  case '&':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '&') {
      nextChar();
      kind = TokKind::amp_amp;
    } else {
      kind = TokKind::amp;
    }
    break;
  case '|':
    if (pos + 1 < buffer.size() && buffer[pos + 1] == '|') {
      nextChar();
      kind = TokKind::pipe_pipe;
    } else {
      kind = TokKind::pipe;
    }
    break;
  default:
    kind = TokKind::unknown;
    break;
  }

  StringRef text = buffer.substr(pos, 1);
  nextChar();
  return makeToken(kind, text, startLine, startCol);
}

Token Lexer::peek() {
  // Cheap lookahead: snapshot state, lex, restore.
  unsigned savedPos = pos;
  unsigned savedLine = curLine;
  unsigned savedCol = curCol;
  Token t = lex();
  pos = savedPos;
  curLine = savedLine;
  curCol = savedCol;
  return t;
}
