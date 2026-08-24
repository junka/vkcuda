//===- Parser.cpp - Recursive-descent parser implementation ---------------===//

#include "vc/Frontend/Parser.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;
using namespace llvm;

namespace {
// Decode a single escape sequence starting at the front of `s` (the character
// after the backslash). Returns the decoded code point. Used for both char and
// string literals.
int64_t decodeChar(StringRef s) {
  if (s.empty()) return 0;
  char c = s.front();
  switch (c) {
  case 'n': return '\n';
  case 't': return '\t';
  case 'r': return '\r';
  case '0': return '\0';
  case '\\': return '\\';
  case '\'': return '\'';
  case '"': return '"';
  case 'a': return '\a';
  case 'b': return '\b';
  case 'f': return '\f';
  case 'v': return '\v';
  default: return (unsigned char)c; // unknown escape -> literal char
  }
}

// Number of source characters consumed by the escape whose first post-backslash
// char is at the front of `s` (always 1 for the simple escapes we support).
unsigned escapeLen(StringRef s) { return 1; }
} // namespace

bool Parser::parseTranslationUnit() {
  while (!curTok.is(TokKind::eof)) {
    if (!parseTopLevelDecl())
      return false;
  }
  return tu.diagnostics.empty();
}

bool Parser::consume(TokKind k) {
  if (curTok.is(k)) {
    advance();
    return true;
  }
  return false;
}

bool Parser::expect(TokKind k, const char *what) {
  if (consume(k))
    return true;
  error(curTok, std::string("expected ") + what);
  return false;
}

Diagnostic Parser::error(const Token &at, std::string msg) {
  Diagnostic d{DiagnosticKind::Error, toSourceLoc(at), std::move(msg)};
  tu.diagnostics.push_back(d);
  // Associate with the SourceMgr so mlir/LLVM diagnostics can render it.
  srcMgr.PrintMessage(SMLoc::getFromPointer(at.text.data()),
                      SourceMgr::DK_Error, tu.diagnostics.back().message);
  return d;
}

//===----------------------------------------------------------------------===//
// Top level
//===----------------------------------------------------------------------===//

bool Parser::parseTopLevelDecl() {
  if (curTok.is(TokKind::hash_line)) {
    // Preprocessor directive (`#include ...`, `#define ...`): store verbatim
    // for the host C++ backend to emit at the top of the generated .cpp.
    tu.hostPpLines.push_back(std::string(curTok.text));
    advance();
    return true;
  }
  if (curTok.is(TokKind::kw_struct))
    return parseStructDecl();
  if (curTok.is(TokKind::kw_typedef))
    return parseTypedefDecl();
  return parseFunctionOrKernel();
}

bool Parser::parseDeviceAttrs(DeviceAttr &out) {
  out = DeviceAttr::None;
  bool sawAny = false;
  while (true) {
    switch (curTok.kind) {
    case TokKind::kw_global: out = DeviceAttr::Global; advance(); sawAny = true; break;
    case TokKind::kw_device: out = DeviceAttr::Device; advance(); sawAny = true; break;
    case TokKind::kw_host: out = DeviceAttr::Host; advance(); sawAny = true; break;
    default: return sawAny;
    }
  }
}

bool Parser::parseFunctionOrKernel() {
  DeviceAttr dattr;
  parseDeviceAttrs(dattr);

  Type *retTy = parseType();
  if (!retTy) {
    error(curTok, "expected return type");
    return false;
  }
  if (!curTok.is(TokKind::identifier)) {
    error(curTok, "expected function name");
    return false;
  }
  Token nameTok = curTok;
  advance();

  auto *fn = new FunctionDecl(toSourceLoc(nameTok));
  fn->returnType = retTy;
  fn->name = nameTok.text;
  fn->deviceAttr = dattr;

  if (!expect(TokKind::l_paren, "'('"))
    return false;
  if (!curTok.is(TokKind::r_paren)) {
    while (true) {
      if (auto *p = parseParam())
        fn->params.push_back(p);
      else
        return false;
      if (consume(TokKind::comma))
        continue;
      break;
    }
  }
  if (!expect(TokKind::r_paren, "')'"))
    return false;

  if (curTok.is(TokKind::semi)) {
    advance(); // prototype only
    tu.decls.emplace_back(fn);
    return true;
  }

  if (curTok.is(TokKind::l_brace)) {
    fn->body = parseCompoundStmt();
  }

  tu.decls.emplace_back(fn);

  // If this was a __global__, also record a KernelDecl wrapper.
  if (dattr == DeviceAttr::Global) {
    auto *k = new KernelDecl(fn->getLoc());
    k->func = fn;
    tu.decls.emplace_back(k);
  }
  return true;
}

// struct Name { Type field; Type field[N]; ... };
// The StructDecl and its RecordType are registered in `typeNames` so later
// declarations can use `Name` as a type. The StructDecl is added to tu.decls
// (it owns its FieldDecls).
bool Parser::parseStructDecl() {
  Token structTok = curTok;
  advance(); // 'struct'
  if (!curTok.is(TokKind::identifier)) {
    error(curTok, "expected struct name");
    return false;
  }
  Token nameTok = curTok;
  advance();

  auto *sd = new StructDecl(toSourceLoc(structTok), nameTok.text);
  auto *recTy = new RecordType(sd);
  // Register before parsing the body so self-/mutually-referential typedefs
  // and pointer fields can name the struct.
  typeNames[nameTok.text] = recTy;

  if (!expect(TokKind::l_brace, "'{'"))
    return false;
  while (!curTok.is(TokKind::r_brace) && !curTok.is(TokKind::eof)) {
    Type *fty = parseType();
    if (!fty) {
      error(curTok, "expected field type");
      return false;
    }
    if (!curTok.is(TokKind::identifier)) {
      error(curTok, "expected field name");
      return false;
    }
    Token fieldTok = curTok;
    advance();
    auto *fd = new FieldDecl(toSourceLoc(fieldTok), fty, fieldTok.text);
    // optional trailing array dims: field[N][M]
    while (consume(TokKind::l_square)) {
      if (!curTok.is(TokKind::int_literal)) {
        error(curTok, "expected array size");
        return false;
      }
      int64_t dim = 0;
      curTok.text.getAsInteger(10, dim);
      fd->arrayDims.push_back(dim);
      advance();
      if (!expect(TokKind::r_square, "']'"))
        return false;
    }
    if (!expect(TokKind::semi, "';' after field"))
      return false;
    sd->fields.push_back(fd);
  }
  if (!expect(TokKind::r_brace, "'}'"))
    return false;
  if (!expect(TokKind::semi, "';' after struct definition"))
    return false;

  tu.decls.emplace_back(sd);
  return true;
}

// typedef <underlying> <name>;
// Registers a TypedefType(name -> underlying) in `typeNames`.
bool Parser::parseTypedefDecl() {
  Token typedefTok = curTok;
  advance(); // 'typedef'
  Type *underlying = parseType();
  if (!underlying) {
    error(curTok, "expected type after 'typedef'");
    return false;
  }
  if (!curTok.is(TokKind::identifier)) {
    error(curTok, "expected typedef name");
    return false;
  }
  Token nameTok = curTok;
  advance();

  auto *td = new TypedefDecl(toSourceLoc(typedefTok), nameTok.text, underlying);
  typeNames[nameTok.text] = new TypedefType(td);

  if (!expect(TokKind::semi, "';' after typedef"))
    return false;
  tu.decls.emplace_back(td);
  return true;
}

ParamDecl *Parser::parseParam() {
  // optional __restrict__
  consume(TokKind::kw_restrict);
  Type *ty = parseType();
  if (!ty)
    return nullptr;
  // pointer params: a trailing '*' may have been folded into parseType.
  Token nameTok = curTok;
  if (!expect(TokKind::identifier, "parameter name"))
    return nullptr;
  // expect() already consumed the identifier; do not advance again.
  return new ParamDecl(toSourceLoc(nameTok), ty, nameTok.text);
}

// Best-effort compile-time integer evaluation: folds IntegerLiteral and
// arithmetic/bitwise expressions over literals. Used for array dimensions
// (`float a[32 * 32]`) so the host backend emits a concrete size instead of 0.
bool Parser::evalConstInt(const ASTNode *e, int64_t &out) {
  if (!e) return false;
  switch (e->getNodeType()) {
  case ASTNode::NodeKind::IntegerLiteral:
    out = static_cast<const IntegerLiteral *>(e)->value;
    return true;
  case ASTNode::NodeKind::UnaryExpr: {
    auto *u = static_cast<const UnaryExpr *>(e);
    int64_t v;
    if (!evalConstInt(u->operand.get(), v)) return false;
    switch (u->op) {
    case UnaryOp::Neg: out = -v; return true;
    case UnaryOp::Not: out = ~v; return true;
    case UnaryOp::LNot: out = !v; return true;
    default: return false; // Deref/AddrOf/Inc/Dec are not constant.
    }
  }
  case ASTNode::NodeKind::BinaryExpr: {
    auto *b = static_cast<const BinaryExpr *>(e);
    int64_t l, r;
    if (!evalConstInt(b->lhs.get(), l) || !evalConstInt(b->rhs.get(), r))
      return false;
    switch (b->op) {
    case BinaryOp::Add: out = l + r; return true;
    case BinaryOp::Sub: out = l - r; return true;
    case BinaryOp::Mul: out = l * r; return true;
    case BinaryOp::Div: out = l / r; return true;
    case BinaryOp::Mod: out = l % r; return true;
    case BinaryOp::Shl: out = l << r; return true;
    case BinaryOp::Shr: out = l >> r; return true;
    case BinaryOp::And: out = l & r; return true;
    case BinaryOp::Or: out = l | r; return true;
    case BinaryOp::Xor: out = l ^ r; return true;
    case BinaryOp::LAnd: out = l && r; return true;
    case BinaryOp::LOr: out = l || r; return true;
    default: return false; // comparisons/assign not needed for sizes.
    }
  }
  default:
    return false;
  }
}

VarDecl *Parser::parseVarDecl(Type *ty) {
  // Per-declarator leading pointer stars: in `int *a, *b;` each declarator
  // may carry its own `*`s applied to the shared base type.
  Type *declTy = ty;
  while (consume(TokKind::star))
    declTy = new PointerType(declTy);
  Token nameTok = curTok;
  if (!expect(TokKind::identifier, "variable name"))
    return nullptr;
  // expect() already consumed the identifier.
  auto *v = new VarDecl(toSourceLoc(nameTok), declTy, nameTok.text);
  // Trailing array dimensions: "name[16][8]" or "name[32 * 32]". Each is
  // constant-folded to a concrete size; a non-constant dim falls back to 0.
  while (curTok.is(TokKind::l_square)) {
    advance();
    NodePtr dim = parseExpression();
    int64_t sz = 0;
    if (!evalConstInt(dim.get(), sz))
      sz = 0; // unsized / non-constant
    v->arrayDims.push_back(sz);
    expect(TokKind::r_square, "']'");
  }
  if (curTok.is(TokKind::assign)) {
    advance();
    // Brace-enclosed initializer list: { a, b, c }.
    if (curTok.is(TokKind::l_brace)) {
      Token lb = curTok;
      advance();
      auto *il = new InitListExpr(toSourceLoc(lb));
      if (!curTok.is(TokKind::r_brace)) {
        while (true) {
          // Elements may themselves be init lists (nested) or expressions.
          if (curTok.is(TokKind::l_brace)) {
            // Recurse by parsing an expression whose primary handles {...}?
            // Simpler: parse nested via a temporary by re-entering parseExpr
            // won't see '{'. Parse nested init list inline.
            // (Kept minimal: one level of nesting for vec-of-vec / array.)
            Token lb2 = curTok;
            advance();
            auto *il2 = new InitListExpr(toSourceLoc(lb2));
            if (!curTok.is(TokKind::r_brace)) {
              while (true) {
                auto e = parseExpression();
                if (e) il2->elements.push_back(std::move(e));
                if (consume(TokKind::comma)) continue;
                break;
              }
            }
            expect(TokKind::r_brace, "'}'");
            il->elements.push_back(NodePtr(il2));
          } else {
            auto e = parseExpression();
            if (e) il->elements.push_back(std::move(e));
          }
          if (consume(TokKind::comma)) continue;
          break;
        }
      }
      expect(TokKind::r_brace, "'}'");
      v->init = NodePtr(il);
    } else {
      v->init = parseExpression();
    }
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

Type *Parser::makeVectorType(StringRef name) {
  // CUDA/HLSL-style vector names: <base><count>, count in 2..4.
  // Recognized bases: float, int, uint, double, bool.
  struct Base { const char *prefix; BuiltinTypeKind kind; };
  static constexpr Base bases[] = {
      {"float", BuiltinTypeKind::Float32},
      {"int", BuiltinTypeKind::Int32},
      {"uint", BuiltinTypeKind::UInt32},
      {"double", BuiltinTypeKind::Float64},
      {"bool", BuiltinTypeKind::Bool},
  };
  for (const Base &b : bases) {
    StringRef p = b.prefix;
    if (name.size() == p.size() + 1 && name.starts_with(p)) {
      char d = name.back();
      if (d >= '2' && d <= '4') {
        unsigned count = d - '0';
        return new VectorType(new BuiltinType(b.kind), count);
      }
    }
  }
  return nullptr;
}

Type *Parser::parseBaseType() {
  // Discard leading qualifiers (`const`) — VC's type system doesn't track
  // cv-qualifiers; the GLSL/host backends don't need them. Multiple `const`
  // and `__restrict__` tokens are tolerated.
  while (curTok.is(TokKind::kw_const) || curTok.is(TokKind::kw_restrict))
    advance();
  Type *base = nullptr;
  switch (curTok.kind) {
  case TokKind::kw_void: base = new BuiltinType(BuiltinTypeKind::Void); break;
  case TokKind::kw_bool: base = new BuiltinType(BuiltinTypeKind::Bool); break;
  case TokKind::kw_int: base = new BuiltinType(BuiltinTypeKind::Int32); break;
  case TokKind::kw_uint: base = new BuiltinType(BuiltinTypeKind::UInt32); break;
  case TokKind::kw_long: base = new BuiltinType(BuiltinTypeKind::Int64); break;
  case TokKind::kw_float: base = new BuiltinType(BuiltinTypeKind::Float32); break;
  case TokKind::kw_double: base = new BuiltinType(BuiltinTypeKind::Float64); break;
  case TokKind::identifier:
    // A user-named type (struct/typedef) or a vector name like float4.
    if (auto *vec = makeVectorType(curTok.text))
      base = vec;
    else {
      auto it = typeNames.find(curTok.text);
      if (it != typeNames.end())
        base = it->second;
      else {
        // Unknown named type (e.g. a runtime typedef like VCStreamHandle or
        // VCKernelHandle from VCRuntime.h, or an opaque host type). Treat it
        // as a forward-declared record so the host backend emits the name
        // verbatim; the real definition comes from the #included header.
        auto *sd = new StructDecl(toSourceLoc(curTok), curTok.text);
        base = new RecordType(sd);
        typeNames[curTok.text] = base;
      }
    }
    break;
  default: return nullptr;
  }
  advance();
  return base;
}

Type *Parser::parseType() {
  Type *base = parseBaseType();
  if (!base) return nullptr;
  // pointer levels: '*' '*'
  Type *ty = base;
  while (consume(TokKind::star))
    ty = new PointerType(ty);
  // reference levels: '&' (host-only, e.g. `const Point &p` in a helper).
  while (consume(TokKind::amp))
    ty = new ReferenceType(ty);
  return ty;
}

bool Parser::startsType(const Token &t) {
  switch (t.kind) {
  case TokKind::kw_const:
    // `const` qualifies a following type; peek through it.
    return true;
  case TokKind::kw_void:
  case TokKind::kw_bool:
  case TokKind::kw_int:
  case TokKind::kw_uint:
  case TokKind::kw_long:
  case TokKind::kw_float:
  case TokKind::kw_double:
    return true;
  case TokKind::identifier:
    // Vector names (float4, ...) and known struct/typedef names start types.
    if (makeVectorType(t.text)) return true;
    if (typeNames.count(t.text) > 0) return true;
    // An unknown capitalized identifier likely names a runtime/opaque type
    // (e.g. VCStreamHandle from VCRuntime.h). Treat it as a type only when the
    // next token looks like a declarator name or `*`, so plain expression
    // statements like `s = 5;` aren't misread as declarations. We deliberately
    // do NOT treat a following `&` as a declarator indicator: `name & expr` is
    // overwhelmingly a bitwise-and expression (e.g. `(k & 1)`), not a reference
    // declaration. Reference parameters only occur in host helper signatures
    // like `const Point &p`, where the type name is already known (struct/
    // typedef) or capitalized — handled by the branches above.
    {
      const Token &nx = lexer.peek();
      if (nx.is(TokKind::identifier) && !makeVectorType(nx.text))
        return true;
      if (nx.is(TokKind::star))
        return true;
    }
    return false;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

NodePtr Parser::parseStatement() {
  switch (curTok.kind) {
  case TokKind::l_brace: return parseCompoundStmt();
  case TokKind::kw_return: return parseReturnStmt();
  case TokKind::kw_for: return parseForStmt();
  case TokKind::kw_while: return parseWhileStmt();
  case TokKind::kw_do: return parseDoStmt();
  case TokKind::kw_switch: return parseSwitchStmt();
  case TokKind::kw_break: {
    Token t = curTok;
    advance();
    expect(TokKind::semi, "';'");
    return NodePtr(new BreakStmt(toSourceLoc(t)));
  }
  case TokKind::kw_continue: {
    Token t = curTok;
    advance();
    expect(TokKind::semi, "';'");
    return NodePtr(new ContinueStmt(toSourceLoc(t)));
  }
  case TokKind::kw_if: {
    Token t = curTok;
    advance();
    expect(TokKind::l_paren, "'('");
    auto cond = parseExpression();
    expect(TokKind::r_paren, "')'");
    auto *iff = new IfStmt(toSourceLoc(t));
    iff->cond = std::move(cond);
    iff->thenStmt = parseStatement();
    if (consume(TokKind::kw_else))
      iff->elseStmt = parseStatement();
    return NodePtr(iff);
  }
  case TokKind::kw_shared: {
    advance();
    Type *ty = parseType();
    auto *v = parseVarDecl(ty);
    if (v) v->isShared = true;
    expect(TokKind::semi, "';'");
    SourceLocation loc = v ? v->getLoc() : toSourceLoc(curTok);
    return NodePtr(new DeclStmt(loc, v));
  }
  default: return parseDeclOrExprStmt();
  }
}

NodePtr Parser::parseCompoundStmt() {
  Token lb = curTok;
  if (!expect(TokKind::l_brace, "'{'"))
    return nullptr;
  auto *cs = new CompoundStmt(toSourceLoc(lb));
  while (!curTok.is(TokKind::r_brace) && !curTok.is(TokKind::eof)) {
    if (auto s = parseStatement())
      cs->statements.push_back(std::move(s));
    else
      return nullptr;
  }
  expect(TokKind::r_brace, "'}'");
  return NodePtr(cs);
}

NodePtr Parser::parseReturnStmt() {
  Token r = curTok;
  advance();
  auto *rs = new ReturnStmt(toSourceLoc(r));
  if (!curTok.is(TokKind::semi))
    rs->value = parseExpression();
  expect(TokKind::semi, "';'");
  return NodePtr(rs);
}

// for (init; cond; step) body
// init: a declaration or expression statement (consumes its ';')
// cond: optional expression
// step: optional expression
NodePtr Parser::parseForStmt() {
  Token f = curTok;
  advance();
  auto *fs = new ForStmt(toSourceLoc(f));
  expect(TokKind::l_paren, "'('");

  // init: empty, a declaration, or an expression.
  if (!curTok.is(TokKind::semi)) {
    fs->init = parseDeclOrExprStmt(); // consumes the ';'
  } else {
    advance(); // consume ';'
  }

  // cond: optional expression followed by ';'.
  if (!curTok.is(TokKind::semi))
    fs->cond = parseExpression();
  expect(TokKind::semi, "';'");

  // step: optional expression followed by ')'.
  if (!curTok.is(TokKind::r_paren))
    fs->step = parseExpression();
  expect(TokKind::r_paren, "')'");

  fs->body = parseStatement();
  return NodePtr(fs);
}

NodePtr Parser::parseWhileStmt() {
  Token w = curTok;
  advance();
  expect(TokKind::l_paren, "'('");
  auto *ws = new WhileStmt(toSourceLoc(w));
  ws->cond = parseExpression();
  expect(TokKind::r_paren, "')'");
  ws->body = parseStatement();
  return NodePtr(ws);
}

// do body while (cond);
NodePtr Parser::parseDoStmt() {
  Token d = curTok;
  advance();
  auto *ds = new DoStmt(toSourceLoc(d));
  ds->body = parseStatement();
  if (!expect(TokKind::kw_while, "'while'"))
    return NodePtr(ds);
  expect(TokKind::l_paren, "'('");
  ds->cond = parseExpression();
  expect(TokKind::r_paren, "')'");
  expect(TokKind::semi, "';'");
  return NodePtr(ds);
}

NodePtr Parser::parseSwitchStmt() {
  Token s = curTok;
  advance(); // 'switch'
  expect(TokKind::l_paren, "'('");
  auto *sw = new SwitchStmt(toSourceLoc(s));
  sw->cond = parseExpression();
  expect(TokKind::r_paren, "')'");
  expect(TokKind::l_brace, "'{'");
  auto *body = new CompoundStmt(toSourceLoc(s));
  while (!curTok.is(TokKind::r_brace) && !curTok.is(TokKind::eof)) {
    if (curTok.is(TokKind::kw_case) || curTok.is(TokKind::kw_default)) {
      Token c = curTok;
      advance();
      auto *cs = new CaseStmt(toSourceLoc(c));
      if (c.kind == TokKind::kw_case) {
        cs->value = parseExpression();
        expect(TokKind::colon, "':'");
      } else {
        expect(TokKind::colon, "':'");
      }
      cs->sub = parseStatement();
      body->statements.push_back(NodePtr(cs));
    } else {
      // A statement not under an explicit case label (rare in C, allowed).
      auto st = parseStatement();
      if (st) body->statements.push_back(std::move(st));
    }
  }
  expect(TokKind::r_brace, "'}'");
  sw->body = NodePtr(body);
  return NodePtr(sw);
}

NodePtr Parser::parseDeclOrExprStmt() {
  // If the current token starts a type, parse a declaration.
  Token save = curTok;
  if (startsType(curTok)) {
    // Parse the base type once; each declarator carries its own pointer `*`s
    // (so `int *a, b;` gives a=int*, b=int; `void *x, *y;` gives both void*).
    Type *baseTy = parseBaseType();
    VarDecl *first = parseVarDecl(baseTy);
    // A DeclStmt may hold several declarators sharing one base type
    // (`float a[4], b, c[8];`). They share the enclosing scope (no extra {}),
    // so build one DeclStmt whose `decls` lists them all.
    auto *ds = new DeclStmt(toSourceLoc(save), first);
    while (consume(TokKind::comma)) {
      if (VarDecl *v = parseVarDecl(baseTy))
        ds->decls.push_back(v);
    }
    expect(TokKind::semi, "';'");
    return NodePtr(ds);
  }
  auto *es = new ExprStmt(toSourceLoc(save));
  es->expr = parseExpression();
  // Error recovery: if we made no progress, skip a token to avoid an
  // infinite loop on an unrecognized statement.
  if (!es->expr && curTok.kind == save.kind) {
    error(curTok, "unexpected token in statement");
    if (!curTok.is(TokKind::eof)) advance();
  }
  expect(TokKind::semi, "';'");
  return NodePtr(es);
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

NodePtr Parser::parseExpression() { return parseAssignment(); }

NodePtr Parser::parseAssignment() {
  auto lhs = parseConditional();
  if (!lhs) return nullptr;
  if (curTok.is(TokKind::assign)) {
    Token op = curTok;
    advance();
    auto rhs = parseAssignment();
    return NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::Assign,
                                  std::move(lhs), std::move(rhs)));
  }
  // Compound assignment: `a op= b`  ==>  `a = a op b`. We clone the lhs so the
  // base op and the assign can each own a copy. (Postfix value-semantics of
  // `a++` as a sub-expression are not modeled; see plan TODO.)
  static const struct { TokKind tok; BinaryOp base; } compound[] = {
    {TokKind::plus_equal, BinaryOp::Add},
    {TokKind::minus_equal, BinaryOp::Sub},
    {TokKind::star_equal, BinaryOp::Mul},
    {TokKind::slash_equal, BinaryOp::Div},
    {TokKind::percent_equal, BinaryOp::Mod},
    {TokKind::lessless_equal, BinaryOp::Shl},
    {TokKind::greatergreater_equal, BinaryOp::Shr},
    {TokKind::amp_equal, BinaryOp::And},
    {TokKind::pipe_equal, BinaryOp::Or},
    {TokKind::caret_equal, BinaryOp::Xor},
  };
  for (auto &c : compound) {
    if (curTok.is(c.tok)) {
      Token op = curTok;
      advance();
      auto rhs = parseAssignment();
      auto lhsCopy = cloneExpr(lhs.get());
      auto base = NodePtr(new BinaryExpr(toSourceLoc(op), c.base,
                                         std::move(lhsCopy), std::move(rhs)));
      return NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::Assign,
                                    std::move(lhs), std::move(base)));
    }
  }
  return lhs;
}

// C precedence: assignment  <  conditional (?:)  <  logical-or.
// `cond ? a : b` is right-associative; the else-branch may itself be a
// conditional, and the then-branch is a full assignment expression.
NodePtr Parser::parseConditional() {
  auto cond = parseLogicalOr();
  if (!cond) return nullptr;
  if (curTok.is(TokKind::question)) {
    Token q = curTok;
    advance();
    auto thenE = parseAssignment();
    expect(TokKind::colon, "':' in ?: expression");
    auto elseE = parseConditional();
    return NodePtr(new ConditionalExpr(toSourceLoc(q), std::move(cond),
                                       std::move(thenE), std::move(elseE)));
  }
  return cond;
}

NodePtr Parser::parseLogicalOr() {
  auto lhs = parseBitwiseOr();
  while (curTok.is(TokKind::pipe_pipe)) {
    Token op = curTok;
    advance();
    auto rhs = parseBitwiseOr();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::LOr,
                                 std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

// C precedence: bitwise-OR  <  bitwise-XOR  <  bitwise-AND  <  equality.
NodePtr Parser::parseBitwiseOr() {
  auto lhs = parseBitwiseXor();
  while (curTok.is(TokKind::pipe)) {
    Token op = curTok;
    advance();
    auto rhs = parseBitwiseXor();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::Or,
                                 std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseBitwiseXor() {
  auto lhs = parseBitwiseAnd();
  while (curTok.is(TokKind::caret)) {
    Token op = curTok;
    advance();
    auto rhs = parseBitwiseAnd();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::Xor,
                                 std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseBitwiseAnd() {
  auto lhs = parseLogicalAnd();
  while (curTok.is(TokKind::amp)) {
    Token op = curTok;
    advance();
    auto rhs = parseLogicalAnd();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::And,
                                 std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseLogicalAnd() {
  auto lhs = parseEquality();
  while (curTok.is(TokKind::amp_amp)) {
    Token op = curTok;
    advance();
    auto rhs = parseEquality();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::LAnd,
                                 std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseEquality() {
  auto lhs = parseRelational();
  while (curTok.isOneOf(TokKind::eq, TokKind::ne)) {
    Token op = curTok;
    BinaryOp bop = op.is(TokKind::eq) ? BinaryOp::Eq : BinaryOp::NEq;
    advance();
    auto rhs = parseRelational();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseRelational() {
  auto lhs = parseShift();
  while (curTok.isOneOf(TokKind::lt, TokKind::gt, TokKind::le, TokKind::ge)) {
    Token op = curTok;
    BinaryOp bop;
    switch (curTok.kind) {
    case TokKind::lt: bop = BinaryOp::Lt; break;
    case TokKind::gt: bop = BinaryOp::Gt; break;
    case TokKind::le: bop = BinaryOp::Le; break;
    default: bop = BinaryOp::Ge; break;
    }
    advance();
    auto rhs = parseShift();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

// C precedence: relational  <  shift  <  additive.
NodePtr Parser::parseShift() {
  auto lhs = parseAdditive();
  while (curTok.isOneOf(TokKind::lessless, TokKind::greatergreater)) {
    Token op = curTok;
    BinaryOp bop = op.is(TokKind::lessless) ? BinaryOp::Shl : BinaryOp::Shr;
    advance();
    auto rhs = parseAdditive();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseAdditive() {
  auto lhs = parseMultiplicative();
  while (curTok.isOneOf(TokKind::plus, TokKind::minus)) {
    Token op = curTok;
    BinaryOp bop = op.is(TokKind::plus) ? BinaryOp::Add : BinaryOp::Sub;
    advance();
    auto rhs = parseMultiplicative();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseMultiplicative() {
  auto lhs = parseUnary();
  while (curTok.isOneOf(TokKind::star, TokKind::slash, TokKind::percent)) {
    Token op = curTok;
    BinaryOp bop;
    switch (curTok.kind) {
    case TokKind::star: bop = BinaryOp::Mul; break;
    case TokKind::slash: bop = BinaryOp::Div; break;
    default: bop = BinaryOp::Mod; break; // percent -> real modulo
    }
    advance();
    auto rhs = parseUnary();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseUnary() {
  // C-style cast: ( type ) expr. Disambiguate from a parenthesized expression
  // by speculatively parsing a type between '(' and ')'; if the token after
  // ')' starts an expression, it's a cast, otherwise roll back and treat the
  // '(' as grouping (handled by parsePostfix -> parsePrimary).
  if (curTok.is(TokKind::l_paren) && startsType(lexer.peek())) {
    Token lp = curTok;
    Token savedTok = curTok;
    Lexer::Pos savedPos = lexer.savePos(); // pos is just past curTok '('
    // The speculative parseType() below may register an unknown identifier
    // (e.g. a variable name like `k` in `(k & 1)`) as a synthetic RecordType in
    // `typeNames`. That registration is NOT rolled back by lexer.restorePos, so
    // it would leak and make every later occurrence of that name parse as a
    // type. Snapshot whether the peeked base name is already known so we can
    // undo any speculative insertion on rollback.
    std::string specName;
    bool specWasKnown = false;
    {
      const Token &pk = lexer.peek();
      if (pk.is(TokKind::identifier)) {
        specName = pk.text.str();
        specWasKnown = typeNames.count(specName) > 0;
      }
    }
    advance(); // '('
    Type *ty = parseType();
    if (ty && curTok.is(TokKind::r_paren)) {
      // Look one past ')' to confirm it starts an expression (cast operand).
      Token after = lexer.peek();
      if (after.isOneOf(TokKind::identifier, TokKind::int_literal,
                        TokKind::float_literal, TokKind::l_paren,
                        TokKind::minus, TokKind::bang, TokKind::tilde,
                        TokKind::star, TokKind::amp, TokKind::plus_plus,
                        TokKind::minus_minus, TokKind::char_literal)) {
        advance(); // ')'
        NodePtr sub = parseUnary();
        return NodePtr(new CStyleCastExpr(toSourceLoc(lp), ty, std::move(sub)));
      }
    }
    // Not a cast: roll back curTok, the lexer position, AND any speculative
    // typeNames insertion so the '(' is reprocessed cleanly as grouping.
    if (!specName.empty() && !specWasKnown)
      typeNames.erase(specName);
    curTok = savedTok;
    lexer.restorePos(savedPos);
  }
  // Prefix ++ / -- : model as a real unary node. GLSL natively implements
  // `++a` (yields the new value), so pass it through instead of rewriting to
  // `a = a + 1` (which also modeled the value correctly but loses the form).
  if (curTok.isOneOf(TokKind::plus_plus, TokKind::minus_minus)) {
    Token op = curTok;
    UnaryOp uop = op.is(TokKind::plus_plus) ? UnaryOp::PreInc : UnaryOp::PreDec;
    advance();
    auto operand = parseUnary();
    return NodePtr(new UnaryExpr(toSourceLoc(op), uop, std::move(operand)));
  }
  if (curTok.isOneOf(TokKind::minus, TokKind::bang, TokKind::amp,
                     TokKind::star, TokKind::tilde)) {
    Token op = curTok;
    UnaryOp uop;
    switch (curTok.kind) {
    case TokKind::minus: uop = UnaryOp::Neg; break;
    case TokKind::bang: uop = UnaryOp::LNot; break;
    case TokKind::amp: uop = UnaryOp::AddrOf; break;
    case TokKind::tilde: uop = UnaryOp::Not; break;
    default: uop = UnaryOp::Deref; break;
    }
    advance();
    auto operand = parseUnary();
    return NodePtr(new UnaryExpr(toSourceLoc(op), uop, std::move(operand)));
  }
  return parsePostfix();
}

NodePtr Parser::parsePostfix() {
  auto base = parsePrimary();
  while (base) {
    // CUDA launch `kernel<<<grid,block>>>(args)` must be detected before any
    // other postfix handling: the `<<<` opens with a `lessless` token.
    if (curTok.is(TokKind::lessless) &&
        lexer.peek().is(TokKind::lt)) {
      if (auto launch = tryParseLaunch(base))
        return launch;
    }
    if (curTok.is(TokKind::l_square)) {
      Token lb = curTok;
      advance();
      auto idx = parseExpression();
      expect(TokKind::r_square, "']'");
      base = NodePtr(new IndexExpr(toSourceLoc(lb), std::move(base), std::move(idx)));
    } else if (curTok.is(TokKind::dot)) {
      advance();
      Token m = curTok;
      if (!expect(TokKind::identifier, "member name"))
        return nullptr;
      // expect() already consumed the member identifier.
      base = NodePtr(new MemberAccessExpr(toSourceLoc(m), std::move(base), m.text));
    } else if (curTok.is(TokKind::colon) && lexer.peek().is(TokKind::colon)) {
      // Scope/resolution operator `::` (lexed as two `colon` tokens), as in
      // `VCMemcpyKind::HostToDevice`. Lowered to a MemberAccessExpr so the
      // host backend can emit it verbatim; the device backend rarely sees it.
      advance();
      advance();
      Token m = curTok;
      if (!expect(TokKind::identifier, "scoped name"))
        return nullptr;
      auto *ma = new MemberAccessExpr(toSourceLoc(m), std::move(base), m.text);
      ma->isScope = true; // source used `::`, not `.`
      base = NodePtr(ma);
    } else if (curTok.is(TokKind::l_paren)) {
      // call: callee ( args )
      if (auto launch = tryParseLaunch(base))
        return launch; // launch consumes its own args
      Token lp = curTok;
      advance();
      auto *call = new CallExpr(toSourceLoc(lp), std::move(base));
      if (!curTok.is(TokKind::r_paren)) {
        while (true) {
          auto a = parseExpression();
          if (a) call->args.push_back(std::move(a));
          if (consume(TokKind::comma)) continue;
          break;
        }
      }
      expect(TokKind::r_paren, "')'");
      base = NodePtr(call);
    } else if (curTok.isOneOf(TokKind::plus_plus, TokKind::minus_minus)) {
      // Postfix ++ / -- : model as a real unary node. GLSL natively yields the
      // OLD value for `a++` (the value before increment), which the previous
      // parser-rewrite to `a = a + 1` got wrong as a sub-expression. Passing
      // the node through lets GLSL implement the correct value semantics.
      Token op = curTok;
      UnaryOp uop = op.is(TokKind::plus_plus) ? UnaryOp::PostInc : UnaryOp::PostDec;
      advance();
      base = NodePtr(new UnaryExpr(toSourceLoc(op), uop, std::move(base)));
    } else {
      break;
    }
  }
  return base;
}

NodePtr Parser::tryParseLaunch(NodePtr &callee) {
  // CUDA launch uses '<<<' ... '>>>'. The lexer tokenizes '<<' as a single
  // lessless token and the trailing '<' as lt, so '<<<' = lessless + lt.
  if (!(curTok.is(TokKind::lessless) &&
        lexer.peek().is(TokKind::lt)))
    return nullptr;
  Token openLt = curTok;
  advance(); // '<<'
  advance(); // '<'
  auto grid = parseExpression();
  expect(TokKind::comma, "','");
  auto block = parseExpression();
  // Optional extra launch arguments: <<<g, b, sharedMem, stream>>>. The 3rd
  // (dynamic shared memory) is unused by the VC runtime and dropped; the 4th
  // (stream handle) is captured so the host backend can emit vcLaunchKernelS.
  NodePtr stream;
  if (consume(TokKind::comma)) {
    parseExpression(); // shared-mem size — ignored
    if (consume(TokKind::comma))
      stream = parseExpression(); // stream handle
  }
  expect(TokKind::launch_close, "'>>>'");
  expect(TokKind::l_paren, "'(' after >>>");

  // Split a `dim3(a, b)` call into its X and Y components so the host backend
  // can emit vcLaunchKernel2D. A plain scalar grid/block stays 1D (Y = null).
  auto splitDim3 = [this](NodePtr n) -> std::pair<NodePtr, NodePtr> {
    if (n && n->getNodeType() == ASTNode::NodeKind::CallExpr) {
      auto *c = static_cast<CallExpr *>(n.get());
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr &&
          static_cast<DeclRefExpr *>(c->callee.get())->name == "dim3" &&
          c->args.size() >= 2)
        return {std::move(c->args[0]), std::move(c->args[1])};
    }
    return {std::move(n), nullptr};
  };
  auto [gx, gy] = splitDim3(std::move(grid));
  auto [bx, by] = splitDim3(std::move(block));

  auto *launch = new LaunchExpr(toSourceLoc(openLt), std::move(callee),
                                std::move(gx), std::move(bx),
                                std::move(gy), std::move(by), std::move(stream));
  if (!curTok.is(TokKind::r_paren)) {
    while (true) {
      auto a = parseExpression();
      if (a) launch->args.push_back(std::move(a));
      if (consume(TokKind::comma)) continue;
      break;
    }
  }
  expect(TokKind::r_paren, "')'");
  return NodePtr(launch);
}

NodePtr Parser::parsePrimary() {
  Token t = curTok;
  switch (curTok.kind) {
  case TokKind::int_literal: {
    advance();
    int64_t v = 0;
    StringRef txt = t.text;
    // Hex literal: 0x... -> parse base 16. Strip the 0x prefix and any
    // trailing integer suffix (u/U/l/L) before converting.
    if (txt.size() > 2 && txt[0] == '0' && (txt[1] == 'x' || txt[1] == 'X')) {
      StringRef hex = txt.substr(2);
      while (!hex.empty() &&
             (hex.back() == 'u' || hex.back() == 'U' ||
              hex.back() == 'l' || hex.back() == 'L'))
        hex = hex.drop_back();
      hex.getAsInteger(16, v);
    } else {
      // Strip trailing integer suffixes before decimal parse.
      while (!txt.empty() &&
             (txt.back() == 'u' || txt.back() == 'U' ||
              txt.back() == 'l' || txt.back() == 'L'))
        txt = txt.drop_back();
      txt.getAsInteger(10, v);
    }
    return NodePtr(new IntegerLiteral(toSourceLoc(t), v));
  }
  case TokKind::float_literal: {
    advance();
    double v = 0;
    StringRef txt = t.text;
    if (txt.ends_with("f") || txt.ends_with("F"))
      txt = txt.drop_back();
    txt.getAsDouble(v);
    return NodePtr(new FloatLiteral(toSourceLoc(t), v));
  }
  case TokKind::identifier:
    advance();
    return NodePtr(new DeclRefExpr(toSourceLoc(t), t.text));
  case TokKind::kw_syncthreads:
    advance();
    // represent as a call to a builtin named __syncthreads
    return NodePtr(new DeclRefExpr(toSourceLoc(t), "__syncthreads"));
  case TokKind::l_paren: {
    advance();
    auto e = parseExpression();
    expect(TokKind::r_paren, "')'");
    return e;
  }
  case TokKind::char_literal: {
    advance();
    // t.text includes the surrounding quotes; decode the body into an int.
    int64_t v = 0;
    StringRef body = t.text;
    if (!body.empty() && body.front() == '\'') body = body.drop_front();
    if (!body.empty() && body.back() == '\'') body = body.drop_back();
    if (!body.empty()) {
      if (body.front() == '\\' && body.size() >= 2)
        v = decodeChar(body.substr(1)); // escape: char after backslash
      else
        v = (unsigned char)body.front();
    }
    return NodePtr(new CharLiteral(toSourceLoc(t), v));
  }
  case TokKind::string_literal: {
    advance();
    StringRef body = t.text;
    if (!body.empty() && body.front() == '"') body = body.drop_front();
    if (!body.empty() && body.back() == '"') body = body.drop_back();
    std::string decoded;
    for (size_t i = 0; i < body.size(); ++i) {
      if (body[i] == '\\' && i + 1 < body.size()) {
        // Reuse decodeChar on the escape (it handles \n, \t, \\, \", ...).
        StringRef esc = body.substr(i + 1);
        decoded.push_back((char)decodeChar(esc));
        // Skip the escape character plus however many decodeChar consumed.
        i += escapeLen(esc);
      } else {
        decoded.push_back(body[i]);
      }
    }
    return NodePtr(new StringLiteral(toSourceLoc(t), std::move(decoded)));
  }
  default:
    // Functional cast: `int(x)`, `float(x)`, `vec4(x)` written with a type
    // keyword. (Vector names like float4 already route through `identifier`
    // above as a CallExpr.) A scalar type keyword followed by '(' is a cast.
    if (startsType(curTok) && lexer.peek().is(TokKind::l_paren)) {
      Token tt = curTok;
      Type *ty = parseType();
      expect(TokKind::l_paren, "'('");
      NodePtr sub = parseExpression();
      expect(TokKind::r_paren, "')'");
      return NodePtr(new CStyleCastExpr(toSourceLoc(tt), ty, std::move(sub)));
    }
    (void)error(t, "expected expression");
    return nullptr;
  }
}
