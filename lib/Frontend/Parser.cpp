//===- Parser.cpp - Recursive-descent parser implementation ---------------===//

#include "vc/Frontend/Parser.h"

#include "llvm/Support/raw_ostream.h"

using namespace vc;
using namespace llvm;

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

VarDecl *Parser::parseVarDecl(Type *ty) {
  Token nameTok = curTok;
  if (!expect(TokKind::identifier, "variable name"))
    return nullptr;
  // expect() already consumed the identifier.
  auto *v = new VarDecl(toSourceLoc(nameTok), ty, nameTok.text);
  if (curTok.is(TokKind::assign)) {
    advance();
    v->init = parseExpression();
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

Type *Parser::parseType() {
  BuiltinTypeKind bk;
  switch (curTok.kind) {
  case TokKind::kw_void: bk = BuiltinTypeKind::Void; break;
  case TokKind::kw_bool: bk = BuiltinTypeKind::Bool; break;
  case TokKind::kw_int: bk = BuiltinTypeKind::Int32; break;
  case TokKind::kw_uint: bk = BuiltinTypeKind::UInt32; break;
  case TokKind::kw_long: bk = BuiltinTypeKind::Int64; break;
  case TokKind::kw_float: bk = BuiltinTypeKind::Float32; break;
  case TokKind::kw_double: bk = BuiltinTypeKind::Float64; break;
  default: return nullptr;
  }
  advance();
  auto *base = new BuiltinType(bk);
  // pointer levels: '*' '*'
  Type *ty = base;
  while (consume(TokKind::star))
    ty = new PointerType(ty);
  return ty;
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

NodePtr Parser::parseStatement() {
  switch (curTok.kind) {
  case TokKind::l_brace: return parseCompoundStmt();
  case TokKind::kw_return: return parseReturnStmt();
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

NodePtr Parser::parseDeclOrExprStmt() {
  // If the current token starts a type, parse a declaration.
  Token save = curTok;
  switch (curTok.kind) {
  case TokKind::kw_void:
  case TokKind::kw_bool:
  case TokKind::kw_int:
  case TokKind::kw_uint:
  case TokKind::kw_long:
  case TokKind::kw_float:
  case TokKind::kw_double: {
    Type *ty = parseType();
    // Could be "T name = ..." or a cast-like expr; here we assume decl.
    VarDecl *v = parseVarDecl(ty);
    expect(TokKind::semi, "';'");
    return NodePtr(new DeclStmt(v ? v->getLoc() : toSourceLoc(save), v));
  }
  default: {
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
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

NodePtr Parser::parseExpression() { return parseAssignment(); }

NodePtr Parser::parseAssignment() {
  auto lhs = parseEquality();
  if (!lhs) return nullptr;
  if (curTok.is(TokKind::assign)) {
    Token op = curTok;
    advance();
    auto rhs = parseAssignment();
    return NodePtr(new BinaryExpr(toSourceLoc(op), BinaryOp::Assign,
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
  auto lhs = parseAdditive();
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
    default: bop = BinaryOp::Div; break; // percent -> Div placeholder
    }
    advance();
    auto rhs = parseUnary();
    lhs = NodePtr(new BinaryExpr(toSourceLoc(op), bop, std::move(lhs), std::move(rhs)));
  }
  return lhs;
}

NodePtr Parser::parseUnary() {
  if (curTok.isOneOf(TokKind::minus, TokKind::bang, TokKind::amp, TokKind::star)) {
    Token op = curTok;
    UnaryOp uop;
    switch (curTok.kind) {
    case TokKind::minus: uop = UnaryOp::Neg; break;
    case TokKind::bang: uop = UnaryOp::LNot; break;
    case TokKind::amp: uop = UnaryOp::AddrOf; break;
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
    } else {
      break;
    }
  }
  return base;
}

NodePtr Parser::tryParseLaunch(NodePtr &callee) {
  // CUDA launch uses '<<<' ... '>>>'. We lex '<' '<' '<' as three tokens
  // since the lexer doesn't have a dedicated launch-open.
  if (!(curTok.is(TokKind::lt) &&
        lexer.peek().is(TokKind::lt)))
    return nullptr;
  // Confirm it's actually '<<<' by checking the peek-ahead is '<' followed
  // by non-'<'. Cheap heuristic sufficient for the demo.
  Token openLt = curTok;
  advance(); // first <
  advance(); // second <
  advance(); // third <
  auto grid = parseExpression();
  expect(TokKind::comma, "','");
  auto block = parseExpression();
  // optional shared-mem stream args ignored for now
  if (!curTok.is(TokKind::launch_close)) {
    // tolerate extra commas
    while (consume(TokKind::comma)) parseExpression();
  }
  expect(TokKind::launch_close, "'>>>'");
  expect(TokKind::l_paren, "'(' after >>>");
  auto *launch = new LaunchExpr(toSourceLoc(openLt), std::move(callee),
                                std::move(grid), std::move(block));
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
    t.text.getAsInteger(10, v);
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
  default:
    (void)error(t, "expected expression");
    return nullptr;
  }
}
