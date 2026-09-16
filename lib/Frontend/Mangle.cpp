//===- Mangle.cpp - Device-side function name mangling -------------------===//
//
// Implementation of the shared device mangling helpers declared in Mangle.h.
// The rule is deliberately identical to the four previously-duplicated sites
// (Sema mangledFuncName, GLSL deviceFuncName, MLIR deviceFuncName, and the
// vc.cpp deviceSymName lambda) so this refactor changes no existing symbol.
//
//===----------------------------------------------------------------------===//

#include "vc/Frontend/Mangle.h"

#include "vc/Frontend/AST.h"

using namespace vc;

std::string vc::mangleScopeName(llvm::StringRef scopedName) {
  std::string out;
  llvm::StringRef rest = scopedName;
  while (!rest.empty()) {
    auto pair = rest.split("::");
    if (!out.empty()) out += '_';
    out += pair.first.str();
    rest = pair.second;
  }
  return out;
}

std::string vc::deviceMangledName(const FunctionDecl *f) {
  // Base name: a method lowers to the free-function spelling `Class_method`
  // (the device has no member functions); a free function keeps its name.
  std::string base;
  if (f->isMethod && !f->className.empty())
    base = f->className.str() + "_" + f->name.str();
  else
    base = f->name.str();

  if (f->nsName.empty())
    return base;
  return mangleScopeName(f->nsName) + "_" + base;
}
