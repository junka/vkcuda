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

std::string vc::mangleType(const Type *t) {
  if (!t) return "v"; // unknown -> void/placeholder
  // Strip typedefs: a TypedefType mangling uses its underlying type so that
  // `typedef int I; f(I)` and `f(int)` produce the same symbol.
  if (t->getKind() == TypeKind::Typedef) {
    auto *td = static_cast<const TypedefType *>(t)->decl;
    return mangleType(td ? td->underlying : nullptr);
  }
  switch (t->getKind()) {
  case TypeKind::Builtin: {
    switch (static_cast<const BuiltinType *>(t)->builtin) {
    case BuiltinTypeKind::Void: return "v";
    case BuiltinTypeKind::Bool: return "b";
    case BuiltinTypeKind::Int32: return "i";
    case BuiltinTypeKind::UInt32: return "u";
    case BuiltinTypeKind::Int64: return "I";
    case BuiltinTypeKind::UInt64: return "U";
    case BuiltinTypeKind::Float16: return "h";
    case BuiltinTypeKind::Float32: return "f";
    case BuiltinTypeKind::Float64: return "d";
    }
    return "?";
  }
  case TypeKind::Pointer:
    return "p" + mangleType(static_cast<const PointerType *>(t)->pointee);
  case TypeKind::Reference:
    return "r" + mangleType(static_cast<const ReferenceType *>(t)->pointee);
  case TypeKind::Vector: {
    auto *v = static_cast<const VectorType *>(t);
    return "V" + std::to_string(v->count) + mangleType(v->elem);
  }
  case TypeKind::Record:
    // Records have no overload-distinct spelling here; use a stable placeholder
    // keyed on the decl name so distinct structs differ.
    if (auto *d = static_cast<const RecordType *>(t)->decl)
      return "S" + d->name.str();
    return "S";
  case TypeKind::Typedef:
    return "?"; // resolved above; unreachable
  }
  return "?";
}

std::string vc::deviceBaseName(const FunctionDecl *f) {
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

std::string vc::deviceMangledName(const FunctionDecl *f) {
  std::string name = deviceBaseName(f);

  // Kernels keep the bare mangled name (no parameter suffix): CUDA disallows
  // overloading __global__ functions, and the host launch handle (which has no
  // argument type info) must match the device symbol exactly.
  if (f->deviceAttr == DeviceAttr::Global)
    return name;

  // __device__ helpers get a parameter-type suffix so overloads emit distinct
  // symbols (f(int)->name_i, f(float)->name_f). An empty parameter list gets
  // _v (void) so f() and f(int) differ.
  //
  // `const T&` and `T&` are the same Type (const is modeled on ParamDecl, not
  // in the Type itself — see ReferenceType), so mangleType alone can't tell them
  // apart. A const-qualified reference parameter gets a `K` marker before the
  // reference encoding so f(int&) -> _ri and f(const int&) -> _Kri emit distinct
  // device symbols and can coexist in an overload set.
  name += "_";
  if (f->params.empty())
    name += "v";
  else {
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (i) name += "_";
      ParamDecl *p = f->params[i];
      // const on a reference parameter (const T&) must be part of the symbol so
      // f(int&) and f(const int&) differ. const lives on ParamDecl, not in the
      // Type, so resolve typedefs here to spot a reference behind an alias.
      const Type *pt = p->type;
      while (pt && pt->getKind() == TypeKind::Typedef) {
        auto *td = static_cast<const TypedefType *>(pt);
        pt = td->decl ? td->decl->underlying : nullptr;
      }
      if (p->isConst && pt && pt->getKind() == TypeKind::Reference)
        name += "K";
      name += mangleType(p->type);
    }
  }
  return name;
}
