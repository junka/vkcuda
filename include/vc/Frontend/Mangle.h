//===- Mangle.h - Device-side function name mangling ---------------------===//
//
// Shared device mangling: produces the underscore-joined symbol name used as
//   - the Sema device function-table key,
//   - the GLSL backend's emitted function symbol,
//   - the MLIR backend's `func.func` / kernel symbol,
//   - the `vc -emit=host`/`-emit=full` driver's MLIR entry-point name.
//
// Host C++ lookup spelling (`Class::method`, verbatim `::`) is NOT mangling
// and stays in the host backend — see ASTToHost::hostFuncKey.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_MANGLE_H
#define VC_FRONTEND_MANGLE_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace vc {

class FunctionDecl;

/// Mangle a `::`-scoped name to underscore form: `outer::inner` -> `outer_inner`.
/// Empty input returns an empty string. This is the atomic device mangling
/// operation shared by decl-site naming and call-site scope chains.
std::string mangleScopeName(llvm::StringRef scopedName);

/// The device symbol name of a function. A class method `Class::method`
/// becomes `Class_method` (GLSL/SPIR-V have no member functions, so this is
/// the free-function spelling the device needs). A namespace member
/// `ns::func` (read from `FunctionDecl::nsName`, colon form) becomes
/// `ns_func`; nested `outer::inner::func` -> `outer_inner_func`. Top-level
/// free functions keep their bare name.
///
/// This preserves the current non-overload ABI. Parameter type encoding is
/// intentionally NOT included here — it belongs to overload work.
std::string deviceMangledName(const FunctionDecl *f);

} // namespace vc

#endif // VC_FRONTEND_MANGLE_H
