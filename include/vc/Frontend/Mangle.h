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
class Type;

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
/// __device__ functions (not __global__ kernels) get a parameter-type suffix so
/// overloaded helpers (e.g. `f(int)` vs `f(float)`) emit distinct device symbols.
/// Kernels are excluded: CUDA does not allow overloading __global__ functions
/// (the launch syntax `k<<<...>>>(args)` cannot disambiguate), and excluding
/// them keeps the kernel symbol matching the host's launch handle (which has no
/// parameter type information).
std::string deviceMangledName(const FunctionDecl *f);

/// The namespace+name mangling WITHOUT the parameter-type suffix. This is the
/// overload-set key: `f(int)` and `f(float)` share a base name so they form one
/// overload set. (deviceMangledName adds the parameter suffix on top to produce
/// the distinct emitted symbol.)
std::string deviceBaseName(const FunctionDecl *f);

/// Encode a Type as a compact mangling token (e.g. int->_i, float->_f,
/// int*->_pi, float4->_f4). Used by deviceMangledName for overload distinction.
/// Not Itanium-ABI compatible — just unique and readable.
std::string mangleType(const Type *t);

} // namespace vc

#endif // VC_FRONTEND_MANGLE_H
