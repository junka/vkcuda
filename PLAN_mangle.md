# Device Mangling Consolidation Plan

## Goal

Consolidate duplicated device-side function name mangling so Sema lookup keys,
GLSL symbols, MLIR symbols, and MLIR host/full entry-point names are derived
from one rule.

Keep host-side C++ lookup/naming semantics separate.

## Current Duplicate Sites

There are four naming/mangling sites that look similar, but not all have the
same semantics:

1. `Sema` `mangledFuncName`
   - Device-side function-table key.
   - Currently derives namespace prefix from recursive `nsPrefix`, already in
     underscore form.
2. GLSL backend `deviceFuncName`
   - Device-side emitted function symbol.
   - Reads `FunctionDecl::nsName`, which is stored in colon form.
3. MLIR backend `deviceFuncName`
   - Device-side `func.func` / kernel symbol.
   - Reads `FunctionDecl::nsName`, colon form.
4. `tools/vc/vc.cpp` local `deviceSymName`
   - MLIR `-emit=host` / `-emit=full` entry-point name for each kernel.
   - Reads `FunctionDecl::nsName`, colon form.

A fifth related site exists but has different semantics:

5. Host backend `hostFuncKey`
   - Host C++ lookup key.
   - Uses native host C++ spelling, especially `Class::method`.
   - Must not be merged with device mangling.

## Merge Scope

Merge the semantically device-side sites into a shared helper:

```cpp
std::string deviceMangledName(const FunctionDecl *f);
```

Use it from:

- `Sema` function registration / lookup keys,
- GLSL backend device function emission / lookup,
- MLIR backend device function emission / lookup,
- `tools/vc/vc.cpp` MLIR host/full kernel entry-point collection.

Do not merge:

- Host backend `hostFuncKey`.
- Existing `mangleScoped` behavior for structs, typedefs, variables, enum
  constants, or other non-function symbols.

Partially reuse:

- Host backend `launchHandleName` may reuse a generic scope-name helper because
  it only needs a safe generated identifier matching kernel handles, not host
  C++ lookup spelling.

## Helper API

Add shared helpers in a frontend/codegen-visible location, preferably a small
new header rather than a backend-local file:

```cpp
// include/vc/Frontend/Mangle.h
std::string mangleScopeName(llvm::StringRef scopedName);
std::string deviceMangledName(const FunctionDecl *f);
```

Implementation placement options:

- Preferred: `lib/Frontend/Mangle.cpp` plus CMake target wiring into
  `VCFrontend`.
- Acceptable if keeping file count low: implement in `lib/Frontend/ASTHelpers.cpp`
  and declare in a new/nearby header.

## Device Mangling Rules

`mangleScopeName(scopedName)`:

- Input is colon-scoped form, e.g. `outer::inner`.
- Output is underscore form, e.g. `outer_inner`.
- Empty input returns empty string.

`deviceMangledName(f)`:

1. Compute base name:
   - If `f->isMethod && !f->className.empty()`: `Class_method`.
   - Else: `f->name`.
2. If `f->nsName` is non-empty:
   - Return `mangleScopeName(f->nsName) + "_" + base`.
3. Otherwise return `base`.

This preserves the current non-overload ABI. Do not add parameter type suffixes
in this change.

## Host Naming Rules

Do not merge `hostFuncKey`.

`hostFuncKey` is host-facing and uses native C++ scoped spelling with `::`,
especially for class methods. It must remain independent.

`launchHandleName` may reuse `mangleScopeName` when converting a scoped launch
callee like `outer::inner::kernel` to `outer_inner_kernel`, but it should not
call `deviceMangledName` because launch sites have an AST callee expression,
not a `FunctionDecl *`.

## Main Risk: Namespace Source Of Truth

Sema currently builds device function-table keys from a recursive `nsPrefix` in
underscore form:

```text
namespace a { namespace b { f } } -> a_b_f
```

The shared helper will instead read from `FunctionDecl::nsName`, which Parser
stores in colon form:

```text
a::b + f -> a_b_f
```

These must produce the same key. If not, Sema call lookup and backend emitted
symbols diverge.

Before replacing Sema lookup keys, verify nested namespace behavior with a
regression/demo.

## Secondary Risk: Methods Inside Namespaces

Current behavior may differ for methods inside namespaces:

- Sema's old `mangledFuncName(f, nsPrefix)` can prefix methods with namespace,
  producing something like `ns_Class_method`.
- Parser `stampNamespace` currently skips methods when assigning
  `FunctionDecl::nsName`, so a shared helper based only on `f->nsName` may
  produce `Class_method` instead.

Before changing Sema to use `deviceMangledName`, verify the current supported
method-in-namespace behavior.

If methods inside namespaces are unsupported or untested, keep this change from
silently changing their behavior by either:

- adding/stamping `nsName` for methods where correct, or
- documenting and testing that namespace methods remain unsupported / unchanged.

Do not let this refactor accidentally change class-method device symbols.

## Out Of Scope

Do not include these in this change:

- Parameter type encoding; this belongs to P3A overload work.
- Overload resolution changes.
- Canonical type equality changes.
- Builtin registry changes.
- `mangleScoped` for non-function declarations.
- `hostFuncKey`.

## Implementation Steps

### Step 1: Add Shared Helper

Add `mangleScopeName` and `deviceMangledName` with the rules above.

Add small focused unit-style coverage indirectly through frontend/backend tests;
there is no standalone unit-test framework in this repo.

### Step 2: Verify Namespace Key Equivalence Before Replacement

Add or run a nested namespace device demo before replacing Sema keys:

```cpp
namespace a {
namespace b {
__device__ int f(int x) { return x + 1; }
__global__ void k(int *out) { out[0] = f(41); }
}
}
```

Expected device key/symbol:

```text
a_b_f
a_b_k
```

Verify:

- Sema resolves `f` inside `k`,
- GLSL emits matching device symbol names,
- MLIR emits matching `func.func` / entry-point symbols,
- `vc -emit=full` records the matching MLIR entry point via the shared helper.

### Step 3: Verify Method/Namespace Boundary

Before replacing Sema's method key path, add or inspect a regression for class
methods, including namespace-contained class/method if currently supported.

Acceptance for this step:

- `Class::method` device symbol remains `Class_method` where that is current
  behavior.
- No accidental change to host `Class::method` spelling.
- If namespace methods are supported, Sema and backends agree on whether the
  device symbol is `ns_Class_method` or `Class_method`.

### Step 4: Replace Sema Function Key

Replace local `mangledFuncName` usage with `deviceMangledName(f)` for function
registration and lookup.

Keep recursive namespace traversal for collecting declarations, but it should
no longer define a separate function-mangling scheme.

Keep `mangleScoped` in Sema for non-function declarations.

### Step 5: Replace GLSL Device Naming

Replace GLSL backend `deviceFuncName` with `deviceMangledName(f)`.

Keep GLSL-specific reserved-word escaping separate if needed for non-function
identifiers.

### Step 6: Replace MLIR Device Naming

Replace MLIR backend `deviceFuncName` with `deviceMangledName(f)`.

Ensure kernel wrapper ops, call lowering, and emitted function symbols all use
the same helper.

### Step 7: Replace MLIR Driver Entry-Point Naming

Replace `tools/vc/vc.cpp` local `deviceSymName` lambda with
`deviceMangledName(fn)`.

This prevents `-emit=full` / `-emit=host` from drifting from MLIR backend symbol
emission.

### Step 8: Reuse Scope Helper For Launch Handles

Update host `launchHandleName` to build scoped launch identifiers through
`mangleScopeName`, but do not call `deviceMangledName` and do not change
`hostFuncKey`.

### Step 9: Regression Tests

Add or update tests for:

- nested namespace device function call,
- nested namespace kernel launch,
- MLIR `-emit=full` entry-point consistency for namespaced kernels,
- class/host method behavior unchanged,
- method-in-namespace behavior, if supported.

Suggested validation commands:

```bash
cmake --build build --target vc-check
cmake --build build-mlir --target mlir-check
cmake --build build-mlir --target vc-e2e-check
```

## Acceptance Criteria

- One shared device-side function mangling helper is used by Sema, GLSL, MLIR,
  and the MLIR driver entry-point collection.
- `hostFuncKey` is untouched.
- `launchHandleName` reuses only the generic scope-name helper.
- Nested namespace device calls resolve and emit matching symbols.
- Namespaced kernel launches work for both GLSL and MLIR full/host emission.
- Class method device symbols and host C++ spelling do not change accidentally.
- Existing frontend, MLIR, and E2E tests continue to pass.

## Status: DONE

All steps executed. Verification results:

- New files: `include/vc/Frontend/Mangle.h`, `lib/Frontend/Mangle.cpp` (added to
  VCFrontend in CMake).
- Replaced 5 duplicate sites with `deviceMangledName` / `mangleScopeName`:
  Sema `mangledFuncName` + `mangleScopeChain`, GLSL `deviceFuncName` +
  `mangleScopeChainGLSL`, MLIR `deviceFuncName` + inline call-site mangle,
  `tools/vc/vc.cpp` `deviceSymName`, host `launchHandleName` (scope helper only).
- `hostFuncKey` untouched (host C++ spelling, different semantics).
- lit suite: 32/32 PASS (Frontend + MLIR).
- e2e: 66 PASS; the single FAIL is `printf.vc :: glsl` — confirmed PRE-EXISTING
  on the clean tree (undefined `vc::vcEnableKernelPrintf` runtime symbol, unrelated
  to mangling).
- New regression demo `test/namespace_nested.vc` (nested `namespace a { namespace
  b { ... } }`) PASS on both backends.
- Method-in-namespace boundary: `deviceMangledName` produces `Class_method` for
  methods (no `ns_` prefix), identical to the pre-change behavior on the clean
  tree. namespace+class method remains an unsupported/pre-existing limitation,
  unchanged by this refactor.

