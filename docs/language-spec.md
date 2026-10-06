# VC Language — Draft Specification

> Status: the frontend covers a CUDA-like subset and lowers it to SPIR-V by
> **two independent backends** — GLSL (`vcc`: AST → GLSL → `glslc`) and
> MLIR (`vc`: AST → VC dialect → gpu → SPIR-V). This documents the currently
> implemented surface; gaps are noted as "(planned)".
>
> The authoritative record of what works is the e2e demo suite: `README.md`'s
> *Implemented language surface* tables name the `.vc` demo proving each
> construct on each backend. Where this document and a passing demo disagree,
> the demo wins — and this document is the bug.

VC is a CUDA-like language whose kernels compile to Vulkan SPIR-V.

## Keywords (CUDA-compatible)

| Keyword | Meaning |
|---------|---------|
| `__global__` | device kernel, launchable from host |
| `__device__` | device-only function |
| `__host__` | host function (default) |
| `__shared__` | workgroup-shared memory |
| `__constant__` | compile-time-initialized read-only global |
| `__restrict__` | pointer alias hint |
| `__syncthreads()` | workgroup barrier |
| `struct` / `class` / `typedef` / `using` | user record types and type aliases |
| `namespace` | named scope (`ns::f` device-mangles to `ns_f`) |
| `enum` | const-folded to `int` |
| `switch` / `case` / `default` | switch control flow |
| `break` / `continue` / `do` / `while` / `for` / `if` / `else` / `return` | control flow |
| `static` / `extern` | storage classes (host passthrough; see Limitations) |

## Built-in variables

| Variable | VC op | SPIR-V builtin |
|----------|-------|----------------|
| `threadIdx.{x,y,z}` | `vc.thread_id` | `LocalInvocationId` |
| `blockIdx.{x,y,z}` | `vc.block_id` | `WorkgroupId` |
| `blockDim.{x,y,z}` | `vc.block_dim` | `WorkgroupSize` |
| `gridDim.{x,y,z}` | `vc.grid_dim` | `NumWorkgroups` |

## Types

- Scalars: `void bool int unsigned uint long float double`, plus `__half`/`f16`
  (→ GLSL `float16_t`; MLIR widens f16 SSBOs and narrows post-conversion).
- Pointers: `T*` — pointer params become SSBO bindings; scalars become a
  push-constant block. Reference params `T&` / `const T&` are supported too
  (GLSL `inout`/`in`; MLIR Function-storage memref / `spirv.ptr`).
- CUDA-style vectors: `float4`, `float3`, `float2`, `int4`, `uint3`, `double2`,
  `bool4`, ... (any of float/int/uint/double/bool + count 2–4). These lower to
  GLSL `vec4`/`ivec3`/... and support native swizzles (`.x`, `.xyz`, `.xz`,
  `.rgba`, etc.), `make_float4`, `dot`, `cross`, and swizzle-writes through a
  vector pointer (`(*p).x = v`).
- `struct Name { Type field; Type field[N]; ... };` — lowers to a GLSL `struct`
  emitted before `main()`; on MLIR a struct-as-value is a flat memref (locals)
  or a scalarized by-value param. Nested structs (`o.in.a`) and struct writes
  through SSBOs work.
- `class` — methods are inlined at the call site (`_this` becomes an `inout`
  parameter on GLSL).
- `typedef <Type> <Name>;` and `using <Name> = <Type>;` — type aliases. GLSL has
  no typedef, so an alias resolves to the underlying type's spelling at emit.

## Expressions

- Full C operator set with C precedence: arithmetic (`+ - * / %`), bitwise
  (`& | ^ ~`), shifts (`<< >>`), relational/equality, logical (`&& || !`),
  assignment and compound assignment (`+= -= *= /= %= <<= >>= &= |= ^=`),
  ternary (`?:`), increment/decrement (`++ --`, both prefix and postfix with
  correct old/new value semantics), and the comma operator.
- C-style cast `(float)x` and functional cast `float(x)`.
- Initializer lists `{ a, b, c }`, including 2D nesting
  (`int b[2][3] = {{1,2,3},{4,5,6}}`) and multi-dim partial subscripts
  (`b[0][i]`). Note: init lists must have **no trailing comma** — `{1,2,3,}`
  is a hard parse error.
- Vector constructors: `float4(...)` → `vec4(...)`.
- Character literals `'A'`, `'\n'` (decoded to int; GLSL has no char).
- String literals `"..."` are tokenized and parsed; the only kernel use is as a
  `printf` format string.
- `sizeof(Type)` / `sizeof expr`, including after a cast (`(int)sizeof(x)`).
- `#define NAME <literal>` — an object-like macro over a single literal, folded
  at parse time. No function-like macros, no `#ifdef`.
- CUDA/math builtins lower to GLSL: `__sinf`→`sin`, `sinf`→`sin`, `sqrtf`→`sqrt`,
  `__syncthreads`→`barrier()`, etc.

## Functions

- `__device__` helper functions are inlined (GLSL) or hoisted into `gpu.module`
  and converted with ConvertFuncToSPIRV (MLIR).
- **Overload resolution** is implemented: an overload set is resolved by
  `conversionRank`, `CallExpr::resolvedCallee` records the winner, and each
  `__device__` signature gets a parameter-mangled symbol (`sq_i` / `sq_f`).
  Kernels are excluded from mangling. `const T&` participates in the ranking.
- **Bounded device recursion** is supported: SPIR-V forbids recursion, so an
  `AstTransforms` pass (`unrollDeviceRecursion`) rewrites bounded linear
  self-recursion — single/multi-parameter, optional combine, strictly
  decreasing — into an accumulator loop before codegen. Branching or indirect
  recursion is rejected loudly rather than miscompiled.
- **Sub-array-to-pointer decay** (`f(b[i])` with an `int *` parameter) works on
  MLIR (the callee is inlined at the call site, plus a dead-helper DCE pass);
  GLSL rejects it with a `#error` via a pre-pass.
- Reference returns `int &f()` are lvalues: `pick(x,y,1) = v`.

## Semantic analysis

`Sema` runs after parsing (drivers abort on error). It builds a scoped symbol
table, infers expression types into a side table, and reports:
- undeclared identifiers (error),
- unknown struct fields (error),
- invalid swizzles like `.xyza` (error),
- calls to user functions with wrong arity (error),
- assignment to a `const` (error),
- default-argument lists that are not right-to-left contiguous (error),
- out-of-bounds constant array subscripts (error),
- type-mismatch / unknown callee (warning, non-fatal).

Thread-index builtins (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`) and the
math-builtin set are treated as implicitly declared. The builtin name lists and
the device-name mangling now live in single shared places
(`Frontend/BuiltinRegistry.*`, `Frontend/Mangle.h`) rather than being duplicated
per backend.

## Launch syntax
```
kernel<<<grid, block>>>(args...)
kernel<<<grid, block, 0, stream>>>(args...)
```
maps to `vkCmdDispatch` with workgroup count = `ceil(grid/block)` — i.e. `grid`
is an **element count**, not a block count, and the runtime divides by
`blockDim`. `vcLaunchKernelIndirect` takes a device buffer holding a
`VkDispatchIndirectCommand` (`{x,y,z}` in *workgroups*, not elements).

The third argument (CUDA's dynamic shared-memory byte count) must be the
literal `0` when present: no launch path carries a byte count, and both
backends size `extern __shared__` from `blockDim.x`, so a nonzero request is a
parse error rather than a dropped field (launches live in host function bodies,
which Sema does not type-check). The fourth argument is the stream handle.

## Hardware features

- **Atomics, warp shuffle/vote, and block-wide vote intrinsics** are
  implemented (both backends): `spirv.Atomic*` / `memref.atomic_rmw` /
  `GroupNonUniform*`, vote via shared-array reduction. Warp ops need SPIR-V 1.3
  (vulkan1.1), enabled on demand. The MLIR backend lowers **32-bit integer**
  atomics only — a float/double/64-bit target or value is rejected (VC stores a
  `double` as two i32 slots, so such an atomic would add bit patterns or half a
  value); the GLSL backend supports float atomics.
- **`wmma::*` / cooperative matrix** is implemented:
  `gpu.subgroup_mma` → `spirv.KHR.CooperativeMatrix`, with post-conversion
  fixups (f16 SSBO narrowing, entry-point interfaces, marked atomics) and an
  opportunistic runtime request for `coopMatrix` + `shaderFloat16`. The
  `wmma_gemm.vc` demo passes e2e on the MLIR backend.
- **Kernel `printf`** works on the GLSL backend only, via `debugPrintfEXT` +
  `GL_EXT_debug_printf`, enabled at runtime with `vcEnableKernelPrintf()` (or
  `VC_KERNEL_PRINTF=1`) before `vcInit()`. The MLIR backend emits a
  diagnostic and is skip-listed in `test/run_e2e.py`.
- `__threadfence()` lowers to `spirv.MemoryBarrier` / `memoryBarrier*()`
  **without** a `barrier()` — put an explicit `__syncthreads()` after it if you
  need block-wide visibility.

## Limitations

Not implemented; each is rejected with a parse or semantic error rather than
silently miscompiled:

- `union` — `expected '('` at the `union` keyword.
- `goto` / labels — `expected ';'` at the label.
- Initializer nesting beyond two levels: `int a[2][2][2] = {{{...}}}` fails to
  parse; 2D `{{...},{...}}` and flat multi-dim lists work.
- Function-local `static` on the device side (use `__shared__`); `static` /
  `extern` at file scope are host passthrough.
- No string type beyond `printf` format strings.
- `__constant__` is compile-time-initialized only — there is no
  `cudaMemcpyToSymbol` runtime path.

Two behaviours that earlier revisions of this document listed as limitations
are in fact **fixed** and should not be worked around:

- Kernel parameter names that collide with GLSL reserved words (e.g. `out`) are
  renamed by the backend's `glslName()` mangling — no need to avoid them.
- A `__device__` helper parameter that shares a name with a kernel scalar
  parameter correctly emits the `pc.` push-constant prefix; there is no
  shadowing bug.

## Host API

See `include/vc/Runtime/VCRuntime.h` (`vcMalloc`, `vcFree`, `vcMemcpy`,
`vcLaunchKernel`, `vcDeviceSynchronize`, ...). Demos live in `test/*.vc` and
are documented in `examples/README.md`; the self-checking suite is
`python3 test/run_e2e.py`.
