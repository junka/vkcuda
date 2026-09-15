# VC Language — Draft Specification

> Status: the frontend covers a substantial CUDA-like subset and lowers it to
> GLSL → SPIR-V → Vulkan compute. This documents the currently implemented
> surface; gaps are noted as "(planned)".

VC is a CUDA-like language whose kernels compile to Vulkan SPIR-V.

## Keywords (CUDA-compatible)

| Keyword | Meaning |
|---------|---------|
| `__global__` | device kernel, launchable from host |
| `__device__` | device-only function |
| `__host__` | host function (default) |
| `__shared__` | workgroup-shared memory |
| `__restrict__` | pointer alias hint |
| `__syncthreads()` | workgroup barrier |
| `struct` / `typedef` | user record types and type aliases |
| `switch` / `case` / `default` | switch control flow |
| `break` / `continue` / `do` / `while` / `for` / `if` / `else` / `return` | control flow |

## Built-in variables

| Variable | VC op | SPIR-V builtin |
|----------|-------|----------------|
| `threadIdx.{x,y,z}` | `vc.thread_id` | `LocalInvocationId` |
| `blockIdx.{x,y,z}` | `vc.block_id` | `WorkgroupId` |
| `blockDim.{x,y,z}` | `vc.block_dim` | `WorkgroupSize` |
| `gridDim.{x,y,z}` | `vc.grid_dim` | `NumWorkgroups` |

## Types

- Scalars: `void bool int unsigned uint long float double`.
- Pointers: `T*` — pointer params become SSBO bindings; scalars become a
  push-constant block.
- CUDA-style vectors: `float4`, `float3`, `float2`, `int4`, `uint3`, `double2`,
  `bool4`, ... (any of float/int/uint/double/bool + count 2–4). These lower to
  GLSL `vec4`/`ivec3`/... and support native swizzles (`.x`, `.xyz`, `.xz`,
  `.rgba`, etc.).
- `struct Name { Type field; Type field[N]; ... };` — lowers to a GLSL `struct`
  emitted before `main()`. Struct pointer params become SSBOs of structs.
- `typedef <Type> <Name>;` — a type alias; GLSL has no typedef, so the alias
  resolves to the underlying type's spelling at emit time.

## Expressions

- Full C operator set with C precedence: arithmetic (`+ - * / %`), bitwise
  (`& | ^ ~`), shifts (`<< >>`), relational/equality, logical (`&& || !`),
  assignment and compound assignment (`+= -= *= /= %= <<= >>= &= |= ^=`),
  ternary (`?:`), increment/decrement (`++ --`, both prefix and postfix with
  correct old/new value semantics).
- C-style cast `(float)x` and functional cast `float(x)`.
- Initializer lists `{ a, b, c }` (one level of nesting supported).
- Vector constructors: `float4(...)` → `vec4(...)`.
- Character literals `'A'`, `'\n'` (decoded to int; GLSL has no char).
- String literals `"..."` are tokenized and parsed but have no kernel use; the
  backend emits a placeholder.
- CUDA/math builtins lower to GLSL: `__sinf`→`sin`, `sinf`→`sin`, `sqrtf`→`sqrt`,
  `__syncthreads`→`barrier()`, etc.

## Semantic analysis

`Sema` runs after parsing (driver aborts on error). It builds a scoped symbol
table, infers expression types into a side table, and reports:
- undeclared identifiers (error),
- unknown struct fields (error),
- invalid swizzles like `.xyza` (error),
- calls to user functions with wrong arity (error),
- type-mismatch / unknown callee (warning, non-fatal).

Thread-index builtins (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`) and a
common math-builtin set are treated as implicitly declared.

## Launch syntax
```
kernel<<<grid, block>>>(args...)
```
maps to `vkCmdDispatch` with workgroup count = `ceil(grid/block)`.

## Hardware features (deferred)
- `wmma::*` -> `spirv.KHR.CooperativeMatrix` (Tensor Core). Placeholder ops
  `vc.wmma.load/store/mma` exist in the dialect; lowering TBD.
- Atomics, warp shuffle/vote, and block-wide vote intrinsics are implemented
  for the currently tested scalar cases; broader CUDA overload coverage and
  backend-specific edge cases remain planned work.

## Limitations
- Kernel parameter names that are GLSL reserved words (e.g. `out`) collide with
  glslc — use a different name (e.g. `result`).
- `__device__` helper parameters that share a name with a kernel scalar param
  are emitted with a `pc.` push-constant prefix; avoid shadowing those names.
- No `union`, no multi-dimensional initializer nesting beyond one level,
  no string type, no full overload resolution.

## Host API
See `include/vc/Runtime/VCRuntime.h` (`vcMalloc`, `vcFree`, `vcMemcpy`,
`vcLaunchKernel`, `vcDeviceSynchronize`, ...). Examples live in `examples/`
(`vector_add`, `block_reduce`, `matmul`, `async_overlap`, `features_demo`,
`struct_demo`, `features2_demo`).
