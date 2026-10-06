# VC — A CUDA-like language targeting Vulkan via MLIR/SPIR-V

> The project has **two SPIR-V backends**:
>
> - **GLSL backend (working, no MLIR needed)**: `.vc` → AST → GLSL compute
>   shader → `glslc` → SPIR-V. The full `test/*.vc` self-checking demo suite
>   runs through this backend.
> - **MLIR backend (working)**: `.vc` → AST → MLIR (vc dialect) → gpu →
>   spirv → binary. The `vc` driver emits MLIR, serialized SPIR-V, and linked
>   host executables (`-emit=mlir` / `-emit=spirv` / `-emit=full`); the same
>   self-checking demos run through this backend when MLIR is enabled.

## Architecture

```
.vc source ──► Frontend (Lexer+Parser) ──► AST ──┬─► GLSL ──glslc──► SPIR-V (.spv)
                                                 │
                                                 └─► MLIR (vc dialect) ──► gpu ──► spirv ──► binary
                                                                                            │
                                                                                            ▼
                                        Vulkan Runtime (vcMalloc/vcLaunchKernel/…)
```

See [docs/language-spec.md](docs/language-spec.md) for the CUDA→Vulkan
mapping table and the planned language surface.

## Prerequisites

Already present on this system: LLVM/Clang 18 dev, Vulkan 1.3.275, CMake,
Ninja, g++ 13.

Still needed (sudo):

```Shell
sudo apt install libmlir-18-dev mlir-18-tools \
                 llvm-spirv-18 libllvmspirvlib-18-dev spirv-headers spirv-tools
```

## Build

```bash
# Without MLIR (frontend + runtime + AST dumper only):
cmake -B build -G Ninja -DVC_ENABLE_MLIR=OFF
cmake --build build

# With MLIR (full compiler, once the package above is installed):
cmake -B build -G Ninja
cmake --build build
```

## Run

```bash
# End-to-end via the GLSL backend (single-file CUDA-style driver):
./build/tools/vc-dump-ast/vc-dump-ast test/vadd.vc  # dump AST
./build/tools/vcc/vcc test/vadd.vc -emit=glsl    # dump GLSL source
./build/tools/vcc/vcc test/vadd.vc -o build/vadd.spv   # GLSL -> glslc -> .spv, link host exe, run

# MLIR backend (vc driver; -emit=mlir dumps IR, -emit=spirv writes binary):
./build-mlir/tools/vc/vc test/vadd.vc -emit=mlir
./build-mlir/tools/vc/vc test/vadd.vc -emit=spirv -o build/vadd_mlir.spv
./build-mlir/tools/vc/vc test/vadd.vc -emit=full -o build/vadd_mlir  # SPIR-V + linked host exe

# Full self-checking demo suite, both backends at once:
python3 test/run_e2e.py --vcc build/tools/vcc/vcc \
                        --mlirc build-mlir/tools/vc/vc test/ examples/
```

The e2e harness builds and runs every `test/*.vc` and `examples/*.vc` demo
through **both** backends and checks the `PASS`/`FAIL` the program prints. As
of this commit: **135 (backend, demo) pairs pass, 3 skipped** (constructs one
backend can't lower are skip-listed rather than failing the suite).

See [examples/README.md](examples/README.md) for per-demo details, the
stream/memcpy semantics table, and the specialization-constant mechanism
that lets 2D block sizes reach the shader.

## MLIR backend

The MLIR backend compiles the device subset end-to-end without glslc:

```
AST ─► translateASTToMLIR  ─►  vc dialect + func/scf/arith/memref core
      lib/Codegen/ASTToMLIR.cpp
      │
      ▼  Stage 1: lowerVCToGPU (lib/Codegen/VCToGPU.cpp)
gpu dialect module (kernel entry, thread/block indexing, barriers)
      │
      ▼  Stage 2: MLIR's built-in conversions (lib/Codegen/LoweringPasses.cpp)
convert-gpu-to-spirv  ─►  spirv-lower-abi-attrs  ─►  spirv-update-vce
      │
      ▼
spirv.module  ─►  spirv::serialize  ─►  SPIR-V binary
```

### The `vc` dialect (`lib/Dialect/VC/VC.td`)

A thin event/directive layer over the standard dialects that models the
CUDA device surface; everything else lowers to `scf`/`arith`/`memref`/`func`.

| VC op          | CUDA source         | SPIR-V target                               |
| -------------- | ------------------- | ------------------------------------------- |
| `vc.kernel`    | `__global__` entry  | `OpEntryPoint(GLCompute)`                   |
| `vc.thread_id` | `threadIdx.{x,y,z}` | `LocalInvocationId` builtin                 |
| `vc.block_id`  | `blockIdx.{x,y,z}`  | `WorkgroupId` builtin                       |
| `vc.block_dim` | `blockDim.{x,y,z}`  | `WorkgroupSize` builtin                     |
| `vc.grid_dim`  | `gridDim.{x,y,z}`   | `NumWorkgroups` builtin                     |
| `vc.barrier`   | `__syncthreads()`   | `OpControlBarrier(Workgroup)`               |
| `vc.wmma.*`    | tensor-core ops     | `spirv.KHR.CooperativeMatrix`               |

### Lowering decisions

- **Two stages.** `lowerVCToGPU` rewrites the VC ops onto the gpu dialect;
  the gpu→spirv leg reuses MLIR's `convert-gpu-to-spirv`, which clones each
  `gpu.module` and legalizes the whole body with bundled patterns. Then
  `spirv-lower-abi-attrs` materializes `spirv.entry_point_abi` /
  `spirv.interface_var_abi` attributes into SPIR-V globals + the entry
  point, and `spirv-update-vce` infers the `vce_triple` the serializer
  requires. Both ABI/VCE passes run on `spirv.module` (nested), not at
  top level.
- **Pointers → StorageBuffer memrefs.** A `T *` parameter becomes
  `memref<?xT, strided<[1]>, #spirv.storage_class<StorageBuffer>>`: the
  static layout is required by `getVulkanElementPtr`, and MemRefToSPIRV
  needs an explicit SPIR-V storage-class attribute (not a numeric memory
  space) on every memref.
- **Locals → Function-storage slots.** Variables become 0-d (scalar) or
  N-element (array) `memref.alloca` tagged `#spirv.storage_class<Function>`,
  initialized element-by-element from the initializer list.
- **`__constant__`** **globals are materialized lazily.** A file-scope
  `__constant__` (array or scalar) gets its own Function-storage slot
  hoisted to the kernel entry block on first reference and filled from the
  compile-time initializer. This is a functional mapping (per-kernel
  read-only storage), not a physical shared global — there is no
  `cudaMemcpyToSymbol` runtime path.
- **No returns inside structured regions.** `if (cond) return;` guards are
  inverted into `scf.if(!cond) { remaining statements }` and the function
  tails out to a single return; SPIR-V forbids `OpReturn` in the middle of
  a block. (This was the `test/constant.vc` converter crash.)
- **`index`** **vs** **`i32`.** Language-level integers stay 32-bit; `index` appears
  only where MLIR structurally needs it (memref subscripts, `scf` trip
  counts), bridged with `arith.index_cast`.
- **`for`** **→** **`scf.while`.** Preserves C evaluation order exactly (cond/step
  may depend on locals) rather than requiring canonical `i < bound` loops.

### Testing

`cmake --build build --target mlir-check` runs `test/MLIR/*.vc` through the
`vc` driver and FileCheck-validates the emitted IR: kernel structure and
indexing ops, `__constant__` array/scalar materialization, local arrays,
early-return guards, structured control flow, and an end-to-end `-emit=spirv`
smoke test. `vc-check` covers the frontend diagnostics.

### Deferred

Optimizer passes and broader cross-driver performance coverage. (The
`vc.barrier`/shared-memory, WMMA/cooperative-matrix, and runtime-driven
MLIR-shader paths listed as deferred in earlier revisions are now implemented
— see the surface table above.)

## Runtime

The Vulkan runtime implements a CUDA-style host API over Vulkan compute:

- **Memory**: `vcMalloc` → device-local (high-bandwidth, not mapped);
  `vcMallocHost` → host-visible pinned (persistently mapped). `vcMemcpy`
  stages through transient host-visible buffers via `vkCmdCopyBuffer`.
- **Streams**: `vcStreamCreate` / `vcStreamSynchronize` give ordered command
  queues backed by a per-stream command-buffer ring + fences. Launches and
  H2D/D2D copies are **asynchronous** (record + submit, no `vkQueueWaitIdle`);
  D2H copies block so the destination is readable on return. `NULL` is the
  default stream; `vcLaunchKernelS` / `vcMemcpyS` take an explicit stream.
- **Kernel args**: pointer args bind as consecutive SSBOs; scalar args go
  through push constants (the GLSL backend emits `layout(push_constant)
  uniform PC { … } pc;`, the runtime packs and `vkCmdPushConstants`s them).
- **Pipelines**: cached per `(blockX, blockY, blockZ)` on each kernel and
  backed by a `VkPipelineCache`; workgroup size is a specialization constant
  so one `.spv` runs at any block shape.

## Layout

- `include/vc/` — public headers (Dialect, Frontend, Runtime)
- `lib/Dialect/VC/` — VC dialect (`VC.td` + registration)
- `lib/Frontend/` — Lexer, Parser, AST, Sema, ASTDumper
- `lib/Codegen/` — AST→MLIR (`ASTToMLIR.cpp`) + lowering pipeline to SPIR-V
  (`VCToGPU.cpp`, `LoweringPasses.cpp`); AST→GLSL
- `lib/Runtime/` — Vulkan runtime (async streams, device-local memory,
  push constants, pipeline cache)
- `tools/vc/` — the `vc` compiler driver (MLIR / SPIR-V emission)
- `tools/vc-dump-ast/` — frontend-only AST dumper
- `tools/vc-glsl/` — the `vc-glsl` GLSL-backend driver
- `tools/vcc/` — single-file CUDA-style driver (host + GLSL backend)
- `test/*.vc` — kernel sources (vadd, reduce, matmul, constant, …); also the
  end-to-end test corpus
- `test/Frontend/`, `test/MLIR/` — FileCheck regression tests (targets
  `vc-check` / `mlir-check`)
- `examples/` — the hand-written demos meant to be read: `singlefile.vc`, and
  `flash_attn.vc` (FlashAttention-2/3) with `compare_flash_attn.py`
  cross-checking it against PyTorch's SDPA kernels (needs torch; the demo
  dumps tensors with `--dump`)
- `docs/language-spec.md` — language design notes
- `docs/plans/` — pre-implementation design plans (all shipped; kept for the
  rationale and the rejected alternatives)
- `examples/README.md` — how to build and run every demo
- `vscode-vc/` — TextMate grammar for `.vc` (syntax highlighting only, no LSP)

## Implemented language surface

Both backends (GLSL and MLIR/SPIR-V) cover the device subset below unless a
row notes otherwise. The e2e demo named in each row is the living spec — if
the demo passes on a backend, that construct works there.

A construct the compiler cannot honor is refused with a diagnostic, never
silently dropped or mis-evaluated. Where the MLIR backend *can* lower something
only at reduced fidelity (an f64 transcendental computed in f32), it says so as
a warning; `-Werror` promotes both the frontend warnings and those codegen
warnings to hard errors.

### Core C/CUDA

| Construct | Demo | Notes |
| --- | --- | --- |
| `if`/`for`/`while`/`do`/`switch` | `cf_cond.vc`, `sync.vc` | `for`→`scf.while` (preserves C eval order); `switch`→`scf.if` chain. In MLIR every `case` must end with `break`/`return` — fallthrough and statements before the first label are rejected, not mis-lowered |
| `break`/`continue` | `cf_cond.vc` | loop-carried i1 flag slots + `scf.if` guards (MLIR) |
| Early returns (`if (cond) return;`) | `recursion.vc`, `ref_return.vc` | inverted into `scf.if(!cond){…}`; value-returning early returns via yield chains |
| Scalars / arrays / `struct` / `class` | `struct.vc`, `class.vc`, `nested_struct.vc` | struct-as-value = flat memref; methods inlined at call site |
| `enum` | `enum.vc` | const-fold to `int` |
| `namespace` / `::` scope | `namespace.vc`, `namespace_nested.vc` | `ns::f`→`ns_f` device mangle |
| `using Name = Type;` aliases | `using_alias.vc` | |
| `sizeof` | `sizeof.vc` | |
| Multi-dim arrays + init lists | `multidim_init.vc`, `multidim_subscript.vc` | nested `InitListExpr`; partial subscript `b[0][i]` |
| Default arguments | `default_arg.vc` | right-to-left contiguity enforced |
| `#define NAME <literal>` macros | `define.vc` | object-like, single literal only |
| Storage classes `static`/`extern` | — | host passthrough; device rejects function-local `static` |
| CUDA keyword compat | `keyword_compat.vc` | `__restrict__`, `nullptr`, `constexpr` accepted |

### Functions / overloading / references

| Construct | Demo | Notes |
| --- | --- | --- |
| `__device__` helper calls | `features.vc` | hoist `func.func` into `gpu.module` + ConvertFuncToSPIRV |
| Overload resolution | `overload.vc` | `conversionRank`/`resolveOverload`; `__device__` param-mangled symbols |
| Bounded device recursion | `recursion.vc` | SPIR-V forbids recursion → AstTransforms pass unrolls bounded linear self-recursion to an accumulator loop |
| Sub-array-to-pointer decay | `sub_array_decay.vc` | MLIR inlines callee at call site; GLSL rejects via `#error` |
| Pointer arithmetic | `pointer_arith.vc` | derived pointers (base+offset), `&x`/`*p` |
| Reference params `T&` / locals | `references.vc`, `ref_swap.vc` | Function-storage memref by-value to `func.call`→`spirv.ptr<struct<array<1×T>>,Function>`; GLSL `inout` |
| `const T&` overload | `const_overload.vc` | `K` mangle marker; const-aware conversion rank; GLSL→`in` (by-value) |
| Reference return `int &f()` | `ref_return.vc` | lvalue call (`pick(x,y,1)=v`); `HoistRefReturnIfYieldsPass` rewrites `scf.if`-yielding-memref to yield `spirv.ptr`; GLSL inlines to ternary/if-else |
| Ref-from-element `f(out[i])` | `ref_elem.vc` | Function-storage temp + copy-in/copy-out for non-const `T&` |
| Struct references `P&`/`const P&` | `ref_struct.vc` | |

### Types / vectors

| Construct | Demo | Notes |
| --- | --- | --- |
| `int`/`float`/`double`/`bool` | `bool.vc`, `math_width.vc` | `bool`→`i32` (i1 has no SPIR-V storage) |
| `__half` / `f16` | `half.vc`, `f16_ssbo.vc` | `float16_t`; f16 SSBO scalar load/store (rtarray widen + post-convert narrow) |
| Vectors `float4`/`int3`/swizzle | `vectors.vc`, `vector_ptr.vc` | `make_float4`/`dot`/`cross`; vector-ptr swizzle write `(*p).x=v` |
| By-value vector kernel args | `vec_arg.vc` | push-constant vector field |

### Memory model

| Construct | Demo | Notes |
| --- | --- | --- |
| `__shared__` workgroup memory | `sync.vc` | `spirv.GlobalVariable`+`addressof`+`AccessChain` (not `memref.global`). Globals are pooled by name module-wide, so two same-named `__shared__` arrays must agree in element type and shape — a conflict is rejected |
| Dynamic `extern __shared__ T s[]` | `dyn_shared.vc` | sized from `blockDim.x`; post-serialize SPIR-V binary patch (`__vc_dynshared_`→`OpTypeArray` length). The `<<<g, b, nbytes>>>` byte count is **rejected at parse time** — no launch path carries it |
| `__constant__` globals | `constant.vc`, `const_local.vc` | lazily materialized per-kernel; **no `cudaMemcpyToSymbol`** (compile-time init only) |
| Atomics | `atomics.vc` | `spirv.Atomic*` / `memref.atomic_rmw`; SSBO `atomicExch` via ordinal-marker rewrite. **MLIR is i32-only** — a float/double/64-bit target or value is rejected (GLSL supports float atomics) |
| `__threadfence()` | `sync.vc` | `spirv.MemoryBarrier` (no `barrier()`) |

### Intrinsics / builtins

| Construct | Demo | Notes |
| --- | --- | --- |
| Math builtins (`sqrt`/`sin`/`pow`/…) | `math_builtins2/3.vc` | `spirv.GL.*`. SPIR-V has no f64 transcendental opcode, so in MLIR a `double` argument is computed at f32 precision — reported as a warning, and a hard error under `-Werror` |
| Warp shuffles / ballot | `warp.vc` | `gpu.subgroup_size`+`GroupNonUniform*`; needs SPIR-V 1.3 (vulkan1.1) on demand |
| Vote (`__syncthreads_count/and/or`) | `vote.vc` | shared-array reduction |
| WMMA / cooperative matrix | `wmma_gemm.vc` | `gpu.subgroup_mma`→`spirv.KHR.CooperativeMatrix`; runtime opportunistic coopMatrix+shaderFloat16 |
| Kernel `printf` | `printf.vc` | `debugPrintfEXT` (GLSL only; MLIR skip-listed) |

### Host runtime API

| Area | Demos | Notes |
| --- | --- | --- |
| Memory | `vadd`, `matmul` | `vcMalloc` (device-local), `vcMallocHost` (pinned), `vcMallocManaged` (unified), `vcMemset`/`vcMemcpy` (+async +2D) |
| Streams / events | `async_overlap`, `async_memcpy`, `event.vc`, `stream_query.vc` | async queues, timeline-semaphore events, `vcLaunchKernelIndirect` |
| Graphs | `graph.vc` | record/replay via Vulkan secondary command buffers |
| Multi-device | `multigpu.vc` | `vcGetDevice/SetDevice/Count`, `vcMemcpyPeer` (host-bridge) |
| Introspection | `profiling.vc`, `occupancy.vc` | `vcGetDeviceProperties`, `vcPointerGetAttributes`, `vcEventElapsedTime`, `vcOccupancyMaxPotentialBlockSize` |
| Host callbacks | `stream_query.vc` | `vcLaunchHostFunc` (timeline sem + bg thread) |

## Long tail — deliberately not done

These are real CUDA constructs VC does **not** support, with the reason each
was left out. They are skip-listed (one backend) or rejected loudly rather
than silently miscompiling.

- **`__device__` mutable globals** (`__device__ int g = 0;`). VC models no
  `cudaMemcpyToSymbol` runtime path, so a mutable device-global would be a
  compile-time-initialized shader global with no host write surface — a
  "looks like CUDA but behaves differently" trap. Use `__constant__` (read-only)
  or a kernel parameter for mutable state. (Parser rejects the declaration.)
- **Reference-to-array parameters** `int (&arr)[4]` (and the `T (&)[N]`
  overload-set form). Niche syntax; the existing pointer-decay path
  (`int *row`, `sub_array_decay.vc`) covers the common case. Parser errors
  with "expected parameter name".
- **3+ level nested init lists** (`int a[2][2][2] = {{{…}}}`). 2-level
  nesting works (`multidim_init.vc`); deeper nesting is a parse error
  ("expected expression") — flatten the literal or use explicit assigns.
- **`union`.** The parser does not accept the keyword at all ("expected '('"
  — it is read as a function name). Model the storage explicitly with a
  `struct` plus a cast, or keep separate typed buffers.
- **`goto` / labels.** Rejected at parse time ("expected ';'" at the label).
  Restructure with a loop plus `break`/`continue`, or a `for` with a flag.
- **Float / 64-bit atomics (MLIR backend only).** SPIR-V's atomic set here is
  the integer one, and VC stores every scalar slot as an i32 (`double` spans
  two slots), so `atomicAdd` on a `float`/`double`/`long` would add bit patterns
  or touch half a value. The translator rejects it; the GLSL backend supports
  float atomics. (`test/MLIR/atomic_reject.vc`)
- **`switch` fallthrough (MLIR backend).** Each case lowers to its own `scf.if`,
  so a case with no terminating `break`/`return` — or statements sitting before
  the first label — is rejected instead of running the wrong statements.
  (`test/MLIR/switch_reject.vc`)
- **`<<<g, b, sharedMemBytes>>>`.** No launch path carries a dynamic shared-memory
  byte count (both backends size `extern __shared__` from `blockDim.x`), so a
  nonzero third argument is a parse error rather than a silently dropped
  field (rejected by the parser, not Sema: launches live in host bodies,
  which Sema deliberately does not type-check).
  Pass `0` (or the 2-argument form) to name a stream without asking for bytes.
  (`test/Frontend/launch-shmem.vc`)
- **Function-like / conditional macros** (`#define F(x) …`, `#ifdef`). Only
  object-like single-literal `#define` is supported.
- **General C++ templates / exceptions / RTTI.** Out of scope for a shader
  language; VC is a CUDA **subset**, not a C++ compiler.
- **True async peer copies / transfer queues.** `vcMemcpyPeer` uses a
  host-bridge fallback (D2H→H2D), not a device-to-device copy engine.
- **General-purpose device-memory sub-allocation.** Each `vcMalloc` is a
  standalone `VkBuffer`; no arena/allocator.
- **Optimizer passes.** No IR-level optimization beyond MLIR's default
  legalization canonicalization; performance-tuning passes are deferred.
- **WMMA beyond the demo shapes.** `wmma_gemm.vc` covers the cooperative-matrix
  path; arbitrary fragment layouts / non-`f16`/`f32` accumulation combos are
  not exercised.

## What is scaffolded vs. TODO

Done: lexer, parser (CUDA subset — see the table above), AST, AST dumper, Sema
(unused-var / argument type / overload / reference / extra diagnostics),
Vulkan runtime (instance/device/queue/buffer/stream/pipeline/dispatch), the
full CUDA-style host API (async streams, events, graphs, managed memory,
multi-device selection, device-group P2P peer copies with host-bridge
fallback, device-local memory, reusable staging buffers, push-constant scalar
args, pipeline caching, indirect dispatch, occupancy/profiling/attributes),
CMake with optional MLIR, and the **MLIR backend**: `vc` dialect, AST→MLIR
translation, `vc→gpu` lowering, shared memory/barriers, atomics, warp/vote
intrinsics, WMMA/cooperative matrix, serialized SPIR-V (`vc -emit=spirv`),
and host executable generation (`vc -emit=full`).

Deferred: see **Long tail** above.

## License

MIT — see [LICENSE](LICENSE). The `.deb` ships the same text as
`/usr/share/doc/vc/copyright`.

