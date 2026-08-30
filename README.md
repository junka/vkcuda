# VC — A CUDA-like language targeting Vulkan via MLIR/SPIR-V

> The project has **two SPIR-V backends**:
>
> - **GLSL backend (working, no MLIR needed)**: `.vc` → AST → GLSL compute
>   shader → `glslc` → SPIR-V. End-to-end `vector_add`, `block_reduce`, and
>   `matmul` run and print PASS; an `async_overlap` demo exercises the
>   stream API.
> - **MLIR backend (working)**: `.vc` → AST → MLIR (vc dialect) → gpu →
>   spirv → binary. The `vc` driver emits MLIR and serialized SPIR-V
>   (`-emit=mlir` / `-emit=spirv`); see [MLIR backend](#mlir-backend) for the
>   design and [test/MLIR](test/MLIR) for the regression tests.

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
# End-to-end via the GLSL backend (works now, no MLIR):
./build/tools/vc-glsl/vc-glsl test/vadd.vc -emit=ast     # dump AST
./build/tools/vc-glsl/vc-glsl test/vadd.vc -emit=glsl    # dump GLSL source
./build/tools/vc-glsl/vc-glsl test/vadd.vc -o build/vadd.spv   # GLSL -> glslc -> .spv
./build/examples/vector_add build/vadd.spv               # runs on Vulkan -> PASS

# Run all four demos (compile kernels + run hosts):
for d in vadd:vector_add reduce:block_reduce matmul:matmul; do
  k=${d%%:*}; b=${d##*:}
  ./build/tools/vc-glsl/vc-glsl test/$k.vc -o build/$k.spv && \
  ./build/examples/$b build/$k.spv
done
./build/examples/async_overlap build/vadd.spv            # async stream overlap -> PASS

# MLIR backend (vc driver; -emit=mlir dumps IR, -emit=spirv writes binary):
./build/tools/vc/vc test/vadd.vc -emit=mlir
./build/tools/vc/vc test/vadd.vc -emit=spirv -o build/vadd_mlir.spv
```

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
| `vc.wmma.*`    | tensor-core ops     | `spirv.KHR.CooperativeMatrix` (placeholder) |

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

`vc.barrier` GPU lowering and shared-memory semantics, WMMA/tensor-core
lowering (`vc.wmma.*` → `spirv.KHR.CooperativeMatrix`), optimizer passes,
and running the MLIR-emitted shaders through the Vulkan runtime
(only the GLSL backend is driveable from host code today).

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
- `test/*.vc` — kernel sources (vadd, reduce, matmul, constant, …)
- `test/Frontend/`, `test/MLIR/` — FileCheck regression tests (targets
  `vc-check` / `mlir-check`)
- `docs/language-spec.md` — language design notes

## What is scaffolded vs. TODO

Done: lexer, parser (CUDA subset + `if`/`for`/`while`/`__shared__`/
`__syncthreads`/2D launch), AST, AST dumper, Sema (unused-var / argument
type / extra diagnostics), Vulkan runtime (instance/device/queue/buffer/
stream/pipeline/dispatch), CUDA-style host API (async streams,
device-local + staging memory, push-constant scalar args, pipeline
caching), CMake with optional MLIR, end-to-end demos, and the **MLIR
backend**: `vc` dialect, AST→MLIR translation, `vc→gpu` lowering, and
the gpu→spirv pipeline producing serialized SPIR-V (`vc -emit=spirv`,
validated by `constant.vc`), plus `mlir-check` IR regression tests.

Deferred (placeholders present): `vc.barrier`/`__shared__` GPU lowering
and SPIR-V semantics, WMMA/tensor-core lowering (`vc.wmma.*` →
`spirv.KHR.CooperativeMatrix`), full type system, optimizer passes,
async D2H memcpy, independent transfer queue + cross-queue semaphores,
memory sub-allocation pool, multi-device, driving MLIR-emitted shaders
from host code.
