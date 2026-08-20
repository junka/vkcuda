# VC — A CUDA-like language targeting Vulkan via MLIR/SPIR-V

> The project has **two SPIR-V backends**:
> - **GLSL backend (working, no MLIR needed)**: `.vc` → AST → GLSL compute
>   shader → `glslc` → SPIR-V. End-to-end `vector_add`, `block_reduce`, and
>   `matmul` run and print PASS; an `async_overlap` demo exercises the
>   stream API.
> - **MLIR backend (scaffold)**: `.vc` → AST → MLIR (vc dialect) → gpu →
>   spirv. Requires `libmlir-18-dev`; the vc→gpu lowering pass is a TODO.

## Architecture

```
.vc source ──► Frontend (Lexer+Parser) ──► AST ──┬─► GLSL ──glslc──► SPIR-V (.spv)
                                                 │
                                                 └─► MLIR (vc dialect) ──► gpu ──► spirv
                                                                              │
                                                                              ▼
                                                                      SPIR-V binary
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

```bash
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

# MLIR backend (needs libmlir-18-dev; vc->gpu lowering still TODO):
./build/bin/vc test/vadd.vc -emit=mlir
./build/bin/vc test/vadd.vc -emit=spirv -o vadd.spv
```

See [examples/README.md](examples/README.md) for per-demo details, the
stream/memcpy semantics table, and the specialization-constant mechanism
that lets 2D block sizes reach the shader.

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
- `lib/Codegen/` — AST→MLIR + lowering pipeline (vc→gpu→spirv); AST→GLSL
- `lib/Runtime/` — Vulkan runtime (async streams, device-local memory,
  push constants, pipeline cache)
- `tools/vc/` — the `vc` compiler driver
- `tools/vc-dump-ast/` — frontend-only AST dumper
- `tools/vc-glsl/` — the `vc-glsl` GLSL-backend driver
- `test/*.vc` — kernel sources (vadd, reduce, matmul)
- `examples/` — host demos (vector_add, block_reduce, matmul, async_overlap)
- `docs/language-spec.md` — language design notes

## What is scaffolded vs. TODO

Done: lexer, parser (vector-add subset + `if`/`for`/`while`/`__shared__`/
`__syncthreads`/2D launch), AST, AST dumper, Sema skeleton, Vulkan runtime
(instance/device/queue/buffer/stream/pipeline/dispatch), CUDA-style host API
(async streams, device-local + staging memory, push-constant scalar args,
pipeline caching), CMake with optional MLIR, four end-to-end demos.

Deferred (placeholders present): VC→GPU lowering pass, WMMA/tensor-core
lowering (`vc.wmma.*` → `spirv.KHR.CooperativeMatrix`), full type system,
`__shared__`/`__syncthreads` IR semantics, control flow IR, optimization
passes, async D2H memcpy, independent transfer queue + cross-queue
semaphores, memory sub-allocation pool, multi-device.

