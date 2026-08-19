# VC — A CUDA-like language targeting Vulkan via MLIR/SPIR-V

> The project has **two SPIR-V backends**:
> - **GLSL backend (working, no MLIR needed)**: `.vc` → AST → GLSL compute
>   shader → `glslc` → SPIR-V. End-to-end `vector_add` runs and prints PASS.
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

# MLIR backend (needs libmlir-18-dev; vc->gpu lowering still TODO):
./build/bin/vc test/vadd.vc -emit=mlir
./build/bin/vc test/vadd.vc -emit=spirv -o vadd.spv
```

## Layout

- `include/vc/` — public headers (Dialect, Frontend, Runtime)
- `lib/Dialect/VC/` — VC dialect (`VC.td` + registration)
- `lib/Frontend/` — Lexer, Parser, AST, Sema, ASTDumper
- `lib/Codegen/` — AST→MLIR + lowering pipeline (vc→gpu→spirv)
- `lib/Runtime/` — Vulkan runtime implementing the CUDA-style API
- `tools/vc/` — the `vc` compiler driver
- `tools/vc-dump-ast/` — frontend-only AST dumper
- `test/vadd.vc`, `examples/vector_add.cpp` — minimal demo
- `docs/language-spec.md` — language design notes

## What is scaffolded vs. TODO

Done: lexer, parser (vector-add subset + `if`), AST, AST dumper, Sema
skeleton, Vulkan runtime (instance/device/queue/buffer/pipeline/dispatch),
CUDA-style host API, CMake with optional MLIR.

Deferred (placeholders present): VC→GPU lowering pass, WMMA/tensor-core
lowering (`vc.wmma.*` → `spirv.KHR.CooperativeMatrix`), full type system,
`__shared__`/`__syncthreads` IR semantics, control flow IR, optimization
passes, multi-stream/multi-device.
