# VC — A CUDA-like language targeting Vulkan via MLIR/SPIR-V

> Scaffold status: the **frontend** (lexer/parser/AST/Sema) and the
> **Vulkan runtime** (CUDA-style host API) build and run. The
> **MLIR dialect / codegen / `vc` compiler driver** require
> `libmlir-18-dev` to be installed before they can build.

## Architecture

```
.vc source ──► Frontend (Lexer+Parser) ──► AST ──► Codegen ──► MLIR (vc dialect)
                                                                  │  lower: vc→gpu→spirv
                                                                  ▼
                                                          SPIR-V binary (.spv)
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
# Frontend smoke test (works now):
./build/tools/vc-dump-ast/vc-dump-ast test/vadd.vc

# End-to-end (needs MLIR-built `vc`):
./build/bin/vc test/vadd.vc -emit=ast    # dump AST
./build/bin/vc test/vadd.vc -emit=mlir   # dump VC-dialect MLIR
./build/bin/vc test/vadd.vc -emit=spirv -o vadd.spv
./build/examples/vector_add vadd.spv     # runs on Vulkan, prints PASS/FAIL
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
