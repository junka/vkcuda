# VC Examples

Four end-to-end demos that compile a `.vc` kernel to SPIR-V (via the GLSL
backend) and run it on Vulkan through the CUDA-style runtime API. Each is a
self-contained host program: allocate buffers, launch the kernel, copy back,
print `PASS`/`FAIL`. The first three are synchronous correctness demos; the
fourth (`async_overlap`) exercises the async stream API.

## Prerequisites

- A working build of the project (see the top-level [README](../README.md)):
  ```bash
  cmake -B build -G Ninja -DVC_ENABLE_MLIR=OFF
  cmake --build build
  ```
- A Vulkan 1.2+ capable device with a compute queue.
- `glslc` (shaderc) on `PATH` — the GLSL backend invokes it to assemble SPIR-V.

## Common flow

Every demo follows the same two steps:

```bash
# 1. Compile the .vc kernel -> SPIR-V (.spv)
./build/tools/vc-glsl/vc-glsl test/<name>.vc -o build/<name>.spv

# 2. Run the host program, passing the .spv path
./build/examples/<binary> build/<name>.spv
```

The host program takes the `.spv` path as its first argument (defaults to
`<name>.spv` in the cwd if omitted). Expected output on success:
`<name>: PASS`.

---

## vector_add

**Kernel:** [`test/vadd.vc`](../test/vadd.vc)
**Host:** [`vector_add.cpp`](vector_add.cpp)

Element-wise `c[i] = a[i] + b[i]` over 64 floats. The simplest demo —
exercises a 1D launch, a scalar argument passed by value (`int n`), and an
`if` guard.

```bash
./build/tools/vc-glsl/vc-glsl test/vadd.vc -o build/vadd.spv
./build/examples/vector_add build/vadd.spv
# devices: 1
# vector_add: PASS
```

| param | value |
|-------|-------|
| N | 64 |
| block | 32 |
| grid | N (= 2 workgroups) |
| args | `a*`, `b*`, `c*`, `n` (scalar) |

---

## block_reduce

**Kernel:** [`test/reduce.vc`](../test/reduce.vc)
**Host:** [`block_reduce.cpp`](block_reduce.cpp)

Block-wise sum reduction. 128 elements are split into 4 blocks of 32 threads;
each block tree-reduces its slice to one output element. Exercises a `for`
loop, `__shared__` memory, and `__syncthreads()`.

```bash
./build/tools/vc-glsl/vc-glsl test/reduce.vc -o build/reduce.spv
./build/examples/block_reduce build/reduce.spv
# devices: 1
# block_reduce: PASS
```

| param | value |
|-------|-------|
| N | 128 |
| block | 32 |
| grid | N (= 4 workgroups) |
| args | `inBuf*`, `outBuf*`, `N` (scalar) |

The output buffer holds 4 partial sums (one per block); the host verifies
each against the corresponding slice of the input.

---

## matmul

**Kernel:** [`test/matmul.vc`](../test/matmul.vc)
**Host:** [`matmul.cpp`](matmul.cpp)

Tiled matrix multiply `C = A * B` for a 32×32 matrix with 16×16 tiles. The
only 2D demo — exercises a 2D launch (`vcLaunchKernel2D`), 2D thread indices
(`threadIdx.y`, `blockIdx.x`, …), 2D `__shared__` arrays, `&&`, nested `for`
loops, and two `__syncthreads()` per tile.

```bash
./build/tools/vc-glsl/vc-glsl test/matmul.vc -o build/matmul.spv
./build/examples/matmul build/matmul.spv
# devices: 1
# matmul: PASS
```

| param | value |
|-------|-------|
| N | 32 |
| tile/block | 16 × 16 |
| grid | N × N (= 2 × 2 workgroups) |
| args | `A*`, `B*`, `C*`, `N` (scalar) |

The host feeds `A = I` (identity), so `C` should equal `B` exactly and the
check is a tolerance-free comparison.

### How 2D block sizes reach the shader

The GLSL backend emits the workgroup size as specialization constants
(`local_size_x_id = 0[, local_size_y_id = 1[, local_size_z_id = 2]]`) based
on which `.x`/`.y`/`.z` components the kernel reads. At launch,
`vcLaunchKernel2D` specializes the compute pipeline to the given block
dimensions, so the same `.spv` runs at any block shape without recompiling.

---

## async_overlap

**Kernel:** [`test/vadd.vc`](../test/vadd.vc) (reused)
**Host:** [`async_overlap.cpp`](async_overlap.cpp)

Async stream overlap demo. Two independent `vector_add` workloads are
dispatched back-to-back on **two separate streams** (`vcStreamCreate`), with no
synchronization between them, then each stream is synchronized and verified
independently. This exercises the stream API added by the runtime refactor:

- `vcStreamCreate` / `vcStreamDestroy` / `vcStreamSynchronize`
- `vcLaunchKernelS(..., stream)` — explicit-stream 1D launch (`NULL` = default)
- `vcMemcpyS(..., stream)` — explicit-stream memcpy

```bash
./build/tools/vc-glsl/vc-glsl test/vadd.vc -o build/vadd.spv
./build/examples/async_overlap build/vadd.spv
# async_overlap: PASS
```

It proves launches no longer block the host (no per-launch `vkQueueWaitIdle`),
that per-stream command ordering holds (memcpy-then-launch within a stream is
correct without cross-stream sync), and that work on different streams stays
independent.

---

## Stream & memcpy semantics

The runtime mirrors a subset of the CUDA Runtime API semantics:

| API | Memory | mapped? | Async? |
|-----|--------|---------|--------|
| `vcMalloc` | device-local | no | — |
| `vcMallocHost` | host-visible (pinned) | yes | — |
| `vcMemcpy` H2D | — | — | async on stream |
| `vcMemcpy` D2D | — | — | async on stream |
| `vcMemcpy` D2H | — | — | **synchronous** (waits so the host can read) |

- **Streams** are ordered command queues. `NULL` (or omitting the stream arg)
  is the default stream. Commands issued to the same stream execute in order;
  different streams may execute concurrently.
- **`_S` suffix** variants (`vcLaunchKernelS`, `vcLaunchKernel2DS`,
  `vcMemcpyS`) take a `VCStreamHandle` as their last argument. The unsuffixed
  `vcLaunchKernel` / `vcLaunchKernel2D` / `vcMemcpy` are wrappers that pass
  `NULL` (default stream) — so existing programs work unchanged.
- **D2H memcpy blocks** until the copy completes, matching `cudaMemcpy(D2H)`,
  so the destination is readable on return. Truly async D2H is a future TODO.
- **Scalar kernel args** are passed via push constants (`pc.<name>` in the
  emitted GLSL), not staging buffers; pointer args get consecutive SSBO
  bindings. This is handled jointly by the GLSL backend and the runtime, and
  is transparent to `.vc` source and host code.

---

## Run all four


```bash
for d in vadd:vector_add reduce:block_reduce matmul:matmul; do
  k=${d%%:*}; b=${d##*:}
  ./build/tools/vc-glsl/vc-glsl test/$k.vc -o build/$k.spv && \
  ./build/examples/$b build/$k.spv
done
# async_overlap reuses the vadd kernel:
./build/examples/async_overlap build/vadd.spv
```

## Troubleshooting

- **`glslc not found`** — install shaderc (`sudo apt install glslc` or
  equivalent), or ensure it is on `PATH`.
- **`vcInit failed`** — no Vulkan instance/device. Check that a Vulkan loader
  and a compute-capable GPU are present (`vulkaninfo`).
- **`could not load kernel`** — the `.spv` path argument is wrong or the
  kernel wasn't compiled first.
- **`FAIL`** — rerun the host binary; on failure the demos print the first few
  mismatched elements for diagnosis.
