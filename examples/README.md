# VC Examples

Single-file CUDA-style demos. Every `.vc` source holds **both** the
`__global__` kernel and the host `int main()`, exactly like `nvcc`: the
`vcc` driver lowers the device subset to SPIR-V, lowers the host subset to
C++ embedding that SPIR-V, and drives g++ to produce a self-contained
executable. There are no hand-written host `.cpp` files.

Most demos live under [`test/`](../test/) (which doubles as the demo
directory and the e2e test corpus). The hand-written examples kept here are
[`singlefile.vc`](singlefile.vc) — the canonical minimal program — and
[`flash_attn.vc`](flash_attn.vc) — a worked FlashAttention-2/3 kernel, with
[`compare_flash_attn.py`](compare_flash_attn.py) cross-checking it against
PyTorch.

## Prerequisites

- A working build of the project (see the top-level [README](../README.md)):
  ```bash
  cmake -B build -G Ninja -DVC_ENABLE_MLIR=OFF   # GLSL backend only
  cmake --build build
  ```
- A Vulkan 1.2+ capable device with a compute queue.
- `glslc` (shaderc) on `PATH` — the GLSL backend invokes it.

## Build and run

`vcc` compiles, links, and produces an executable in one step:

```bash
./build/tools/vcc/vcc test/vadd.vc -o build/vadd_sf
./build/vadd_sf
# vector_add: PASS
```

`cmake --build build` already does this for the demo set in
[`CMakeLists.txt`](CMakeLists.txt) — `build/<demo>_sf` for the `test/` demos
and `build/singlefile` / `build/flash_attn` for the two kept here. To inspect
the intermediate stages instead of linking:

```bash
./build/tools/vc-dump-ast/vc-dump-ast test/vadd.vc   # pretty-print the AST
./build/tools/vcc/vcc test/vadd.vc -emit=glsl        # dump the generated GLSL
./build/tools/vcc/vcc test/vadd.vc -emit=spirv -o v.spv   # raw SPIR-V, one file per kernel (v.spv, v.2.spv…)
./build/tools/vcc/vcc test/vadd.vc -emit=full -o build/vadd_sf
```

`vcc -emit=` accepts `host`, `glsl`, `spirv`, `full` — not `ast`. The AST
dump is a separate tool, `vc-dump-ast`.

For the MLIR backend (`.vc` → MLIR → gpu → SPIR-V, no `glslc`), use the `vc`
driver built in the MLIR configuration:

```bash
./build-mlir/tools/vc/vc test/vadd.vc -emit=mlir           # dump IR
./build-mlir/tools/vc/vc test/vadd.vc -emit=full -o build/vadd_mlir
./build/vadd_mlir
```

The full self-verifying suite runs every demo through **both** backends:

```bash
python3 test/run_e2e.py --vcc build/tools/vcc/vcc \
                        --mlirc build-mlir/tools/vc/vc test/
```

---

## vadd — element-wise add

**Source:** [`test/vadd.vc`](../test/vadd.vc)

`c[i] = a[i] + b[i]` over 64 floats. The simplest demo — a 1D launch, a
scalar argument passed by value (`int n`), and an `if` guard.

```bash
./build/tools/vcc/vcc test/vadd.vc -o build/vadd_sf && ./build/vadd_sf
# vector_add: PASS
```

| param | value |
|-------|-------|
| N | 64 |
| block | 32 |
| grid | N (the runtime divides `gridDim` by `blockDim`) |
| args | `a*`, `b*`, `c*`, `n` (scalar) |

---

## reduce — block reduction

**Source:** [`test/reduce.vc`](../test/reduce.vc)

Block-wise sum reduction. 128 elements are split into 4 blocks of 32 threads;
each block tree-reduces its slice to one output element. Exercises a `for`
loop, `__shared__` memory, and `__syncthreads()`.

```bash
./build/tools/vcc/vcc test/reduce.vc -o build/reduce_sf && ./build/reduce_sf
# block_reduce: PASS
```

| param | value |
|-------|-------|
| N | 128 |
| block | 32 |
| grid | N (4 workgroups) |
| args | `inBuf*`, `outBuf*`, `N` (scalar) |

The output buffer holds 4 partial sums (one per block); the host verifies
each against the corresponding slice of the input.

---

## matmul — tiled matrix multiply

**Source:** [`test/matmul.vc`](../test/matmul.vc)

Tiled `C = A * B` for a 32×32 matrix with 16×16 tiles. The only 2D demo —
exercises a 2D launch, 2D thread indices (`threadIdx.y`, `blockIdx.x`, …), 2D
`__shared__` arrays, `&&`, nested `for` loops, and two `__syncthreads()` per
tile.

```bash
./build/tools/vcc/vcc test/matmul.vc -o build/matmul_sf && ./build/matmul_sf
# matmul: PASS
```

| param | value |
|-------|-------|
| N | 32 |
| tile/block | 16 × 16 |
| grid | N × N (2 × 2 workgroups) |
| args | `A*`, `B*`, `C*`, `N` (scalar) |

The host feeds `A = I` (identity), so `C` should equal `B` exactly and the
check is a tolerance-free comparison.

### How 2D block sizes reach the shader

The GLSL backend emits the workgroup size as specialization constants
(`local_size_x_id = 0[, local_size_y_id = 1[, local_size_z_id = 2]]`) based
on which `.x`/`.y`/`.z` components the kernel reads. At launch, a 2D launch
specializes the compute pipeline to the given block dimensions, so the same
`.spv` runs at any block shape without recompiling.

---

## flash_attn — FlashAttention (online softmax)

**Source:** [`flash_attn.vc`](flash_attn.vc)

The most involved kernel demo: the core of FlashAttention-2/3 — tiled K/V
traversal with a **running max and running sum**, so the full N×N score
matrix is never materialized. One thread owns one query row; the kernel walks
the key/value sequence in `BC`-wide tiles, staging each tile in `__shared__`
memory, and rescales its accumulator whenever a new tile raises the running
max.

```bash
./build/tools/vcc/vcc examples/flash_attn.vc -o build/flash_attn_sf
./build/flash_attn_sf
# flash_attn: PASS
```

| param | value |
|-------|-------|
| heads×seq×dim | 2 × 64 × 16 |
| tile | 16 query rows × 16 keys |
| grid | `NHEAD * SEQ` (element count → `NHEAD * SEQ / BR` workgroups) |
| args | `Q*`, `K*`, `V*`, `O*`, `scale` (scalar), `causal` (scalar) |

The demo runs both a full (non-causal) and a causal pass, and checks each
against a straight O(N²) double-precision reference computed on the host.

Two things about it are worth calling out, because they are what makes the
flash-attention formulation behave:

- **The base-2 domain.** The kernel computes `exp2` everywhere, never `exp`.
  `log2(e)` is folded into the scale factor (`scale = log2(e) / sqrt(d)`), so
  `exp2(x * scale) == exp(x / sqrt(d))` and the softmax stays exact while
  needing only one transcendental instruction. The host reference must use
  `std::exp2` on the same scores — using `std::exp` there computes a
  different, steeper softmax and disagrees with the kernel by ~1e-2.
- **`alpha = exp2(m_old - m_new)`.** When a new tile raises the running max,
  the accumulator and denominator accumulated so far are scaled by that
  factor — the classic online-softmax correction that lets one pass over K/V
  produce the same result as a two-pass (max, then sum) softmax.

### Cross-checking against PyTorch

The host reference above is the demo's own, so it shares the kernel's author and
its assumptions. For an independent check, the demo can dump its tensors and
[`compare_flash_attn.py`](compare_flash_attn.py) hands them to
PyTorch's own attention:

```bash
mkdir -p /tmp/fa_tensors
./build/flash_attn_sf --dump /tmp/fa_tensors
python3 examples/compare_flash_attn.py /tmp/fa_tensors -v   # needs torch
```

```text
torch 2.10.0+cu128 — cuda_available=False
  torch-vs-kernel    max_abs_err=2.384e-07  [ok]
  torch-vs-hostref   max_abs_err=3.576e-07  [ok]
  kernel-vs-hostref  max_abs_err=3.576e-07  [ok]
  torch-vs-math      max_abs_err=3.322e-07  [ok]
causal0.bin causal=0 (2x64x16): PASS
compare_flash_attn: PASS
```

`torch-vs-kernel` is the claim under test; `torch-vs-math` is a sanity check on
PyTorch rather than on VC. SDPA is pinned with
`torch.nn.attention.sdpa_kernel`, and the two backends are genuinely different
code — verified by profiling rather than assumed. On a CPU-only host
`FLASH_ATTENTION` dispatches to `aten::_scaled_dot_product_flash_attention_for_cpu`
(a real tiled CPU FlashAttention), while `MATH` materializes the N×N scores, so
it is a ground truth that shares no algorithm with the kernel under test.

The `~3e-7` floor is expected, not a defect: the VC kernel accumulates in
float32 while both references accumulate in double. The script asserts `1e-5`,
an order of magnitude above that floor and two below the demo's own `1e-4`.
A GPU is not required, and `torch.cuda.is_available()` is printed so it is
clear which PyTorch kernel ran.

The dump is opt-in at **run time** (`--dump <dir>`), not compile time: VC has
no conditional compilation, and a file-scope `#ifdef` would not enclose the
host code it needs to guard. A run without the flag writes nothing.

What this demo deliberately does *not* model: FA3's warp-specialized pingpong
scheduling and FP8 paths. Those need warp-level producer/consumer stages,
named barriers, and register-level pipelining that VC does not expose — the
demo implements the FA3 *algorithm*, not its CUDA-kernel engineering.

---

## async_overlap — two streams

**Source:** [`test/async_overlap.vc`](../test/async_overlap.vc)

Two independent `vadd` workloads dispatched back-to-back on **two separate
streams**, with no synchronization between them, then each stream is
synchronized and verified independently. Exercises:

- `vcStreamCreate` / `vcStreamDestroy` / `vcStreamSynchronize`
- `vcLaunchKernelS(..., stream)` — explicit-stream 1D launch (`NULL` = default)
- `vcMemcpyS(..., stream)` — explicit-stream memcpy

```bash
./build/tools/vcc/vcc test/async_overlap.vc -o build/async_overlap_sf
./build/async_overlap_sf
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
  so the destination is readable on return.
- **Scalar kernel args** are passed via push constants (`pc.<name>` in the
  emitted GLSL), not staging buffers; pointer args get consecutive SSBO
  bindings. This is handled jointly by the GLSL backend and the runtime, and
  is transparent to `.vc` source and host code.

---

## Run everything

```bash
# Demos from test/ (the VC_SINGLEFILE_DEMOS list in CMakeLists.txt):
for d in vadd reduce matmul struct features features2 async_overlap atomics \
         vectors sync warp enum half constant vote bool sizeof const_local \
         comma default_arg define async_copy namespace class multi_kernel; do
  ./build/${d}_sf
done

# The hand-written examples in this directory:
./build/singlefile
./build/flash_attn

# Or both backends, all demos, with pass/fail checking (both directories):
python3 test/run_e2e.py --vcc build/tools/vcc/vcc \
                        --mlirc build-mlir/tools/vc/vc test/ examples/
```

## Troubleshooting

- **`glslc not found`** — install shaderc (`sudo apt install glslc` or
  equivalent), or ensure it is on `PATH`.
- **`vcInit failed`** — no Vulkan instance/device. Check that a Vulkan loader
  and a compute-capable GPU are present (`vulkaninfo`).
- **`FAIL`** — rerun the executable; on failure the demos print the first few
  mismatched elements for diagnosis.
- **`no Vulkan device`** at configure time — `vc-e2e-check` is gated on a
  working Vulkan device; leave `-DVC_RUN_E2E_TESTS=OFF` (the default) on
  GPU-less machines.
