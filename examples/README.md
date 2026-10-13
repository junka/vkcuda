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
PyTorch, [`matmul_bench.vc`](matmul_bench.vc) — a GPU-timed tiled matmul
benchmark, and [`matmul_coopmat_bench.vc`](matmul_coopmat_bench.vc) — the
same benchmark on the cooperative-matrix (tensor-core) path.

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

## matmul_bench — tiled matmul, GPU-timed

**Source:** [`matmul_bench.vc`](matmul_bench.vc)

The same tiled `C = A·B` kernel as the [matmul](#matmul--tiled-matrix-multiply)
test demo (16×16 tiles, one workgroup per output tile, `__shared__` A/B tiles,
boundary fill), but scaled to non-trivial sizes and wrapped in a benchmark.
For each N ∈ {128, 256, 512, 1024} it records GPU-side start/end events around
the launch (`vcEvent*`), reports `ms` and `GFLOPS = 2·N³ / (ms·10⁶)`, and
verifies every element against a naive O(N³) host reference accumulated in
`double`.

```bash
./build/tools/vcc/vcc examples/matmul_bench.vc -o build/matmul_bench && ./build/matmul_bench
# matmul_bench: tiled 16x16, float32
#   N= 128    0.18 ms      22.8 GFLOPS   maxrelerr=5.8e-07   PASS
#   N= 256    0.40 ms      84.1 GFLOPS   maxrelerr=8.3e-07   PASS
#   N= 512    1.31 ms     204.6 GFLOPS   maxrelerr=1.9e-06   PASS
#   N=1024   12.78 ms     168.0 GFLOPS   maxrelerr=4.4e-06   PASS
# matmul_bench: PASS
```

| param | value |
|-------|-------|
| sizes | 128, 256, 512, 1024 (square, multiple of TS) |
| tile/block | 16 × 16 |
| grid | N × N (element count → N/16 workgroups per axis) |
| args | `A*`, `B*`, `C*`, `N` (scalar) |
| timing | `vcEventRecord`/`vcEventElapsedTime` (GPU-side) |

Two things worth calling out, because they are what makes the numbers
trustworthy:

- **Warmup before timing.** The first compute-pipeline instantiation on a
  freshly-created device is dominated by shader compile / `vkCreateComputePipelines`,
  not by the matmul. A tiny 16×16 launch runs (untimed) before the timed loop,
  so the N=128 number is the kernel itself, not one-time setup. Without it the
  first size reads ~15 ms regardless of N.
- **Relative-error correctness.** Each `C[i][j]` is a sum of N products of
  O(N)-magnitude inputs; the largest terms reach ~1e7 at N=1024, so an absolute
  error of ~10 there is meaningless. The check is relative: `|ref − got| / |ref|`
  against `1e-4`, which is comfortably above fp32 accumulation drift (~1e-7 eps
  × N terms ≈ 1e-4 worst case) and below any real bug.

The benchmark runs identically on the MLIR backend
(`vc examples/matmul_bench.vc -emit=full -o build/matmul_bench_mlir`); the two
backends emit different SPIR-V and can be compared head-to-head this way.

---
## matmul_coopmat_bench — cooperative-matrix matmul, GPU-timed

**Source:** [`matmul_coopmat_bench.vc`](matmul_coopmat_bench.vc)

The [`matmul_bench`](#matmul_bench--tiled-matmul-gpu-timed) problem solved a
second way: instead of `__shared__` tiles and per-thread FMAs, the inner
product runs on the Vulkan **cooperative-matrix** units via the CUDA `wmma::`
API. Same host harness — same sizes, same GPU-side `vcEvent*` timing, same
GFLOPS formula — so the two benchmarks are directly comparable.

```bash
./build-mlir/tools/vc/vc examples/matmul_coopmat_bench.vc -emit=full -o build/mc && ./build/mc
# matmul_coopmat_bench: 16x16 tile (2x2 of 8x8x16 fp16->fp32), accumulator streamed through C
#   N= 128    0.2879 ms      14.56 GFLOPS   PASS
#   N= 256    0.3687 ms      91.00 GFLOPS   PASS
#   N= 512    0.5248 ms     511.53 GFLOPS   PASS
#   N=1024    1.4559 ms    1474.98 GFLOPS   PASS
# matmul_coopmat_bench: PASS
```

No `vcc` variant: the GLSL backend rejects the `wmma::` namespace
(`Sema::usesCoopMatrix()`), so this one is MLIR-only by design. It is built by
a separate `vc`-driven target in `CMakeLists.txt` for the same reason.

| param | value |
|-------|-------|
| sizes | 128, 256, 512, 1024 (square, multiple of 16) |
| tile/block | 16 × 16 output tile per workgroup, 32 threads |
| MMA shape | 8×8×16, fp16 → fp32 (device-native) |
| grid | `((N/16)*32) × (N/16)` (element count — x scaled by block size) |
| args | `A*` `__half`, `B*` `__half`, `C*` `float`, `N` (scalar) |
| timing | `vcEventRecord`/`vcEventElapsedTime` (GPU-side) |

### Side by side

| N | matmul_bench (SIMT fp32) | matmul_coopmat_bench (wmma fp16) | speedup |
|------|--------------------------|----------------------------------|---------|
| 128 | 28.4 GFLOPS (0.147 ms) | 14.6 GFLOPS (0.288 ms) | 0.51× |
| 256 | 90.1 GFLOPS (0.373 ms) | 91.0 GFLOPS (0.369 ms) | 1.01× |
| 512 | 250.6 GFLOPS (1.071 ms) | 511.5 GFLOPS (0.525 ms) | 2.0× |
| 1024 | 331.6 GFLOPS (6.476 ms) | 1475.0 GFLOPS (1.456 ms) | 4.4× |

(Both columns measured through the **MLIR** backend — `vc ... -emit=full` — back to
back on the same device, best of 5 runs per size, so the comparison is
like-for-like. Even so these are wall-clock numbers: read them as
order-of-magnitude ratios, not exact. This device's run-to-run variance is
large; see [matmul — tile shape and performance](#matmul--tile-shape-and-performance) for the measured spread
and what is actually limiting both kernels.)

The shape of the curve is the interesting part. At N=128 the coopmat kernel is
*slower*: the problem is far too small to fill the machine, and it pays fixed
costs the SIMT kernel does not — a fresh cooperative-matrix pipeline, the
tile zeroing pass, and 8 fragment loads plus 4 accumulator round-trips per K
step. Up to N=256 they are roughly even. From N=512 the tensor cores pull
decisively ahead, and at N=1024 they win ~4.4× while the SIMT kernel gets no
faster (251 → 332 GFLOPS from 512 to 1024): it walks the same four N×N
operand row ranges per workgroup at every size, so the miss traffic per
output grows with N, whereas each coopmat workgroup reads its panels the
same way regardless. The SIMT column is the `TS=16` shape; the section
[matmul — tile shape and performance](#matmul--tile-shape-and-performance)
below measures what a better-shaped fp32 kernel gets instead.

### What the kernel does

One workgroup is one subgroup (32 threads). It computes a 16× 16 output tile
as **four** 8×8 accumulator fragments — `8x8x16` is the device-native
cooperative shape, so a 16× 16 tile is 2× 2 of them. The four MMAs share the
A/B fragment loads exactly as the SIMT kernel shares `As`/`Bs`:

```
for k in 0..N step 16:
    load a0,a1  <- A[rows ..+16][k ..+16]   (2 fragments)
    load b0,b1  <- B[k ..+16][cols ..+16]   (2 fragments)
    load c00..c11 <- C[rows ..+16][cols ..+16]
    mma(c00,a0,b0) mma(c01,a0,b1) mma(c10,a1,b0) mma(c11,a1,b1)
    store c00..c11 -> C
```

### Two constraints this kernel is shaped around

- **The accumulator lives in global memory.** CUDA WMMA keeps it in registers
  across the K loop. Here that would mean threading a `!gpu.mma_matrix` through
  `scf.while` iter_args across every nested structured region in the loop body.
  Spilling to per-thread (Function) storage instead is *illegal Vulkan*:
  `OpCooperativeMatrixStoreKHR`'s pointer must be Workgroup, StorageBuffer or
  PhysicalStorageBuffer (`VUID-StandaloneSpirv-Pointer-08973`) — `spirv-val`
  rejects it, and this driver silently drops the store. `__shared__` is not an
  option either: WMMA load/store take a memref, and `__shared__` lowers to a
  `spirv.ptr<..., Workgroup>`. So the accumulation is streamed through `C`
  itself: zero the tile once, then each K step does load-C → mma → store-C.
  Each workgroup owns a disjoint tile, so there is no race. The extra 8×8 fp32
  load+store per MMA per K step is the visible price, and the honest cost of
  register blocking this backend can currently express.
- **The leading dimension is a compile-time constant.** `leadDimension` is a
  static `IndexAttr` on the `gpu.subgroup_mma_*` ops, not an operand, so a
  single kernel cannot take the row stride as a runtime argument. `L` is set to
  the largest benchmarked N and every size reuses it; tiles only ever address
  the N×N corner, so the oversize stride costs no traffic.

The correctness check is *exact*: inputs are small integers (exactly
representable in fp16) and every partial sum stays inside fp32's
exact-integer range, so each `C[i][j]` is compared for bit equality against a
`double` reference rather than against a tolerance.

---

## matmul — tile shape and performance

`matmul_bench` above answers *"can VC express a tiled matmul?"* and
`matmul_coopmat_bench` answers *"and on the tensor cores?"*. This section
answers the follow-up: **what shape should a hand-written fp32 matmul take on
this device, and what do the numbers mean.**

The table below comes from a scratch harness — four kernels, same `C = A·B`,
each changing exactly one thing, back to back on one device with `vcEvent`
timing. To reproduce it, copy the four kernels into one `.vc` file and build it
with `vc ... -emit=full`.

| variant | what it changes | N=256 | N=512 | N=1024 | N=1024 (best) |
|---|---|---|---|---|---|
| `v0` — `TS=16`, 1 output/thread | *the current `matmul_bench` shape* | 64 | 167 | 238 | 342 |
| `v1` — no shared memory | drop the staging entirely | 18 | 39 | 44 | 93 |
| `v2` — `TS=32`, 2×2 register blocking | bigger tile + 4 accumulators/thread | **72** | **215** | 253 | **618** |
| `v3` — `TS=64`, 4×4 register blocking | the CUDA-textbook shape | 60 | 165 | **289** | 416 |

All values are GFLOPS (fp32), **median of 12 runs**, with the best-of-12 in the
last column. `v1` walks K straight out of global memory. `v2`/`v3` stage a tile
through `__shared__` with `__syncthreads()` on both sides of the inner loop, and
give each thread a 2×2 / 4×4 quad of outputs so the shared loads are amortized
over more FMAs.

**Shared staging is not optional, and it matters more as N grows.** At N=1024
the staged kernel beats the unstaged one by ~3.7× (342 vs 93, best case). That
gap is the entire reason the tile exists — the unstaged version re-reads
`A[row][k]` and `B[k][col]` from global memory on every FMA.

**Register blocking is the single biggest win.** `v2` (2×2) is roughly 1.5–2×
`v0` at every size. The surprise is that **4×4 is not better than 2×2**: at
N=512 the two are within noise, and at N=1024 `v3` leads the median by only
~14% while `v2` wins the best case by 1.5×. The cause is occupancy, not
arithmetic — this device reports `sharedMemPerBlock = 65536`, so `TS=64` needs
32 KB per workgroup (2 resident) where `TS=32` needs 8 KB (8 resident). The
extra register blocking buys fewer shared loads but pays for them in a much
smaller working set. **`TS=32` with a 2×2 per-thread quad is the right default
here**; the textbook 4×4 shape is not worth the shared-memory cost on an
integrated GPU.

### Reading these numbers: this device is very noisy

Every figure above is a **median**, and that is not incidental. On the Intel
(ARL) integrated GPU used here, the *same* kernel at the *same* size varies by
3–10× between consecutive runs — `v2` at N=1024 ranged from 113 to 618 GFLOPS
across 12 runs of one binary. This is a power- and thermally-shared iGPU, so
single-shot timings (and even "best of 3") are not meaningful; the medians are.
Anything reported in this README as a single run should be read as
order-of-magnitude only.

### What was actually bounding the kernel (now fixed)

Two costs used to live in the compiler rather than in the kernel source, and
both were larger than the tile-shape choice above. Both have since been fixed:

- **The MLIR backend ran no canonicalize / SROA / CSE pass.**
  `lib/Codegen/LoweringPasses.cpp` registered only the conversion passes
  (`ConvertFuncToSPIRV`, `ConvertGPUToSPIRV`, `SPIRVLowerABIAttributes`, …).
  The consequence was that **every local variable was a real memory access**:
  disassembling the SPIR-V for `matmul_bench.vc` showed 13
  `OpVariable ... Function` slots — the loop induction variables, the derived
  indices, and the accumulator array — each loaded and stored around the
  arithmetic. **Fixed** by a `PromoteScalarAllocasPass` + canonicalize + CSE
  inserted as stage 1b of `runLoweringPipeline`
  ([`lib/Codegen/LoweringPasses.cpp`](../lib/Codegen/LoweringPasses.cpp)).
  On `mmvar.vc` this promoted 43 scalar allocas to SSA (Function slots
  126→81, load/store 670→410) and lifted N=1024 `v2` from ~253 to ~348 median.
- **Every `for` lowered to `scf.while`,** with the induction variable in a
  `memref<i32, Function>` slot, and each guarded statement wrapped in an
  `scf.if`. **Fixed** by `tryEmitCanonicalFor` in
  [`lib/Codegen/ASTToMLIR.cpp`](../lib/Codegen/ASTToMLIR.cpp): a `for` of the
  shape `for (int i = start; i < bound; i = i + step)` with no `break`/
  `continue`/write-to-`i` in the body now lowers directly to `scf.for`, with
  `start` allowed to be any integer expression (not just a literal). This
  dropped `scf.while` 23→0, Function slots 81→9, load/store 410→87, and SPIR-V
  134K→59K; N=1024 `v2` went from ~212 to ~347 median.

Both fixes apply to **every** kernel, not just matmul. They are the reason the
GFLOPS figures in the table above are reproducible at all — earlier runs on the
unfixed pipeline were dominated by slot traffic.

### Split-K: does splitting K help on an iGPU? (no)

Split-K parallelises the K reduction: instead of one workgroup owning a whole
output tile and walking all of K, P workgroups collaborate on each tile, each
computing the partial sum over its `1/P` slice of K. The P partials are then
summed into C. This helps when **M and N are small but K is large** — the output
tile grid is too small to fill the device, so splitting K along its long axis
creates more parallelism.

The VC MLIR backend does not yet lower `float atomicAdd` (only i32), so the
prototype uses a **two-stage reduction**: each part writes its partial sum into
its own slot of a `partials[P][N*N]` buffer (no atomics), and a second small
kernel sums the P slots into C. Measured (median of 12, same `v2` tile shape
`TS=32`/`BLK=16`/2×2-per-thread, N=K square):

| N | `v2_reg2x2` | split P=1 | split P=4 | split P=16 |
|---|---|---|---|---|
| 256 | 40 | 66 | 60 | 41 |
| 512 | 79 | 123 | 137 | 95 |
| 1024 | 101 | 212 | 204 | 140 |

(GFLOPS, median; the device's 3–10× run-to-run noise means treat these as
order-of-magnitude.) Two takeaways:

- **Split-K does not help on this integrated GPU, and large P actively hurts.**
  P=16 loses ~30% at every size. The reason is that an iGPU has very few SMs
  (and Vulkan does not even report the count), so the output tile grid already
  fills the device at these N — splitting K does not raise occupancy, it only
  adds the `partials` memory traffic and a second kernel launch. The benefit
  case (small M×N, large K) does not arise for a square matmul at N≥256.
- **P=1 ≈ the plain kernel, as it should be.** The P=1 column is the same tile
  and the same 2×2 register blocking as `v2_reg2x2`, just routed through the
  split launcher (`tc = blockIdx.x % numTileCols`, `partIdx = 0`). Its
  higher-looking medians are run-to-run noise plus the warmup-then-time
  back-to-back pattern keeping the GPU clocked up — not a real win.

The two-stage reduction is the right design if Split-K is ever wanted on this
backend (it sidesteps the float-atomic limitation cleanly), but on this device
the conclusion is: **don't split K for square matmul; spend the parallelism on
a bigger output grid instead.** Stream-K (the load-balanced variant) is the
remaining unexplored option and is listed as a future step.


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
