# VC Language — Draft Specification (scaffold)

> Status: scaffold. This documents the intended surface; the compiler
> currently implements only the vector-add subset. Extend alongside the
> implementation.

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

## Built-in variables

| Variable | VC op | SPIR-V builtin |
|----------|-------|----------------|
| `threadIdx.{x,y,z}` | `vc.thread_id` | `LocalInvocationId` |
| `blockIdx.{x,y,z}` | `vc.block_id` | `WorkgroupId` |
| `blockDim.{x,y,z}` | `vc.block_dim` | `WorkgroupSize` |
| `gridDim.{x,y,z}` | `vc.grid_dim` | `NumWorkgroups` |

## Types
`void bool int unsigned uint long float double`; pointers; (planned)
`struct`, templates.

## Launch syntax
```
kernel<<<grid, block>>>(args...)
```
maps to `vkCmdDispatch` with workgroup count = `ceil(grid/block)`.

## Hardware features (deferred)
- `wmma::*` -> `spirv.KHR.CooperativeMatrix` (Tensor Core). Placeholder ops
  `vc.wmma.load/store/mma` exist in the dialect; lowering TBD.
- `__shfl_*`, atomics, `__ballot` — TBD.

## Host API
See `include/vc/Runtime/VCRuntime.h` (`vcMalloc`, `vcFree`, `vcMemcpy`,
`vcLaunchKernel`, `vcDeviceSynchronize`, ...).
