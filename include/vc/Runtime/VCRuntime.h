//===- VCRuntime.h - CUDA-style host API over Vulkan ----------------------===//
//
// The host-facing runtime. Mirrors a subset of the CUDA Runtime API
// (vcMalloc / vcFree / vcMemcpy / vcLaunchKernel / vcDeviceSynchronize)
// but is implemented entirely on top of Vulkan compute.
//
// Design: kernels are precompiled to SPIR-V (.spv); the host loads a .spv,
// binds the device buffers as descriptor set resources, and dispatches via
// vkCmdDispatch. The <<<grid,block>>> launch grid is translated to the
// workgroup count / workgroup size.
//
//===----------------------------------------------------------------------===//

#ifndef VC_RUNTIME_VCRUNTIME_H
#define VC_RUNTIME_VCRUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vc {

/// Opaque handle to a device allocation.
struct VCBuffer;
using VCBufferHandle = VCBuffer *;

/// Opaque handle to a loaded kernel (SPIR-V shader module + pipeline).
struct VCKernel;
using VCKernelHandle = VCKernel *;

/// Error codes mirroring cudaError_t style.
enum class VCError {
  Success = 0,
  InvalidValue,
  OutOfMemory,
  InvalidKernel,
  InvalidDevice,
  InitializationError,
  MapFailed,
  Unknown,
};

/// Memory transfer direction.
enum class VCMemcpyKind {
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
};

/// Initialize the runtime (creates the Vulkan instance/device/queue).
/// Returns Success on first or repeated calls.
VCError vcInit();

/// Tear down the runtime and release all device resources.
VCError vcShutdown();

/// Query the number of available Vulkan devices.
VCError vcGetDeviceCount(int *count);

/// Allocate `bytes` of device-local, host-visible memory.
VCError vcMalloc(void **devPtr, size_t bytes);

/// Free a device allocation.
VCError vcFree(void *devPtr);

/// Copy memory. `count` is bytes.
VCError vcMemcpy(void *dst, const void *src, size_t count, VCMemcpyKind kind);

/// Block until all queued device work is complete.
VCError vcDeviceSynchronize();

/// Load a SPIR-V binary (already assembled into a .spv blob) as a kernel.
/// `entryPoint` names the spir-v entry function (default "main").
VCError vcLoadKernel(const uint32_t *spirvWords, size_t wordCount,
                     const char *entryPoint, VCKernelHandle *outKernel);

/// Load a kernel from a .spv file on disk.
VCError vcLoadKernelFromFile(const char *path, const char *entryPoint,
                             VCKernelHandle *outKernel);

/// Release a loaded kernel.
VCError vcReleaseKernel(VCKernelHandle kernel);

/// Argument passed to a kernel launch: a device pointer (from vcMalloc) or
/// an immediate scalar. Scalars are copied into a uniform/SSBO binding.
struct VCKernelArg {
  enum Kind { Pointer, Scalar } kind = Pointer;
  const void *data = nullptr; // device ptr, or pointer to scalar bytes
  size_t size = 0;            // bytes (for scalar; ignored for Pointer)
};

/// Launch a kernel with a 1D grid/block. `gridDim`/`blockDim` are element
/// counts; the workgroup count = ceil(gridDim/blockDim).
VCError vcLaunchKernel(VCKernelHandle kernel, unsigned gridDim,
                       unsigned blockDim, const VCKernelArg *args,
                       int argCount);

/// Launch a kernel with a 2D grid/block. `gridDimX/Y` and `blockDimX/Y` are
/// element counts; workgroup counts are ceil(grid/block) per axis. Use this
/// for kernels that read threadIdx.y / blockIdx.y.
VCError vcLaunchKernel2D(VCKernelHandle kernel, unsigned gridDimX,
                         unsigned gridDimY, unsigned blockDimX,
                         unsigned blockDimY, const VCKernelArg *args,
                         int argCount);

/// Human-readable string for an error code.
const char *vcErrorString(VCError err);

} // namespace vc

#endif // VC_RUNTIME_VCRUNTIME_H
