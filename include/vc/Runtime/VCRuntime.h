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

/// Opaque handle to an execution stream (an ordered command queue). Commands
/// issued to the same stream execute in order; different streams may execute
/// concurrently. NULL is the default stream.
struct VCStream;
using VCStreamHandle = VCStream *;

/// Opaque handle to a recorded command graph (CUDA-Graph-style "record once,
/// replay many"). A graph captures a sequence of kernel launches and
/// device-to-device copies issued while recording is active; vcGraphLaunch
/// replays the whole sequence with a single queue submit. NULL is never a
/// valid graph handle.
struct VCGraph;
using VCGraphHandle = VCGraph *;

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

/// Allocate `bytes` of device-local memory. Not host-accessible; use
/// vcMemcpy to move data in/out.
VCError vcMalloc(void **devPtr, size_t bytes);

/// Allocate `bytes` of host-visible (pinned) memory, persistently mapped.
/// Useful for staging buffers the host reads/writes directly.
VCError vcMallocHost(void **hostPtr, size_t bytes);

/// Free a device or host allocation.
VCError vcFree(void *devPtr);

/// Copy memory. `count` is bytes. D2D is asynchronous on the default stream;
/// H2D and D2H block until the copy completes so the host can read/write the
/// host side immediately (staging is transient). For non-blocking copies use
/// vcMemcpyAsync.
VCError vcMemcpy(void *dst, const void *src, size_t count, VCMemcpyKind kind);

/// Like vcMemcpy but on an explicit `stream` (NULL = default stream).
VCError vcMemcpyS(void *dst, const void *src, size_t count, VCMemcpyKind kind,
                  VCStreamHandle stream);

/// Asynchronous copy on an explicit `stream` (NULL = default stream). Returns
/// immediately without waiting for the GPU:
///   - H2D: the host source is snapshot into a staging buffer (owned by the
///     stream's frame) before returning, so the caller may overwrite `src`
///     immediately. The device copy executes in stream order.
///   - D2H: the device->staging copy is submitted to the stream, but the
///     final host memcpy (staging -> dst) is deferred until the copy finishes.
///     The caller MUST vcStreamSynchronize before reading `dst`.
///   - D2D: same as the synchronous path (already async).
/// Staging buffers live on the stream's current frame and are reclaimed when
/// that frame is recycled (after its fence signals), so they never block.
VCError vcMemcpyAsyncS(void *dst, const void *src, size_t count,
                       VCMemcpyKind kind, VCStreamHandle stream);

/// Asynchronous copy on the default stream. Convenience wrapper.
VCError vcMemcpyAsync(void *dst, const void *src, size_t count,
                      VCMemcpyKind kind);

/// Block until all queued device work is complete.
VCError vcDeviceSynchronize();

/// Create an execution stream. Commands on the same stream run in order.
VCError vcStreamCreate(VCStreamHandle *out);

/// Destroy a stream. Implicitly waits for pending work on it.
VCError vcStreamDestroy(VCStreamHandle stream);

/// Block until all work queued on `stream` is complete. NULL = default stream.
VCError vcStreamSynchronize(VCStreamHandle stream);

//----------------------------------------------------------------------------
// Command graphs (CUDA-Graph-style record/replay)
//----------------------------------------------------------------------------

/// Create a new empty command graph. The graph owns its own command pool,
/// descriptor pool, and (after recording) a secondary command buffer, all
/// released by vcGraphDestroy.
VCError vcGraphCreate(VCGraphHandle *out);

/// Destroy a graph and all resources it owns. The buffers referenced by
/// launches recorded into the graph must outlive it (the graph does not
/// retain them). Safe to call on an un-recorded graph.
VCError vcGraphDestroy(VCGraphHandle graph);

/// Begin recording on the default stream. While recording is active, every
/// vcLaunchKernelS / vcMemcpyS targeting the default stream is appended to
/// the graph instead of submitted. Kernel launches and device-to-device
/// copies are recorded directly; host-to-device copies allocate a persistent
/// staging buffer owned by the graph. Must be paired with vcGraphEndRecord.
VCError vcGraphBeginRecord(VCGraphHandle graph);

/// End recording. Finalizes the graph's recorded command buffer so it is
/// ready for vcGraphLaunch. Recording must currently be active on this graph.
VCError vcGraphEndRecord(VCGraphHandle graph);

/// Replay the recorded graph on `stream` (NULL = default stream) as a single
/// asynchronous queue submit. The graph must have finished recording.
VCError vcGraphLaunch(VCGraphHandle graph, VCStreamHandle stream);

/// Reset a graph to empty so it can be re-recorded. Destroys the recorded
/// command buffer, descriptor sets, and staging buffers but keeps the graph
/// handle valid for a fresh vcGraphBeginRecord.
VCError vcGraphReset(VCGraphHandle graph);

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
/// counts; the workgroup count = ceil(gridDim/blockDim). Asynchronous on the
/// default stream.
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

/// Asynchronous 1D launch on an explicit `stream` (NULL = default stream).
VCError vcLaunchKernelS(VCKernelHandle kernel, unsigned gridDim,
                        unsigned blockDim, const VCKernelArg *args,
                        int argCount, VCStreamHandle stream);

/// Asynchronous 2D launch on an explicit `stream` (NULL = default stream).
VCError vcLaunchKernel2DS(VCKernelHandle kernel, unsigned gridDimX,
                          unsigned gridDimY, unsigned blockDimX,
                          unsigned blockDimY, const VCKernelArg *args,
                          int argCount, VCStreamHandle stream);

/// Human-readable string for an error code.
const char *vcErrorString(VCError err);

} // namespace vc

#endif // VC_RUNTIME_VCRUNTIME_H
