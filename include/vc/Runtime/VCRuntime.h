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

/// Opaque handle to a stream event: a synchronization primitive recording a
/// point in a stream's execution. Events let one stream wait on work recorded
/// in another, expressing cross-stream dependencies without a full device
/// sync (CUDA cudaEvent / cudaStreamWaitEvent). Backed by a Vulkan timeline
/// semaphore so a single event can be recorded repeatedly (each record bumps
/// a monotonic counter). NULL is never a valid event handle.
struct VCEvent;
using VCEventHandle = VCEvent *;

/// Error codes mirroring cudaError_t style.
enum class VCError {
  Success = 0,
  InvalidValue,
  OutOfMemory,
  InvalidKernel,
  InvalidDevice,
  InitializationError,
  MapFailed,
  NotReady, // a non-blocking query found work still in flight
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

/// Asynchronous allocation on `stream` (NULL = default stream). The returned
/// handle is immediately usable for subsequent operations on the same stream
/// (which execute in stream order). The allocation itself is host-side and
/// immediate; the `stream` argument orders subsequent use, matching
/// cudaMallocAsync's contract.
VCError vcMallocAsync(void **devPtr, size_t bytes, VCStreamHandle stream);

/// Asynchronous host-visible (pinned) allocation on `stream`. See
/// vcMallocAsync.
VCError vcMallocHostAsync(void **hostPtr, size_t bytes, VCStreamHandle stream);

/// Asynchronous free on `stream` (NULL = default stream). Defers the actual
/// release until all work already submitted to `stream` has completed, so it
/// is safe to call while GPU work referencing `devPtr` is still in flight on
/// that stream (mirrors cudaFreeAsync — no use-after-free). The buffer is
/// reclaimed at the next vcStreamSynchronize / vcDeviceSynchronize / vcShutdown
/// once its stream's work is done.
VCError vcFreeAsync(void *devPtr, VCStreamHandle stream);

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

/// Fill the first `count` bytes of a device allocation with `value` (taken as
/// a byte, broadcast to every byte) — equivalent to cudaMemset. `count` MUST
/// be a multiple of 4 (Vulkan vkCmdFillBuffer constraint). Synchronous: blocks
/// until the fill completes. Safe on both device-local and host-visible
/// allocations.
VCError vcMemset(void *devPtr, int value, size_t count);

/// Like vcMemset but on an explicit `stream` (NULL = default stream).
VCError vcMemsetS(void *devPtr, int value, size_t count, VCStreamHandle stream);

/// Asynchronous fill on `stream` (NULL = default stream). Returns immediately;
/// the fill executes in stream order. The caller MUST vcStreamSynchronize
/// before reading the buffer. Equivalent to cudaMemsetAsync.
VCError vcMemsetAsyncS(void *devPtr, int value, size_t count,
                       VCStreamHandle stream);

/// Asynchronous fill on the default stream. Convenience wrapper.
VCError vcMemsetAsync(void *devPtr, int value, size_t count);

/// 2D pitched copy of a `width`x`height` byte region: row `r` of the source
/// (at offset `r*spitch`) is copied to row `r` of the destination (at offset
/// `r*dpitch`). Only DeviceToDevice is supported (both pointers from vcMalloc).
/// Synchronous on the default stream. Equivalent to cudaMemcpy2D (D2D).
VCError vcMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                   size_t width, size_t height, VCMemcpyKind kind);

/// Like vcMemcpy2D but on an explicit `stream` (NULL = default stream).
VCError vcMemcpy2DS(void *dst, size_t dpitch, const void *src, size_t spitch,
                    size_t width, size_t height, VCMemcpyKind kind,
                    VCStreamHandle stream);

/// Asynchronous 2D pitched copy on `stream` (NULL = default stream). Only
/// DeviceToDevice. Caller must vcStreamSynchronize before reading.
VCError vcMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                        size_t width, size_t height, VCMemcpyKind kind,
                        VCStreamHandle stream);

/// 2D pitched fill: `width`x`height` bytes of `dst` (row `r` at `r*pitch`)
/// set to `value` (byte, broadcast). `width` must be a multiple of 4.
/// Synchronous on the default stream. Equivalent to cudaMemset2D.
VCError vcMemset2D(void *dst, size_t pitch, int value, size_t width,
                   size_t height);

/// Asynchronous 2D pitched fill on `stream` (NULL = default stream). `width`
/// must be a multiple of 4. Caller must vcStreamSynchronize before reading.
VCError vcMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                        size_t height, VCStreamHandle stream);

/// Block until all queued device work is complete.
VCError vcDeviceSynchronize();

/// Create an execution stream. Commands on the same stream run in order.
VCError vcStreamCreate(VCStreamHandle *out);

/// Destroy a stream. Implicitly waits for pending work on it.
VCError vcStreamDestroy(VCStreamHandle stream);

/// Block until all work queued on `stream` is complete. NULL = default stream.
VCError vcStreamSynchronize(VCStreamHandle stream);

/// Non-blocking query: `done` is set to 1 if all work queued on `stream` has
/// completed, 0 otherwise. NULL = default stream. Equivalent to
/// cudaStreamQuery (returns Success with *done rather than a NotReady error).
VCError vcStreamQuery(VCStreamHandle stream, int *done);

/// Host callback type for vcLaunchHostFunc.
using VCHostFn = void (*)(void *userData);

/// Queue a host function `fn` to run on `stream` (NULL = default stream) after
/// all work already submitted to the stream completes. This is stream-ordered:
/// the callback runs once the GPU reaches this point, without the host polling.
/// Mirrors cudaLaunchHostFunc. The callback runs on an internal runtime thread;
/// it must not block or call vcShutdown. It MAY submit further work to streams.
VCError vcLaunchHostFunc(VCStreamHandle stream, VCHostFn fn, void *userData);

//----------------------------------------------------------------------------
// Stream events (cross-stream synchronization)
//----------------------------------------------------------------------------

/// Create a stream event. An event is a reusable marker backed by a timeline
/// semaphore: each vcEventRecord bumps its counter, and vcStreamWaitEvent
/// makes a stream block until the most recent record completes.
VCError vcEventCreate(VCEventHandle *out);

/// Destroy an event and its timeline semaphore.
VCError vcEventDestroy(VCEventHandle event);

/// Record `event` on `stream`: the event's counter is bumped and signaled at
/// the tail of the stream's NEXT submission (the next launch/copy on that
/// stream). Subsequent vcStreamWaitEvent calls on other streams will wait for
/// this point. Recording an event that already has a pending record on the
/// same stream chains after it (each record is a distinct counter value).
VCError vcEventRecord(VCEventHandle event, VCStreamHandle stream);

/// Make `stream` wait for `event`'s most recent recorded value before
/// executing its NEXT submission. This is the cross-stream dependency: work
/// on `stream` after this call does not start until the recorded work on the
/// event's stream has completed. Multiple waits accumulate. NULL stream =
/// default stream.
VCError vcStreamWaitEvent(VCStreamHandle stream, VCEventHandle event);

/// Query whether `event`'s most recent record has completed (GPU side).
/// `done` is set to 1 if complete, 0 otherwise. Non-blocking.
VCError vcEventQuery(VCEventHandle event, int *done);

/// Block the host until `event`'s most recent record has completed.
VCError vcEventSynchronize(VCEventHandle event);

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
