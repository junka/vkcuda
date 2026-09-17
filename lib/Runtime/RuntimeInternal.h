//===- RuntimeInternal.h - Internal Vulkan runtime state -----------------===//

#ifndef VC_RUNTIME_INTERNAL_H
#define VC_RUNTIME_INTERNAL_H

#include "vc/Runtime/VCRuntime.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vc {

struct VCStream; // forward: VulkanDevice owns a per-device default stream

// Process-global toggle for kernel-internal printf (debugPrintfEXT). Set by
// vcEnableKernelPrintf OR the VC_KERNEL_PRINTF=1 env var BEFORE vcInit; read
// once during init() to decide whether to enable the validation layer, the
// VK_KHR_shader_non_semantic_info device extension, and a debug messenger that
// captures NonSemantic.DebugPrintf output. Lives at namespace scope (not on the
// Runtime singleton) because vcEnableKernelPrintf may be called before the
// singleton is constructed.
extern bool g_kernelPrintfEnabled;


/// All long-lived Vulkan handles for one logical device + compute queue.
struct VulkanDevice {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t computeQueueFamily = 0;
  VkQueue computeQueue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE; // backs the default stream
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  bool ownsDevice = true; // false for secondary entries sharing a device group
  bool deviceGroup = false;
  uint32_t groupId = UINT32_MAX;
  uint32_t groupLocalIndex = 0;
  uint32_t groupSize = 1;
  uint32_t groupDeviceMask = 1;
  VkPhysicalDeviceMemoryProperties memProps{};
  VkPhysicalDeviceProperties physProps{}; // device name, limits, timestampPeriod
  int subgroupSize = 1;          // queried via VkPhysicalDeviceSubgroupProperties
  bool timestampAvailable = false; // limits.timestampComputeAndGraphics
  bool headless = true; // no surface/swapchain
  bool timelineSemaphore = false; // VK_KHR_timeline_semaphore / Vulkan 1.2 core
  bool coopMatrix = false;       // VK_KHR_cooperative_matrix + shaderFloat16
  bool f16Storage = false;       // shaderFloat16 + shaderStorageBuffer16BitAccess
                                 // (scalar __half SSBO load/store)
  // Per-device default stream. CUDA gives each device its own default stream;
  // vcSetDevice(i) makes resolveStream(NULL) return devices_[i]->defaultStream.
  // unique_ptr because VCStream is forward-declared at this point.
  std::unique_ptr<VCStream> defaultStream;
};

/// A device buffer + its backing memory. May be device-local (mapped==nullptr)
/// or host-visible (mapped!=nullptr, persistently mapped). `managed` marks a
/// unified-memory allocation (vcMallocManaged): device storage persistently
/// mapped + host-coherent, so host and device share the same payload with no
/// vcMemcpy.
struct VCBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  size_t size = 0;
  void *mapped = nullptr; // non-null when persistently mapped (host-visible)
  bool hostVisible = false;
  bool managed = false; // unified memory (vcMallocManaged)
  int deviceIdx = 0; // which device owns this allocation (for free/peer copy)
  uint32_t memoryHeapIndex = UINT32_MAX;
};

/// A frame in a stream's command-buffer ring. One command buffer + the fence
/// that signals when the GPU has finished executing it. Frames are recycled:
/// beginFrame waits on the fence of the frame it is about to reuse, then
/// resets the command buffer.
struct StreamFrame {
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  // Descriptor sets allocated on this frame's pool; all freed together when
  // the frame is reset via vkResetDescriptorPool.
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  // Staging buffers owned by this frame for async H2D/D2H copies. They must
  // outlive the GPU work recorded into this frame, so they live here and are
  // reclaimed (after the fence signals) when the frame is recycled in
  // beginFrame. This is what makes vcMemcpyAsync non-blocking: the staging
  // is not freed in the copy call, it rides the frame's lifetime.
  std::vector<std::unique_ptr<VCBuffer>> stagingBuffers;

  // Deferred D2H readbacks recorded on this frame. Each entry is a
  // device->staging copy already submitted in this frame's command buffer;
  // the final staging->host memcpy can only run once that copy has executed
  // (i.e. once the fence signals). beginFrame drains this list before
  // recycling the frame, delivering the data to the caller's host pointer.
  // The caller must have synchronized (vcStreamSynchronize) before reading.
  struct D2HReadback { void *hostDst; VCBuffer *staging; size_t bytes; };
  std::vector<D2HReadback> d2hReadbacks;
};

/// An execution stream: an ordered command queue. Backed by the device's
/// compute queue (all streams share it) but with its own command pool, a
/// ring of frames, and a per-frame descriptor pool. Submitting does NOT wait
/// for the GPU — commands queue up and execute asynchronously.
struct VCStream {
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  int deviceIdx = 0; // which device this stream's queue belongs to
  VkCommandPool commandPool = VK_NULL_HANDLE;
  std::vector<StreamFrame> frames;
  size_t frameIdx = 0; // next frame to record into
  // Reusable host-visible staging buffers. Completed frame staging buffers move
  // here instead of immediately destroying VkBuffer/VkDeviceMemory, reducing
  // allocation churn for repeated async H2D/D2H copies.
  std::vector<std::unique_ptr<VCBuffer>> stagingCache;
  size_t stagingCacheBytes = 0;
  // When non-null, the stream is in graph-capture mode: launches and copies
  // are appended to this graph's secondary command buffer instead of being
  // submitted. Set by vcGraphBeginRecord on the default stream.
  struct VCGraph *captureTarget = nullptr;

  // Pending timeline-semaphore operations to attach to the NEXT endFrame
  // submission on this stream. vcStreamWaitEvent pushes a wait here;
  // vcEventRecord pushes a signal. endFrame drains both into the VkSubmitInfo
  // (via VkTimelineSemaphoreSubmitInfo) and clears them. This lets events
  // express cross-stream dependencies without forcing an extra submit.
  struct PendingWait { VkSemaphore sem; uint64_t value; };
  struct PendingSignal { VkSemaphore sem; uint64_t value; };
  std::vector<PendingWait> pendingWaits;
  std::vector<PendingSignal> pendingSignals;
};

/// A loaded kernel: shader module + descriptor/pipeline layouts + a cache of
/// already-built pipelines keyed by (block dims, scalar-arg layout).
/// Per-device Vulkan objects for a kernel. A VCKernel can be launched on any
/// device: each device it touches lazily gets its own shader module, layout,
/// and pipeline cache (all VkHandles are device-local and cannot be shared
/// across VkDevices). The SPIR-V words themselves are device-independent and
/// held once on the VCKernel.
struct VCKernelDeviceState {
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  bool layoutBuilt = false;
  // Cache: key = packed (blockX, blockY, blockZ) -> pipeline. Pipelines are
  // specialized on workgroup size, so one per distinct block shape.
  std::unordered_map<uint64_t, VkPipeline> pipelines;
};

struct VCKernel {
  std::vector<uint32_t> spirvWords; // device-independent SPIR-V
  std::string entryPoint;
  int argCount = 0;
  // Reflected SSBO bindings in API pointer-argument order. Reflection keeps the
  // runtime ABI tied to the shader's actual DescriptorSet/Binding decorations
  // instead of assuming bindings are dense from zero.
  std::vector<uint32_t> storageBufferBindings;
  bool hasResourceReflection = false;
  // Push-constant range (offset/size) for scalar args, in bytes. size==0
  // means no push constants. Layout is built once per device and cached there.
  uint32_t pcOffset = 0;
  uint32_t pcSize = 0;
  // Reflected push-constant member offsets in API scalar-argument order. Empty
  // means fall back to the legacy tightly-packed scalar ABI.
  std::vector<uint32_t> pushConstantOffsets;
  // Per-device state, lazily created on first launch on that device.
  std::unordered_map<int, std::unique_ptr<VCKernelDeviceState>> perDevice;
};

/// A recorded command graph (CUDA-Graph-style record/replay). While
/// recording is active on the default stream, launches and D2D copies are
/// appended to `secondaryCB` and descriptor sets are allocated from
/// `descriptorPool` (both owned by the graph, surviving across replays).
/// Host-to-device copies recorded into the graph allocate persistent staging
/// buffers owned by the graph (transient staging would be freed before the
/// graph ever executes). The graph is replayed by executing `secondaryCB`
/// inside a primary command buffer on the target stream.
struct VCGraph {
  VkCommandPool commandPool = VK_NULL_HANDLE;       // backs secondaryCB
  VkCommandBuffer secondaryCB = VK_NULL_HANDLE;     // the recorded commands
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE; // graph-lifetime sets
  bool recording = false;
  bool recorded = false;
  int deviceIdx = 0; // which device this graph was recorded on
  // Persistent staging for H2D (and D2H) copies recorded into the graph.
  // These must outlive the graph's replays, so they live here, not as
  // transient scratch.
  std::vector<std::unique_ptr<VCBuffer>> stagingBuffers;
  // D2H copies recorded into the graph: device->staging copy is in the
  // secondary command buffer, and the host readback (staging.mapped ->
  // hostDst) must happen after the graph executes + the stream is synced.
  // vcGraphLaunch does NOT auto-readback (it is async); the caller syncs
  // then calls vcGraphReadback.
  struct D2HReadback { void *hostDst; VCBuffer *staging; size_t bytes; };
  std::vector<D2HReadback> d2hReadbacks;
};

/// A stream event: a reusable synchronization marker backed by a Vulkan
/// timeline semaphore. Each vcEventRecord bumps `value` and arranges for the
/// recording stream's next submission to signal the semaphore at that value;
/// vcStreamWaitEvent arranges for the waiting stream's next submission to
/// wait on that value. `value` is the counter to be signaled by the next
/// record (0 = never recorded); `lastRecorded` is the value most recently
/// recorded, used for waits/queries.
struct VCEvent {
  VkSemaphore semaphore = VK_NULL_HANDLE;
  uint64_t value = 1;          // next counter to signal on record
  uint64_t lastRecorded = 0;   // counter of the most recent record (0 = none)
  int deviceIdx = 0;           // which device this event's semaphore lives on
  // GPU timestamp query for vcEventElapsedTime. One-slot query pool written by
  // vkCmdWriteTimestamp at record time; the counter is read back (with WAIT) by
  // eventElapsedTime. VC events always carry timing (no disable-timing flag).
  VkQueryPool queryPool = VK_NULL_HANDLE;
};

class Runtime {
public:
  static Runtime &get();

  // Safety net: if the program exits without calling vcShutdown, the Meyers
  // singleton is destroyed and the background host-func thread would be joinable
  // -> std::terminate. The destructor signals stop + joins so that path is safe.
  ~Runtime();

  VCError init();
  VCError shutdown();
  bool initialized() const { return !devices_.empty(); }

  // The current device (set by vcSetDevice). Most host operations target it;
  // operations that take a stream/buffer use that object's own deviceIdx
  // instead so they stay correct after a device switch.
  VulkanDevice &device() { return *devices_[currentDeviceIdx_]; }
  const VulkanDevice &device() const { return *devices_[currentDeviceIdx_]; }
  size_t deviceCount() const { return devices_.size(); }
  int currentDevice() const { return currentDeviceIdx_; }
  VCError setDevice(int idx);
  VulkanDevice &deviceAt(int idx) { return *devices_[idx]; }

  // Resolve a public stream handle to the internal stream; NULL -> the
  // current device's default stream.
  VCStream &resolveStream(VCStreamHandle h);

  VCError createStream(VCStreamHandle *out);
  VCError destroyStream(VCStreamHandle stream);
  VCError streamSynchronize(VCStream &s);
  // Non-blocking: *done = 1 if all the stream's frame fences are signaled.
  VCError streamQuery(const VCStream &s, int *done) const;
  // Stream-ordered host callback: signal a one-shot timeline semaphore at the
  // stream's current tail (empty submit), record {sem, value, fn, userData}
  // for the background thread to dispatch once the semaphore signals.
  VCError launchHostFunc(VCStream &s, VCHostFn fn, void *userData);

  // Frame lifecycle on a stream.
  // beginFrame: wait for the next frame's previous submission, reset its
  //             command buffer + descriptor pool, begin recording. Returns
  //             the command buffer to record into.
  VkCommandBuffer beginFrame(VCStream &s);
  // endFrame: end recording, submit to the stream's queue with the frame's
  //           fence. Does NOT wait.
  void endFrame(VCStream &s);

  // Allocate a descriptor set on the current frame's pool for the kernel's
  // set layout. The set is reclaimed when the frame is reset.
  VkDescriptorSet allocFrameDescriptorSet(VCStream &s,
                                          VkDescriptorSetLayout layout);

  VCError mallocBuffer(size_t bytes, VCBuffer &out);          // device-local
  VCError mallocHostBuffer(size_t bytes, VCBuffer &out);      // host-visible
  VCError mallocHostBufferOn(int deviceIdx, size_t bytes,
                             VCBuffer &out); // host-visible on a specific device
  VCError mallocManagedBuffer(size_t bytes, VCBuffer &out); // unified (shared ptr)
  VCError freeBuffer(VCBuffer &buf);
  // Fill `bytes` of `buf` (offset 0) with byte `value` broadcast to uint32.
  // bytes must be a multiple of 4 (vkCmdFillBuffer constraint). Sync blocks;
  // Async records and returns (caller syncs before reading). Staging-free:
  // the buffer has TRANSFER_DST usage, so vkCmdFillBuffer writes it directly.
  VCError memsetBuffer(VCBuffer &buf, int value, size_t bytes,
                       VCStream &s);                          // sync
  VCError memsetBufferAsync(VCBuffer &buf, int value, size_t bytes,
                            VCStream &s);                     // async
  // Deferred free: the buffer is reclaimed once all work already submitted to
  // `s` has completed (its frame fences signal). Safe to call while GPU work
  // referencing the buffer is still in flight on `s` (cudaFreeAsync semantics).
  VCError freeBufferAsync(VCBuffer *buf, VCStream &s);
  // 2D pitched D2D copy: height rows of `width` bytes, src stride spitch, dst
  // stride dpitch. One submit with height VkBufferCopy regions (no kernel).
  VCError copy2D(VCBuffer &dst, size_t dpitch, const VCBuffer &src,
                 size_t spitch, size_t width, size_t height,
                 VCStream &s);                          // sync
  VCError copy2DAsync(VCBuffer &dst, size_t dpitch, const VCBuffer &src,
                      size_t spitch, size_t width, size_t height,
                      VCStream &s);                     // async
  // 2D pitched fill: height rows of `width` bytes at stride `pitch`. width
  // must be a multiple of 4. height vkCmdFillBuffer calls, one per row.
  VCError memset2DBuffer(VCBuffer &buf, size_t pitch, int value, size_t width,
                         size_t height, VCStream &s);   // sync
  VCError memset2DBufferAsync(VCBuffer &buf, size_t pitch, int value,
                              size_t width, size_t height,
                              VCStream &s);              // async
  // Copy primitives, all recorded on stream `s`:
  VCError copyDeviceToDevice(VCBuffer &dst, const VCBuffer &src, size_t bytes,
                             VCStream &s);                 // async
  VCError copyHostToDevice(VCBuffer &dst, const void *hostSrc, size_t bytes,
                           VCStream &s);                   // sync (staging freed)
  VCError copyDeviceToHost(void *hostDst, const VCBuffer &src, size_t bytes,
                           VCStream &s);                   // sync (readable on return)

  // Async variants: staging is owned by the stream's current frame and
  // reclaimed when that frame is recycled (after its fence signals). H2D
  // snapshots the host source into staging before returning; D2H defers the
  // staging->host memcpy to frame recycle time (caller must sync first).
  VCError copyHostToDeviceAsync(VCBuffer &dst, const void *hostSrc,
                                size_t bytes, VCStream &s);  // async
  VCError copyDeviceToHostAsync(void *hostDst, const VCBuffer &src,
                                size_t bytes, VCStream &s);  // async (sync to read)

  VCError loadKernel(const uint32_t *words, size_t wordCount,
                     const char *entryPoint, VCKernel &out);
  VCError loadKernelFromFile(const char *path, const char *entryPoint,
                             VCKernel &out);
  void releaseKernel(VCKernel &k);
  // Lazily create the per-device shader module for `k` on `deviceIdx`.
  VCKernelDeviceState *getOrCreateKernelDeviceState(VCKernel &k, int deviceIdx);

  // Build (or fetch cached) the compute pipeline specialized to the block
  // size, on `deviceIdx`. Lazily builds the descriptor/pipeline layout on
  // first call using the scalar/pointer arrangement of `args`.
  VkPipeline getPipeline(VCKernel &k, int deviceIdx, unsigned blockX,
                         unsigned blockY, unsigned blockZ,
                         const VCKernelArg *args, int argCount);

  // Core launch: record bind+dispatch into the stream's current frame.
  VCError dispatch(VCKernel &k, unsigned wgX, unsigned wgY, unsigned wgZ,
                   unsigned blockX, unsigned blockY, unsigned blockZ,
                   const VCKernelArg *args, int argCount, VCStream &s);

  // Record bind+dispatch into a specific command buffer using a specific
  // descriptor pool (used by graph capture, which records into a secondary
  // command buffer instead of submitting).
  VCError recordDispatchInto(VkCommandBuffer cb, VkDescriptorPool dpool,
                              VCKernel &k, int deviceIdx, unsigned wgX,
                              unsigned wgY, unsigned wgZ, unsigned blockX,
                              unsigned blockY, unsigned blockZ,
                              const VCKernelArg *args, int argCount);

  // Shared bind sequence for direct + indirect dispatch: get/create the
  // pipeline (specialized to blockX/Y/Z), allocate + write the descriptor
  // set for pointer args, pack scalars into push constants, and record
  // vkCmdBindPipeline + vkCmdBindDescriptorSets + vkCmdPushConstants into
  // `cb`. Returns the bound pipeline (VK_NULL_HANDLE on failure). Does NOT
  // dispatch — the caller records vkCmdDispatch(Indirect).
  VkPipeline bindKernelForDispatch(VkCommandBuffer cb, VkDescriptorPool dpool,
                                   VCKernel &k, int deviceIdx, unsigned blockX,
                                   unsigned blockY, unsigned blockZ,
                                   const VCKernelArg *args, int argCount);

  // Indirect dispatch: same bind sequence as recordDispatchInto, but the
  // workgroup counts {x,y,z} are read from `indirectArgs` at `offset`
  // (a VkDispatchIndirectCommand) via vkCmdDispatchIndirect. The grid is
  // device-driven — no host round-trip. `indirectArgs` must be on the same
  // device as the stream/graph (`deviceIdx`).
  VCError recordDispatchIndirectInto(VkCommandBuffer cb, VkDescriptorPool dpool,
                                     VCKernel &k, int deviceIdx,
                                     const VCBuffer &indirectArgs,
                                     size_t offset, unsigned blockX,
                                     unsigned blockY, unsigned blockZ,
                                     const VCKernelArg *args, int argCount);
  VCError dispatchIndirect(VCKernel &k, const VCBuffer &indirectArgs,
                           size_t offset, unsigned blockX, unsigned blockY,
                           unsigned blockZ, const VCKernelArg *args,
                           int argCount, VCStream &s);

  VCError synchronize();

  // ---- Command graphs (record/replay) ----
  VCError createGraph(VCGraphHandle *out);
  VCError destroyGraph(VCGraphHandle graph);
  VCError beginRecord(VCGraph &g);
  VCError endRecord(VCGraph &g);
  VCError launchGraph(VCGraph &g, VCStream &s);
  VCError resetGraph(VCGraph &g);

  // ---- Stream events (timeline semaphore) ----
  VCError createEvent(VCEventHandle *out);
  VCError destroyEvent(VCEventHandle event);
  VCError recordEvent(VCEvent &e, VCStream &s);
  VCError streamWaitEvent(VCStream &s, VCEvent &e);
  VCError eventQuery(const VCEvent &e, int *done) const;
  VCError eventSynchronize(const VCEvent &e) const;
  // GPU-side elapsed time between two recorded events (ms). Reads back each
  // event's timestamp query (blocking) and converts ticks→ms via timestampPeriod.
  VCError eventElapsedTime(float *ms, const VCEvent &start,
                           const VCEvent &end) const;

  // ---- Device / pointer queries ----
  VCError getDeviceProperties(VCDeviceProperties *out, int device) const;
  // Limit-based block-size heuristic for vcOccupancyMaxPotentialBlockSize.
  // Not real SM occupancy (Vulkan exposes no SM count / per-kernel resources).
  VCError occupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                         size_t dynamicSharedMemPerBlock,
                                         int blockSizeLimit) const;
  VCError pointerGetAttributes(VCPointerAttributes *out,
                               const void *ptr) const;
  // Register/unregister a buffer for vcPointerGetAttributes. Called by the
  // vcMalloc/MallocHost/MallocManaged/Async C wrappers so the registry tracks
  // every live allocation and its kind (Device / Host / Managed).
  void registerBuffer(VCBuffer *b, VCMemoryType kind);
  void unregisterBuffer(VCBuffer *b);

  // Resolve a host pointer used as a vcMemcpy H2D source or D2H destination.
  // If `hostPtr` is a registered host-visible/managed VCBuffer (from
  // vcMallocHost / vcMallocManaged), the caller passed the *handle*
  // (VCBuffer*), not the mapped payload address — reading/writing `bytes`
  // there would read/clobber the struct. Return its persistently mapped host
  // pointer instead. Otherwise `hostPtr` is a plain host pointer (stack/heap
  // array) and is returned unchanged. Used by both copyHostToDevice (read
  // source) and copyDeviceToHost (write target) so the same handle/payload
  // distinction applies symmetrically.
  void *resolveHostAccess(void *hostPtr) const;

  // ---- Cross-device copy (device-group P2P when available, host fallback) ----
  // Copies `bytes` from src buffer (on srcDevice) to dst buffer (on dstDevice).
  // Same-device falls through to the normal D2D path. Cross-device routes
  // through a VK_KHR_device_group/Vulkan 1.1 peer copy when both buffers belong
  // to the same logical device group and the source heap reports COPY_SRC peer
  // access from the destination device. Otherwise it falls back to host staging.
  VCError copyPeer(VCBuffer &dst, int dstDevice, const VCBuffer &src,
                   int srcDevice, size_t bytes);

private:
  // All enumerated Vulkan devices; currentDeviceIdx_ selects the active one
  // (CUDA cudaSetDevice model). Each VulkanDevice owns its own default stream.
  std::vector<std::unique_ptr<VulkanDevice>> devices_;
  int currentDeviceIdx_ = 0;
  std::vector<std::unique_ptr<VCStream>> streams_; // owns created streams
  std::vector<std::unique_ptr<VCGraph>> graphs_;   // owns created graphs
  std::vector<std::unique_ptr<VCEvent>> events_;   // owns created events
  bool init_ = false;

  // Every VCKernel the runtime has built per-device state for (shader module +
  // pipelines + layouts). VCKernelHandle is a raw pointer the caller owns (the
  // `__vc_k_<name>` local in host main), so the runtime does NOT own the
  // VCKernel struct — but it DOES own the Vulkan handles inside it. We track
  // the pointers here so shutdown() can releaseKernel() them all before
  // destroying the devices, instead of leaking VkShaderModule/Pipeline/Layout.
  // Deduped: a kernel launched on multiple devices registers once.
  std::vector<VCKernel*> kernelRegistry_;
  void trackKernel(VCKernel *k);

  // Debug messenger for the validation layer (kernel printf). Created in init()
  // when g_kernelPrintfEnabled is set; destroyed at the top of shutdown() before
  // the instance is torn down. The single shared instance lives on devices_[0].
  VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

  // Registry of live allocations for vcPointerGetAttributes. Maps the buffer
  // handle (the void* VC hands out) to its kind. Mutex-guarded because allocs
  // and frees can happen from the host-func background thread (callbacks may
  // re-enter the runtime).
  std::unordered_map<VCBuffer*, VCMemoryType> allocRegistry_; // value: kind
  mutable std::mutex allocRegistryMu_;

  // Buffers whose release was deferred by vcFreeAsync. Each is freed once all
  // work on its stream has completed (frame fences signal). Drained from
  // streamSynchronize / synchronize / shutdown.
  struct PendingFree { VCBuffer *buf; VCStream *stream; };
  std::vector<PendingFree> pendingFrees_;
  void drainPendingFrees();

  // Stream-ordered host callbacks (vcLaunchHostFunc). Each entry's semaphore
  // is signaled at the stream's tail when the preceding work completes; the
  // background thread polls the counter and invokes the callback (unlocked)
  // once it reaches `value`, then destroys the semaphore.
  struct PendingHostFunc {
    VkSemaphore sem;
    uint64_t value;
    VCHostFn fn;
    void *userData;
    int deviceIdx = 0; // device the semaphore was created on (must query/destroy there)
  };
  std::vector<PendingHostFunc> pendingHostFuncs_;
  std::mutex hostFuncMu_;
  std::atomic<bool> hostFuncStop_{false};
  std::thread hostFuncThread_;
  void hostFuncLoop(); // background thread entry: poll + dispatch callbacks
  // Dispatch any pending host callbacks whose timeline semaphore has signaled.
  // Called by streamSynchronize (after the stream's fences are done) so that
  // stream-ordered callbacks have run by the time sync returns. Moves entries
  // out of the shared list under hostFuncMu_ so the background thread cannot
  // double-dispatch them.
  void drainHostFuncs();

  // Enumerate all physical devices with a compute queue and build a
  // VulkanDevice (logical device + default stream + pipeline cache) for each.
  bool enumerateDevices(VkInstance instance);
  // Create the logical device + compute queue for one VulkanDevice. When
  // `deviceGroupMembers` has more than one physical device, the logical device
  // is created as a Vulkan device group rooted at `vd.physical`.
  bool setupLogicalDevice(VulkanDevice &vd,
                          const std::vector<VkPhysicalDevice> &deviceGroupMembers = {});
  bool buildLayout(VCKernel &k, int deviceIdx, const VCKernelArg *args,
                    int argCount);
  // Find a memory type index satisfying `flags` among `reqBits`.
  uint32_t findMemoryType(uint32_t reqBits,
                          VkMemoryPropertyFlags flags) const;
  static uint32_t findMemoryTypeOn(const VulkanDevice &vd, uint32_t reqBits,
                                   VkMemoryPropertyFlags flags);
};

} // namespace vc

#endif // VC_RUNTIME_INTERNAL_H
