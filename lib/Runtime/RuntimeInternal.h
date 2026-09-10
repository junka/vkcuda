//===- RuntimeInternal.h - Internal Vulkan runtime state -----------------===//

#ifndef VC_RUNTIME_INTERNAL_H
#define VC_RUNTIME_INTERNAL_H

#include "vc/Runtime/VCRuntime.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vc {

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
  VkPhysicalDeviceMemoryProperties memProps{};
  bool headless = true; // no surface/swapchain
};

/// A device buffer + its backing memory. May be device-local (mapped==nullptr)
/// or host-visible (mapped!=nullptr, persistently mapped).
struct VCBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  size_t size = 0;
  void *mapped = nullptr; // non-null when persistently mapped (host-visible)
  bool hostVisible = false;
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
  VkCommandPool commandPool = VK_NULL_HANDLE;
  std::vector<StreamFrame> frames;
  size_t frameIdx = 0; // next frame to record into
  // When non-null, the stream is in graph-capture mode: launches and copies
  // are appended to this graph's secondary command buffer instead of being
  // submitted. Set by vcGraphBeginRecord on the default stream.
  struct VCGraph *captureTarget = nullptr;
};

/// A loaded kernel: shader module + descriptor/pipeline layouts + a cache of
/// already-built pipelines keyed by (block dims, scalar-arg layout).
struct VCKernel {
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  std::string entryPoint;
  int argCount = 0;
  // Push-constant range (offset/size) for scalar args, in bytes. size==0
  // means no push constants. Layout is built lazily on first launch once the
  // scalar/pointer arg arrangement is known.
  uint32_t pcOffset = 0;
  uint32_t pcSize = 0;
  bool layoutBuilt = false;
  // Cache: key = packed (blockX, blockY, blockZ) -> pipeline. Pipelines are
  // specialized on workgroup size, so one per distinct block shape.
  std::unordered_map<uint64_t, VkPipeline> pipelines;
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

class Runtime {
public:
  static Runtime &get();

  VCError init();
  VCError shutdown();
  bool initialized() const { return bool(device_); }

  VulkanDevice &device() { return *device_; }

  // Resolve a public stream handle to the internal stream; NULL -> default.
  VCStream &resolveStream(VCStreamHandle h);

  VCError createStream(VCStreamHandle *out);
  VCError destroyStream(VCStreamHandle stream);
  VCError streamSynchronize(VCStream &s);

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
  VCError freeBuffer(VCBuffer &buf);
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

  // Build (or fetch cached) the compute pipeline specialized to the block
  // size. Lazily builds the descriptor/pipeline layout on first call using
  // the scalar/pointer arrangement of `args`.
  VkPipeline getPipeline(VCKernel &k, unsigned blockX, unsigned blockY,
                         unsigned blockZ, const VCKernelArg *args,
                         int argCount);

  // Core launch: record bind+dispatch into the stream's current frame.
  VCError dispatch(VCKernel &k, unsigned wgX, unsigned wgY, unsigned wgZ,
                   unsigned blockX, unsigned blockY, unsigned blockZ,
                   const VCKernelArg *args, int argCount, VCStream &s);

  // Record bind+dispatch into a specific command buffer using a specific
  // descriptor pool (used by graph capture, which records into a secondary
  // command buffer instead of submitting).
  VCError recordDispatchInto(VkCommandBuffer cb, VkDescriptorPool dpool,
                              VCKernel &k, unsigned wgX, unsigned wgY,
                              unsigned wgZ, unsigned blockX, unsigned blockY,
                              unsigned blockZ, const VCKernelArg *args,
                              int argCount);

  VCError synchronize();

  // ---- Command graphs (record/replay) ----
  VCError createGraph(VCGraphHandle *out);
  VCError destroyGraph(VCGraphHandle graph);
  VCError beginRecord(VCGraph &g);
  VCError endRecord(VCGraph &g);
  VCError launchGraph(VCGraph &g, VCStream &s);
  VCError resetGraph(VCGraph &g);

private:
  std::unique_ptr<VulkanDevice> device_;
  std::unique_ptr<VCStream> defaultStream_;
  std::vector<std::unique_ptr<VCStream>> streams_; // owns created streams
  std::vector<std::unique_ptr<VCGraph>> graphs_;   // owns created graphs
  bool init_ = false;

  bool pickPhysicalDevice();
  bool createLogicalDevice();
  bool createDefaultStream();
  bool createPipelineCache();
  bool buildLayout(VCKernel &k, const VCKernelArg *args, int argCount);
  // Find a memory type index satisfying `flags` among `reqBits`.
  uint32_t findMemoryType(uint32_t reqBits,
                          VkMemoryPropertyFlags flags) const;
};

} // namespace vc

#endif // VC_RUNTIME_INTERNAL_H
