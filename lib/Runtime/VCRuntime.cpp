//===- VCRuntime.cpp - CUDA-style runtime over Vulkan (impl) --------------===//

#include "vc/Runtime/VCRuntime.h"
#include "RuntimeInternal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace vc {

// Forward decl: defined in the graphs section below; used by shutdown().
static void resetGraphState(VulkanDevice &dev, VCGraph &g);

//----------------------------------------------------------------------------
// Error helpers
//----------------------------------------------------------------------------

const char *vcErrorString(VCError err) {
  switch (err) {
  case VCError::Success: return "success";
  case VCError::InvalidValue: return "invalid value";
  case VCError::OutOfMemory: return "out of memory";
  case VCError::InvalidKernel: return "invalid kernel";
  case VCError::InvalidDevice: return "invalid device";
  case VCError::InitializationError: return "initialization error";
  case VCError::MapFailed: return "map failed";
  default: return "unknown error";
  }
}

//----------------------------------------------------------------------------
// Runtime singleton
//----------------------------------------------------------------------------

Runtime &Runtime::get() {
  static Runtime inst;
  return inst;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT,
              VkDebugUtilsMessageTypeFlagsEXT,
              const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  (void)data;
  return VK_FALSE;
}

VCError Runtime::init() {
  if (init_) return VCError::Success;
  device_ = std::make_unique<VulkanDevice>();

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vc-runtime";
  app.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  // macOS (MoltenVK) exposes a "portability subset" driver; the loader only
  // enumerates it when this flag + extension are set. Without them
  // vkCreateInstance finds no devices and everything silently no-ops.
  ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  const char *exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = exts;

  VkResult r = vkCreateInstance(&ici, nullptr, &device_->instance);
  if (r != VK_SUCCESS) {
    // Retry without debug utils extension (portability ext still required).
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = &exts[1];
    r = vkCreateInstance(&ici, nullptr, &device_->instance);
    if (r != VK_SUCCESS) return VCError::InitializationError;
  }

  if (!pickPhysicalDevice()) return VCError::InvalidDevice;
  if (!createLogicalDevice()) return VCError::InitializationError;
  if (!createDefaultStream()) return VCError::InitializationError;
  if (!createPipelineCache()) return VCError::InitializationError;

  init_ = true;
  return VCError::Success;
}

bool Runtime::pickPhysicalDevice() {
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(device_->instance, &n, nullptr);
  if (n == 0) return false;
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(device_->instance, &n, devs.data());
  // Prefer a device with a compute queue.
  for (auto d : devs) {
    uint32_t qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qf, nullptr);
    std::vector<VkQueueFamilyProperties> props(qf);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qf, props.data());
    for (uint32_t i = 0; i < qf; ++i) {
      if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        device_->physical = d;
        device_->computeQueueFamily = i;
        vkGetPhysicalDeviceMemoryProperties(d, &device_->memProps);
        return true;
      }
    }
  }
  return false;
}

bool Runtime::createLogicalDevice() {
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qi{};
  qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qi.queueFamilyIndex = device_->computeQueueFamily;
  qi.queueCount = 1;
  qi.pQueuePriorities = &prio;

  // Enable timeline semaphores (Vulkan 1.2 core feature). Used by stream
  // events (vcEvent*) to express cross-stream dependencies without a full
  // device sync. Probe the physical device first so we don't request an
  // unsupported feature (every conformant 1.2+ driver has it).
  VkPhysicalDeviceVulkan12Features feats12{};
  feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceVulkan12Features supported12{};
  supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceFeatures2 feats2{};
  feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  feats2.pNext = &supported12;
  vkGetPhysicalDeviceFeatures2(device_->physical, &feats2);
  if (supported12.timelineSemaphore) {
    feats12.timelineSemaphore = VK_TRUE;
    device_->timelineSemaphore = true;
  }

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qi;
  if (device_->timelineSemaphore) {
    dci.pNext = &feats12; // replaces pEnabledFeatures chain
  } else {
    VkPhysicalDeviceFeatures feats{};
    dci.pEnabledFeatures = &feats;
  }

  if (vkCreateDevice(device_->physical, &dci, nullptr, &device_->device) !=
      VK_SUCCESS)
    return false;
  vkGetDeviceQueue(device_->device, device_->computeQueueFamily, 0,
                   &device_->computeQueue);
  return true;
}

// Build a stream over the compute queue with a FRAME_RING-sized frame ring.
// Each frame has its own command buffer, fence, and descriptor pool so it can
// be reset independently when recycled.
static bool initStream(VulkanDevice &dev, VCStream &s,
                       size_t frameRing = 2) {
  s.queue = dev.computeQueue;
  s.queueFamily = dev.computeQueueFamily;
  s.frames.resize(frameRing);
  s.frameIdx = 0;

  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  // Reset command buffer bit so frames can be vkResetCommandBuffer'd.
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = s.queueFamily;
  if (vkCreateCommandPool(dev.device, &pci, nullptr, &s.commandPool) !=
      VK_SUCCESS)
    return false;

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = s.commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first beginFrame doesn't wait

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 64;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 32;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  for (auto &f : s.frames) {
    if (vkAllocateCommandBuffers(dev.device, &ai, &f.cb) != VK_SUCCESS)
      return false;
    if (vkCreateFence(dev.device, &fci, nullptr, &f.fence) != VK_SUCCESS)
      return false;
    if (vkCreateDescriptorPool(dev.device, &dpci, nullptr,
                               &f.descriptorPool) != VK_SUCCESS)
      return false;
  }
  return true;
}

static void teardownStream(VulkanDevice &dev, VCStream &s) {
  // Collect live command-buffer handles into a contiguous array before
  // freeing: StreamFrame.cb fields are not contiguous in memory (other
  // members sit between them), so &frames[0].cb is not a valid handle array.
  std::vector<VkCommandBuffer> cbs;
  for (auto &f : s.frames) {
    // Free any staging buffers still attached to this frame (callers are
    // expected to have idled the queue/device before teardown, so the GPU is
    // done with them). d2hReadbacks are host-side bookkeeping, dropped here.
    for (auto &sb : f.stagingBuffers) {
      if (sb->mapped) vkUnmapMemory(dev.device, sb->memory);
      if (sb->buffer) vkDestroyBuffer(dev.device, sb->buffer, nullptr);
      if (sb->memory) vkFreeMemory(dev.device, sb->memory, nullptr);
    }
    f.stagingBuffers.clear();
    f.d2hReadbacks.clear();
    if (f.descriptorPool)
      vkDestroyDescriptorPool(dev.device, f.descriptorPool, nullptr);
    if (f.fence) vkDestroyFence(dev.device, f.fence, nullptr);
    if (f.cb) cbs.push_back(f.cb);
  }
  if (!cbs.empty() && s.commandPool)
    vkFreeCommandBuffers(dev.device, s.commandPool,
                         static_cast<uint32_t>(cbs.size()), cbs.data());
  if (s.commandPool) vkDestroyCommandPool(dev.device, s.commandPool, nullptr);
}

bool Runtime::createDefaultStream() {
  defaultStream_ = std::make_unique<VCStream>();
  return initStream(*device_, *defaultStream_);
}

bool Runtime::createPipelineCache() {
  VkPipelineCacheCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  return vkCreatePipelineCache(device_->device, &ci, nullptr,
                               &device_->pipelineCache) == VK_SUCCESS;
}

VCError Runtime::shutdown() {
  if (!init_) return VCError::Success;
  vkDeviceWaitIdle(device_->device);
  for (auto &g : graphs_) {
    if (!g) continue;
    resetGraphState(*device_, *g);
    if (g->secondaryCB && g->commandPool)
      vkFreeCommandBuffers(device_->device, g->commandPool, 1, &g->secondaryCB);
    if (g->descriptorPool)
      vkDestroyDescriptorPool(device_->device, g->descriptorPool, nullptr);
    if (g->commandPool)
      vkDestroyCommandPool(device_->device, g->commandPool, nullptr);
  }
  graphs_.clear();
  for (auto &e : events_) {
    if (e && e->semaphore)
      vkDestroySemaphore(device_->device, e->semaphore, nullptr);
  }
  events_.clear();
  for (auto &s : streams_) if (s) teardownStream(*device_, *s);
  streams_.clear();
  if (defaultStream_) { teardownStream(*device_, *defaultStream_); defaultStream_.reset(); }
  if (device_->pipelineCache)
    vkDestroyPipelineCache(device_->device, device_->pipelineCache, nullptr);
  vkDestroyDevice(device_->device, nullptr);
  vkDestroyInstance(device_->instance, nullptr);
  device_.reset();
  init_ = false;
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Streams
//----------------------------------------------------------------------------

VCStream &Runtime::resolveStream(VCStreamHandle h) {
  return h ? *reinterpret_cast<VCStream *>(h) : *defaultStream_;
}

VCError Runtime::createStream(VCStreamHandle *out) {
  if (!init_ || !out) return VCError::InitializationError;
  auto s = std::make_unique<VCStream>();
  if (!initStream(*device_, *s)) return VCError::InitializationError;
  *out = reinterpret_cast<VCStreamHandle>(s.get());
  streams_.push_back(std::move(s));
  return VCError::Success;
}

VCError Runtime::destroyStream(VCStreamHandle stream) {
  if (!init_ || !stream) return VCError::Success;
  auto *s = reinterpret_cast<VCStream *>(stream);
  vkQueueWaitIdle(s->queue);
  teardownStream(*device_, *s);
  // Remove from ownership vector.
  for (auto it = streams_.begin(); it != streams_.end(); ++it) {
    if (it->get() == s) { streams_.erase(it); break; }
  }
  return VCError::Success;
}

// Drain deferred D2H readbacks from any frame whose fence is signaled (the
// device->staging copy has completed). Used by streamSynchronize (so the
// caller can read hostDst immediately after sync) and beginFrame (which
// drains the specific frame it is about to recycle).
static void drainReadyReadbacks(VulkanDevice &dev, VCStream &s) {
  for (auto &f : s.frames) {
    if (!f.d2hReadbacks.empty() && f.fence != VK_NULL_HANDLE) {
      if (vkGetFenceStatus(dev.device, f.fence) == VK_SUCCESS) {
        for (auto &rb : f.d2hReadbacks)
          std::memcpy(rb.hostDst, rb.staging->mapped, rb.bytes);
        f.d2hReadbacks.clear();
      }
    }
  }
}

VCError Runtime::streamSynchronize(VCStream &s) {
  // Wait for every in-flight frame's fence so all queued work is done.
  for (size_t i = 0; i < s.frames.size(); ++i) {
    // The frame at frameIdx may be unsignaled (about to be reused); waiting
    // on an already-signaled fence is cheap, so just wait on all of them.
    if (vkWaitForFences(device_->device, 1, &s.frames[i].fence, VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS)
      return VCError::Unknown;
  }
  // Now every fence is signaled: deliver all deferred D2H readbacks so the
  // caller can read the host destinations immediately after sync returns.
  drainReadyReadbacks(*device_, s);
  return VCError::Success;
}

// Wait for the frame we're about to reuse, reset it, begin recording.
VkCommandBuffer Runtime::beginFrame(VCStream &s) {
  StreamFrame &f = s.frames[s.frameIdx];
  // Wait for the GPU to finish with this frame's previous submission.
  vkWaitForFences(device_->device, 1, &f.fence, VK_TRUE, UINT64_MAX);
  // The fence just signaled, so every device->staging copy recorded in this
  // frame has executed: deliver deferred D2H readbacks to their host targets.
  // (The caller is responsible for having synchronized before reading.)
  for (auto &rb : f.d2hReadbacks)
    std::memcpy(rb.hostDst, rb.staging->mapped, rb.bytes);
  f.d2hReadbacks.clear();
  // Drop the frame's staging buffers now that the GPU is done with them.
  for (auto &sb : f.stagingBuffers) freeBuffer(*sb);
  f.stagingBuffers.clear();

  vkResetFences(device_->device, 1, &f.fence);
  vkResetCommandBuffer(f.cb, 0);
  vkResetDescriptorPool(device_->device, f.descriptorPool, 0);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(f.cb, &bi);
  return f.cb;
}

void Runtime::endFrame(VCStream &s) {
  StreamFrame &f = s.frames[s.frameIdx];
  vkEndCommandBuffer(f.cb);

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &f.cb;

  // Attach pending timeline-semaphore waits/signals (from vcEventRecord /
  // vcStreamWaitEvent) to this submission. Timeline semaphores carry a
  // 64-bit counter value per signal/wait, threaded via the pNext chain.
  VkTimelineSemaphoreSubmitInfo ti{};
  std::vector<VkSemaphore> waitSems;
  std::vector<uint64_t> waitVals;
  std::vector<VkPipelineStageFlags> waitStages;
  std::vector<VkSemaphore> signalSems;
  std::vector<uint64_t> signalVals;
  if (device_->timelineSemaphore && !s.pendingWaits.empty()) {
    waitSems.reserve(s.pendingWaits.size());
    waitVals.reserve(s.pendingWaits.size());
    waitStages.assign(s.pendingWaits.size(),
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    for (auto &w : s.pendingWaits) {
      waitSems.push_back(w.sem);
      waitVals.push_back(w.value);
    }
    si.waitSemaphoreCount = static_cast<uint32_t>(waitSems.size());
    si.pWaitSemaphores = waitSems.data();
    si.pWaitDstStageMask = waitStages.data();
  }
  if (device_->timelineSemaphore && !s.pendingSignals.empty()) {
    signalSems.reserve(s.pendingSignals.size());
    signalVals.reserve(s.pendingSignals.size());
    for (auto &sig : s.pendingSignals) {
      signalSems.push_back(sig.sem);
      signalVals.push_back(sig.value);
    }
    si.signalSemaphoreCount = static_cast<uint32_t>(signalSems.size());
    si.pSignalSemaphores = signalSems.data();
  }
  if (!waitVals.empty() || !signalVals.empty()) {
    ti.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ti.waitSemaphoreValueCount = static_cast<uint32_t>(waitVals.size());
    ti.pWaitSemaphoreValues = waitVals.empty() ? nullptr : waitVals.data();
    ti.signalSemaphoreValueCount = static_cast<uint32_t>(signalVals.size());
    ti.pSignalSemaphoreValues = signalVals.empty() ? nullptr : signalVals.data();
    si.pNext = &ti;
  }

  s.pendingWaits.clear();
  s.pendingSignals.clear();

  vkQueueSubmit(s.queue, 1, &si, f.fence);
  s.frameIdx = (s.frameIdx + 1) % s.frames.size();
}

VkDescriptorSet Runtime::allocFrameDescriptorSet(VCStream &s,
                                                 VkDescriptorSetLayout layout) {
  StreamFrame &f = s.frames[s.frameIdx];
  VkDescriptorSetAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = f.descriptorPool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  vkAllocateDescriptorSets(device_->device, &ai, &set);
  return set;
}

//----------------------------------------------------------------------------
// Memory
//----------------------------------------------------------------------------

uint32_t Runtime::findMemoryType(uint32_t reqBits,
                                 VkMemoryPropertyFlags flags) const {
  for (uint32_t i = 0; i < device_->memProps.memoryTypeCount; ++i) {
    if ((reqBits & (1u << i)) &&
        (device_->memProps.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  }
  return UINT32_MAX;
}

static VkBuffer createBuffer(VkDevice dev, size_t bytes) {
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buf = VK_NULL_HANDLE;
  vkCreateBuffer(dev, &bci, nullptr, &buf);
  return buf;
}

// device-local: not mapped. Falls back to host-visible if no device-local
// type satisfies the buffer (e.g. on integrated GPUs device-local == host).
VCError Runtime::mallocBuffer(size_t bytes, VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  out.size = bytes;
  out.buffer = createBuffer(device_->device, bytes);
  if (!out.buffer) return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(device_->device, out.buffer, &reqs);
  uint32_t typeIdx = findMemoryType(
      reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  bool hostFallback = false;
  if (typeIdx == UINT32_MAX) {
    typeIdx = findMemoryType(
        reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hostFallback = true;
    if (typeIdx == UINT32_MAX) {
      vkDestroyBuffer(device_->device, out.buffer, nullptr);
      return VCError::OutOfMemory;
    }
  }

  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = reqs.size;
  mai.memoryTypeIndex = typeIdx;
  if (vkAllocateMemory(device_->device, &mai, nullptr, &out.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_->device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  vkBindBufferMemory(device_->device, out.buffer, out.memory, 0);
  out.hostVisible = hostFallback;
  if (hostFallback)
    vkMapMemory(device_->device, out.memory, 0, bytes, 0, &out.mapped);
  return VCError::Success;
}

// host-visible + coherent, persistently mapped (pinned staging).
VCError Runtime::mallocHostBuffer(size_t bytes, VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  out.size = bytes;
  out.buffer = createBuffer(device_->device, bytes);
  if (!out.buffer) return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(device_->device, out.buffer, &reqs);
  uint32_t typeIdx = findMemoryType(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (typeIdx == UINT32_MAX) {
    vkDestroyBuffer(device_->device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = reqs.size;
  mai.memoryTypeIndex = typeIdx;
  if (vkAllocateMemory(device_->device, &mai, nullptr, &out.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_->device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  vkBindBufferMemory(device_->device, out.buffer, out.memory, 0);
  vkMapMemory(device_->device, out.memory, 0, bytes, 0, &out.mapped);
  out.hostVisible = true;
  return VCError::Success;
}

VCError Runtime::freeBuffer(VCBuffer &buf) {
  if (!init_) return VCError::InitializationError;
  if (buf.mapped) {
    vkUnmapMemory(device_->device, buf.memory);
    buf.mapped = nullptr;
  }
  if (buf.buffer) vkDestroyBuffer(device_->device, buf.buffer, nullptr);
  if (buf.memory) vkFreeMemory(device_->device, buf.memory, nullptr);
  buf = VCBuffer{};
  return VCError::Success;
}

// Primitive: record vkCmdCopyBuffer(src.buf -> dst.buf) on stream s.
VCError Runtime::copyDeviceToDevice(VCBuffer &dst, const VCBuffer &src,
                                    size_t bytes, VCStream &s) {
  // Graph capture: append the copy to the secondary command buffer.
  if (s.captureTarget) {
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(s.captureTarget->secondaryCB, src.buffer, dst.buffer, 1,
                    &region);
    // Make the transfer write visible to subsequent dispatches in the graph.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(s.captureTarget->secondaryCB,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    return VCError::Success;
  }
  VkCommandBuffer cb = beginFrame(s);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &region);
  endFrame(s);
  return VCError::Success;
}

// H2D: copy `bytes` from a host pointer into device buffer `dst` on stream s.
// Stages through a transient host-visible scratch buffer.
VCError Runtime::copyHostToDevice(VCBuffer &dst, const void *hostSrc,
                                  size_t bytes, VCStream &s) {
  // Graph capture: allocate a persistent staging buffer owned by the graph
  // (transient staging would be freed before the graph ever executes), fill
  // it now, and record the device copy into the secondary command buffer.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    auto staging = std::make_unique<VCBuffer>();
    if (mallocHostBuffer(bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    std::memcpy(staging->mapped, hostSrc, bytes);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, staging->buffer, dst.buffer, 1, &region);
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  VCBuffer scratch{};
  if (mallocHostBuffer(bytes, scratch) != VCError::Success)
    return VCError::OutOfMemory;
  std::memcpy(scratch.mapped, hostSrc, bytes);
  VkCommandBuffer cb = beginFrame(s);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, scratch.buffer, dst.buffer, 1, &region);
  endFrame(s);
  streamSynchronize(s); // scratch must survive until the copy executes
  freeBuffer(scratch);
  return VCError::Success;
}

// D2H: copy `bytes` from device buffer `src` into a host pointer. Syncs so
// the caller can read the result immediately.
VCError Runtime::copyDeviceToHost(void *hostDst, const VCBuffer &src,
                                  size_t bytes, VCStream &s) {
  // Graph capture: record a device->staging copy into the secondary command
  // buffer using a persistent staging buffer owned by the graph. The host
  // readback is deferred: the caller must synchronize after vcGraphLaunch
  // and then memcpy from the staging buffer's mapped pointer. We stash the
  // host destination + staging buffer so vcGraphLaunch can expose them, but
  // the simple contract is: D2H inside a graph is recorded as a copy to a
  // graph-owned staging buffer; the caller reads it post-sync via the mapped
  // pointer stored on the graph.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    auto staging = std::make_unique<VCBuffer>();
    if (mallocHostBuffer(bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, src.buffer, staging->buffer, 1, &region);
    // Record where the host wants the data so a post-launch readback can
    // deliver it. We can't copy now (the GPU copy hasn't executed).
    VCBuffer *raw = staging.get();
    g.d2hReadbacks.push_back({hostDst, raw, bytes});
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  VCBuffer scratch{};
  if (mallocHostBuffer(bytes, scratch) != VCError::Success)
    return VCError::OutOfMemory;
  VkCommandBuffer cb = beginFrame(s);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, src.buffer, scratch.buffer, 1, &region);
  endFrame(s);
  streamSynchronize(s);
  std::memcpy(hostDst, scratch.mapped, bytes);
  freeBuffer(scratch);
  return VCError::Success;
}

// Async H2D: snapshot host data into a frame-owned staging buffer, submit the
// device copy, return immediately. The staging lives on the frame and is
// reclaimed after the fence signals, so this never blocks the host.
VCError Runtime::copyHostToDeviceAsync(VCBuffer &dst, const void *hostSrc,
                                       size_t bytes, VCStream &s) {
  // Graph capture: same as the sync path's graph branch — persistent staging
  // owned by the graph.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    auto staging = std::make_unique<VCBuffer>();
    if (mallocHostBuffer(bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    std::memcpy(staging->mapped, hostSrc, bytes);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, staging->buffer, dst.buffer, 1, &region);
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  auto staging = std::make_unique<VCBuffer>();
  if (mallocHostBuffer(bytes, *staging) != VCError::Success)
    return VCError::OutOfMemory;
  // Snapshot the host source NOW so the caller can overwrite it immediately.
  std::memcpy(staging->mapped, hostSrc, bytes);
  // Record which frame we submit into so we can attach staging to it. endFrame
  // advances frameIdx, so capture the index before calling it.
  size_t submitIdx = s.frameIdx;
  VkCommandBuffer cb = beginFrame(s);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, staging->buffer, dst.buffer, 1, &region);
  endFrame(s);
  s.frames[submitIdx].stagingBuffers.push_back(std::move(staging));
  return VCError::Success;
}

// Async D2H: submit the device->staging copy, defer the staging->host memcpy
// to when the frame is recycled (after the fence signals). The caller MUST
// vcStreamSynchronize before reading hostDst.
VCError Runtime::copyDeviceToHostAsync(void *hostDst, const VCBuffer &src,
                                       size_t bytes, VCStream &s) {
  // Graph capture: record device->staging copy + a deferred readback entry.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    auto staging = std::make_unique<VCBuffer>();
    if (mallocHostBuffer(bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, src.buffer, staging->buffer, 1, &region);
    VCBuffer *raw = staging.get();
    g.d2hReadbacks.push_back({hostDst, raw, bytes});
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  auto staging = std::make_unique<VCBuffer>();
  if (mallocHostBuffer(bytes, *staging) != VCError::Success)
    return VCError::OutOfMemory;
  size_t submitIdx = s.frameIdx;
  VkCommandBuffer cb = beginFrame(s);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, src.buffer, staging->buffer, 1, &region);
  endFrame(s);
  // Defer the host memcpy: the device copy hasn't run yet. beginFrame will
  // drain this entry when the fence signals.
  VCBuffer *raw = staging.get();
  s.frames[submitIdx].d2hReadbacks.push_back({hostDst, raw, bytes});
  s.frames[submitIdx].stagingBuffers.push_back(std::move(staging));
  return VCError::Success;
}

VCError Runtime::synchronize() {
  if (!init_) return VCError::InitializationError;
  vkDeviceWaitIdle(device_->device);
  // Deliver deferred D2H readbacks on every stream (the caller may read host
  // destinations immediately after a device sync).
  if (defaultStream_) drainReadyReadbacks(*device_, *defaultStream_);
  for (auto &s : streams_) if (s) drainReadyReadbacks(*device_, *s);
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Command graphs (record/replay)
//----------------------------------------------------------------------------

// Size the graph descriptor pool generously: one set per recorded launch that
// has pointer args, each with up to a handful of SSBO bindings.
static VkDescriptorPool createGraphDescriptorPool(VkDevice dev) {
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 1024;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // FREE_DESCRIPTOR_SET_BIT so resetGraph can free sets without destroying
  // the pool (allows re-recording).
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 256;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
  return pool;
}

VCError Runtime::createGraph(VCGraphHandle *out) {
  if (!init_ || !out) return VCError::InitializationError;
  auto g = std::make_unique<VCGraph>();

  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = device_->computeQueueFamily;
  if (vkCreateCommandPool(device_->device, &pci, nullptr, &g->commandPool) !=
      VK_SUCCESS)
    return VCError::Unknown;

  g->descriptorPool = createGraphDescriptorPool(device_->device);
  if (!g->descriptorPool) {
    vkDestroyCommandPool(device_->device, g->commandPool, nullptr);
    return VCError::Unknown;
  }

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = g->commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  ai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device_->device, &ai, &g->secondaryCB) !=
      VK_SUCCESS) {
    vkDestroyDescriptorPool(device_->device, g->descriptorPool, nullptr);
    vkDestroyCommandPool(device_->device, g->commandPool, nullptr);
    return VCError::Unknown;
  }

  *out = reinterpret_cast<VCGraphHandle>(g.get());
  graphs_.push_back(std::move(g));
  return VCError::Success;
}

// Tear down a graph's recorded contents but keep the handle valid for
// re-recording. Frees the secondary command buffer's recorded state by
// resetting it, resets the descriptor pool, drops staging.
static void resetGraphState(VulkanDevice &dev, VCGraph &g) {
  if (g.secondaryCB) vkResetCommandBuffer(g.secondaryCB, 0);
  if (g.descriptorPool)
    vkResetDescriptorPool(dev.device, g.descriptorPool, 0);
  g.stagingBuffers.clear();
  g.d2hReadbacks.clear();
  g.recorded = false;
  g.recording = false;
}

VCError Runtime::destroyGraph(VCGraphHandle graph) {
  if (!init_ || !graph) return VCError::Success;
  auto *g = reinterpret_cast<VCGraph *>(graph);
  // Clear capture on the default stream if this graph is mid-record.
  if (defaultStream_ && defaultStream_->captureTarget == g)
    defaultStream_->captureTarget = nullptr;
  vkDeviceWaitIdle(device_->device);
  resetGraphState(*device_, *g);
  if (g->secondaryCB && g->commandPool)
    vkFreeCommandBuffers(device_->device, g->commandPool, 1, &g->secondaryCB);
  if (g->descriptorPool)
    vkDestroyDescriptorPool(device_->device, g->descriptorPool, nullptr);
  if (g->commandPool)
    vkDestroyCommandPool(device_->device, g->commandPool, nullptr);
  for (auto it = graphs_.begin(); it != graphs_.end(); ++it) {
    if (it->get() == g) { graphs_.erase(it); break; }
  }
  return VCError::Success;
}

VCError Runtime::beginRecord(VCGraph &g) {
  if (g.recording) return VCError::Unknown; // already recording
  // Start fresh: drop any previous recording. Only reset if the buffer was
  // previously recorded (avoid touching a freshly-allocated cb in its
  // initial state, which some drivers handle poorly on reset).
  if (g.recorded)
    resetGraphState(*device_, g);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  // SIMULTANEOUS_USE lets the same graph be replayed on multiple streams
  // concurrently (or re-launched before a prior launch finishes).
  bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  // Secondary command buffers referenced by vkCmdExecuteCommands do not
  // inherit a render pass (compute-only), but provide an (empty) inheritance
  // info to satisfy drivers that expect it for secondary level.
  VkCommandBufferInheritanceInfo inh{};
  inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
  bi.pInheritanceInfo = &inh;
  if (vkBeginCommandBuffer(g.secondaryCB, &bi) != VK_SUCCESS)
    return VCError::Unknown;
  g.recording = true;
  defaultStream_->captureTarget = &g;
  return VCError::Success;
}

VCError Runtime::endRecord(VCGraph &g) {
  if (!g.recording) return VCError::Unknown; // not recording
  if (vkEndCommandBuffer(g.secondaryCB) != VK_SUCCESS) {
    defaultStream_->captureTarget = nullptr;
    g.recording = false;
    return VCError::Unknown;
  }
  g.recording = false;
  g.recorded = true;
  if (defaultStream_->captureTarget == &g)
    defaultStream_->captureTarget = nullptr;
  return VCError::Success;
}

VCError Runtime::launchGraph(VCGraph &g, VCStream &s) {
  if (!g.recorded) return VCError::Unknown;
  // Execute the recorded secondary command buffer inside a primary frame on
  // the target stream — a single queue submit for the whole graph.
  VkCommandBuffer cb = beginFrame(s);
  vkCmdExecuteCommands(cb, 1, &g.secondaryCB);
  endFrame(s);
  return VCError::Success;
}

VCError Runtime::resetGraph(VCGraph &g) {
  if (defaultStream_ && defaultStream_->captureTarget == &g)
    defaultStream_->captureTarget = nullptr;
  vkDeviceWaitIdle(device_->device);
  resetGraphState(*device_, g);
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Stream events (timeline semaphore)
//----------------------------------------------------------------------------

VCError Runtime::createEvent(VCEventHandle *out) {
  if (!init_ || !out) return VCError::InitializationError;
  if (!device_->timelineSemaphore) return VCError::Unknown; // need timeline sem
  auto e = std::make_unique<VCEvent>();
  VkSemaphoreTypeCreateInfo ti{};
  ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  ti.initialValue = 0;
  VkSemaphoreCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  ci.pNext = &ti;
  if (vkCreateSemaphore(device_->device, &ci, nullptr, &e->semaphore) !=
      VK_SUCCESS)
    return VCError::Unknown;
  e->value = 1;          // next signal value
  e->lastRecorded = 0;   // nothing recorded yet
  *out = reinterpret_cast<VCEventHandle>(e.get());
  events_.push_back(std::move(e));
  return VCError::Success;
}

VCError Runtime::destroyEvent(VCEventHandle event) {
  if (!init_ || !event) return VCError::Success;
  auto *e = reinterpret_cast<VCEvent *>(event);
  vkDeviceWaitIdle(device_->device);
  if (e->semaphore)
    vkDestroySemaphore(device_->device, e->semaphore, nullptr);
  for (auto it = events_.begin(); it != events_.end(); ++it) {
    if (it->get() == e) { events_.erase(it); break; }
  }
  return VCError::Success;
}

// Record: the event marks the current tail of `stream` — all work submitted
// to the stream so far must complete before the event signals. We bump the
// event's counter, push a pending signal, and immediately flush an empty
// submission so the signal is actually emitted (VC's model is one submit per
// operation; without a flush, the signal would wait for the next operation
// that may never come). The empty submit still carries the frame's fence, so
// the stream's recycle semantics are unchanged.
VCError Runtime::recordEvent(VCEvent &e, VCStream &s) {
  if (!device_->timelineSemaphore) return VCError::Unknown;
  uint64_t v = e.value++;
  e.lastRecorded = v;
  s.pendingSignals.push_back({e.semaphore, v});
  // Flush: begin/end a frame with no recorded commands. endFrame attaches the
  // pending signal to the submit and clears it.
  beginFrame(s);
  endFrame(s);
  return VCError::Success;
}

// Wait: arrange for the stream's NEXT submission to wait on the event's most
// recently recorded value before executing. Does NOT flush — the wait rides
// the caller's next launch/copy on this stream. Cross-stream dependency.
VCError Runtime::streamWaitEvent(VCStream &s, VCEvent &e) {
  if (!device_->timelineSemaphore) return VCError::Unknown;
  if (e.lastRecorded == 0) return VCError::Success; // never recorded: no-op
  s.pendingWaits.push_back({e.semaphore, e.lastRecorded});
  return VCError::Success;
}

VCError Runtime::eventQuery(const VCEvent &e, int *done) const {
  if (!done) return VCError::InvalidValue;
  if (e.lastRecorded == 0) { *done = 1; return VCError::Success; }
  uint64_t counter = 0;
  if (vkGetSemaphoreCounterValue(device_->device, e.semaphore, &counter) !=
      VK_SUCCESS)
    return VCError::Unknown;
  *done = counter >= e.lastRecorded ? 1 : 0;
  return VCError::Success;
}

VCError Runtime::eventSynchronize(const VCEvent &e) const {
  if (e.lastRecorded == 0) return VCError::Success;
  VkSemaphoreWaitInfo wi{};
  wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wi.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
  wi.semaphoreCount = 1;
  wi.pSemaphores = &e.semaphore;
  wi.pValues = &e.lastRecorded;
  if (vkWaitSemaphores(device_->device, &wi, UINT64_MAX) != VK_SUCCESS)
    return VCError::Unknown;
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Kernels / pipelines
//----------------------------------------------------------------------------

VCError Runtime::loadKernel(const uint32_t *words, size_t wordCount,
                            const char *entryPoint, VCKernel &out) {
  if (!init_) return VCError::InitializationError;
  if (!words || wordCount == 0) return VCError::InvalidValue;

  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = wordCount * sizeof(uint32_t);
  ci.pCode = words;
  if (vkCreateShaderModule(device_->device, &ci, nullptr,
                           &out.shaderModule) != VK_SUCCESS)
    return VCError::InvalidKernel;
  out.entryPoint = entryPoint ? entryPoint : "main";
  return VCError::Success;
}

VCError Runtime::loadKernelFromFile(const char *path, const char *entryPoint,
                                    VCKernel &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return VCError::InvalidValue;
  std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> words((size + 3) / 4, 0);
  f.read(reinterpret_cast<char *>(words.data()), size);
  return loadKernel(words.data(), words.size(), entryPoint, out);
}

void Runtime::releaseKernel(VCKernel &k) {
  if (!init_) return;
  for (auto &kv : k.pipelines)
    if (kv.second) vkDestroyPipeline(device_->device, kv.second, nullptr);
  k.pipelines.clear();
  if (k.pipelineLayout)
    vkDestroyPipelineLayout(device_->device, k.pipelineLayout, nullptr);
  if (k.descriptorSetLayout)
    vkDestroyDescriptorSetLayout(device_->device, k.descriptorSetLayout,
                                 nullptr);
  if (k.shaderModule)
    vkDestroyShaderModule(device_->device, k.shaderModule, nullptr);
  k = VCKernel{};
}

// Build the descriptor-set + pipeline layouts. Pointer args get consecutive
// SSBO bindings; scalar args are packed into a single push-constant range
// (so they don't need per-launch staging buffers). `args`/`argCount` define
// the arrangement; the layout is built once and cached on the kernel.
bool Runtime::buildLayout(VCKernel &k, const VCKernelArg *args, int argCount) {
  k.argCount = argCount;
  // Collect pointer bindings + measure push-constant size for scalars.
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t pcSize = 0;
  for (int i = 0; i < argCount; ++i) {
    if (args[i].kind == VCKernelArg::Pointer) {
      VkDescriptorSetLayoutBinding b{};
      b.binding = static_cast<uint32_t>(bindings.size());
      b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b.descriptorCount = 1;
      b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(b);
    } else {
      // Round scalar size up to 4 bytes for std140-friendly packing.
      uint32_t sz = static_cast<uint32_t>((args[i].size + 3) & ~size_t(3));
      pcSize += sz;
    }
  }

  VkDescriptorSetLayoutCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dci.bindingCount = static_cast<uint32_t>(bindings.size());
  dci.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(device_->device, &dci, nullptr,
                                  &k.descriptorSetLayout) != VK_SUCCESS)
    return false;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &k.descriptorSetLayout;
  VkPushConstantRange pcr{};
  if (pcSize > 0) {
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = pcSize;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
  }
  if (vkCreatePipelineLayout(device_->device, &plci, nullptr,
                             &k.pipelineLayout) != VK_SUCCESS)
    return false;
  k.pcSize = pcSize;
  k.layoutBuilt = true;
  return true;
}

// Fetch or create a pipeline specialized to the block size. Layout must be
// built first (via buildLayout) — done in dispatch on first use.
VkPipeline Runtime::getPipeline(VCKernel &k, unsigned blockX,
                                unsigned blockY, unsigned blockZ,
                                const VCKernelArg *args, int argCount) {
  if (!k.layoutBuilt && !buildLayout(k, args, argCount))
    return VK_NULL_HANDLE;

  // Key: pack block dims into 64 bits (16 bits each + reserved).
  uint64_t key = (uint64_t(blockX) << 32) | (uint64_t(blockY) << 16) | blockZ;
  auto it = k.pipelines.find(key);
  if (it != k.pipelines.end()) return it->second;

  // Specialization constants: 0->x, 1->y, 2->z (matches GLSL backend).
  VkSpecializationMapEntry entries[3];
  unsigned data[3] = {blockX ? blockX : 1, blockY ? blockY : 1,
                      blockZ ? blockZ : 1};
  for (int i = 0; i < 3; ++i) {
    entries[i].constantID = i;
    entries[i].offset = i * sizeof(unsigned);
    entries[i].size = sizeof(unsigned);
  }
  VkSpecializationInfo spec{};
  spec.mapEntryCount = 3;
  spec.pMapEntries = entries;
  spec.dataSize = sizeof(data);
  spec.pData = data;

  VkComputePipelineCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = k.shaderModule;
  pci.stage.pName = k.entryPoint.c_str();
  pci.stage.pSpecializationInfo = &spec;
  pci.layout = k.pipelineLayout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(device_->device, device_->pipelineCache, 1,
                               &pci, nullptr, &pipeline) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  k.pipelines[key] = pipeline;
  return pipeline;
}

//----------------------------------------------------------------------------
// Launch
//----------------------------------------------------------------------------

VCError Runtime::recordDispatchInto(VkCommandBuffer cb, VkDescriptorPool dpool,
                                     VCKernel &k, unsigned wgX, unsigned wgY,
                                     unsigned wgZ, unsigned blockX,
                                     unsigned blockY, unsigned blockZ,
                                     const VCKernelArg *args, int argCount) {
  if (!k.shaderModule) return VCError::InvalidKernel;
  VkPipeline pipeline = getPipeline(k, blockX, blockY, blockZ, args, argCount);
  if (!pipeline) return VCError::InvalidKernel;

  VkDescriptorSet set = VK_NULL_HANDLE;
  // Only allocate a set if there are pointer args (SSBO bindings).
  bool hasPointer = false;
  for (int i = 0; i < argCount; ++i)
    if (args[i].kind == VCKernelArg::Pointer) { hasPointer = true; break; }
  if (hasPointer) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &k.descriptorSetLayout;
    vkAllocateDescriptorSets(device_->device, &ai, &set);
  }
  if (hasPointer && !set) return VCError::Unknown;

  // Write descriptor bindings for pointer args (consecutive binding index).
  std::vector<VkDescriptorBufferInfo> bufInfos;
  std::vector<VkWriteDescriptorSet> writes;
  bufInfos.reserve(argCount);
  writes.reserve(argCount);
  uint32_t bindIdx = 0;
  for (int i = 0; i < argCount; ++i) {
    if (args[i].kind != VCKernelArg::Pointer) continue;
    auto *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(args[i].data));
    VkDescriptorBufferInfo bi{};
    bi.buffer = b ? b->buffer : VK_NULL_HANDLE;
    bi.offset = 0;
    bi.range = b ? b->size : 0;
    bufInfos.push_back(bi);
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set;
    w.dstBinding = bindIdx++;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bufInfos.back();
    writes.push_back(w);
  }
  if (!writes.empty())
    vkUpdateDescriptorSets(device_->device, writes.size(), writes.data(), 0,
                           nullptr);

  // Pack scalar args into a push-constant buffer.
  std::vector<uint8_t> pcData;
  pcData.reserve(k.pcSize);
  if (k.pcSize > 0) {
    for (int i = 0; i < argCount; ++i) {
      if (args[i].kind != VCKernelArg::Scalar) continue;
      uint32_t sz = static_cast<uint32_t>((args[i].size + 3) & ~size_t(3));
      size_t off = pcData.size();
      pcData.resize(off + sz, 0);
      std::memcpy(pcData.data() + off, args[i].data, args[i].size);
    }
  }

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  if (set)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            k.pipelineLayout, 0, 1, &set, 0, nullptr);
  if (!pcData.empty())
    vkCmdPushConstants(cb, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(pcData.size()), pcData.data());
  vkCmdDispatch(cb, wgX, wgY, wgZ);
  // Make the dispatch's SSBO writes visible to subsequent commands in the
  // same command buffer (copies, later dispatches). Within a single Vulkan
  // command buffer, a later vkCmdCopyBuffer/vkCmdDispatch does NOT
  // automatically see an earlier dispatch's memory writes without a barrier.
  // For graph capture (secondary cb) this is essential: the recorded ops must
  // form a correct pipeline. For the normal per-frame path the barrier is
  // harmless (the submit's fence is the only cross-cb sync the caller needs).
  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
  return VCError::Success;
}

VCError Runtime::dispatch(VCKernel &k, unsigned wgX, unsigned wgY,
                          unsigned wgZ, unsigned blockX, unsigned blockY,
                          unsigned blockZ, const VCKernelArg *args,
                          int argCount, VCStream &s) {
  // Graph capture: record into the graph's secondary command buffer using its
  // own descriptor pool, without submitting.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    return recordDispatchInto(g.secondaryCB, g.descriptorPool, k, wgX, wgY,
                              wgZ, blockX, blockY, blockZ, args, argCount);
  }
  VkCommandBuffer cb = beginFrame(s);
  VCError e = recordDispatchInto(cb, s.frames[s.frameIdx].descriptorPool, k,
                                  wgX, wgY, wgZ, blockX, blockY, blockZ, args,
                                  argCount);
  if (e != VCError::Success) return e;
  endFrame(s);
  return VCError::Success;
}

//----------------------------------------------------------------------------
// C API wrappers
//----------------------------------------------------------------------------

VCError vcInit() { return Runtime::get().init(); }
VCError vcShutdown() { return Runtime::get().shutdown(); }

VCError vcGetDeviceCount(int *count) {
  if (!Runtime::get().initialized()) {
    if (Runtime::get().init() != VCError::Success) {
      *count = 0;
      return VCError::Success;
    }
  }
  *count = 1; // scaffold: single-device
  return VCError::Success;
}

VCError vcMalloc(void **devPtr, size_t bytes) {
  if (!devPtr) return VCError::InvalidValue;
  auto *b = new VCBuffer{};
  VCError e = Runtime::get().mallocBuffer(bytes, *b);
  if (e != VCError::Success) { delete b; return e; }
  *devPtr = b;
  return VCError::Success;
}

VCError vcMallocHost(void **hostPtr, size_t bytes) {
  if (!hostPtr) return VCError::InvalidValue;
  auto *b = new VCBuffer{};
  VCError e = Runtime::get().mallocHostBuffer(bytes, *b);
  if (e != VCError::Success) { delete b; return e; }
  *hostPtr = b;
  return VCError::Success;
}

VCError vcFree(void *devPtr) {
  if (!devPtr) return VCError::Success;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  VCError e = Runtime::get().freeBuffer(*b);
  delete b;
  return e;
}

VCError vcMemcpyS(void *dst, const void *src, size_t count,
                  VCMemcpyKind kind, VCStreamHandle stream) {
  auto &rt = Runtime::get();
  auto &s = rt.resolveStream(stream);
  if (kind == VCMemcpyKind::HostToDevice) {
    auto *b = reinterpret_cast<VCBuffer *>(dst);
    if (!b) return VCError::InvalidValue;
    return rt.copyHostToDevice(*b, src, count, s);
  } else if (kind == VCMemcpyKind::DeviceToHost) {
    auto *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!b) return VCError::InvalidValue;
    return rt.copyDeviceToHost(dst, *b, count, s);
  } else { // DeviceToDevice
    auto *db = reinterpret_cast<VCBuffer *>(dst);
    auto *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!db || !sb) return VCError::InvalidValue;
    return rt.copyDeviceToDevice(*db, *sb, count, s);
  }
}

VCError vcMemcpy(void *dst, const void *src, size_t count,
                 VCMemcpyKind kind) {
  return vcMemcpyS(dst, src, count, kind, nullptr);
}

VCError vcMemcpyAsyncS(void *dst, const void *src, size_t count,
                       VCMemcpyKind kind, VCStreamHandle stream) {
  auto &rt = Runtime::get();
  auto &s = rt.resolveStream(stream);
  if (kind == VCMemcpyKind::HostToDevice) {
    auto *b = reinterpret_cast<VCBuffer *>(dst);
    if (!b) return VCError::InvalidValue;
    return rt.copyHostToDeviceAsync(*b, src, count, s);
  } else if (kind == VCMemcpyKind::DeviceToHost) {
    auto *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!b) return VCError::InvalidValue;
    return rt.copyDeviceToHostAsync(dst, *b, count, s);
  } else { // DeviceToDevice — already async
    auto *db = reinterpret_cast<VCBuffer *>(dst);
    auto *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!db || !sb) return VCError::InvalidValue;
    return rt.copyDeviceToDevice(*db, *sb, count, s);
  }
}

VCError vcMemcpyAsync(void *dst, const void *src, size_t count,
                      VCMemcpyKind kind) {
  return vcMemcpyAsyncS(dst, src, count, kind, nullptr);
}

VCError vcDeviceSynchronize() { return Runtime::get().synchronize(); }

VCError vcStreamCreate(VCStreamHandle *out) {
  return Runtime::get().createStream(out);
}

VCError vcStreamDestroy(VCStreamHandle stream) {
  return Runtime::get().destroyStream(stream);
}

VCError vcStreamSynchronize(VCStreamHandle stream) {
  auto &rt = Runtime::get();
  return rt.streamSynchronize(rt.resolveStream(stream));
}

//----------------------------------------------------------------------------
// Command graphs (C API)
//----------------------------------------------------------------------------

VCError vcGraphCreate(VCGraphHandle *out) {
  return Runtime::get().createGraph(out);
}

VCError vcGraphDestroy(VCGraphHandle graph) {
  return Runtime::get().destroyGraph(graph);
}

VCError vcGraphBeginRecord(VCGraphHandle graph) {
  if (!graph) return VCError::InvalidValue;
  return Runtime::get().beginRecord(*reinterpret_cast<VCGraph *>(graph));
}

VCError vcGraphEndRecord(VCGraphHandle graph) {
  if (!graph) return VCError::InvalidValue;
  return Runtime::get().endRecord(*reinterpret_cast<VCGraph *>(graph));
}

VCError vcGraphLaunch(VCGraphHandle graph, VCStreamHandle stream) {
  if (!graph) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  return rt.launchGraph(*reinterpret_cast<VCGraph *>(graph),
                        rt.resolveStream(stream));
}

VCError vcGraphReset(VCGraphHandle graph) {
  if (!graph) return VCError::InvalidValue;
  return Runtime::get().resetGraph(*reinterpret_cast<VCGraph *>(graph));
}

//----------------------------------------------------------------------------
// Stream events (C API)
//----------------------------------------------------------------------------

VCError vcEventCreate(VCEventHandle *out) {
  return Runtime::get().createEvent(out);
}

VCError vcEventDestroy(VCEventHandle event) {
  return Runtime::get().destroyEvent(event);
}

VCError vcEventRecord(VCEventHandle event, VCStreamHandle stream) {
  if (!event) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  return rt.recordEvent(*reinterpret_cast<VCEvent *>(event),
                        rt.resolveStream(stream));
}

VCError vcStreamWaitEvent(VCStreamHandle stream, VCEventHandle event) {
  if (!event) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  return rt.streamWaitEvent(rt.resolveStream(stream),
                            *reinterpret_cast<VCEvent *>(event));
}

VCError vcEventQuery(VCEventHandle event, int *done) {
  if (!event) return VCError::InvalidValue;
  return Runtime::get().eventQuery(*reinterpret_cast<VCEvent *>(event), done);
}

VCError vcEventSynchronize(VCEventHandle event) {
  if (!event) return VCError::InvalidValue;
  return Runtime::get().eventSynchronize(*reinterpret_cast<VCEvent *>(event));
}

VCError vcLoadKernel(const uint32_t *spirvWords, size_t wordCount,
                     const char *entryPoint, VCKernelHandle *outKernel) {
  if (!outKernel) return VCError::InvalidValue;
  auto *k = new VCKernel{};
  VCError e = Runtime::get().loadKernel(spirvWords, wordCount, entryPoint, *k);
  if (e != VCError::Success) { delete k; return e; }
  *outKernel = k;
  return VCError::Success;
}

VCError vcLoadKernelFromFile(const char *path, const char *entryPoint,
                             VCKernelHandle *outKernel) {
  if (!outKernel) return VCError::InvalidValue;
  auto *k = new VCKernel{};
  VCError e = Runtime::get().loadKernelFromFile(path, entryPoint, *k);
  if (e != VCError::Success) { delete k; return e; }
  *outKernel = k;
  return VCError::Success;
}

VCError vcReleaseKernel(VCKernelHandle kernel) {
  if (!kernel) return VCError::Success;
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  Runtime::get().releaseKernel(*k);
  delete k;
  return VCError::Success;
}

VCError vcLaunchKernelS(VCKernelHandle kernel, unsigned gridDim,
                        unsigned blockDim, const VCKernelArg *args,
                        int argCount, VCStreamHandle stream) {
  if (!kernel) return VCError::InvalidKernel;
  auto &rt = Runtime::get();
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  unsigned wg = (gridDim + blockDim - 1) / blockDim;
  return rt.dispatch(*k, wg, 1, 1, blockDim, 1, 1, args, argCount,
                     rt.resolveStream(stream));
}

VCError vcLaunchKernel2DS(VCKernelHandle kernel, unsigned gridDimX,
                          unsigned gridDimY, unsigned blockDimX,
                          unsigned blockDimY, const VCKernelArg *args,
                          int argCount, VCStreamHandle stream) {
  if (!kernel) return VCError::InvalidKernel;
  auto &rt = Runtime::get();
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  unsigned wgX = (gridDimX + blockDimX - 1) / blockDimX;
  unsigned wgY = (gridDimY + blockDimY - 1) / blockDimY;
  return rt.dispatch(*k, wgX, wgY, 1, blockDimX, blockDimY, 1, args, argCount,
                     rt.resolveStream(stream));
}

VCError vcLaunchKernel(VCKernelHandle kernel, unsigned gridDim,
                       unsigned blockDim, const VCKernelArg *args,
                       int argCount) {
  return vcLaunchKernelS(kernel, gridDim, blockDim, args, argCount, nullptr);
}

VCError vcLaunchKernel2D(VCKernelHandle kernel, unsigned gridDimX,
                         unsigned gridDimY, unsigned blockDimX,
                         unsigned blockDimY, const VCKernelArg *args,
                         int argCount) {
  return vcLaunchKernel2DS(kernel, gridDimX, gridDimY, blockDimX, blockDimY,
                           args, argCount, nullptr);
}

} // namespace vc

