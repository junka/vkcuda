//===- VCRuntime.cpp - CUDA-style runtime over Vulkan (impl) --------------===//

#include "vc/Runtime/VCRuntime.h"
#include "RuntimeInternal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace vc {

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

  VkPhysicalDeviceFeatures feats{};
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qi;
  dci.pEnabledFeatures = &feats;

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

VCError Runtime::streamSynchronize(VCStream &s) {
  // Wait on the in-flight frame's fence (the one most recently submitted).
  if (!s.frames.empty()) {
    auto &f = s.frames[(s.frameIdx + s.frames.size() - 1) % s.frames.size()];
    if (vkWaitForFences(device_->device, 1, &f.fence, VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS)
      return VCError::Unknown;
  }
  return VCError::Success;
}

// Wait for the frame we're about to reuse, reset it, begin recording.
VkCommandBuffer Runtime::beginFrame(VCStream &s) {
  StreamFrame &f = s.frames[s.frameIdx];
  // Wait for the GPU to finish with this frame's previous submission.
  vkWaitForFences(device_->device, 1, &f.fence, VK_TRUE, UINT64_MAX);
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

VCError Runtime::synchronize() {
  if (!init_) return VCError::InitializationError;
  vkDeviceWaitIdle(device_->device);
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

VCError Runtime::dispatch(VCKernel &k, unsigned wgX, unsigned wgY,
                          unsigned wgZ, unsigned blockX, unsigned blockY,
                          unsigned blockZ, const VCKernelArg *args,
                          int argCount, VCStream &s) {
  if (!k.shaderModule) return VCError::InvalidKernel;
  VkPipeline pipeline = getPipeline(k, blockX, blockY, blockZ, args, argCount);
  if (!pipeline) return VCError::InvalidKernel;

  VkDescriptorSet set = VK_NULL_HANDLE;
  // Only allocate a set if there are pointer args (SSBO bindings).
  bool hasPointer = false;
  for (int i = 0; i < argCount; ++i)
    if (args[i].kind == VCKernelArg::Pointer) { hasPointer = true; break; }
  if (hasPointer)
    set = allocFrameDescriptorSet(s, k.descriptorSetLayout);
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

  VkCommandBuffer cb = beginFrame(s);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  if (set)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            k.pipelineLayout, 0, 1, &set, 0, nullptr);
  if (!pcData.empty())
    vkCmdPushConstants(cb, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(pcData.size()), pcData.data());
  vkCmdDispatch(cb, wgX, wgY, wgZ);
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

