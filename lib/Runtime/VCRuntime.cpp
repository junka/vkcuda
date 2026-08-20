//===- VCRuntime.cpp - CUDA-style runtime over Vulkan (impl) --------------===//

#include "vc/Runtime/VCRuntime.h"
#include "RuntimeInternal.h"

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

static VCError vkErr(VkResult r) {
  return r == VK_SUCCESS ? VCError::Success : VCError::Unknown;
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
  //fprintf(stderr, "[vc-vk] %s\n", data->pMessage);
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
  const char *exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  ici.enabledExtensionCount = 1;
  ici.ppEnabledExtensionNames = exts;

  VkResult r = vkCreateInstance(&ici, nullptr, &device_->instance);
  if (r != VK_SUCCESS) {
    // Retry without debug utils extension.
    ici.enabledExtensionCount = 0;
    r = vkCreateInstance(&ici, nullptr, &device_->instance);
    if (r != VK_SUCCESS) return VCError::InitializationError;
  }

  if (!pickPhysicalDevice()) return VCError::InvalidDevice;
  if (!createLogicalDevice()) return VCError::InitializationError;
  if (!createCommandPool()) return VCError::InitializationError;
  if (!createDescriptorPool()) return VCError::InitializationError;

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

bool Runtime::createCommandPool() {
  VkCommandPoolCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  ci.queueFamilyIndex = device_->computeQueueFamily;
  return vkCreateCommandPool(device_->device, &ci, nullptr,
                             &device_->commandPool) == VK_SUCCESS;
}

bool Runtime::createDescriptorPool() {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 256;

  VkDescriptorPoolCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // Allow individual sets to be freed back to the pool so repeated launches
  // do not exhaust the 64-set cap.
  ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  ci.maxSets = 64;
  ci.poolSizeCount = 1;
  ci.pPoolSizes = &poolSize;
  return vkCreateDescriptorPool(device_->device, &ci, nullptr,
                                &device_->descriptorPool) == VK_SUCCESS;
}

VCError Runtime::shutdown() {
  if (!init_) return VCError::Success;
  vkDeviceWaitIdle(device_->device);
  vkDestroyDescriptorPool(device_->device, device_->descriptorPool, nullptr);
  vkDestroyCommandPool(device_->device, device_->commandPool, nullptr);
  vkDestroyDevice(device_->device, nullptr);
  vkDestroyInstance(device_->instance, nullptr);
  device_.reset();
  init_ = false;
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Buffers
//----------------------------------------------------------------------------

VCError Runtime::mallocBuffer(size_t bytes, VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  out.size = bytes;

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device_->device, &bci, nullptr, &out.buffer) != VK_SUCCESS)
    return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(device_->device, out.buffer, &reqs);

  // Find host-visible memory type.
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(device_->physical, &memProps);
  uint32_t typeIdx = UINT32_MAX;
  VkMemoryPropertyFlags wanted =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((reqs.memoryTypeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & wanted) == wanted) {
      typeIdx = i;
      break;
    }
  }
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

  // Persistently map for simple host access (staging in place).
  vkMapMemory(device_->device, out.memory, 0, bytes, 0, &out.mapped);
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

VCError Runtime::copy(void *dst, const void *src, size_t bytes,
                      VCMemcpyKind kind) {
  if (!init_) return VCError::InitializationError;
  // Buffers are host-visible/coherent, so a simple memcpy through the
  // mapped pointer suffices for the scaffold. A real runtime would issue
  // transfer commands for device-local memory.
  if (kind == VCMemcpyKind::HostToDevice) {
    VCBuffer *b = reinterpret_cast<VCBuffer *>(dst);
    if (!b || !b->mapped) return VCError::InvalidValue;
    std::memcpy(b->mapped, src, bytes);
  } else if (kind == VCMemcpyKind::DeviceToHost) {
    VCBuffer *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!b || !b->mapped) return VCError::InvalidValue;
    std::memcpy(dst, b->mapped, bytes);
  } else {
    VCBuffer *db = reinterpret_cast<VCBuffer *>(dst);
    VCBuffer *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
    if (!db || !sb || !db->mapped || !sb->mapped) return VCError::InvalidValue;
    std::memcpy(db->mapped, sb->mapped, bytes);
  }
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Kernels
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
  if (k.pipeline) vkDestroyPipeline(device_->device, k.pipeline, nullptr);
  if (k.pipelineLayout)
    vkDestroyPipelineLayout(device_->device, k.pipelineLayout, nullptr);
  if (k.descriptorSetLayout)
    vkDestroyDescriptorSetLayout(device_->device, k.descriptorSetLayout,
                                 nullptr);
  if (k.shaderModule) vkDestroyShaderModule(device_->device, k.shaderModule, nullptr);
  k = VCKernel{};
}

// Build the descriptor-set + pipeline layouts for a kernel given its arg
// count. These are layout-only and reusable across launches; the actual
// compute pipeline (which carries the workgroup-size specialization) is
// built per launch in buildPipelineWithSpec().
bool Runtime::buildPipelineForKernel(VCKernel &k, int argCount) {
  // One storage-buffer binding per pointer argument.
  std::vector<VkDescriptorSetLayoutBinding> bindings(argCount);
  for (int i = 0; i < argCount; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dci.bindingCount = argCount;
  dci.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(device_->device, &dci, nullptr,
                                  &k.descriptorSetLayout) != VK_SUCCESS)
    return false;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &k.descriptorSetLayout;
  if (vkCreatePipelineLayout(device_->device, &plci, nullptr,
                             &k.pipelineLayout) != VK_SUCCESS)
    return false;
  k.argCount = argCount;
  return true;
}

// Build a compute pipeline specializing the workgroup size to the given
// block dimensions. Specialization constant IDs match what the GLSL
// backend emits: 0 -> local_size_x, 1 -> local_size_y, 2 -> local_size_z.
bool Runtime::buildPipelineWithSpec(VCKernel &k, unsigned blockX,
                                    unsigned blockY, unsigned blockZ) {
  if (k.pipeline) {
    vkDestroyPipeline(device_->device, k.pipeline, nullptr);
    k.pipeline = VK_NULL_HANDLE;
  }

  // Always provide all three; the shader only reads the IDs it declared,
  // and extra map entries for undeclared IDs are ignored by the driver.
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
  return vkCreateComputePipelines(device_->device, VK_NULL_HANDLE, 1, &pci,
                                  nullptr, &k.pipeline) == VK_SUCCESS;
}

VkCommandBuffer Runtime::beginOneTime() const {
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = device_->commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(device_->device, &ai, &cb);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  return cb;
}

void Runtime::endOneTime(VkCommandBuffer cb) const {
  vkEndCommandBuffer(cb);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  vkQueueSubmit(device_->computeQueue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(device_->computeQueue);
  vkFreeCommandBuffers(device_->device, device_->commandPool, 1, &cb);
}

VCError Runtime::launch(VCKernel &k, unsigned gridDim, unsigned blockDim,
                        const VCKernelArg *args, int argCount) {
  if (!init_) return VCError::InitializationError;
  if (!k.shaderModule) return VCError::InvalidKernel;
  if (!k.pipelineLayout && !buildPipelineForKernel(k, argCount))
    return VCError::InvalidKernel;

  unsigned workgroups = (gridDim + blockDim - 1) / blockDim;
  if (!buildPipelineWithSpec(k, blockDim, 1, 1))
    return VCError::InvalidKernel;
  return dispatchBound(k, args, argCount, workgroups, 1, 1);
}

VCError Runtime::launch2D(VCKernel &k, unsigned gridDimX, unsigned gridDimY,
                          unsigned blockDimX, unsigned blockDimY,
                          const VCKernelArg *args, int argCount) {
  if (!init_) return VCError::InitializationError;
  if (!k.shaderModule) return VCError::InvalidKernel;
  if (!k.pipelineLayout && !buildPipelineForKernel(k, argCount))
    return VCError::InvalidKernel;

  unsigned wgX = (gridDimX + blockDimX - 1) / blockDimX;
  unsigned wgY = (gridDimY + blockDimY - 1) / blockDimY;
  if (!buildPipelineWithSpec(k, blockDimX, blockDimY, 1))
    return VCError::InvalidKernel;
  return dispatchBound(k, args, argCount, wgX, wgY, 1);
}

VCError Runtime::dispatchBound(VCKernel &k, const VCKernelArg *args,
                               int argCount, unsigned wgX, unsigned wgY,
                               unsigned wgZ) {
  // Allocate + write a descriptor set.
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = device_->descriptorPool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &k.descriptorSetLayout;
  VkDescriptorSet set;
  if (vkAllocateDescriptorSets(device_->device, &dsai, &set) != VK_SUCCESS)
    return VCError::Unknown;

  std::vector<VkDescriptorBufferInfo> bufInfos;
  std::vector<VkWriteDescriptorSet> writes;
  // Temporary buffers backing scalar-by-value arguments (each is bound as
  // a storage buffer to match the shader's SSBO binding).
  std::vector<VCBuffer> scalarBufs;
  bufInfos.reserve(argCount);
  writes.reserve(argCount);
  for (int i = 0; i < argCount; ++i) {
    VkDescriptorBufferInfo bi{};
    if (args[i].kind == VCKernelArg::Pointer) {
      auto *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(args[i].data));
      bi.buffer = b ? b->buffer : VK_NULL_HANDLE;
      bi.offset = 0;
      bi.range = b ? b->size : 0;
    } else {
      // Scalar: stage into a small device buffer.
      scalarBufs.emplace_back();
      if (mallocBuffer(std::max<size_t>(args[i].size, 4), scalarBufs.back()) !=
          VCError::Success) {
        for (auto &sb : scalarBufs) freeBuffer(sb);
        return VCError::OutOfMemory;
      }
      std::memcpy(scalarBufs.back().mapped, args[i].data, args[i].size);
      bi.buffer = scalarBufs.back().buffer;
      bi.offset = 0;
      bi.range = args[i].size;
    }
    bufInfos.push_back(bi);
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set;
    w.dstBinding = i;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bufInfos.back();
    writes.push_back(w);
  }
  vkUpdateDescriptorSets(device_->device, writes.size(), writes.data(), 0,
                         nullptr);

  VkCommandBuffer cb = beginOneTime();
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout,
                          0, 1, &set, 0, nullptr);
  vkCmdDispatch(cb, wgX, wgY, wgZ);
  endOneTime(cb); // waits for the queue to idle, so scalar buffers are safe
                 // to release now.
  for (auto &sb : scalarBufs) freeBuffer(sb);
  // Return the descriptor set to the pool; the pool was created with
  // FREE_DESCRIPTOR_SET_BIT so this won't exhaust it across launches.
  vkFreeDescriptorSets(device_->device, device_->descriptorPool, 1, &set);
  return VCError::Success;
}

VCError Runtime::synchronize() {
  if (!init_) return VCError::InitializationError;
  vkDeviceWaitIdle(device_->device);
  return VCError::Success;
}

} // namespace vc

//----------------------------------------------------------------------------
// C API wrappers
//----------------------------------------------------------------------------

namespace vc {

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

VCError vcFree(void *devPtr) {
  if (!devPtr) return VCError::Success;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  VCError e = Runtime::get().freeBuffer(*b);
  delete b;
  return e;
}

VCError vcMemcpy(void *dst, const void *src, size_t count, VCMemcpyKind kind) {
  return Runtime::get().copy(dst, src, count, kind);
}

VCError vcDeviceSynchronize() { return Runtime::get().synchronize(); }

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

VCError vcLaunchKernel(VCKernelHandle kernel, unsigned gridDim,
                       unsigned blockDim, const VCKernelArg *args,
                       int argCount) {
  if (!kernel) return VCError::InvalidKernel;
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  return Runtime::get().launch(*k, gridDim, blockDim, args, argCount);
}

VCError vcLaunchKernel2D(VCKernelHandle kernel, unsigned gridDimX,
                         unsigned gridDimY, unsigned blockDimX,
                         unsigned blockDimY, const VCKernelArg *args,
                         int argCount) {
  if (!kernel) return VCError::InvalidKernel;
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  return Runtime::get().launch2D(*k, gridDimX, gridDimY, blockDimX, blockDimY,
                                 args, argCount);
}

} // namespace vc
