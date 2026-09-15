//===- VCRuntime.cpp - CUDA-style runtime over Vulkan (impl) --------------===//

#include "vc/Runtime/VCRuntime.h"
#include "RuntimeInternal.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace vc {

// Process-global kernel-printf toggle (see RuntimeInternal.h). Read once in
// init(); defaults off (validation layer has a runtime/perf cost).
bool g_kernelPrintfEnabled = false;

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
  case VCError::NotReady: return "not ready";
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

// debugPrintfEXT output from kernels arrives here as a validation-layer INFO
// message. We print the formatted text to stderr so it is visible alongside the
// program's own stdout. WARNING/ERROR messages (real Vulkan misuse) are also
// surfaced when the layer is active; VERBOSE is dropped to avoid noise.
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
              VkDebugUtilsMessageTypeFlagsEXT,
              const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  if (!data) return VK_FALSE;
  // Kernel printf comes through as INFO; real misuse is WARNING/ERROR. Always
  // forward those; drop VERBOSE (loader chatter).
  if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) {
    if (data->pMessage) {
      std::fprintf(stderr, "%s\n", data->pMessage);
      std::fflush(stderr);
    }
  }
  return VK_FALSE;
}

// Forward declarations: enumerateDevices (called from init) builds default
// streams before the static definitions below.
static bool initStream(VulkanDevice &dev, VCStream &s, size_t frameRing = 2);
static void teardownStream(VulkanDevice &dev, VCStream &s);
static bool canUseDeviceGroupPeerCopy(const VulkanDevice &dstDevice,
                                      const VulkanDevice &srcDevice,
                                      const VCBuffer &src);

static constexpr size_t kMaxCachedStagingBuffers = 16;
static constexpr size_t kMaxCachedStagingBytes = 64ull * 1024ull * 1024ull;

static uint32_t selectComputeQueueFamily(VkPhysicalDevice physical) {
  uint32_t qf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf, nullptr);
  std::vector<VkQueueFamilyProperties> props(qf);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf, props.data());
  for (uint32_t i = 0; i < qf; ++i)
    if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
  return UINT32_MAX;
}

static void fillPhysicalDeviceInfo(VulkanDevice &vd, VkInstance instance,
                                   VkPhysicalDevice physical,
                                   uint32_t queueFamily) {
  vd.instance = instance;
  vd.physical = physical;
  vd.computeQueueFamily = queueFamily;
  vkGetPhysicalDeviceMemoryProperties(physical, &vd.memProps);
  VkPhysicalDeviceSubgroupProperties sub{};
  sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 p2{};
  p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  p2.pNext = &sub;
  vkGetPhysicalDeviceProperties2(physical, &p2);
  vd.physProps = p2.properties;
  vd.subgroupSize = (int)sub.subgroupSize;
  if (vd.subgroupSize == 0) vd.subgroupSize = 1;
  vd.timestampAvailable =
      p2.properties.limits.timestampComputeAndGraphics == VK_TRUE;
}

Runtime::~Runtime() {
  // Safety net: if the program exits without vcShutdown, the singleton's
  // destructor still runs. Join the host-func thread to avoid std::terminate
  // from destroying a joinable std::thread. If init() was never called or
  // shutdown() already ran, there is nothing to do.
  if (!init_) return;
  hostFuncStop_ = true;
  if (hostFuncThread_.joinable()) hostFuncThread_.join();
  // Release kernel Vulkan handles before devices are destroyed (same reason as
  // shutdown(): the host's VCKernel locals have no destructor to do this).
  for (VCKernel *k : kernelRegistry_) if (k) releaseKernel(*k);
  kernelRegistry_.clear();
  // Best-effort device teardown (shutdown() does the full job normally).
  for (auto &d : devices_) {
    if (!d) continue;
    vkDeviceWaitIdle(d->device);
    if (d->ownsDevice)
      vkDestroyDevice(d->device, nullptr);
  }
  if (!devices_.empty()) {
    VkInstance inst = devices_[0]->instance;
    if (debugMessenger_ && inst) {
      auto vkDestroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
              inst, "vkDestroyDebugUtilsMessengerEXT");
      if (vkDestroyDebugUtilsMessengerEXT)
        vkDestroyDebugUtilsMessengerEXT(inst, debugMessenger_, nullptr);
      debugMessenger_ = VK_NULL_HANDLE;
    }
    vkDestroyInstance(inst, nullptr);
  }
  devices_.clear();
}

VCError Runtime::init() {
  if (init_) return VCError::Success;

  // VC_KERNEL_PRINTF=1 is an alternative entry to vcEnableKernelPrintf (which
  // must run before init). Read once here; the flag gates validation-layer +
  // debugPrintfEXT support for the whole instance/device lifetime.
  if (!g_kernelPrintfEnabled) {
    const char *env = std::getenv("VC_KERNEL_PRINTF");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y'))
      g_kernelPrintfEnabled = true;
  }

  // One shared instance across all devices.
  auto instDev = std::make_unique<VulkanDevice>();
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

  // Kernel printf (debugPrintfEXT) is forwarded to a debug messenger by the
  // Khronos validation layer. When enabled, activate the layer at instance
  // creation and request GPU-based debug printf via the validation-features
  // pNext chain. The layer + DEBUG_PRINTF_EXT feature must be present on the
  // INSTANCE so the layer intercepts NonSemantic.DebugPrintf ExtInst during
  // shader execution and routes the formatted text to the messenger.
  const char *layer = "VK_LAYER_KHRONOS_validation";
  VkValidationFeatureEnableEXT printfEnable =
      VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
  VkValidationFeaturesEXT valFeats{};
  valFeats.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
  valFeats.enabledValidationFeatureCount = 1;
  valFeats.pEnabledValidationFeatures = &printfEnable;
  bool wantPrintf = g_kernelPrintfEnabled;
  if (wantPrintf) {
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = &layer;
    ici.pNext = &valFeats;
  }

  VkResult r = vkCreateInstance(&ici, nullptr, &instDev->instance);
  if (r != VK_SUCCESS) {
    // Retry without debug utils extension (portability ext still required).
    // Keep the validation layer + printf feature if requested (they don't
    // depend on debug_utils being in the instance extension list; the messenger
    // is created via vkGetInstanceProcAddr below and needs the layer, not the
    // instance extension, to receive printf — but debug_utils IS required for
    // the messenger itself, so if it was stripped the messenger just won't be
    // created and printf output is lost; correctness is unaffected).
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = &exts[1];
    r = vkCreateInstance(&ici, nullptr, &instDev->instance);
    if (r != VK_SUCCESS) return VCError::InitializationError;
  }
  VkInstance instance = instDev->instance;

  // When kernel printf is on, register a debug messenger on the instance so the
  // validation layer can deliver debugPrintfEXT output (and any misuse) to
  // debugCallback. INFO severity is required: debugPrintfEXT arrives as INFO.
  if (wantPrintf) {
    auto vkCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (vkCreateDebugUtilsMessengerEXT) {
      VkDebugUtilsMessengerCreateInfoEXT mci{};
      mci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      mci.messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      mci.pfnUserCallback = debugCallback;
      // Non-fatal if messenger creation fails: kernels still run, just without
      // printf output capture. Don't disable printf globally on this — the
      // layer may still print via printf_to_stdout.
      vkCreateDebugUtilsMessengerEXT(instance, &mci, nullptr, &debugMessenger_);
    }
  }

  if (!enumerateDevices(instance)) {
    if (debugMessenger_) {
      auto vkDestroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
              instance, "vkDestroyDebugUtilsMessengerEXT");
      if (vkDestroyDebugUtilsMessengerEXT)
        vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger_, nullptr);
      debugMessenger_ = VK_NULL_HANDLE;
    }
    vkDestroyInstance(instance, nullptr);
    return VCError::InvalidDevice;
  }
  instDev.reset(); // devices_ entries each carry `instance` (shared)

  init_ = true;
  // Background thread dispatches vcLaunchHostFunc callbacks once their
  // stream-ordered semaphore signals. Started after init_ so it sees a valid
  // device; joined at the top of shutdown().
  hostFuncThread_ = std::thread(&Runtime::hostFuncLoop, this);
  return VCError::Success;
}

// Enumerate every physical device with a compute queue and build a full
// VulkanDevice (logical device + default stream + pipeline cache) for each.
// Returns false if none were found. Each entry shares the single instance.
bool Runtime::enumerateDevices(VkInstance instance) {
  uint32_t groupCount = 0;
  vkEnumeratePhysicalDeviceGroups(instance, &groupCount, nullptr);
  std::vector<VkPhysicalDeviceGroupProperties> groups(groupCount);
  for (auto &g : groups)
    g.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
  if (groupCount)
    vkEnumeratePhysicalDeviceGroups(instance, &groupCount, groups.data());

  if (groups.empty()) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    if (n == 0) return false;
    groups.resize(n);
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(instance, &n, phys.data());
    for (uint32_t i = 0; i < n; ++i) {
      groups[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
      groups[i].physicalDeviceCount = 1;
      groups[i].physicalDevices[0] = phys[i];
    }
  }

  for (uint32_t gi = 0; gi < groups.size(); ++gi) {
    VkPhysicalDeviceGroupProperties &group = groups[gi];
    std::vector<VkPhysicalDevice> members;
    std::vector<uint32_t> queueFamilies;
    members.reserve(group.physicalDeviceCount);
    queueFamilies.reserve(group.physicalDeviceCount);
    for (uint32_t i = 0; i < group.physicalDeviceCount; ++i) {
      VkPhysicalDevice physical = group.physicalDevices[i];
      uint32_t qf = selectComputeQueueFamily(physical);
      if (qf == UINT32_MAX) continue;
      members.push_back(physical);
      queueFamilies.push_back(qf);
    }
    if (members.empty()) continue;

    bool canUseDeviceGroup = members.size() > 1;
    for (uint32_t qf : queueFamilies)
      if (qf != queueFamilies[0]) canUseDeviceGroup = false;

    if (!canUseDeviceGroup) {
      for (size_t i = 0; i < members.size(); ++i) {
        auto vd = std::make_unique<VulkanDevice>();
        fillPhysicalDeviceInfo(*vd, instance, members[i], queueFamilies[i]);
        if (!setupLogicalDevice(*vd)) continue;
        vd->defaultStream = std::make_unique<VCStream>();
        if (!initStream(*vd, *vd->defaultStream)) continue;
        vd->defaultStream->deviceIdx = (int)devices_.size();
        VkPipelineCacheCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vkCreatePipelineCache(vd->device, &pci, nullptr, &vd->pipelineCache);
        devices_.push_back(std::move(vd));
      }
      continue;
    }

    auto owner = std::make_unique<VulkanDevice>();
    fillPhysicalDeviceInfo(*owner, instance, members[0], queueFamilies[0]);
    owner->deviceGroup = true;
    owner->groupId = gi;
    owner->groupLocalIndex = 0;
    owner->groupSize = static_cast<uint32_t>(members.size());
    owner->groupDeviceMask = 1u;
    if (!setupLogicalDevice(*owner, members)) continue;
    VkDevice sharedDevice = owner->device;
    VkQueue sharedQueue = owner->computeQueue;
    bool timelineSemaphore = owner->timelineSemaphore;

    std::vector<std::unique_ptr<VulkanDevice>> groupDevices;
    groupDevices.push_back(std::move(owner));
    for (uint32_t i = 1; i < members.size(); ++i) {
      auto vd = std::make_unique<VulkanDevice>();
      fillPhysicalDeviceInfo(*vd, instance, members[i], queueFamilies[i]);
      vd->device = sharedDevice;
      vd->computeQueue = sharedQueue;
      vd->ownsDevice = false;
      vd->deviceGroup = true;
      vd->groupId = gi;
      vd->groupLocalIndex = i;
      vd->groupSize = static_cast<uint32_t>(members.size());
      vd->groupDeviceMask = 1u << i;
      vd->timelineSemaphore = timelineSemaphore;
      groupDevices.push_back(std::move(vd));
    }

    for (auto &vd : groupDevices) {
      vd->defaultStream = std::make_unique<VCStream>();
      if (!initStream(*vd, *vd->defaultStream)) continue;
      vd->defaultStream->deviceIdx = (int)devices_.size();
      VkPipelineCacheCreateInfo pci{};
      pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
      vkCreatePipelineCache(vd->device, &pci, nullptr, &vd->pipelineCache);
      devices_.push_back(std::move(vd));
    }
  }
  return !devices_.empty();
}

// Build the logical device + compute queue for one VulkanDevice (formerly
// createLogicalDevice, now per-device).
bool Runtime::setupLogicalDevice(
    VulkanDevice &vd, const std::vector<VkPhysicalDevice> &deviceGroupMembers) {
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qi{};
  qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qi.queueFamilyIndex = vd.computeQueueFamily;
  qi.queueCount = 1;
  qi.pQueuePriorities = &prio;

  // Enable timeline semaphores (Vulkan 1.2 core feature). Used by stream
  // events (vcEvent*) to express cross-stream dependencies without a full
  // device sync.
  VkPhysicalDeviceVulkan12Features feats12{};
  feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceVulkan12Features supported12{};
  supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceFeatures2 feats2{};
  feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  feats2.pNext = &supported12;
  vkGetPhysicalDeviceFeatures2(vd.physical, &feats2);
  if (supported12.timelineSemaphore) {
    feats12.timelineSemaphore = VK_TRUE;
    vd.timelineSemaphore = true;
  }

  VkDeviceGroupDeviceCreateInfo dgci{};
  bool useDeviceGroup = deviceGroupMembers.size() > 1;
  if (useDeviceGroup) {
    dgci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO;
    dgci.physicalDeviceCount = static_cast<uint32_t>(deviceGroupMembers.size());
    dgci.pPhysicalDevices = deviceGroupMembers.data();
  }

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qi;

  // Device extensions. When kernel printf is on, enable
  // VK_KHR_shader_non_semantic_info so the validation layer recognizes the
  // NonSemantic.DebugPrintf extended instruction set the shaders emit. It is
  // core in Vulkan 1.1+ but enabled explicitly for layer matching on any
  // device (harmless if unsupported — vkCreateDevice would just fail, and we
  // only set it when the user opted into printf).
  const char *devExts[1];
  if (g_kernelPrintfEnabled) {
    devExts[0] = VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;
  }

  if (vd.timelineSemaphore) {
    feats12.pNext = useDeviceGroup ? &dgci : nullptr;
    dci.pNext = &feats12; // replaces pEnabledFeatures chain
  } else {
    if (useDeviceGroup) dci.pNext = &dgci;
    VkPhysicalDeviceFeatures feats{};
    dci.pEnabledFeatures = &feats;
  }

  if (vkCreateDevice(vd.physical, &dci, nullptr, &vd.device) != VK_SUCCESS)
    return false;
  vkGetDeviceQueue(vd.device, vd.computeQueueFamily, 0, &vd.computeQueue);
  return true;
}

VCError Runtime::shutdown() {
  if (!init_) return VCError::Success;
  // Idle every device before tearing anything down.
  for (auto &d : devices_) if (d) vkDeviceWaitIdle(d->device);
  // Stop the host-callback thread first: it dispatches any remaining
  // vcLaunchHostFunc callbacks (the devices are idle, so their semaphores have
  // all signaled) then exits. Must join before tearing down the devices the
  // thread polls.
  hostFuncStop_ = true;
  if (hostFuncThread_.joinable()) hostFuncThread_.join();
  // Reclaim any buffers whose release was deferred by vcFreeAsync — the devices
  // are idle, so they are all safe to free now (avoids leaking them).
  drainPendingFrees();
  // Graphs/events/streams each belong to a specific device (their deviceIdx);
  // tear them down with that device's handle.
  for (auto &g : graphs_) {
    if (!g) continue;
    VulkanDevice &gd = *devices_[g->deviceIdx];
    resetGraphState(gd, *g);
    if (g->secondaryCB && g->commandPool)
      vkFreeCommandBuffers(gd.device, g->commandPool, 1, &g->secondaryCB);
    if (g->descriptorPool)
      vkDestroyDescriptorPool(gd.device, g->descriptorPool, nullptr);
    if (g->commandPool)
      vkDestroyCommandPool(gd.device, g->commandPool, nullptr);
  }
  graphs_.clear();
  for (auto &e : events_) {
    if (!e) continue;
    VulkanDevice &ed = *devices_[e->deviceIdx];
    if (e->semaphore) vkDestroySemaphore(ed.device, e->semaphore, nullptr);
    if (e->queryPool) vkDestroyQueryPool(ed.device, e->queryPool, nullptr);
  }
  events_.clear();
  for (auto &s : streams_) if (s) teardownStream(*devices_[s->deviceIdx], *s);
  streams_.clear();
  // Release every kernel's per-device Vulkan handles (shader module, pipelines,
  // layouts) BEFORE destroying the devices they belong to. The host caller's
  // VCKernel is a raw local with no destructor, so without this the handles
  // leak (visible as OBJ_ERRORs when the validation layer is on).
  for (VCKernel *k : kernelRegistry_) if (k) releaseKernel(*k);
  kernelRegistry_.clear();
  // Destroy each device: default stream, pipeline cache, logical device.
  VkInstance instance = devices_.empty() ? VK_NULL_HANDLE : devices_[0]->instance;
  for (auto &d : devices_) {
    if (!d) continue;
    if (d->defaultStream) teardownStream(*d, *d->defaultStream);
    if (d->pipelineCache) vkDestroyPipelineCache(d->device, d->pipelineCache, nullptr);
    if (d->ownsDevice)
      vkDestroyDevice(d->device, nullptr);
  }
  devices_.clear();
  // Destroy the debug messenger BEFORE the instance it belongs to. The messenger
  // is on the shared instance (devices_[0]->instance, captured in `instance`).
  if (debugMessenger_ && instance) {
    auto vkDestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (vkDestroyDebugUtilsMessengerEXT)
      vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger_, nullptr);
    debugMessenger_ = VK_NULL_HANDLE;
  }
  if (instance) vkDestroyInstance(instance, nullptr);
  currentDeviceIdx_ = 0;
  init_ = false;
  return VCError::Success;
}

static bool initStream(VulkanDevice &dev, VCStream &s,
                       size_t frameRing) {
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
  for (auto &sb : s.stagingCache) {
    if (sb->mapped) vkUnmapMemory(dev.device, sb->memory);
    if (sb->buffer) vkDestroyBuffer(dev.device, sb->buffer, nullptr);
    if (sb->memory) vkFreeMemory(dev.device, sb->memory, nullptr);
  }
  s.stagingCache.clear();
  s.stagingCacheBytes = 0;
  if (!cbs.empty() && s.commandPool)
    vkFreeCommandBuffers(dev.device, s.commandPool,
                         static_cast<uint32_t>(cbs.size()), cbs.data());
  if (s.commandPool) vkDestroyCommandPool(dev.device, s.commandPool, nullptr);
}

static bool shouldCacheStaging(const VCBuffer &buf) {
  return buf.hostVisible && buf.mapped && !buf.managed && buf.buffer &&
         buf.memory && buf.size <= kMaxCachedStagingBytes;
}

static std::unique_ptr<VCBuffer> takeCachedStaging(VCStream &s, size_t bytes) {
  auto best = s.stagingCache.end();
  for (auto it = s.stagingCache.begin(); it != s.stagingCache.end(); ++it) {
    if (!*it || (*it)->size < bytes) continue;
    if (best == s.stagingCache.end() || (*it)->size < (*best)->size)
      best = it;
  }
  if (best == s.stagingCache.end()) return nullptr;
  std::unique_ptr<VCBuffer> out = std::move(*best);
  s.stagingCacheBytes -= out->size;
  s.stagingCache.erase(best);
  return out;
}

static bool cacheCompletedStaging(VCStream &s, std::unique_ptr<VCBuffer> &buf) {
  if (!buf || !shouldCacheStaging(*buf)) return false;
  if (s.stagingCache.size() >= kMaxCachedStagingBuffers ||
      s.stagingCacheBytes + buf->size > kMaxCachedStagingBytes)
    return false;
  s.stagingCacheBytes += buf->size;
  s.stagingCache.push_back(std::move(buf));
  return true;
}

//----------------------------------------------------------------------------
// Streams
//----------------------------------------------------------------------------

VCStream &Runtime::resolveStream(VCStreamHandle h) {
  return h ? *reinterpret_cast<VCStream *>(h) : *device().defaultStream;
}

VCError Runtime::createStream(VCStreamHandle *out) {
  if (!init_ || !out) return VCError::InitializationError;
  auto s = std::make_unique<VCStream>();
  if (!initStream(device(), *s)) return VCError::InitializationError;
  s->deviceIdx = currentDeviceIdx_;
  *out = reinterpret_cast<VCStreamHandle>(s.get());
  streams_.push_back(std::move(s));
  return VCError::Success;
}

VCError Runtime::destroyStream(VCStreamHandle stream) {
  if (!init_ || !stream) return VCError::Success;
  auto *s = reinterpret_cast<VCStream *>(stream);
  vkQueueWaitIdle(s->queue);
  teardownStream(*devices_[s->deviceIdx], *s);
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
          std::memcpy(Runtime::get().resolveHostAccess(rb.hostDst),
                      rb.staging->mapped, rb.bytes);
        f.d2hReadbacks.clear();
      }
    }
  }
}

VCError Runtime::streamSynchronize(VCStream &s) {
  VkDevice dev = devices_[s.deviceIdx]->device;
  // Wait for every in-flight frame's fence so all queued work is done.
  for (size_t i = 0; i < s.frames.size(); ++i) {
    // The frame at frameIdx may be unsignaled (about to be reused); waiting
    // on an already-signaled fence is cheap, so just wait on all of them.
    if (vkWaitForFences(dev, 1, &s.frames[i].fence, VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS)
      return VCError::Unknown;
  }
  // Now every fence is signaled: deliver all deferred D2H readbacks so the
  // caller can read the host destinations immediately after sync returns.
  drainReadyReadbacks(*devices_[s.deviceIdx], s);
  // Dispatch any stream-ordered host callbacks (vcLaunchHostFunc) whose
  // semaphore signaled with the stream's submissions — they are due now.
  // Also covers callbacks whose semaphore has not yet signaled: the stream's
  // fences are done, so the empty-submit signal must have fired too; loop a
  // few polls to let the timeline counter propagate.
  for (int i = 0; i < 64; ++i) {
    size_t before = 0;
    { std::lock_guard<std::mutex> lk(hostFuncMu_); before = pendingHostFuncs_.size(); }
    if (before == 0) break;
    drainHostFuncs();
    size_t after = 0;
    { std::lock_guard<std::mutex> lk(hostFuncMu_); after = pendingHostFuncs_.size(); }
    if (after == 0) break;
    // Still pending (semaphore counter not yet visible): wait briefly for the
    // GPU/driver to publish it, then retry.
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  // Reclaim buffers whose release was deferred by vcFreeAsync on this stream —
  // its work is now done, so they are safe to free.
  drainPendingFrees();
  return VCError::Success;
}

// Non-blocking completion query: a stream is "done" when every frame's fence
// has signaled (i.e. no submission is in flight). Mirrors cudaStreamQuery.
VCError Runtime::streamQuery(const VCStream &s, int *done) const {
  if (!done) return VCError::InvalidValue;
  VkDevice dev = devices_[s.deviceIdx]->device;
  *done = 1;
  for (auto &f : s.frames) {
    if (f.fence != VK_NULL_HANDLE &&
        vkGetFenceStatus(dev, f.fence) != VK_SUCCESS) {
      *done = 0; // at least one frame still executing
      break;
    }
  }
  return VCError::Success; // query succeeded; *done reflects the state
}

// Wait for the frame we're about to reuse, reset it, begin recording.
VkCommandBuffer Runtime::beginFrame(VCStream &s) {
  VkDevice dev = devices_[s.deviceIdx]->device;
  StreamFrame &f = s.frames[s.frameIdx];
  // Wait for the GPU to finish with this frame's previous submission.
  vkWaitForFences(dev, 1, &f.fence, VK_TRUE, UINT64_MAX);
  // The fence just signaled, so every device->staging copy recorded in this
  // frame has executed: deliver deferred D2H readbacks to their host targets.
  // (The caller is responsible for having synchronized before reading.)
  for (auto &rb : f.d2hReadbacks)
    std::memcpy(resolveHostAccess(rb.hostDst), rb.staging->mapped, rb.bytes);
  f.d2hReadbacks.clear();
  // Recycle completed staging buffers now that the GPU is done with them.
  for (auto &sb : f.stagingBuffers) {
    if (cacheCompletedStaging(s, sb)) continue;
    if (sb) freeBuffer(*sb);
  }
  f.stagingBuffers.clear();

  vkResetFences(dev, 1, &f.fence);
  vkResetCommandBuffer(f.cb, 0);
  vkResetDescriptorPool(dev, f.descriptorPool, 0);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkDeviceGroupCommandBufferBeginInfo dgbi{};
  if (devices_[s.deviceIdx]->deviceGroup) {
    dgbi.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO;
    dgbi.deviceMask = devices_[s.deviceIdx]->groupDeviceMask;
    bi.pNext = &dgbi;
  }
  vkBeginCommandBuffer(f.cb, &bi);
  return f.cb;
}

void Runtime::endFrame(VCStream &s) {
  VulkanDevice &vd = *devices_[s.deviceIdx];
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
  if (vd.timelineSemaphore && !s.pendingWaits.empty()) {
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
  if (vd.timelineSemaphore && !s.pendingSignals.empty()) {
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

  VkDeviceGroupSubmitInfo dgsi{};
  uint32_t commandDeviceMask = vd.groupDeviceMask;
  std::vector<uint32_t> waitDeviceIndices;
  std::vector<uint32_t> signalDeviceIndices;
  if (vd.deviceGroup) {
    waitDeviceIndices.assign(waitSems.size(), vd.groupLocalIndex);
    signalDeviceIndices.assign(signalSems.size(), vd.groupLocalIndex);
    dgsi.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO;
    dgsi.waitSemaphoreCount = static_cast<uint32_t>(waitDeviceIndices.size());
    dgsi.pWaitSemaphoreDeviceIndices = waitDeviceIndices.empty()
                                           ? nullptr
                                           : waitDeviceIndices.data();
    dgsi.commandBufferCount = 1;
    dgsi.pCommandBufferDeviceMasks = &commandDeviceMask;
    dgsi.signalSemaphoreCount =
        static_cast<uint32_t>(signalDeviceIndices.size());
    dgsi.pSignalSemaphoreDeviceIndices = signalDeviceIndices.empty()
                                             ? nullptr
                                             : signalDeviceIndices.data();
    dgsi.pNext = si.pNext;
    si.pNext = &dgsi;
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
  vkAllocateDescriptorSets(devices_[s.deviceIdx]->device, &ai, &set);
  return set;
}

//----------------------------------------------------------------------------
// Memory
//----------------------------------------------------------------------------

uint32_t Runtime::findMemoryType(uint32_t reqBits,
                                 VkMemoryPropertyFlags flags) const {
  return findMemoryTypeOn(device(), reqBits, flags);
}

uint32_t Runtime::findMemoryTypeOn(const VulkanDevice &vd, uint32_t reqBits,
                                   VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < vd.memProps.memoryTypeCount; ++i) {
    if ((reqBits & (1u << i)) &&
        (vd.memProps.memoryTypes[i].propertyFlags & flags) == flags)
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
              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buf = VK_NULL_HANDLE;
  vkCreateBuffer(dev, &bci, nullptr, &buf);
  return buf;
}

static VkResult allocateBufferMemoryOn(const VulkanDevice &vd, VkDeviceSize size,
                                       uint32_t memoryTypeIndex,
                                       VkDeviceMemory *memory) {
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = size;
  mai.memoryTypeIndex = memoryTypeIndex;
  VkMemoryAllocateFlagsInfo flags{};
  if (vd.deviceGroup) {
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT;
    flags.deviceMask = vd.groupDeviceMask;
    mai.pNext = &flags;
  }
  return vkAllocateMemory(vd.device, &mai, nullptr, memory);
}

static VkResult bindBufferMemoryOn(const VulkanDevice &vd, VkBuffer buffer,
                                   VkDeviceMemory memory) {
  if (!vd.deviceGroup)
    return vkBindBufferMemory(vd.device, buffer, memory, 0);

  std::vector<uint32_t> deviceIndices(vd.groupSize, vd.groupLocalIndex);
  VkBindBufferMemoryDeviceGroupInfo dgbi{};
  dgbi.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO;
  dgbi.deviceIndexCount = static_cast<uint32_t>(deviceIndices.size());
  dgbi.pDeviceIndices = deviceIndices.data();

  VkBindBufferMemoryInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
  bi.pNext = &dgbi;
  bi.buffer = buffer;
  bi.memory = memory;
  bi.memoryOffset = 0;
  return vkBindBufferMemory2(vd.device, 1, &bi);
}

// device-local: not mapped. Falls back to host-visible if no device-local
// type satisfies the buffer (e.g. on integrated GPUs device-local == host).
VCError Runtime::mallocBuffer(size_t bytes, VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  VulkanDevice &vd = device();
  out.deviceIdx = currentDeviceIdx_;
  out.size = bytes;
  out.buffer = createBuffer(vd.device, bytes);
  if (!out.buffer) return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(vd.device, out.buffer, &reqs);
  uint32_t typeIdx = findMemoryType(
      reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  bool hostFallback = false;
  if (typeIdx == UINT32_MAX) {
    typeIdx = findMemoryType(
        reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hostFallback = true;
    if (typeIdx == UINT32_MAX) {
      vkDestroyBuffer(vd.device, out.buffer, nullptr);
      return VCError::OutOfMemory;
    }
  }

  if (allocateBufferMemoryOn(vd, reqs.size, typeIdx, &out.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  if (bindBufferMemoryOn(vd, out.buffer, out.memory) != VK_SUCCESS) {
    vkFreeMemory(vd.device, out.memory, nullptr);
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    out = VCBuffer{};
    return VCError::OutOfMemory;
  }
  out.hostVisible = hostFallback;
  out.memoryHeapIndex = vd.memProps.memoryTypes[typeIdx].heapIndex;
  if (hostFallback)
    vkMapMemory(vd.device, out.memory, 0, bytes, 0, &out.mapped);
  return VCError::Success;
}

// host-visible + coherent, persistently mapped (pinned staging), on a
// specific device. Used by the copy paths so staging lives on the SAME device
// as the stream/buffer the copy targets — staging allocated on the "current"
// device would be a VkBuffer on the wrong VkDevice for a cross-device stream.
VCError Runtime::mallocHostBufferOn(int deviceIdx, size_t bytes,
                                    VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  VulkanDevice &vd = *devices_[deviceIdx];
  out.deviceIdx = deviceIdx;
  out.size = bytes;
  out.buffer = createBuffer(vd.device, bytes);
  if (!out.buffer) return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(vd.device, out.buffer, &reqs);
  uint32_t typeIdx = findMemoryTypeOn(vd, reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (typeIdx == UINT32_MAX) {
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  if (allocateBufferMemoryOn(vd, reqs.size, typeIdx, &out.memory) != VK_SUCCESS) {
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  if (bindBufferMemoryOn(vd, out.buffer, out.memory) != VK_SUCCESS) {
    vkFreeMemory(vd.device, out.memory, nullptr);
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    out = VCBuffer{};
    return VCError::OutOfMemory;
  }
  vkMapMemory(vd.device, out.memory, 0, bytes, 0, &out.mapped);
  out.hostVisible = true;
  out.memoryHeapIndex = vd.memProps.memoryTypes[typeIdx].heapIndex;
  return VCError::Success;
}

// host-visible staging on the current device.
VCError Runtime::mallocHostBuffer(size_t bytes, VCBuffer &out) {
  return mallocHostBufferOn(currentDeviceIdx_, bytes, out);
}

// Unified memory: device storage that is also persistently mapped + host-
// coherent, so host and device share the same payload with no vcMemcpy.
// Prefer DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (true unified — UMA, or
// discrete GPU with a sufficiently large BAR window). Fall back to plain
// HOST_VISIBLE | HOST_COHERENT (host-visible, possibly non-device-local) so
// the unified-pointer contract holds even without a device-local host type.
// Vulkan has no page-fault migration, so this is zero-copy shared memory, not
// HMM-style on-demand migration.
VCError Runtime::mallocManagedBuffer(size_t bytes, VCBuffer &out) {
  if (!init_) return VCError::InitializationError;
  VulkanDevice &vd = *devices_[currentDeviceIdx_];
  out.deviceIdx = currentDeviceIdx_;
  out.size = bytes;
  out.buffer = createBuffer(vd.device, bytes);
  if (!out.buffer) return VCError::OutOfMemory;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(vd.device, out.buffer, &reqs);
  uint32_t typeIdx = findMemoryTypeOn(vd, reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (typeIdx == UINT32_MAX) {
    // No device-local host-visible type: fall back to host-visible (still
    // satisfies the shared-pointer contract, just not device-local).
    typeIdx = findMemoryTypeOn(vd, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIdx == UINT32_MAX) {
      vkDestroyBuffer(vd.device, out.buffer, nullptr);
      return VCError::OutOfMemory;
    }
  }
  if (allocateBufferMemoryOn(vd, reqs.size, typeIdx, &out.memory) != VK_SUCCESS) {
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    return VCError::OutOfMemory;
  }
  if (bindBufferMemoryOn(vd, out.buffer, out.memory) != VK_SUCCESS) {
    vkFreeMemory(vd.device, out.memory, nullptr);
    vkDestroyBuffer(vd.device, out.buffer, nullptr);
    out = VCBuffer{};
    return VCError::OutOfMemory;
  }
  vkMapMemory(vd.device, out.memory, 0, bytes, 0, &out.mapped);
  out.hostVisible = true;
  out.managed = true;
  out.memoryHeapIndex = vd.memProps.memoryTypes[typeIdx].heapIndex;
  return VCError::Success;
}

VCError Runtime::freeBuffer(VCBuffer &buf) {
  if (!init_) return VCError::InitializationError;
  VkDevice dev = devices_[buf.deviceIdx]->device;
  if (buf.mapped) {
    vkUnmapMemory(dev, buf.memory);
    buf.mapped = nullptr;
  }
  if (buf.buffer) vkDestroyBuffer(dev, buf.buffer, nullptr);
  if (buf.memory) vkFreeMemory(dev, buf.memory, nullptr);
  buf = VCBuffer{};
  return VCError::Success;
}

// Fill `bytes` of `buf` (offset 0) with byte `value` broadcast to uint32.
// vkCmdFillBuffer writes a 4-byte uint32 repeatedly, so `bytes` must be a
// multiple of 4. The buffer already carries TRANSFER_DST usage (see
// createBuffer), so this writes the device buffer directly — no staging.
VCError Runtime::memsetBuffer(VCBuffer &buf, int value, size_t bytes,
                              VCStream &s) {
  if (bytes == 0) return VCError::Success;
  if (bytes % 4 != 0) return VCError::InvalidValue; // vkCmdFillBuffer constraint
  uint32_t fill = (uint32_t)(uint8_t)value * 0x01010101u;
  if (s.captureTarget) {
    vkCmdFillBuffer(s.captureTarget->secondaryCB, buf.buffer, 0, bytes, fill);
    // Make the transfer write visible to subsequent ops in the graph (same
    // barrier as the D2D copy's capture branch).
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
  vkCmdFillBuffer(cb, buf.buffer, 0, bytes, fill);
  endFrame(s);
  streamSynchronize(s); // sync: fill completes before we return
  return VCError::Success;
}

VCError Runtime::memsetBufferAsync(VCBuffer &buf, int value, size_t bytes,
                                   VCStream &s) {
  if (bytes == 0) return VCError::Success;
  if (bytes % 4 != 0) return VCError::InvalidValue;
  uint32_t fill = (uint32_t)(uint8_t)value * 0x01010101u;
  if (s.captureTarget) {
    vkCmdFillBuffer(s.captureTarget->secondaryCB, buf.buffer, 0, bytes, fill);
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
  vkCmdFillBuffer(cb, buf.buffer, 0, bytes, fill);
  endFrame(s);
  // async: do NOT sync — caller must vcStreamSynchronize before reading.
  return VCError::Success;
}

// Deferred free (cudaFreeAsync semantics): the buffer is not released here.
// It is queued and reclaimed once all work already submitted to `s` has
// finished (its frame fences signal). This makes "submit a launch referencing
// buf, then immediately freeAsync(buf), then keep going" safe — no
// use-after-free, and the host did not stall waiting for the GPU.
VCError Runtime::freeBufferAsync(VCBuffer *buf, VCStream &s) {
  if (!buf) return VCError::Success;
  pendingFrees_.push_back({buf, &s});
  return VCError::Success;
}

// Reclaim every deferred buffer whose stream's in-flight work is done. A
// stream is "done" when all its frame fences are signaled (queried
// non-blockingly with vkGetFenceStatus). Buffers whose stream is still busy
// stay queued for a later drain. Called from streamSynchronize / synchronize /
// shutdown.
void Runtime::drainPendingFrees() {
  for (auto it = pendingFrees_.begin(); it != pendingFrees_.end();) {
    VCStream *s = it->stream;
    bool ready = true;
    if (s) {
      VkDevice sdev = devices_[s->deviceIdx]->device;
      for (auto &f : s->frames) {
        if (f.fence != VK_NULL_HANDLE &&
            vkGetFenceStatus(sdev, f.fence) != VK_SUCCESS) {
          ready = false;
          break;
        }
      }
    }
    if (ready) {
      freeBuffer(*it->buf);
      delete it->buf;
      it = pendingFrees_.erase(it);
    } else {
      ++it;
    }
  }
}

// 2D pitched D2D copy: height rows of `width` bytes, src stride spitch, dst
// stride dpitch. vkCmdCopyBuffer accepts multiple regions in one call, so this
// is a single submit with one region per row — no copy kernel needed.
VCError Runtime::copy2DAsync(VCBuffer &dst, size_t dpitch, const VCBuffer &src,
                             size_t spitch, size_t width, size_t height,
                             VCStream &s) {
  if (width == 0 || height == 0) return VCError::Success;
  std::vector<VkBufferCopy> regions(height);
  for (size_t r = 0; r < height; ++r)
    regions[r] = {r * spitch, r * dpitch, width};
  if (s.captureTarget) {
    vkCmdCopyBuffer(s.captureTarget->secondaryCB, src.buffer, dst.buffer,
                    static_cast<uint32_t>(height), regions.data());
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
  vkCmdCopyBuffer(cb, src.buffer, dst.buffer,
                  static_cast<uint32_t>(height), regions.data());
  endFrame(s);
  return VCError::Success;
}

VCError Runtime::copy2D(VCBuffer &dst, size_t dpitch, const VCBuffer &src,
                        size_t spitch, size_t width, size_t height,
                        VCStream &s) {
  VCError e = copy2DAsync(dst, dpitch, src, spitch, width, height, s);
  if (e != VCError::Success) return e;
  streamSynchronize(s);
  return VCError::Success;
}

// 2D pitched fill: height rows of `width` bytes at stride `pitch`. width must
// be a multiple of 4 (vkCmdFillBuffer). One vkCmdFillBuffer per row, all in the
// same command buffer — Vulkan executes recorded commands in order, so no
// intervening barrier is needed.
VCError Runtime::memset2DBufferAsync(VCBuffer &buf, size_t pitch, int value,
                                     size_t width, size_t height,
                                     VCStream &s) {
  if (width == 0 || height == 0) return VCError::Success;
  if (width % 4 != 0) return VCError::InvalidValue;
  uint32_t fill = (uint32_t)(uint8_t)value * 0x01010101u;
  if (s.captureTarget) {
    for (size_t r = 0; r < height; ++r)
      vkCmdFillBuffer(s.captureTarget->secondaryCB, buf.buffer, r * pitch,
                      width, fill);
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
  for (size_t r = 0; r < height; ++r)
    vkCmdFillBuffer(cb, buf.buffer, r * pitch, width, fill);
  endFrame(s);
  return VCError::Success;
}

VCError Runtime::memset2DBuffer(VCBuffer &buf, size_t pitch, int value,
                                size_t width, size_t height, VCStream &s) {
  VCError e = memset2DBufferAsync(buf, pitch, value, width, height, s);
  if (e != VCError::Success) return e;
  streamSynchronize(s);
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
// Uses the async staging path, then synchronizes for the synchronous API.
VCError Runtime::copyHostToDevice(VCBuffer &dst, const void *hostSrc,
                                  size_t bytes, VCStream &s) {
  // Graph capture: allocate a persistent staging buffer owned by the graph
  // (transient staging would be freed before the graph ever executes), fill
  // it now, and record the device copy into the secondary command buffer.
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    auto staging = std::make_unique<VCBuffer>();
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    std::memcpy(staging->mapped,
                resolveHostAccess(const_cast<void *>(hostSrc)), bytes);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, staging->buffer, dst.buffer, 1, &region);
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  VCError e = copyHostToDeviceAsync(dst, hostSrc, bytes, s);
  if (e != VCError::Success) return e;
  return streamSynchronize(s);
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
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
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
  VCError e = copyDeviceToHostAsync(hostDst, src, bytes, s);
  if (e != VCError::Success) return e;
  return streamSynchronize(s);
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
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    std::memcpy(staging->mapped,
                resolveHostAccess(const_cast<void *>(hostSrc)), bytes);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, staging->buffer, dst.buffer, 1, &region);
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  auto staging = takeCachedStaging(s, bytes);
  if (!staging) {
    staging = std::make_unique<VCBuffer>();
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
  }
  // Snapshot the host source NOW so the caller can overwrite it immediately.
  std::memcpy(staging->mapped,
              resolveHostAccess(const_cast<void *>(hostSrc)), bytes);
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
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g.secondaryCB, src.buffer, staging->buffer, 1, &region);
    VCBuffer *raw = staging.get();
    g.d2hReadbacks.push_back({hostDst, raw, bytes});
    g.stagingBuffers.push_back(std::move(staging));
    return VCError::Success;
  }
  auto staging = takeCachedStaging(s, bytes);
  if (!staging) {
    staging = std::make_unique<VCBuffer>();
    if (mallocHostBufferOn(s.deviceIdx, bytes, *staging) != VCError::Success)
      return VCError::OutOfMemory;
  }
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
  for (auto &d : devices_) if (d) vkDeviceWaitIdle(d->device);
  // Deliver deferred D2H readbacks on every stream across every device (the
  // caller may read host destinations immediately after a device sync).
  for (auto &d : devices_)
    if (d && d->defaultStream) drainReadyReadbacks(*d, *d->defaultStream);
  for (auto &s : streams_) if (s) drainReadyReadbacks(*devices_[s->deviceIdx], *s);
  // The device is idle: every deferred free is now safe to reclaim.
  drainPendingFrees();
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
  g->deviceIdx = currentDeviceIdx_;
  VulkanDevice &vd = *devices_[currentDeviceIdx_];

  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = vd.computeQueueFamily;
  if (vkCreateCommandPool(vd.device, &pci, nullptr, &g->commandPool) !=
      VK_SUCCESS)
    return VCError::Unknown;

  g->descriptorPool = createGraphDescriptorPool(vd.device);
  if (!g->descriptorPool) {
    vkDestroyCommandPool(vd.device, g->commandPool, nullptr);
    return VCError::Unknown;
  }

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = g->commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  ai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(vd.device, &ai, &g->secondaryCB) !=
      VK_SUCCESS) {
    vkDestroyDescriptorPool(vd.device, g->descriptorPool, nullptr);
    vkDestroyCommandPool(vd.device, g->commandPool, nullptr);
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
  VulkanDevice &gd = *devices_[g->deviceIdx];
  // Clear capture on the graph's device default stream if mid-record.
  if (gd.defaultStream && gd.defaultStream->captureTarget == g)
    gd.defaultStream->captureTarget = nullptr;
  vkDeviceWaitIdle(gd.device);
  resetGraphState(gd, *g);
  if (g->secondaryCB && g->commandPool)
    vkFreeCommandBuffers(gd.device, g->commandPool, 1, &g->secondaryCB);
  if (g->descriptorPool)
    vkDestroyDescriptorPool(gd.device, g->descriptorPool, nullptr);
  if (g->commandPool)
    vkDestroyCommandPool(gd.device, g->commandPool, nullptr);
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
    resetGraphState(*devices_[g.deviceIdx], g);
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
  VkDeviceGroupCommandBufferBeginInfo dgbi{};
  if (devices_[g.deviceIdx]->deviceGroup) {
    dgbi.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO;
    dgbi.deviceMask = devices_[g.deviceIdx]->groupDeviceMask;
    bi.pNext = &dgbi;
  }
  if (vkBeginCommandBuffer(g.secondaryCB, &bi) != VK_SUCCESS)
    return VCError::Unknown;
  g.recording = true;
  devices_[g.deviceIdx]->defaultStream->captureTarget = &g;
  return VCError::Success;
}

VCError Runtime::endRecord(VCGraph &g) {
  if (!g.recording) return VCError::Unknown; // not recording
  if (vkEndCommandBuffer(g.secondaryCB) != VK_SUCCESS) {
    devices_[g.deviceIdx]->defaultStream->captureTarget = nullptr;
    g.recording = false;
    return VCError::Unknown;
  }
  g.recording = false;
  g.recorded = true;
  if (devices_[g.deviceIdx]->defaultStream->captureTarget == &g)
    devices_[g.deviceIdx]->defaultStream->captureTarget = nullptr;
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
  if (devices_[g.deviceIdx]->defaultStream && devices_[g.deviceIdx]->defaultStream->captureTarget == &g)
    devices_[g.deviceIdx]->defaultStream->captureTarget = nullptr;
  vkDeviceWaitIdle(devices_[g.deviceIdx]->device);
  resetGraphState(*devices_[g.deviceIdx], g);
  return VCError::Success;
}

//----------------------------------------------------------------------------
// Stream events (timeline semaphore)
//----------------------------------------------------------------------------

VCError Runtime::createEvent(VCEventHandle *out) {
  if (!init_ || !out) return VCError::InitializationError;
  if (!device().timelineSemaphore) return VCError::Unknown; // need timeline sem
  auto e = std::make_unique<VCEvent>();
  VkSemaphoreTypeCreateInfo ti{};
  ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  ti.initialValue = 0;
  VkSemaphoreCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  ci.pNext = &ti;
  if (vkCreateSemaphore(device().device, &ci, nullptr, &e->semaphore) !=
      VK_SUCCESS)
    return VCError::Unknown;
  // One-slot timestamp query pool for vcEventElapsedTime. Created even if the
  // device reports no timestamp support; in that case eventElapsedTime returns
  // an error rather than failing here (so events still work for sync).
  if (device().timestampAvailable) {
    VkQueryPoolCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 1;
    vkCreateQueryPool(device().device, &qci, nullptr, &e->queryPool);
  }
  e->deviceIdx = currentDeviceIdx_;
  e->value = 1;          // next signal value
  e->lastRecorded = 0;   // nothing recorded yet
  *out = reinterpret_cast<VCEventHandle>(e.get());
  events_.push_back(std::move(e));
  return VCError::Success;
}

VCError Runtime::destroyEvent(VCEventHandle event) {
  if (!init_ || !event) return VCError::Success;
  auto *e = reinterpret_cast<VCEvent *>(event);
  VkDevice dev = devices_[e->deviceIdx]->device;
  vkDeviceWaitIdle(dev);
  if (e->semaphore)
    vkDestroySemaphore(dev, e->semaphore, nullptr);
  if (e->queryPool)
    vkDestroyQueryPool(dev, e->queryPool, nullptr);
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
// the stream's recycle semantics are unchanged. We also reset + write a GPU
// timestamp query in the same command buffer so vcEventElapsedTime can later
// read back the GPU-side time at this point.
VCError Runtime::recordEvent(VCEvent &e, VCStream &s) {
  if (!devices_[e.deviceIdx]->timelineSemaphore) return VCError::Unknown;
  uint64_t v = e.value++;
  e.lastRecorded = v;
  s.pendingSignals.push_back({e.semaphore, v});
  // Flush: begin/end a frame with no recorded commands. endFrame attaches the
  // pending signal to the submit and clears it.
  VkCommandBuffer cb = beginFrame(s);
  if (e.queryPool) {
    // Reset then write at the end of the recorded stream work — ALL_COMMANDS
    // captures everything submitted so far on this stream.
    vkCmdResetQueryPool(cb, e.queryPool, 0, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        e.queryPool, 0);
  }
  endFrame(s);
  return VCError::Success;
}

// Wait: arrange for the stream's NEXT submission to wait on the event's most
// recently recorded value before executing. Does NOT flush — the wait rides
// the caller's next launch/copy on this stream. Cross-stream dependency.
VCError Runtime::streamWaitEvent(VCStream &s, VCEvent &e) {
  if (!devices_[s.deviceIdx]->timelineSemaphore) return VCError::Unknown;
  if (e.lastRecorded == 0) return VCError::Success; // never recorded: no-op
  s.pendingWaits.push_back({e.semaphore, e.lastRecorded});
  return VCError::Success;
}

VCError Runtime::eventQuery(const VCEvent &e, int *done) const {
  if (!done) return VCError::InvalidValue;
  if (e.lastRecorded == 0) { *done = 1; return VCError::Success; }
  uint64_t counter = 0;
  if (vkGetSemaphoreCounterValue(devices_[e.deviceIdx]->device, e.semaphore,
                                 &counter) != VK_SUCCESS)
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
  if (vkWaitSemaphores(devices_[e.deviceIdx]->device, &wi, UINT64_MAX) !=
      VK_SUCCESS)
    return VCError::Unknown;
  return VCError::Success;
}

// GPU-side elapsed time between two recorded events. Both must have been
// recorded (lastRecorded != 0) and reached on the GPU; we read each event's
// timestamp query with WAIT_BIT so this blocks until the GPU has written both.
// Ticks are converted to ms via limits.timestampPeriod (ns/tick). Returns
// InvalidValue if either event was never recorded or lacks a query pool, and
// NotReady if the device has no timestamp support.
VCError Runtime::eventElapsedTime(float *ms, const VCEvent &start,
                                  const VCEvent &end) const {
  if (!ms) return VCError::InvalidValue;
  // Both events are expected on the same device; use start's.
  const VulkanDevice &vd = *devices_[start.deviceIdx];
  if (!vd.timestampAvailable) return VCError::NotReady;
  if (start.lastRecorded == 0 || end.lastRecorded == 0)
    return VCError::InvalidValue; // never recorded
  if (!start.queryPool || !end.queryPool) return VCError::NotReady;
  uint64_t t0 = 0, t1 = 0;
  VkResult r0 = vkGetQueryPoolResults(vd.device, start.queryPool, 0, 1,
                                      sizeof(t0), &t0, sizeof(t0),
                                      VK_QUERY_RESULT_64_BIT |
                                          VK_QUERY_RESULT_WAIT_BIT);
  VkResult r1 = vkGetQueryPoolResults(vd.device, end.queryPool, 0, 1,
                                      sizeof(t1), &t1, sizeof(t1),
                                      VK_QUERY_RESULT_64_BIT |
                                          VK_QUERY_RESULT_WAIT_BIT);
  if (r0 != VK_SUCCESS || r1 != VK_SUCCESS) return VCError::Unknown;
  // timestampPeriod is nanoseconds per tick.
  double ns = (double)(int64_t)(t1 - t0) * vd.physProps.limits.timestampPeriod;
  *ms = (float)(ns / 1e6);
  return VCError::Success;
}

// Fill VCDeviceProperties from the cached VkPhysicalDeviceProperties + limits.
// Multi-device: any valid device index is accepted. Fields with no Vulkan
// equivalent (multiProcessorCount, clockRate) are left at 0.
VCError Runtime::getDeviceProperties(VCDeviceProperties *out, int deviceIndex) const {
  if (!out) return VCError::InvalidValue;
  if (deviceIndex < 0 || (size_t)deviceIndex >= devices_.size())
    return VCError::InvalidDevice;
  const VulkanDevice &vd = *devices_[deviceIndex];
  const auto &p = vd.physProps;
  const auto &l = p.limits;
  std::memset(out, 0, sizeof(*out));
  std::strncpy(out->name, p.deviceName, sizeof(out->name) - 1);
  out->warpSize = vd.subgroupSize;
  out->maxThreadsPerBlock = (int)l.maxComputeWorkGroupInvocations;
  for (int i = 0; i < 3; ++i) {
    out->maxThreadsDim[i] = (int)l.maxComputeWorkGroupSize[i];
    out->maxGridSize[i] = (int)l.maxComputeWorkGroupCount[i];
  }
  out->sharedMemPerBlock = l.maxComputeSharedMemorySize;
  out->sharedMemPerMultiprocessor = l.maxComputeSharedMemorySize;
  // totalGlobalMem: size of the first device-local heap.
  for (uint32_t i = 0; i < vd.memProps.memoryHeapCount; ++i) {
    if (vd.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      out->totalGlobalMem = vd.memProps.memoryHeaps[i].size;
      break;
    }
  }
  out->clockRate = 0;            // no Vulkan core query
  out->multiProcessorCount = 0; // Vulkan does not expose SM/CU count
  out->major = (int)VK_API_VERSION_MAJOR(p.apiVersion);
  out->minor = (int)VK_API_VERSION_MINOR(p.apiVersion);
  return VCError::Success;
}

// Limit-based block-size heuristic (not real SM occupancy — Vulkan exposes no
// SM count, per-SM registers, or per-kernel resource usage). Picks the largest
// power of two that fits the invocation/shared/warp constraints, mirroring the
// shape of cudaOccupancyMaxPotentialBlockSize within Vulkan's limited envelope.
VCError Runtime::occupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, size_t dynamicSharedMemPerBlock,
    int blockSizeLimit) const {
  if (!init_) return VCError::InitializationError;
  if (currentDeviceIdx_ < 0 || (size_t)currentDeviceIdx_ >= devices_.size())
    return VCError::InvalidDevice;
  VCDeviceProperties props;
  VCError e = getDeviceProperties(&props, currentDeviceIdx_);
  if (e != VCError::Success) return e;

  const size_t sharedLimit = props.sharedMemPerBlock;
  if (dynamicSharedMemPerBlock > sharedLimit)
    return VCError::OutOfMemory; // shared-mem request exceeds per-block cap

  const int maxInv = props.maxThreadsPerBlock; // maxComputeWorkGroupInvocations
  int cap = (blockSizeLimit > 0 && blockSizeLimit < maxInv) ? blockSizeLimit
                                                            : maxInv;
  if (cap < 1) return VCError::OutOfMemory;

  // Largest power of two <= cap, then rounded down to a multiple of the
  // device warp/subgroup size (so full warps/subgroups are dispatched).
  int blk = 1;
  while (blk * 2 <= cap) blk *= 2;
  const int warp = props.warpSize > 0 ? props.warpSize : 1;
  blk -= blk % warp;
  if (blk < 1) blk = 1;

  if (blockSize) *blockSize = blk;
  if (minGridSize) *minGridSize = props.maxGridSize[0];
  return VCError::Success;
}

// Switch the current device (cudaSetDevice). Subsequent allocations, kernel
// loads, stream creation, and the default stream all target this device. Does
// NOT migrate existing buffers/streams/kernels — each object stays on the
// device it was created on (recorded in its deviceIdx) and is operated on via
// that device regardless of the current selection.
VCError Runtime::setDevice(int idx) {
  if (!init_) return VCError::InitializationError;
  if (idx < 0 || (size_t)idx >= devices_.size())
    return VCError::InvalidDevice;
  currentDeviceIdx_ = idx;
  return VCError::Success;
}

// Cross-device copy (cudaMemcpyPeer): prefer a real Vulkan device-group peer
// copy when both buffers are backed by the same logical device group and the
// source heap exposes COPY_SRC peer access to the destination device. Otherwise
// fall back to the always-correct host bridge. Same-device degenerates to an
// ordinary D2D copy. Sync: returns once the copy lands.
VCError Runtime::copyPeer(VCBuffer &dst, int dstDevice, const VCBuffer &src,
                          int srcDevice, size_t bytes) {
  if (!init_) return VCError::InitializationError;
  if (dstDevice != dst.deviceIdx || srcDevice != src.deviceIdx)
    return VCError::InvalidValue;
  if (bytes == 0) return VCError::Success;
  if (bytes > dst.size || bytes > src.size)
    return VCError::InvalidValue;
  // Same device: ordinary D2D copy on a throwaway synchronization — use the
  // source device's default stream.
  if (srcDevice == dstDevice) {
    VulkanDevice &vd = *devices_[srcDevice];
    VCStream &ds = *vd.defaultStream;
    VkCommandBuffer cb = beginFrame(ds);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &region);
    endFrame(ds);
    streamSynchronize(ds);
    return VCError::Success;
  }
  VulkanDevice &svd = *devices_[srcDevice];
  VulkanDevice &dvd = *devices_[dstDevice];
  if (canUseDeviceGroupPeerCopy(dvd, svd, src)) {
    // The command executes on the destination physical device and reads the
    // source allocation as peer memory inside the same logical device group.
    vkQueueWaitIdle(svd.computeQueue);
    VCStream &ds = *dvd.defaultStream;
    VkCommandBuffer cb = beginFrame(ds);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &region);
    endFrame(ds);
    streamSynchronize(ds);
    return VCError::Success;
  }

  // Cross-device host bridge: D2H on the source device, then H2D on the dest.
  // Idle the source so any pending writes to `src` are visible.
  vkQueueWaitIdle(svd.computeQueue);
  // D2H: staging on the SOURCE device.
  VCBuffer sStaging{};
  if (mallocHostBufferOn(srcDevice, bytes, sStaging) != VCError::Success)
    return VCError::OutOfMemory;
  {
    VCStream &ss = *svd.defaultStream;
    VkCommandBuffer cb = beginFrame(ss);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, src.buffer, sStaging.buffer, 1, &region);
    endFrame(ss);
    streamSynchronize(ss);
  }
  // sStaging.mapped now holds the source bytes on the host.
  // H2D: staging on the DEST device, then D2D into the dst buffer.
  VCBuffer dStaging{};
  if (mallocHostBufferOn(dstDevice, bytes, dStaging) != VCError::Success) {
    freeBuffer(sStaging);
    return VCError::OutOfMemory;
  }
  std::memcpy(dStaging.mapped, sStaging.mapped, bytes);
  {
    VCStream &ds = *dvd.defaultStream;
    VkCommandBuffer cb = beginFrame(ds);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, dStaging.buffer, dst.buffer, 1, &region);
    endFrame(ds);
    streamSynchronize(ds);
  }
  freeBuffer(sStaging);
  freeBuffer(dStaging);
  return VCError::Success;
}

// Look up `ptr` in the allocation registry. VC hands out the VCBuffer* itself
// as the device/host pointer, so a valid pointer is a key in the registry.
VCError Runtime::pointerGetAttributes(VCPointerAttributes *out,
                                      const void *ptr) const {
  if (!out) return VCError::InvalidValue;
  out->memoryType = VCMemoryType::Unregistered;
  out->devicePointer = nullptr;
  out->hostPointer = nullptr;
  out->size = 0;
  auto *b = reinterpret_cast<VCBuffer *>(const_cast<void *>(ptr));
  VCMemoryType kind = VCMemoryType::Unregistered;
  {
    std::lock_guard<std::mutex> lk(allocRegistryMu_);
    auto it = allocRegistry_.find(b);
    if (it == allocRegistry_.end()) return VCError::Success; // unregistered
    kind = it->second;
  }
  out->devicePointer = b;        // the handle VC returned
  out->size = b->size;
  out->memoryType = kind;
  if (kind == VCMemoryType::Host || kind == VCMemoryType::Managed)
    out->hostPointer = b->mapped; // persistently-mapped host address
  return VCError::Success;
}

void *Runtime::resolveHostAccess(void *hostPtr) const {
  if (!hostPtr) return hostPtr;
  auto *b = reinterpret_cast<VCBuffer *>(hostPtr);
  {
    std::lock_guard<std::mutex> lk(allocRegistryMu_);
    auto it = allocRegistry_.find(b);
    if (it != allocRegistry_.end() &&
        (it->second == VCMemoryType::Host ||
         it->second == VCMemoryType::Managed) &&
        b->mapped)
      return b->mapped; // host-visible/managed VCBuffer: use the mapped payload
  }
  return hostPtr; // plain stack/heap pointer: use directly
}

void Runtime::registerBuffer(VCBuffer *b, VCMemoryType kind) {
  std::lock_guard<std::mutex> lk(allocRegistryMu_);
  allocRegistry_[b] = kind;
}

void Runtime::unregisterBuffer(VCBuffer *b) {
  std::lock_guard<std::mutex> lk(allocRegistryMu_);
  allocRegistry_.erase(b);
}

static bool canUseDeviceGroupPeerCopy(const VulkanDevice &dstDevice,
                                      const VulkanDevice &srcDevice,
                                      const VCBuffer &src) {
  if (!dstDevice.deviceGroup || !srcDevice.deviceGroup)
    return false;
  if (dstDevice.device != srcDevice.device ||
      dstDevice.groupId != srcDevice.groupId ||
      dstDevice.groupLocalIndex == srcDevice.groupLocalIndex)
    return false;
  if (src.memoryHeapIndex == UINT32_MAX)
    return false;
  VkPeerMemoryFeatureFlags features = 0;
  vkGetDeviceGroupPeerMemoryFeatures(dstDevice.device, src.memoryHeapIndex,
                                     dstDevice.groupLocalIndex,
                                     srcDevice.groupLocalIndex, &features);
  return (features & VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT) != 0;
}

// Stream-ordered host callback (cudaLaunchHostFunc). Vulkan has no native
// stream-ordered host callback, so we approximate it with a one-shot timeline
// semaphore signaled at the stream's current tail (the same empty-submit
// trick recordEvent uses) plus a background thread that polls the semaphore
// counter and invokes the callback once it signals. This preserves the
// essential contract: the callback runs after all work already submitted to
// the stream, without the host having to poll.
VCError Runtime::launchHostFunc(VCStream &s, VCHostFn fn, void *userData) {
  if (!fn) return VCError::InvalidValue;
  VulkanDevice &vd = *devices_[s.deviceIdx];
  if (!vd.timelineSemaphore) return VCError::Unknown;

  // One-shot timeline semaphore, signaled at value 1.
  VkSemaphoreTypeCreateInfo sti{};
  sti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  sti.initialValue = 0;
  VkSemaphoreCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sci.pNext = &sti;
  VkSemaphore sem = VK_NULL_HANDLE;
  if (vkCreateSemaphore(vd.device, &sci, nullptr, &sem) != VK_SUCCESS)
    return VCError::OutOfMemory;

  // Signal sem@1 at the stream's current tail: push a pending signal, then
  // flush an empty submit (beginFrame+endFrame with no recorded commands).
  // endFrame attaches the pending signal via the timeline submit info path.
  s.pendingSignals.push_back({sem, 1});
  beginFrame(s);
  endFrame(s);

  {
    std::lock_guard<std::mutex> lk(hostFuncMu_);
    pendingHostFuncs_.push_back({sem, 1, fn, userData, s.deviceIdx});
  }
  return VCError::Success;
}

// Background thread: poll every pending callback's semaphore counter; once it
// reaches `value` the preceding stream work is done, so dispatch the callback
// (outside the lock, so callbacks may re-enter the runtime) and destroy the
// semaphore. On shutdown (hostFuncStop_ set) drain whatever remains first.
void Runtime::hostFuncLoop() {
  while (true) {
    // Collect any callbacks whose semaphore has signaled, dispatch them
    // outside the lock (a callback may re-enter the runtime).
    std::vector<PendingHostFunc> ready;
    {
      std::lock_guard<std::mutex> lk(hostFuncMu_);
      for (auto it = pendingHostFuncs_.begin();
           it != pendingHostFuncs_.end();) {
        uint64_t cur = 0;
        if (vkGetSemaphoreCounterValue(devices_[it->deviceIdx]->device,
                                       it->sem, &cur) == VK_SUCCESS
            && cur >= it->value) {
          ready.push_back(*it);
          vkDestroySemaphore(devices_[it->deviceIdx]->device, it->sem, nullptr);
          it = pendingHostFuncs_.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto &h : ready) h.fn(h.userData);

    if (hostFuncStop_.load()) {
      // Shutdown path: the device is idle (shutdown called vkDeviceWaitIdle),
      // so every remaining semaphore has signaled. Dispatch + destroy them.
      std::lock_guard<std::mutex> lk(hostFuncMu_);
      for (auto &h : pendingHostFuncs_) {
        h.fn(h.userData);
        vkDestroySemaphore(devices_[h.deviceIdx]->device, h.sem, nullptr);
      }
      pendingHostFuncs_.clear();
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Dispatch any pending host callbacks whose semaphore has already signaled
// (i.e. the stream work preceding them is done). Called by streamSynchronize
// after waiting the stream's fences: the stream's submissions — including the
// empty submits that signal host-func semaphores — have all completed, so the
// callbacks are due. Dispatching here (rather than waiting for the background
// thread's next poll) makes vcStreamSynchronize also wait for stream-ordered
// host callbacks, which is what callers expect. Safe vs. the background thread:
// we move the entries out of the shared list under the same lock, so the thread
// cannot double-dispatch them.
void Runtime::drainHostFuncs() {
  std::vector<PendingHostFunc> ready;
  {
    std::lock_guard<std::mutex> lk(hostFuncMu_);
    for (auto it = pendingHostFuncs_.begin();
         it != pendingHostFuncs_.end();) {
      uint64_t cur = 0;
      if (vkGetSemaphoreCounterValue(devices_[it->deviceIdx]->device,
                                     it->sem, &cur) == VK_SUCCESS
          && cur >= it->value) {
        ready.push_back(*it);
        vkDestroySemaphore(devices_[it->deviceIdx]->device, it->sem, nullptr);
        it = pendingHostFuncs_.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (auto &h : ready) h.fn(h.userData);
}

//----------------------------------------------------------------------------
// Kernels / pipelines
//----------------------------------------------------------------------------

namespace {

enum SpvOpcode : uint16_t {
  SpvOpName = 5,
  SpvOpMemberName = 6,
  SpvOpDecorate = 71,
  SpvOpMemberDecorate = 72,
  SpvOpTypeInt = 21,
  SpvOpTypeFloat = 22,
  SpvOpTypeVector = 23,
  SpvOpTypeArray = 28,
  SpvOpTypeRuntimeArray = 29,
  SpvOpTypeStruct = 30,
  SpvOpTypePointer = 32,
  SpvOpConstant = 43,
  SpvOpVariable = 59,
};

enum SpvDecoration : uint32_t {
  SpvDecArrayStride = 6,
  SpvDecOffset = 35,
  SpvDecBinding = 33,
  SpvDecDescriptorSet = 34,
};

enum SpvStorageClass : uint32_t {
  SpvStorageUniform = 2,
  SpvStoragePushConstant = 9,
  SpvStorageStorageBuffer = 12,
};

struct SpirvTypeInfo {
  enum Kind { Unknown, Int, Float, Vector, Array, RuntimeArray, Struct, Pointer } kind = Unknown;
  uint32_t bitWidth = 0;
  uint32_t elementType = 0;
  uint32_t elementCount = 0;
  uint32_t arrayStride = 0;
  uint32_t storageClass = 0;
  std::vector<uint32_t> memberTypes;
};

struct SpirvVariableInfo {
  uint32_t resultType = 0;
  uint32_t storageClass = 0;
  uint32_t descriptorSet = 0;
  uint32_t binding = std::numeric_limits<uint32_t>::max();
  std::string name;
};

struct SpirvReflectionInfo {
  std::vector<uint32_t> storageBufferBindings;
  std::vector<uint32_t> pushConstantOffsets;
  uint32_t pushConstantSize = 0;
};

static uint32_t alignTo(uint32_t value, uint32_t alignment) {
  if (alignment == 0) return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

static std::string spirvLiteralString(const uint32_t *words, uint32_t wordCount,
                                      uint32_t firstWord) {
  if (firstWord >= wordCount) return {};
  std::string out;
  const char *bytes = reinterpret_cast<const char *>(words + firstWord);
  uint32_t byteCount = (wordCount - firstWord) * 4;
  for (uint32_t i = 0; i < byteCount && bytes[i] != '\0'; ++i)
    out.push_back(bytes[i]);
  return out;
}

static uint32_t spirvTypeSize(
    uint32_t typeId, const std::unordered_map<uint32_t, SpirvTypeInfo> &types,
    const std::unordered_map<uint32_t, uint32_t> &constants,
    const std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> &memberOffsets);

static uint32_t spirvStructSize(
    uint32_t typeId, const SpirvTypeInfo &type,
    const std::unordered_map<uint32_t, SpirvTypeInfo> &types,
    const std::unordered_map<uint32_t, uint32_t> &constants,
    const std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> &memberOffsets) {
  auto offIt = memberOffsets.find(typeId);
  uint32_t end = 0;
  for (uint32_t i = 0; i < type.memberTypes.size(); ++i) {
    uint32_t offset = 0;
    if (offIt != memberOffsets.end()) {
      auto mit = offIt->second.find(i);
      if (mit != offIt->second.end()) offset = mit->second;
    } else {
      offset = end;
    }
    uint32_t memberSize = spirvTypeSize(type.memberTypes[i], types, constants,
                                        memberOffsets);
    end = std::max(end, offset + memberSize);
  }
  return alignTo(end, 4);
}

static uint32_t spirvTypeSize(
    uint32_t typeId, const std::unordered_map<uint32_t, SpirvTypeInfo> &types,
    const std::unordered_map<uint32_t, uint32_t> &constants,
    const std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> &memberOffsets) {
  auto it = types.find(typeId);
  if (it == types.end()) return 0;
  const SpirvTypeInfo &type = it->second;
  switch (type.kind) {
  case SpirvTypeInfo::Int:
  case SpirvTypeInfo::Float:
    return std::max<uint32_t>(1, type.bitWidth / 8);
  case SpirvTypeInfo::Vector:
    return spirvTypeSize(type.elementType, types, constants, memberOffsets) *
           type.elementCount;
  case SpirvTypeInfo::Array: {
    uint32_t count = type.elementCount;
    auto cit = constants.find(count);
    if (cit != constants.end()) count = cit->second;
    uint32_t stride = type.arrayStride ? type.arrayStride
        : spirvTypeSize(type.elementType, types, constants, memberOffsets);
    return stride * count;
  }
  case SpirvTypeInfo::Struct:
    return spirvStructSize(typeId, type, types, constants, memberOffsets);
  case SpirvTypeInfo::Pointer:
    return spirvTypeSize(type.elementType, types, constants, memberOffsets);
  case SpirvTypeInfo::RuntimeArray:
  case SpirvTypeInfo::Unknown:
    return 0;
  }
  return 0;
}

static SpirvReflectionInfo reflectSpirvResources(const uint32_t *words,
                                                 size_t wordCount,
                                                 const std::string &entryPoint) {
  SpirvReflectionInfo out;
  if (!words || wordCount < 5 || words[0] != 0x07230203) return out;

  std::unordered_map<uint32_t, SpirvTypeInfo> types;
  std::unordered_map<uint32_t, uint32_t> constants;
  std::unordered_map<uint32_t, SpirvVariableInfo> vars;
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> memberOffsets;

  for (size_t offset = 5; offset < wordCount;) {
    uint32_t inst = words[offset];
    uint16_t op = static_cast<uint16_t>(inst & 0xffffu);
    uint16_t wc = static_cast<uint16_t>(inst >> 16);
    if (wc == 0 || offset + wc > wordCount) break;
    const uint32_t *w = words + offset;
    switch (op) {
    case SpvOpName:
      if (wc >= 3)
        vars[w[1]].name = spirvLiteralString(w, wc, 2);
      break;
    case SpvOpMemberName:
      break;
    case SpvOpDecorate:
      if (wc >= 3) {
        uint32_t target = w[1];
        uint32_t decoration = w[2];
        if (decoration == SpvDecBinding && wc >= 4)
          vars[target].binding = w[3];
        else if (decoration == SpvDecDescriptorSet && wc >= 4)
          vars[target].descriptorSet = w[3];
        else if (decoration == SpvDecArrayStride && wc >= 4)
          types[target].arrayStride = w[3];
      }
      break;
    case SpvOpMemberDecorate:
      if (wc >= 5 && w[3] == SpvDecOffset)
        memberOffsets[w[1]][w[2]] = w[4];
      break;
    case SpvOpTypeInt:
      if (wc >= 4) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Int;
        t.bitWidth = w[2];
      }
      break;
    case SpvOpTypeFloat:
      if (wc >= 3) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Float;
        t.bitWidth = w[2];
      }
      break;
    case SpvOpTypeVector:
      if (wc >= 4) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Vector;
        t.elementType = w[2];
        t.elementCount = w[3];
      }
      break;
    case SpvOpTypeArray:
      if (wc >= 4) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Array;
        t.elementType = w[2];
        t.elementCount = w[3];
      }
      break;
    case SpvOpTypeRuntimeArray:
      if (wc >= 3) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::RuntimeArray;
        t.elementType = w[2];
      }
      break;
    case SpvOpTypeStruct:
      if (wc >= 2) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Struct;
        t.memberTypes.assign(w + 2, w + wc);
      }
      break;
    case SpvOpTypePointer:
      if (wc >= 4) {
        auto &t = types[w[1]];
        t.kind = SpirvTypeInfo::Pointer;
        t.storageClass = w[2];
        t.elementType = w[3];
      }
      break;
    case SpvOpConstant:
      if (wc >= 4)
        constants[w[2]] = w[3];
      break;
    case SpvOpVariable:
      if (wc >= 4) {
        auto &v = vars[w[2]];
        v.resultType = w[1];
        v.storageClass = w[3];
      }
      break;
    default:
      break;
    }
    offset += wc;
  }

  struct Binding { uint32_t set; uint32_t binding; uint32_t id; };
  std::vector<Binding> buffers;
  std::vector<Binding> entryNamedBuffers;
  std::string entryArgPrefix = entryPoint + "_arg_";
  for (const auto &kv : vars) {
    const SpirvVariableInfo &v = kv.second;
    if ((v.storageClass == SpvStorageUniform ||
         v.storageClass == SpvStorageStorageBuffer) &&
        v.binding != std::numeric_limits<uint32_t>::max()) {
      buffers.push_back({v.descriptorSet, v.binding, kv.first});
      if (!entryArgPrefix.empty() && v.name.rfind(entryArgPrefix, 0) == 0)
        entryNamedBuffers.push_back({v.descriptorSet, v.binding, kv.first});
    }
    if (v.storageClass == SpvStoragePushConstant) {
      auto typeIt = types.find(v.resultType);
      if (typeIt == types.end() || typeIt->second.kind != SpirvTypeInfo::Pointer)
        continue;
      uint32_t structTypeId = typeIt->second.elementType;
      auto structIt = types.find(structTypeId);
      if (structIt == types.end() || structIt->second.kind != SpirvTypeInfo::Struct)
        continue;
      auto offsetsIt = memberOffsets.find(structTypeId);
      for (uint32_t i = 0; i < structIt->second.memberTypes.size(); ++i) {
        uint32_t memberOffset = out.pushConstantSize;
        if (offsetsIt != memberOffsets.end()) {
          auto mit = offsetsIt->second.find(i);
          if (mit != offsetsIt->second.end()) memberOffset = mit->second;
        }
        out.pushConstantOffsets.push_back(memberOffset);
        uint32_t memberSize = spirvTypeSize(structIt->second.memberTypes[i],
                                            types, constants, memberOffsets);
        out.pushConstantSize = std::max(out.pushConstantSize,
                                        memberOffset + memberSize);
      }
      out.pushConstantSize = std::max(out.pushConstantSize,
          spirvTypeSize(structTypeId, types, constants, memberOffsets));
    }
  }

  if (!entryNamedBuffers.empty())
    buffers = std::move(entryNamedBuffers);

  std::sort(buffers.begin(), buffers.end(), [](const Binding &a, const Binding &b) {
    if (a.set != b.set) return a.set < b.set;
    if (a.binding != b.binding) return a.binding < b.binding;
    return a.id < b.id;
  });
  for (const Binding &b : buffers) {
    if (b.set == 0) out.storageBufferBindings.push_back(b.binding);
  }
  return out;
}

} // namespace

VCError Runtime::loadKernel(const uint32_t *words, size_t wordCount,
                            const char *entryPoint, VCKernel &out) {
  if (!init_) return VCError::InitializationError;
  if (!words || wordCount == 0) return VCError::InvalidValue;
  // Store the device-independent SPIR-V; per-device shader modules / layouts /
  // pipelines are created lazily on first launch on each device.
  out.spirvWords.assign(words, words + wordCount);
  out.entryPoint = entryPoint ? entryPoint : "main";
  SpirvReflectionInfo refl = reflectSpirvResources(words, wordCount,
                                                   out.entryPoint);
  out.storageBufferBindings = std::move(refl.storageBufferBindings);
  out.hasResourceReflection = !out.storageBufferBindings.empty();
  out.pushConstantOffsets = std::move(refl.pushConstantOffsets);
  out.pcSize = refl.pushConstantSize;
  return VCError::Success;
}

// Lazily create the per-device Vulkan state for `k` on `deviceIdx`: a shader
// module built from the kernel's SPIR-V. Layout/pipelines are built later on
// first dispatch. Returns nullptr on shader-module creation failure.
VCKernelDeviceState *Runtime::getOrCreateKernelDeviceState(VCKernel &k,
                                                           int deviceIdx) {
  auto it = k.perDevice.find(deviceIdx);
  if (it != k.perDevice.end()) return it->second.get();
  auto st = std::make_unique<VCKernelDeviceState>();
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = k.spirvWords.size() * sizeof(uint32_t);
  ci.pCode = k.spirvWords.data();
  if (vkCreateShaderModule(devices_[deviceIdx]->device, &ci, nullptr,
                           &st->shaderModule) != VK_SUCCESS)
    return nullptr;
  auto *raw = st.get();
  k.perDevice[deviceIdx] = std::move(st);
  // This kernel now owns Vulkan handles on a device. Register it so shutdown()
  // can releaseKernel() it before vkDestroyDevice — the caller's VCKernel
  // (a raw local in host main) has no destructor that would do this.
  trackKernel(&k);
  return raw;
}

// Remember `k` so its Vulkan handles can be torn down at shutdown. Deduped: a
// kernel launched on N devices calls getOrCreateKernelDeviceState N times but
// registers once. Raw pointer — we don't own the VCKernel struct (the host
// caller does), only the per-device VkHandles inside it.
void Runtime::trackKernel(VCKernel *k) {
  if (!k) return;
  for (VCKernel *existing : kernelRegistry_)
    if (existing == k) return;
  kernelRegistry_.push_back(k);
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
  if (!init_) { k = VCKernel{}; return; }
  for (auto &kv : k.perDevice) {
    VkDevice dev = devices_[kv.first]->device;
    for (auto &p : kv.second->pipelines)
      if (p.second) vkDestroyPipeline(dev, p.second, nullptr);
    if (kv.second->pipelineLayout)
      vkDestroyPipelineLayout(dev, kv.second->pipelineLayout, nullptr);
    if (kv.second->descriptorSetLayout)
      vkDestroyDescriptorSetLayout(dev, kv.second->descriptorSetLayout,
                                   nullptr);
    if (kv.second->shaderModule)
      vkDestroyShaderModule(dev, kv.second->shaderModule, nullptr);
  }
  k = VCKernel{};
}

// Build the descriptor-set + pipeline layouts for `k` on `deviceIdx`. Pointer
// args use reflected SSBO bindings when available (falling back to dense
// zero-based bindings for older shaders); scalar args are packed into a single
// push-constant range using reflected member offsets when present.
// `args`/`argCount` define the arrangement; the layout is built once per
// device and cached on the per-device state.
bool Runtime::buildLayout(VCKernel &k, int deviceIdx,
                          const VCKernelArg *args, int argCount) {
  VCKernelDeviceState *st = getOrCreateKernelDeviceState(k, deviceIdx);
  if (!st) return false;
  VkDevice dev = devices_[deviceIdx]->device;
  k.argCount = argCount;
  // Collect pointer bindings + measure push-constant size for scalars.
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t pointerOrdinal = 0;
  uint32_t scalarOrdinal = 0;
  uint32_t pcSize = k.pcSize;
  for (int i = 0; i < argCount; ++i) {
    if (args[i].kind == VCKernelArg::Pointer) {
      VkDescriptorSetLayoutBinding b{};
      b.binding = pointerOrdinal;
      if (pointerOrdinal < k.storageBufferBindings.size())
        b.binding = k.storageBufferBindings[pointerOrdinal];
      b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b.descriptorCount = 1;
      b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(b);
      ++pointerOrdinal;
    } else {
      // Round scalar size up to 4 bytes for std140-friendly packing.
      uint32_t sz = static_cast<uint32_t>((args[i].size + 3) & ~size_t(3));
      if (scalarOrdinal < k.pushConstantOffsets.size())
        pcSize = std::max(pcSize, k.pushConstantOffsets[scalarOrdinal] + sz);
      else
        pcSize += sz;
      ++scalarOrdinal;
    }
  }

  VkDescriptorSetLayoutCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dci.bindingCount = static_cast<uint32_t>(bindings.size());
  dci.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &dci, nullptr,
                                  &st->descriptorSetLayout) != VK_SUCCESS)
    return false;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &st->descriptorSetLayout;
  VkPushConstantRange pcr{};
  if (pcSize > 0) {
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = pcSize;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
  }
  if (vkCreatePipelineLayout(dev, &plci, nullptr,
                             &st->pipelineLayout) != VK_SUCCESS)
    return false;
  k.pcSize = pcSize;
  st->layoutBuilt = true;
  return true;
}

// Fetch or create a pipeline specialized to the block size, on `deviceIdx`.
// Layout must be built first (via buildLayout) — done in dispatch on first use.
VkPipeline Runtime::getPipeline(VCKernel &k, int deviceIdx, unsigned blockX,
                                unsigned blockY, unsigned blockZ,
                                const VCKernelArg *args, int argCount) {
  VCKernelDeviceState *st = getOrCreateKernelDeviceState(k, deviceIdx);
  if (!st) return VK_NULL_HANDLE;
  if (!st->layoutBuilt && !buildLayout(k, deviceIdx, args, argCount))
    return VK_NULL_HANDLE;

  // Key: pack block dims into 64 bits (16 bits each + reserved).
  uint64_t key = (uint64_t(blockX) << 32) | (uint64_t(blockY) << 16) | blockZ;
  auto it = st->pipelines.find(key);
  if (it != st->pipelines.end()) return it->second;

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
  pci.stage.module = st->shaderModule;
  pci.stage.pName = k.entryPoint.c_str();
  pci.stage.pSpecializationInfo = &spec;
  pci.layout = st->pipelineLayout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(devices_[deviceIdx]->device,
                               devices_[deviceIdx]->pipelineCache, 1,
                               &pci, nullptr, &pipeline) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  st->pipelines[key] = pipeline;
  return pipeline;
}

//----------------------------------------------------------------------------
// Launch
//----------------------------------------------------------------------------

// Shared bind sequence: get/create the pipeline specialized to the block size,
// allocate + write the descriptor set for pointer args, pack scalars into push
// constants, and record vkCmdBindPipeline + vkCmdBindDescriptorSets +
// vkCmdPushConstants into `cb`. Returns the bound pipeline (VK_NULL_HANDLE on
// failure). Does NOT dispatch — the caller records vkCmdDispatch(Indirect).
VkPipeline Runtime::bindKernelForDispatch(VkCommandBuffer cb,
                                          VkDescriptorPool dpool,
                                          VCKernel &k, int deviceIdx,
                                          unsigned blockX, unsigned blockY,
                                          unsigned blockZ,
                                          const VCKernelArg *args,
                                          int argCount) {
  if (k.spirvWords.empty()) return VK_NULL_HANDLE;
  VCKernelDeviceState *st = getOrCreateKernelDeviceState(k, deviceIdx);
  if (!st) return VK_NULL_HANDLE;
  VkDevice dev = devices_[deviceIdx]->device;
  VkPipeline pipeline = getPipeline(k, deviceIdx, blockX, blockY, blockZ,
                                    args, argCount);
  if (!pipeline) return VK_NULL_HANDLE;

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
    ai.pSetLayouts = &st->descriptorSetLayout;
    vkAllocateDescriptorSets(dev, &ai, &set);
  }
  if (hasPointer && !set) return VK_NULL_HANDLE;

  // Write descriptor bindings for pointer args. Reflected binding numbers keep
  // the API argument order independent from SPIR-V declaration order.
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
    w.dstBinding = bindIdx;
    if (bindIdx < k.storageBufferBindings.size())
      w.dstBinding = k.storageBufferBindings[bindIdx];
    ++bindIdx;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bufInfos.back();
    writes.push_back(w);
  }
  if (!writes.empty())
    vkUpdateDescriptorSets(dev, writes.size(), writes.data(), 0, nullptr);

  // Pack scalar args into a push-constant buffer.
  std::vector<uint8_t> pcData;
  pcData.reserve(k.pcSize);
  if (k.pcSize > 0) {
    pcData.resize(k.pcSize, 0);
    uint32_t scalarOrdinal = 0;
    uint32_t sequentialOffset = 0;
    for (int i = 0; i < argCount; ++i) {
      if (args[i].kind != VCKernelArg::Scalar) continue;
      uint32_t sz = static_cast<uint32_t>((args[i].size + 3) & ~size_t(3));
      uint32_t off = sequentialOffset;
      if (scalarOrdinal < k.pushConstantOffsets.size())
        off = k.pushConstantOffsets[scalarOrdinal];
      if (off + sz > pcData.size()) pcData.resize(off + sz, 0);
      std::memcpy(pcData.data() + off, args[i].data, args[i].size);
      sequentialOffset += sz;
      ++scalarOrdinal;
    }
  }

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  if (set)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            st->pipelineLayout, 0, 1, &set, 0, nullptr);
  if (!pcData.empty())
    vkCmdPushConstants(cb, st->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(pcData.size()), pcData.data());
  return pipeline;
}

// Make a dispatch's SSBO writes visible to subsequent commands in the same
// command buffer (copies, later dispatches, indirect reads). Within a single
// Vulkan command buffer, a later vkCmdCopyBuffer/vkCmdDispatch does NOT
// automatically see an earlier dispatch's memory writes without a barrier.
// For graph capture (secondary cb) this is essential: the recorded ops must
// form a correct pipeline. For the normal per-frame path the barrier is
// harmless (the submit's fence is the only cross-cb sync the caller needs).
static void emitPostDispatchBarrier(VkCommandBuffer cb) {
  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
}

VCError Runtime::recordDispatchInto(VkCommandBuffer cb, VkDescriptorPool dpool,
                                     VCKernel &k, int deviceIdx, unsigned wgX,
                                     unsigned wgY, unsigned wgZ,
                                     unsigned blockX, unsigned blockY,
                                     unsigned blockZ,
                                     const VCKernelArg *args, int argCount) {
  VkPipeline pipeline = bindKernelForDispatch(cb, dpool, k, deviceIdx, blockX,
                                              blockY, blockZ, args, argCount);
  if (!pipeline) return VCError::InvalidKernel;
  vkCmdDispatch(cb, wgX, wgY, wgZ);
  emitPostDispatchBarrier(cb);
  return VCError::Success;
}

// Indirect dispatch: grid {x,y,z} read from `indirectArgs` at `offset`
// (a VkDispatchIndirectCommand) via vkCmdDispatchIndirect. The bind sequence
// is identical to the direct path; only the source of the grid dims differs.
// A barrier makes the prior shader/transfer writes to `indirectArgs` visible
// to the indirect-command read (COMPUTE_SHADER|TRANSFER -> DRAW_INDIRECT).
VCError Runtime::recordDispatchIndirectInto(VkCommandBuffer cb,
                                            VkDescriptorPool dpool,
                                            VCKernel &k, int deviceIdx,
                                            const VCBuffer &indirectArgs,
                                            size_t offset, unsigned blockX,
                                            unsigned blockY, unsigned blockZ,
                                            const VCKernelArg *args,
                                            int argCount) {
  if (!indirectArgs.buffer) return VCError::InvalidValue;
  // offset must be within the buffer and leave room for 12 bytes.
  if (offset > indirectArgs.size ||
      indirectArgs.size - offset < 3 * sizeof(uint32_t))
    return VCError::InvalidValue;
  // Ensure prior writes to the args buffer are visible to the indirect read.
  VkMemoryBarrier preMb{};
  preMb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  preMb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT;
  preMb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cb,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0, 1, &preMb, 0, nullptr, 0, nullptr);

  VkPipeline pipeline = bindKernelForDispatch(cb, dpool, k, deviceIdx, blockX,
                                              blockY, blockZ, args, argCount);
  if (!pipeline) return VCError::InvalidKernel;
  vkCmdDispatchIndirect(cb, indirectArgs.buffer, offset);
  emitPostDispatchBarrier(cb);
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
    return recordDispatchInto(g.secondaryCB, g.descriptorPool, k, g.deviceIdx,
                              wgX, wgY, wgZ, blockX, blockY, blockZ, args,
                              argCount);
  }
  VkCommandBuffer cb = beginFrame(s);
  VCError e = recordDispatchInto(cb, s.frames[s.frameIdx].descriptorPool, k,
                                  s.deviceIdx, wgX, wgY, wgZ, blockX, blockY,
                                  blockZ, args, argCount);
  if (e != VCError::Success) return e;
  endFrame(s);
  return VCError::Success;
}

// Indirect dispatch on a stream: grid dims come from `indirectArgs` (written
// by prior device work) instead of host values. Graph capture records into
// the graph's secondary cb; otherwise it is wrapped in beginFrame/endFrame
// like a normal dispatch. `indirectArgs` must be on the same device as the
// stream (vkCmdDispatchIndirect reads a VkBuffer that must belong to the
// VkDevice that owns the command buffer).
VCError Runtime::dispatchIndirect(VCKernel &k, const VCBuffer &indirectArgs,
                                  size_t offset, unsigned blockX,
                                  unsigned blockY, unsigned blockZ,
                                  const VCKernelArg *args, int argCount,
                                  VCStream &s) {
  if (indirectArgs.deviceIdx != s.deviceIdx)
    return VCError::InvalidDevice;
  if (s.captureTarget) {
    VCGraph &g = *s.captureTarget;
    return recordDispatchIndirectInto(g.secondaryCB, g.descriptorPool, k,
                                      g.deviceIdx, indirectArgs, offset,
                                      blockX, blockY, blockZ, args, argCount);
  }
  VkCommandBuffer cb = beginFrame(s);
  VCError e = recordDispatchIndirectInto(cb, s.frames[s.frameIdx].descriptorPool,
                                         k, s.deviceIdx, indirectArgs, offset,
                                         blockX, blockY, blockZ, args,
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

// Configure kernel-internal printf support. Must run before vcInit because the
// flag is read during instance/device creation to enable the validation layer,
// VK_KHR_shader_non_semantic_info, and the debug messenger that captures
// debugPrintfEXT output. Setting it after init() has no effect on the already-
// built instance — return InvalidValue so callers notice the ordering mistake.
VCError vcEnableKernelPrintf(int enable) {
  if (Runtime::get().initialized()) return VCError::InvalidValue;
  g_kernelPrintfEnabled = enable != 0;
  return VCError::Success;
}

VCError vcGetDeviceCount(int *count) {
  if (!count) return VCError::InvalidValue;
  if (!Runtime::get().initialized()) {
    if (Runtime::get().init() != VCError::Success) {
      *count = 0;
      return VCError::Success;
    }
  }
  *count = (int)Runtime::get().deviceCount();
  return VCError::Success;
}

VCError vcGetDevice(int *device) {
  if (!device) return VCError::InvalidValue;
  if (!Runtime::get().initialized())
    return VCError::InitializationError;
  *device = Runtime::get().currentDevice();
  return VCError::Success;
}

VCError vcSetDevice(int device) {
  if (!Runtime::get().initialized())
    return VCError::InitializationError;
  return Runtime::get().setDevice(device);
}

VCError vcGetDeviceProperties(VCDeviceProperties *out, int device) {
  if (!Runtime::get().initialized())
    return VCError::InitializationError;
  return Runtime::get().getDeviceProperties(out, device);
}

VCError vcOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                         VCKernelHandle kernel,
                                         size_t dynamicSharedMemPerBlock,
                                         int blockSizeLimit) {
  if (!Runtime::get().initialized())
    return VCError::InitializationError;
  // Kernel handle is accepted for CUDA-API symmetry and future SPIR-V resource
  // reflection; the current limit-based heuristic does not inspect it.
  if (!kernel) return VCError::InvalidKernel;
  return Runtime::get().occupancyMaxPotentialBlockSize(
      minGridSize, blockSize, dynamicSharedMemPerBlock, blockSizeLimit);
}

VCError vcPointerGetAttributes(VCPointerAttributes *out, const void *ptr) {
  if (!Runtime::get().initialized())
    return VCError::InitializationError;
  return Runtime::get().pointerGetAttributes(out, ptr);
}

VCError vcMalloc(void **devPtr, size_t bytes) {
  if (!devPtr) return VCError::InvalidValue;
  auto *b = new VCBuffer{};
  VCError e = Runtime::get().mallocBuffer(bytes, *b);
  if (e != VCError::Success) { delete b; return e; }
  *devPtr = b;
  Runtime::get().registerBuffer(b, VCMemoryType::Device);
  return VCError::Success;
}

VCError vcMallocHost(void **hostPtr, size_t bytes) {
  if (!hostPtr) return VCError::InvalidValue;
  auto *b = new VCBuffer{};
  VCError e = Runtime::get().mallocHostBuffer(bytes, *b);
  if (e != VCError::Success) { delete b; return e; }
  *hostPtr = b;
  Runtime::get().registerBuffer(b, VCMemoryType::Host);
  return VCError::Success;
}

// Unified memory: same handle contract as vcMalloc (the VCBuffer* is the
// pointer VC tracks), but the backing storage is persistently mapped +
// host-coherent. Host reads/writes the payload directly via the mapped
// address exposed by vcPointerGetAttributes (hostPointer); the device accesses
// the same storage through the buffer handle (devicePointer). No vcMemcpy
// needed between host and device. Stream arg is for API symmetry with
// cudaMallocAsync (allocation is host-immediate; Vulkan has no async alloc).
VCError vcMallocManaged(void **devPtr, size_t bytes) {
  if (!devPtr) return VCError::InvalidValue;
  auto *b = new VCBuffer{};
  VCError e = Runtime::get().mallocManagedBuffer(bytes, *b);
  if (e != VCError::Success) { delete b; return e; }
  *devPtr = b;
  Runtime::get().registerBuffer(b, VCMemoryType::Managed);
  return VCError::Success;
}

VCError vcMallocManagedS(void **devPtr, size_t bytes, VCStreamHandle stream) {
  (void)stream; // stream-orders subsequent use; allocation is host-immediate
  return vcMallocManaged(devPtr, bytes);
}

VCError vcFree(void *devPtr) {
  if (!devPtr) return VCError::Success;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  Runtime::get().unregisterBuffer(b);
  VCError e = Runtime::get().freeBuffer(*b);
  delete b;
  return e;
}

// Allocation is a host-side operation (vkCreateBuffer + vkAllocateMemory); it
// is not recorded on a stream. The `stream` argument is accepted for API
// symmetry with CUDA's cudaMallocAsync and to document that the returned
// handle is usable by subsequent stream-ordered operations — it does not make
// the allocation itself asynchronous. (Vulkan has no true async allocation;
// this matches the contract without pretending otherwise.)
VCError vcMallocAsync(void **devPtr, size_t bytes, VCStreamHandle stream) {
  (void)stream; // stream-orders subsequent use; allocation is host-immediate
  return vcMalloc(devPtr, bytes);
}

VCError vcMallocHostAsync(void **hostPtr, size_t bytes, VCStreamHandle stream) {
  (void)stream;
  return vcMallocHost(hostPtr, bytes);
}

// Deferred free: queue the buffer for release once its stream's pending work
// completes. Safe to call while GPU work referencing the buffer is in flight
// on `stream` (cudaFreeAsync semantics).
VCError vcFreeAsync(void *devPtr, VCStreamHandle stream) {
  if (!devPtr) return VCError::Success;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  auto &rt = Runtime::get();
  // Drop from the registry now: a subsequent vcPointerGetAttributes on this
  // pointer reports Unregistered (matching cudaFreeAsync semantics). The
  // underlying buffer is reclaimed later, once the stream's work completes.
  rt.unregisterBuffer(b);
  return rt.freeBufferAsync(b, rt.resolveStream(stream));
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

// Cross-device copy (cudaMemcpyPeer). Both pointers must be vcMalloc handles;
// `dstDevice`/`srcDevice` must match the buffers' own device indices. Uses a
// device-group peer copy when available, otherwise falls back to host staging.
// This is the synchronous form — it returns once the copy lands.
VCError vcMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice,
                     size_t bytes) {
  auto *db = reinterpret_cast<VCBuffer *>(dst);
  auto *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
  if (!db || !sb) return VCError::InvalidValue;
  return Runtime::get().copyPeer(*db, dstDevice, *sb, srcDevice, bytes);
}

// Async form: CUDA's cudaMemcpyPeerAsync runs on a stream belonging to the
// destination device. VC currently runs the selected peer path synchronously
// internally; the `stream` argument is accepted for API symmetry and validated
// to belong to the destination device. A truly asynchronous, stream-ordered
// peer copy is still future work.
VCError vcMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                          int srcDevice, size_t bytes,
                          VCStreamHandle stream) {
  auto *db = reinterpret_cast<VCBuffer *>(dst);
  auto *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
  if (!db || !sb) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  VCStream &s = rt.resolveStream(stream);
  if (s.deviceIdx != dstDevice) return VCError::InvalidValue;
  return rt.copyPeer(*db, dstDevice, *sb, srcDevice, bytes);
}

// 2D pitched copies. Only DeviceToDevice is supported: both pointers must be
// vcMalloc handles. The 2D region is height rows of width bytes; row r of the
// source is at r*spitch, row r of the dst at r*dpitch.
VCError vcMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                        size_t width, size_t height, VCMemcpyKind kind,
                        VCStreamHandle stream) {
  if (kind != VCMemcpyKind::DeviceToDevice) return VCError::InvalidValue;
  auto *db = reinterpret_cast<VCBuffer *>(dst);
  auto *sb = reinterpret_cast<VCBuffer *>(const_cast<void *>(src));
  if (!db || !sb) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  return rt.copy2DAsync(*db, dpitch, *sb, spitch, width, height,
                        rt.resolveStream(stream));
}

VCError vcMemcpy2DS(void *dst, size_t dpitch, const void *src, size_t spitch,
                    size_t width, size_t height, VCMemcpyKind kind,
                    VCStreamHandle stream) {
  VCError e = vcMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                              stream);
  if (e != VCError::Success) return e;
  auto &rt = Runtime::get();
  return rt.streamSynchronize(rt.resolveStream(stream));
}

VCError vcMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                   size_t width, size_t height, VCMemcpyKind kind) {
  return vcMemcpy2DS(dst, dpitch, src, spitch, width, height, kind, nullptr);
}

VCError vcMemset(void *devPtr, int value, size_t count) {
  return vcMemsetS(devPtr, value, count, nullptr);
}

VCError vcMemsetS(void *devPtr, int value, size_t count, VCStreamHandle stream) {
  if (!devPtr) return VCError::InvalidValue;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  auto &rt = Runtime::get();
  return rt.memsetBuffer(*b, value, count, rt.resolveStream(stream));
}

VCError vcMemsetAsyncS(void *devPtr, int value, size_t count,
                       VCStreamHandle stream) {
  if (!devPtr) return VCError::InvalidValue;
  auto *b = reinterpret_cast<VCBuffer *>(devPtr);
  auto &rt = Runtime::get();
  return rt.memsetBufferAsync(*b, value, count, rt.resolveStream(stream));
}

VCError vcMemsetAsync(void *devPtr, int value, size_t count) {
  return vcMemsetAsyncS(devPtr, value, count, nullptr);
}

// 2D pitched fill: height rows of width bytes at stride pitch. width must be a
// multiple of 4 (vkCmdFillBuffer).
VCError vcMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                        size_t height, VCStreamHandle stream) {
  if (!dst) return VCError::InvalidValue;
  auto *b = reinterpret_cast<VCBuffer *>(dst);
  auto &rt = Runtime::get();
  return rt.memset2DBufferAsync(*b, pitch, value, width, height,
                                rt.resolveStream(stream));
}

VCError vcMemset2D(void *dst, size_t pitch, int value, size_t width,
                   size_t height) {
  VCError e = vcMemset2DAsync(dst, pitch, value, width, height, nullptr);
  if (e != VCError::Success) return e;
  auto &rt = Runtime::get();
  return rt.streamSynchronize(rt.resolveStream(nullptr));
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

VCError vcStreamQuery(VCStreamHandle stream, int *done) {
  auto &rt = Runtime::get();
  return rt.streamQuery(rt.resolveStream(stream), done);
}

VCError vcLaunchHostFunc(VCStreamHandle stream, VCHostFn fn, void *userData) {
  auto &rt = Runtime::get();
  return rt.launchHostFunc(rt.resolveStream(stream), fn, userData);
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

VCError vcEventElapsedTime(float *ms, VCEventHandle start, VCEventHandle end) {
  if (!start || !end) return VCError::InvalidValue;
  return Runtime::get().eventElapsedTime(
      ms, *reinterpret_cast<VCEvent *>(start),
      *reinterpret_cast<VCEvent *>(end));
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

VCError vcLaunchKernelIndirectS(VCKernelHandle kernel, void *indirectArgs,
                                size_t offset, unsigned blockX,
                                unsigned blockY, unsigned blockZ,
                                const VCKernelArg *args, int argCount,
                                VCStreamHandle stream) {
  if (!kernel) return VCError::InvalidKernel;
  if (!indirectArgs) return VCError::InvalidValue;
  auto &rt = Runtime::get();
  auto *k = reinterpret_cast<VCKernel *>(kernel);
  auto *argsBuf = reinterpret_cast<VCBuffer *>(indirectArgs);
  return rt.dispatchIndirect(*k, *argsBuf, offset, blockX, blockY, blockZ,
                             args, argCount, rt.resolveStream(stream));
}

VCError vcLaunchKernelIndirect(VCKernelHandle kernel, void *indirectArgs,
                               size_t offset, unsigned blockX,
                               unsigned blockY, unsigned blockZ,
                               const VCKernelArg *args, int argCount) {
  return vcLaunchKernelIndirectS(kernel, indirectArgs, offset, blockX, blockY,
                                 blockZ, args, argCount, nullptr);
}

} // namespace vc
