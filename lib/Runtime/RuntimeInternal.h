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
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  bool headless = true; // no surface/swapchain
};

/// A device buffer + its backing memory, host-visible for staging.
struct VCBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  size_t size = 0;
  void *mapped = nullptr; // non-null when persistently mapped
};

/// A loaded kernel: shader module + compute pipeline + descriptor set layout
/// derived from the argument count.
struct VCKernel {
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  std::string entryPoint;
  int argCount = 0;
};

class Runtime {
public:
  static Runtime &get();

  VCError init();
  VCError shutdown();
  bool initialized() const { return bool(device_); }

  VulkanDevice &device() { return *device_; }

  VCError mallocBuffer(size_t bytes, VCBuffer &out);
  VCError freeBuffer(VCBuffer &buf);
  VCError copy(void *dst, const void *src, size_t bytes, VCMemcpyKind kind);

  VCError loadKernel(const uint32_t *words, size_t wordCount,
                     const char *entryPoint, VCKernel &out);
  VCError loadKernelFromFile(const char *path, const char *entryPoint,
                             VCKernel &out);
  void releaseKernel(VCKernel &k);

  VCError launch(VCKernel &k, unsigned gridDim, unsigned blockDim,
                 const VCKernelArg *args, int argCount);
  VCError synchronize();

private:
  std::unique_ptr<VulkanDevice> device_;
  bool init_ = false;

  bool pickPhysicalDevice();
  bool createLogicalDevice();
  bool createCommandPool();
  bool createDescriptorPool();
  bool buildPipelineForKernel(VCKernel &k, int argCount);
  VkCommandBuffer beginOneTime() const;
  void endOneTime(VkCommandBuffer cb) const;
};

} // namespace vc

#endif // VC_RUNTIME_INTERNAL_H
