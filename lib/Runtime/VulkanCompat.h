//===- VulkanCompat.h - Select the Vulkan header set ----------------------===//
//
// The VC runtime is written against the core Vulkan API.  Vulkan SC is a
// separate, source-incompatible API: no runtime pipeline compilation, no
// dynamic host allocation, mandatory object reservations.  Its header shares
// most type names with core Vulkan but not all (several VK_KHR_* /
// VK_EXT_* extension blocks are absent outright), so the two cannot be
// included together and there is no single header that covers both.
//
// Everything SC-specific in the runtime is behind VC_VULKAN_SC; this shim is
// the one place that decides which header the rest of the runtime sees.
//
//===----------------------------------------------------------------------===//

#ifndef VC_RUNTIME_VULKAN_COMPAT_H
#define VC_RUNTIME_VULKAN_COMPAT_H

#ifdef VC_VULKAN_SC
// The VulkanSC include root (the directory *containing* `vulkan/`).
#include <vulkan/vulkan_sc_core.h>
#else
#include <vulkan/vulkan.h>
#endif

#endif // VC_RUNTIME_VULKAN_COMPAT_H
