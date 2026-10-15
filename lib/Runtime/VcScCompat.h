//===- VcScCompat.h - Small helpers for the Vulkan SC runtime path --------===//
//
// Vulkan SC differs from core Vulkan in ways that need a few lines of glue at
// every call site rather than a wrapper: the driver reports its own rules in
// VkPhysicalDeviceVulkanSC10Properties and the runtime has to obey them
// instead of managing its own resources. Everything here is a no-op or a
// compile-time constant on the default build, so including this header does
// not perturb the core path at all.
//
//===----------------------------------------------------------------------===//

#ifndef VC_RUNTIME_VC_SC_COMPAT_H
#define VC_RUNTIME_VC_SC_COMPAT_H

#include "vc/Runtime/VCRuntime.h"

#include <cstdint>
#include <cstdlib>

namespace vc {

#ifdef VC_VULKAN_SC

/// Read VC_SC_<FIELD> from the environment as an unsigned integer. Returns
/// `fallback` when the variable is unset, empty, or unparseable, so a typo in
/// an override never silently becomes a zero-sized reservation (which the
/// driver would reject at the first allocation, far from the cause).
inline uint64_t vcScEnvU64(const char *name, uint64_t fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) return fallback;
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(v, &end, 0);
  if (end == v) return fallback;
  return (uint64_t)parsed;
}

/// uint32_t flavour of vcScEnvU64.
inline uint32_t vcScEnvU32(const char *name, uint32_t fallback) {
  return (uint32_t)vcScEnvU64(name, (uint64_t)fallback);
}

/// The pipeline-cache blob path: VC_SC_PIPELINE_CACHE, else the installed
/// reservation's path, else the conventional name in the working directory.
/// The vendor SC samples put pipeline_cache.bin next to the executable and open
/// it by relative name, so a bare filename is the documented default.
inline const char *vcScPipelineCachePath(const char *fromConfig) {
  if (fromConfig && *fromConfig) return fromConfig;
  const char *env = std::getenv("VC_SC_PIPELINE_CACHE");
  if (env && *env) return env;
  return "pipeline_cache.bin";
}

#endif // VC_VULKAN_SC

} // namespace vc

#endif // VC_RUNTIME_VC_SC_COMPAT_H
