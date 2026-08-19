//===- Pipeline.cpp - Compute pipeline / shader module management --------===//
//
// Kernel loading and pipeline construction live in VCRuntime.cpp (Runtime
// methods loadKernel / buildPipelineForKernel / launch). This file is
// reserved for future specialization: descriptor-set layout reflection from
// SPIR-V, specialization constants, push constants, and pipeline caches.
//
//===----------------------------------------------------------------------===//

#include "RuntimeInternal.h"

namespace vc {
// TODO: reflect descriptor bindings from spirv instead of assuming one SSBO
//       per launch argument.
// TODO: pipeline cache (VkPipelineCache) for faster re-loads.
// TODO: push-constant path for small scalar args instead of SSBO binding.
} // namespace vc
