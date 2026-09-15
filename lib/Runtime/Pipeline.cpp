//===- Pipeline.cpp - Compute pipeline / shader module management --------===//
//
// Kernel loading and pipeline construction live in VCRuntime.cpp (Runtime
// methods loadKernel / buildPipelineForKernel / launch). This file is
// reserved for future specialization of the pipeline subsystem.
//
//===----------------------------------------------------------------------===//

#include "RuntimeInternal.h"

namespace vc {
// TODO: persist VkPipelineCache blobs across process runs.
// TODO: move the lightweight SPIR-V reflection helper out of VCRuntime.cpp
//       when the pipeline subsystem grows beyond the current single file.
} // namespace vc
