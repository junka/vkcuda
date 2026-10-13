//===- Buffer.cpp - Device buffer management ------------------------------===//
//
// Buffer allocation/free/copy is implemented in VCRuntime.cpp (Runtime
// methods mallocBuffer / freeBuffer / copy). This file is reserved for
// future buffer sub-allocation and device-local memory policy work. The
// current runtime already keeps a small per-stream cache of host-visible
// staging buffers for repeated H2D/D2H copies.
//
//===----------------------------------------------------------------------===//

#include "RuntimeInternal.h"

namespace vc {
// TODO: VkDeviceMemory sub-allocation pool.
// The dedicated-transfer-queue TODO that used to live here is done: see
// Runtime::canRouteCopyToTransfer / beginTransferCopy in VCRuntime.cpp.
} // namespace vc
