//===- Buffer.cpp - Device buffer management ------------------------------===//
//
// Buffer allocation/free/copy is implemented in VCRuntime.cpp (Runtime
// methods mallocBuffer / freeBuffer / copy). This file is reserved for
// future buffer sub-allocation, ring/staging buffers, and device-local
// (non-host-visible) memory paths with explicit transfer commands.
//
//===----------------------------------------------------------------------===//

#include "RuntimeInternal.h"

namespace vc {
// TODO: VkDeviceMemory sub-allocation pool.
// TODO: dedicated staging buffer with transfer queue for device-local memory.
} // namespace vc
