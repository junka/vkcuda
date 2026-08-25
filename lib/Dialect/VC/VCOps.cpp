//===- VCOps.cpp - VC dialect op implementations ---------------------------===//
//
// Implements non-generated op methods (verifiers, interfaces) for the vc
// dialect. Most op behavior comes from VCOps.cpp.inc.
//
//===----------------------------------------------------------------------===//

#include "vc/Dialect/VC/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace vc;

#define GET_OP_CLASSES
#include "VCOps.cpp.inc"

// Verifiers / interface methods that need hand-written bodies go here.
// Most ops (including BarrierOp's MemoryEffectsOpInterface) rely on the
// default behavior generated into VCOps.cpp.inc; extend here as the
// lowering pipeline is filled in.
// (No custom verifiers yet; all ops use the generated ones.)
