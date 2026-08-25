//===- VCDialect.cpp - VC dialect registration -----------------------------===//
//
// Registers the `vc` dialect and its ops/types with MLIR.
//
//===----------------------------------------------------------------------===//

#include "vc/Dialect/VC/Dialect.h"
#include "vc/Dialect/VC/Ops.h"

using namespace mlir;
using namespace vc;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void VCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VCOps.cpp.inc"
  >();
}

// The VCDialect class body (constructor, namespace accessor, registration
// macros) is supplied by VCDialect.cpp.inc generated from VC.td.
#include "VCDialect.cpp.inc"

// Enum attribute class definitions (vc::DimAttr etc.) generated from VC.td.
// No guard macro: definitions are included exactly once here.
#include "VCEnumAttrs.cpp.inc"
