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
  registerOperations();
  // Types and attrs would be registered here once added to VC.td.
}

// The VCDialect class body (constructor, namespace accessor, registration
// macros) is supplied by VCDialect.cpp.inc generated from VC.td.
#include "vc/Dialect/VC/VCDialect.cpp.inc"
