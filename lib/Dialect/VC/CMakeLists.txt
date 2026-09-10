# VC dialect: built with mlir-tblgen. Requires libmlir-18-dev.
# tablegen() reads its input .td from LLVM_TARGET_DEFINITIONS; VC.td includes
# mlir/IR/OpBase.td, so MLIR include dirs must be on the tablegen -I path.
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS VC.td)
mlir_tablegen(VCOps.h.inc -gen-op-decls)
mlir_tablegen(VCOps.cpp.inc -gen-op-defs)
mlir_tablegen(VCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(VCDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(VCTypes.h.inc -gen-typedef-decls)
mlir_tablegen(VCTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(VCEnumAttrs.h.inc -gen-enum-decls)
mlir_tablegen(VCEnumAttrs.cpp.inc -gen-enum-defs)

add_public_tablegen_target(VCDialectIncGen)

add_mlir_library(VCDialect
  VCDialect.cpp
  VCOps.cpp

  DEPENDS
  VCDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFunctionInterfaces
)

target_include_directories(VCDialect PUBLIC
  ${VC_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
# MLIR headers are upstream; SYSTEM suppresses their build warnings (e.g. the
# C++20 TypeRange.h ambiguity note under GCC 13).
target_include_directories(VCDialect SYSTEM PUBLIC
  ${MLIR_INCLUDE_DIRS}
)
