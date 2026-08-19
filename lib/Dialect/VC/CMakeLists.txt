# VC dialect: built with mlir-tblgen. Requires libmlir-18-dev.
mlir_tablegen(VCOps.h.inc -gen-op-decls)
mlir_tablegen(VCOps.cpp.inc -gen-op-defs)
mlir_tablegen(VCDialect.h.inc -gen-dialect-decls)
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
