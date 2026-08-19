//===- Passes.h - Codegen pipeline entry points ---------------------------===//
//
// Declares the AST->MLIR codegen and the lowering pass pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_PASSES_H
#define VC_CODEGEN_PASSES_H

#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace vc {
class TranslationUnit;

namespace codegen {

/// Translate an AST translation unit into a VC-dialect MLIR module.
::mlir::OwningOpRef<::mlir::ModuleOp>
translateASTToMLIR(const TranslationUnit &tu, ::mlir::MLIRContext &ctx);

/// Run the full lowering pipeline: VC -> (standard/gpu) -> SPIR-V.
/// The VC->standard step is a TODO scaffold; gpu->spirv reuses MLIR's
/// built-in conversion passes.
void runLoweringPipeline(::mlir::ModuleOp module);

} // namespace codegen
} // namespace vc

#endif // VC_CODEGEN_PASSES_H
