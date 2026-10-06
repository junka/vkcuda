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
/// `warningsAsErrors` upgrades the reduced-fidelity warnings the translator
/// emits (e.g. an f64 transcendental computed at f32 precision) to hard
/// errors, matching the driver's -Werror contract for the frontend.
::mlir::OwningOpRef<::mlir::ModuleOp>
translateASTToMLIR(const TranslationUnit &tu, ::mlir::MLIRContext &ctx,
                   bool warningsAsErrors = false);

/// Stage 1 of the lowering pipeline: lower the VC dialect to the GPU dialect
/// (vc.kernel -> gpu.module/gpu.func; thread/block/barrier ops -> gpu.*).
void lowerVCToGPU(::mlir::ModuleOp module);

/// Run the full lowering pipeline: VC -> (standard/gpu) -> SPIR-V.
/// Stage 1 is `lowerVCToGPU`; the gpu->spirv stage reuses MLIR's built-in
/// conversion passes.
void runLoweringPipeline(::mlir::ModuleOp module);

} // namespace codegen
} // namespace vc

#endif // VC_CODEGEN_PASSES_H
