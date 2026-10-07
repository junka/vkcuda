//===- Passes.h - Codegen pipeline entry points ---------------------------===//
//
// Declares the AST->MLIR codegen and the lowering pass pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef VC_CODEGEN_PASSES_H
#define VC_CODEGEN_PASSES_H

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace vc {
class TranslationUnit;

namespace codegen {

/// Translate an AST translation unit into a VC-dialect MLIR module.
/// `warningsAsErrors` upgrades the reduced-fidelity warnings the translator
/// emits to hard errors, matching the driver's -Werror contract for the
/// frontend.
///
/// `allowF64MathF32` opts into computing a `double` transcendental (`sin(d)`,
/// `pow(d, e)`, ...) at f32 precision and extending the result back to f64.
/// Without it the translator rejects the call: SPIR-V's GLSLstd450
/// transcendental set is f16/f32-only, so a f64-accurate form does not exist.
::mlir::OwningOpRef<::mlir::ModuleOp>
translateASTToMLIR(const TranslationUnit &tu, ::mlir::MLIRContext &ctx,
                   bool warningsAsErrors = false,
                   bool allowF64MathF32 = false);

/// Stage 1 of the lowering pipeline: lower the VC dialect to the GPU dialect
/// (vc.kernel -> gpu.module/gpu.func; thread/block/barrier ops -> gpu.*).
void lowerVCToGPU(::mlir::ModuleOp module);

/// Run the full lowering pipeline: VC -> (standard/gpu) -> SPIR-V.
/// Stage 1 is `lowerVCToGPU`; the gpu->spirv stage reuses MLIR's built-in
/// conversion passes.
///
/// Fails when the lowering leaves the module in a shape that would serialize
/// into an unloadable — or, worse, silently wrong — SPIR-V binary, so the
/// driver does not emit a binary it knows to be bad.
::mlir::LogicalResult runLoweringPipeline(::mlir::ModuleOp module);

} // namespace codegen
} // namespace vc

#endif // VC_CODEGEN_PASSES_H
