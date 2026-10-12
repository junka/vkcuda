//===- ImageFormatInfo.h - shared storage-image format table --------------===//
//
// One source of truth for storage-image texel formats. A `textureND<T>` image's
// format was historically derived from its element type (float->rgba32f,
// half->rgba16f, int->r32i, uint->r32ui) in FIVE separate switches (GLSL
// qualifier, GLSL texel vec, MLIR spirv ImageFormat, runtime VkFormat, runtime
// texel size). This header collapses them into one table.
//
// `ImageType::format` holds an `ImageFormat` (Default = derive from elemType).
// `resolveImageFormat(Default, elem)` returns the legacy derived format so the
// five switches become one lookup. `imageFormatInfo(f)` then yields every
// backend's spelling of that format.
//
// Design note: the table does NOT store `mlir::spirv::ImageFormat` directly,
// because that would force every includer of AST.h to pull in MLIR headers.
// Instead the MLIR backend maps the VC `ImageFormat` to the spirv enum at use
// (see spirvImageFormat() in ASTToMLIR.cpp).
//
// Texel vec width is ALWAYS 4 on the device side: GLSL `imageLoad` and SPIR-V
// `OpImageRead` return a 4-component vector regardless of how many channels the
// format actually has (unused channels are 0/1). The `channels` field here is
// for HOST staging only — the host stores `channels` scalars per texel, the
// device reads/writes the leading N of the 4-wide vec. This keeps the backends'
// vec4 assumption intact.
//
//===----------------------------------------------------------------------===//

#ifndef VC_FRONTEND_IMAGEFORMATINFO_H
#define VC_FRONTEND_IMAGEFORMATINFO_H

#include <cstdint>

namespace vc {

/// Forward decl — the runtime enum (defined in VCRuntime.h). Kept opaque here
/// so this header doesn't depend on the runtime header; the value is stored as
/// a plain int and cast at use sites. See VCPixelFormat in VCRuntime.h.
using VCPixelFormatId = int;

/// Storage-image texel format. `Default` means "derive from elemType" (the
/// legacy behavior, preserved so `texture2D<float>` etc. are unchanged). Any
/// other value is an explicit format chosen by the source spelling
/// (`texture2D<r16f half>`, `texture2D<rg16f float>`, ...).
enum class ImageFormat {
  Default,
  // Float (4-channel)
  Rgba32f, Rgba16f,
  // Float (1/2/3-channel)
  R32f, R16f, Rg32f, Rg16f, R11fG11fB10f,
  // Float (8-bit / packed)
  Rgba8, Rgba8Snorm, Rg8, Rg8Snorm, R8, R8Snorm, Rgba16Snorm, Rg16Snorm, R16Snorm,
  // Signed int
  Rgba32i, Rgba16i, Rgba8i, R32i, Rg32i, Rg16i, Rg8i, R16i, R8i,
  // Unsigned int
  Rgba32ui, Rgba16ui, Rgba8ui, R32ui, Rgb10a2ui, Rg32ui, Rg16ui, Rg8ui, R16ui, R8ui,
};

/// Element-type class for format compatibility checking (a format's channel
/// bit depth must be compatible with the declared T).
enum class ImageFormatElemClass { Float, Half, SInt, UInt };

struct ImageFormatInfo {
  const char *glslQualifier;   // "rgba32f", "r32i", "r11f_g11b10f", ...
  ImageFormatElemClass elemClass; // which elemType the format expects
  uint32_t channels;           // 1/2/3/4 — host staging width (device vec is always 4)
  uint32_t bytesPerTexel;      // host staging size
};

/// The shared table lookup. Returns info for a resolved (non-Default) format.
const ImageFormatInfo &imageFormatInfo(ImageFormat f);

/// Resolve a (possibly Default) format against the element type. `Default`
/// yields the legacy derived format (float->Rgba32f, half->Rgba16f,
/// int->R32i, uint->R32ui). `elem` is the BuiltinTypeKind of T (the frontend
/// passes the int value to avoid pulling BuiltinTypeKind into this header).
ImageFormat resolveImageFormat(ImageFormat f, int elem);

/// Map a GLSL format-spelling token (as it appears in source, e.g.
/// "r16f", "rg16f", "r11f_g11b10f", "rgba8") to an ImageFormat, or Default
/// if the token is not a recognized format spelling (so the caller can treat
/// it as a non-format identifier, e.g. an access qualifier or elem type).
ImageFormat parseImageFormatSpelling(const char *spelling);

} // namespace vc

#endif // VC_FRONTEND_IMAGEFORMATINFO_H
