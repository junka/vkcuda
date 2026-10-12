//===- ImageFormatInfo.cpp - shared storage-image format table ------------===//

#include "vc/Frontend/ImageFormatInfo.h"
#include "vc/Frontend/AST.h" // BuiltinTypeKind

#include <cstring>

namespace vc {

// One row per ImageFormat. The GLSL qualifier spelling matches what glslc
// accepts as an image format qualifier (and what spirv-dis prints after
// OpTypeImage, modulo case — spirv uses Rgba32f, GLSL uses rgba32f; the MLIR
// backend lower-cases nothing, the GLSL backend uses these lower-case
// spellings directly). elemClass = which T the format is compatible with.
// channels = real channel count for HOST staging (device vec is always 4).
// bytesPerTexel = host staging size.
//
// Index by ImageFormat enum order. Default is row 0 but never looked up via
// imageFormatInfo (resolveImageFormat maps Default -> a concrete format first).
static const ImageFormatInfo kTable[] = {
  // Default (placeholder; resolveImageFormat replaces it before lookup)
  {"rgba32f", ImageFormatElemClass::Float, 4, 16},

  // Float (4-channel)
  {"rgba32f", ImageFormatElemClass::Float, 4, 16}, // Rgba32f
  {"rgba16f", ImageFormatElemClass::Half,  4, 8},  // Rgba16f (half vec4)

  // Float (1/2/3-channel)
  {"r32f",    ImageFormatElemClass::Float, 1, 4},  // R32f
  {"r16f",    ImageFormatElemClass::Half,  1, 2},  // R16f
  {"rg32f",   ImageFormatElemClass::Float, 2, 8},  // Rg32f
  {"rg16f",   ImageFormatElemClass::Half,  2, 4},  // Rg16f
  {"r11f_g11b10f", ImageFormatElemClass::Float, 3, 4}, // R11fG11fB10f

  // Float (8-bit / packed / snorm)
  {"rgba8",        ImageFormatElemClass::Float, 4, 4},  // Rgba8 (unorm)
  {"rgba8_snorm",  ImageFormatElemClass::Float, 4, 4},  // Rgba8Snorm
  {"rg8",          ImageFormatElemClass::Float, 2, 2},  // Rg8 (unorm)
  {"rg8_snorm",    ImageFormatElemClass::Float, 2, 2},  // Rg8Snorm
  {"r8",           ImageFormatElemClass::Float, 1, 1},  // R8 (unorm)
  {"r8_snorm",     ImageFormatElemClass::Float, 1, 1},  // R8Snorm
  {"rgba16_snorm", ImageFormatElemClass::Half,  4, 8},  // Rgba16Snorm
  {"rg16_snorm",   ImageFormatElemClass::Half,  2, 4},  // Rg16Snorm
  {"r16_snorm",    ImageFormatElemClass::Half,  1, 2},  // R16Snorm

  // Signed int
  {"rgba32i", ImageFormatElemClass::SInt, 4, 16}, // Rgba32i
  {"rgba16i", ImageFormatElemClass::SInt, 4, 8},  // Rgba16i
  {"rgba8i",  ImageFormatElemClass::SInt, 4, 4},  // Rgba8i
  {"r32i",    ImageFormatElemClass::SInt, 1, 4},  // R32i  (atomics)
  {"rg32i",   ImageFormatElemClass::SInt, 2, 8},  // Rg32i
  {"rg16i",   ImageFormatElemClass::SInt, 2, 4},  // Rg16i
  {"rg8i",    ImageFormatElemClass::SInt, 2, 2},  // Rg8i
  {"r16i",    ImageFormatElemClass::SInt, 1, 2},  // R16i
  {"r8i",     ImageFormatElemClass::SInt, 1, 1},  // R8i

  // Unsigned int
  {"rgba32ui", ImageFormatElemClass::UInt, 4, 16}, // Rgba32ui
  {"rgba16ui", ImageFormatElemClass::UInt, 4, 8},  // Rgba16ui
  {"rgba8ui",  ImageFormatElemClass::UInt, 4, 4},  // Rgba8ui
  {"r32ui",    ImageFormatElemClass::UInt, 1, 4},  // R32ui (atomics)
  {"rgb10_a2ui", ImageFormatElemClass::UInt, 4, 4},// Rgb10a2ui
  {"rg32ui",   ImageFormatElemClass::UInt, 2, 8},  // Rg32ui
  {"rg16ui",   ImageFormatElemClass::UInt, 2, 4},  // Rg16ui
  {"rg8ui",    ImageFormatElemClass::UInt, 2, 2},  // Rg8ui
  {"r16ui",    ImageFormatElemClass::UInt, 1, 2},  // R16ui
  {"r8ui",     ImageFormatElemClass::UInt, 1, 1},  // R8ui
};

const ImageFormatInfo &imageFormatInfo(ImageFormat f) {
  int idx = static_cast<int>(f);
  // Default (0) is a placeholder; callers must resolve first. Map it to
  // Rgba32f defensively so a stray lookup never reads off-table.
  if (idx <= 0 || idx >= static_cast<int>(std::size(kTable)))
    idx = static_cast<int>(ImageFormat::Rgba32f);
  return kTable[idx];
}

ImageFormat resolveImageFormat(ImageFormat f, int elem) {
  if (f != ImageFormat::Default)
    return f;
  // Legacy derivation from elemType.
  switch (static_cast<BuiltinTypeKind>(elem)) {
  case BuiltinTypeKind::Float32: return ImageFormat::Rgba32f;
  case BuiltinTypeKind::Float16: return ImageFormat::Rgba16f;
  case BuiltinTypeKind::Int32:   return ImageFormat::R32i;
  case BuiltinTypeKind::UInt32:  return ImageFormat::R32ui;
  default:                       return ImageFormat::Rgba32f;
  }
}

// Source spellings -> ImageFormat. Must match the GLSL format qualifier
// spellings (lower-case, with underscores for packed formats) so a user can
// write `texture2D<r16f half>` and have it map to R16f. Ordered loosely by
// likelihood; linear scan is fine (small set, only hit on image type parse).
ImageFormat parseImageFormatSpelling(const char *s) {
  if (!s || !*s) return ImageFormat::Default;
  auto eq = [&](const char *k) { return std::strcmp(s, k) == 0; };
  // Float
  if (eq("rgba32f")) return ImageFormat::Rgba32f;
  if (eq("rgba16f")) return ImageFormat::Rgba16f;
  if (eq("r32f"))    return ImageFormat::R32f;
  if (eq("r16f"))    return ImageFormat::R16f;
  if (eq("rg32f"))   return ImageFormat::Rg32f;
  if (eq("rg16f"))   return ImageFormat::Rg16f;
  if (eq("r11f_g11b10f")) return ImageFormat::R11fG11fB10f;
  if (eq("rgba8"))        return ImageFormat::Rgba8;
  if (eq("rgba8_snorm"))  return ImageFormat::Rgba8Snorm;
  if (eq("rg8"))          return ImageFormat::Rg8;
  if (eq("rg8_snorm"))    return ImageFormat::Rg8Snorm;
  if (eq("r8"))           return ImageFormat::R8;
  if (eq("r8_snorm"))     return ImageFormat::R8Snorm;
  if (eq("rgba16_snorm")) return ImageFormat::Rgba16Snorm;
  if (eq("rg16_snorm"))   return ImageFormat::Rg16Snorm;
  if (eq("r16_snorm"))    return ImageFormat::R16Snorm;
  // Signed int
  if (eq("rgba32i")) return ImageFormat::Rgba32i;
  if (eq("rgba16i")) return ImageFormat::Rgba16i;
  if (eq("rgba8i"))  return ImageFormat::Rgba8i;
  if (eq("r32i"))    return ImageFormat::R32i;
  if (eq("rg32i"))   return ImageFormat::Rg32i;
  if (eq("rg16i"))   return ImageFormat::Rg16i;
  if (eq("rg8i"))    return ImageFormat::Rg8i;
  if (eq("r16i"))    return ImageFormat::R16i;
  if (eq("r8i"))     return ImageFormat::R8i;
  // Unsigned int
  if (eq("rgba32ui")) return ImageFormat::Rgba32ui;
  if (eq("rgba16ui")) return ImageFormat::Rgba16ui;
  if (eq("rgba8ui"))  return ImageFormat::Rgba8ui;
  if (eq("r32ui"))    return ImageFormat::R32ui;
  if (eq("rgb10_a2ui")) return ImageFormat::Rgb10a2ui;
  if (eq("rg32ui"))   return ImageFormat::Rg32ui;
  if (eq("rg16ui"))   return ImageFormat::Rg16ui;
  if (eq("rg8ui"))    return ImageFormat::Rg8ui;
  if (eq("r16ui"))    return ImageFormat::R16ui;
  if (eq("r8ui"))     return ImageFormat::R8ui;
  return ImageFormat::Default;
}

} // namespace vc
