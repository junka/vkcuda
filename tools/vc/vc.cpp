//===- vc.cpp - Compiler driver -------------------------------------------===//
//
// Usage:
//   vc <input.vc> [-emit=ast|mlir|spirv|host|full|sc] [-o <out>]
//
// Pipeline: lex+parse -> AST -> (Sema) -> MLIR(VC) -> lower -> SPIR-V.
// The SPIR-V emission reuses MLIR's spirv-to-binary translation.
//   -emit=ast    dump the parsed AST
//   -emit=mlir   dump the VC-dialect MLIR (default)
//   -emit=spirv  lower + serialize to a .spv binary
//   -emit=host   lower + serialize, then print the embedded host C++ (no link)
//   -emit=full   lower + serialize + embed + drive g++ -> standalone executable
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/Passes.h"
#include "vc/Codegen/HostLink.h"
#include "vc/Dialect/VC/Dialect.h"
#include "vc/Frontend/AST.h"
#include "vc/Frontend/ASTDumper.h"
#include "vc/Frontend/Lexer.h"
#include "vc/Frontend/Mangle.h"
#include "vc/Frontend/Parser.h"
#include "vc/Frontend/Sema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Target/SPIRV/Target.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"

#include <cstring>
#include <memory>
#include <optional>
#include <functional>
#include <set>
#include <string>
#include <tuple>

using namespace vc;
using namespace mlir;

namespace {
enum class EmitKind { AST, MLIR, SPIRV, Host, Full, SC };
} // namespace

// Patch the serialized SPIR-V binary to decorate the `gl_WorkGroupSize`
// SpecConstantComposite with BuiltIn WorkgroupSize. This is what makes the
// runtime-supplied specialization constants (SpecId 0/1/2 = block x/y/z) drive
// the actual hardware workgroup size, mirroring the GLSL backend's
// `layout(local_size_x_id = 0, ...)` -> `gl_WorkGroupSize` builtin.
//
// Why a binary patch: MLIR's spirv dialect has no way to attach a BuiltIn
// decoration to a SpecConstantComposite from IR (SpecConstantCompositeOp
// carries no built_in attr; only GlobalVariableOp does, and WorkgroupSize must
// be a spec-constant composite, not a variable). The serializer thus emits the
// composite + its scalar SpecId decorations but omits the WorkgroupSize builtin
// decoration. We inject it by name: locate the OpName record for
// "gl_WorkGroupSize", read its target <id>, then insert an OpDecorate
// <id> BuiltIn WorkgroupSize instruction.
//
// Insertion point: SPIR-V's logical layout puts all OpDecorate records in an
// annotation section that follows the debug section (OpName / ...) and the
// entry-point header (OpCapability ... OpExecutionMode). The serializer already
// emits a valid block of OpDecorate records there, so we insert ours
// immediately before the first existing OpDecorate to keep the layout valid
// (verified by spirv-val).
static void patchWorkgroupSizeBuiltin(SmallVectorImpl<uint32_t> &binary) {
  // OpName = opcode 5; OpDecorate = opcode 71; BuiltIn decoration = 11;
  // BuiltIn.WorkgroupSize = 25.
  constexpr uint32_t kOpName = 5;
  constexpr uint32_t kOpDecorate = 71;
  constexpr uint32_t kDecorationBuiltIn = 11;
  constexpr uint32_t kBuiltinWorkgroupSize = 25;

  if (binary.size() < 5)
    return;

  // Walk instructions to find `OpName <id> "gl_WorkGroupSize"`.
  // OpName record: [wordCount|opcode, target_id, name...]. The name is the
  // remaining (wordCount - 2) words, null-padded to a word boundary.
  size_t i = 5;
  std::optional<uint32_t> wgsizeId;
  while (i < binary.size()) {
    uint32_t word = binary[i];
    uint32_t opcode = word & 0xFFFF;
    uint32_t wordCount = word >> 16;
    if (wordCount == 0)
      break; // malformed
    if (opcode == kOpName && wordCount >= 3 && i + wordCount <= binary.size()) {
      uint32_t targetId = binary[i + 1];
      // Reassemble the name string from the trailing words.
      std::string name;
      for (uint32_t w = 2; w < wordCount; ++w) {
        uint32_t val = binary[i + w];
        for (int b = 0; b < 4; ++b) {
          char c = static_cast<char>((val >> (8 * b)) & 0xFF);
          if (c == 0)
            goto name_done;
          name.push_back(c);
        }
      }
    name_done:
      if (name == "gl_WorkGroupSize") {
        wgsizeId = targetId;
        break;
      }
    }
    i += wordCount;
  }

  if (!wgsizeId)
    return; // No gl_WorkGroupSize in this module (no kernels / not patched).

  // Build the OpDecorate instruction: OpDecorate <target_id> BuiltIn
  // WorkgroupSize -> 4 words: [4<<16|71, target_id, 11, 25].
  uint32_t decor[4] = {(4u << 16) | kOpDecorate, *wgsizeId,
                       kDecorationBuiltIn, kBuiltinWorkgroupSize};
  // SPIR-V layout (logical): OpCapability, OpExtension, OpExtInstImport,
  // OpMemoryModel, OpEntryPoint, OpExecutionMode, then a debug section
  // (OpName / OpMemberName / OpString / OpLine / ...), then annotations
  // (OpDecorate / OpMemberDecorate / OpGroupDecorate / ...), then types &
  // constants, then functions. OpDecorate must land in the annotation section,
  // i.e. after all OpName lines. The serializer already emits a valid block of
  // OpDecorate records there, so the simplest correct insertion is right
  // before the first existing OpDecorate.
  size_t insertPos = 0;
  size_t j = 5;
  while (j < binary.size()) {
    uint32_t w = binary[j];
    uint32_t op = w & 0xFFFF;
    uint32_t wc = w >> 16;
    if (wc == 0)
      break;
    if (op == kOpDecorate) {
      insertPos = j;
      break;
    }
    j += wc;
  }
  if (insertPos == 0)
    return; // No existing OpDecorate — unexpected; don't risk a bad patch.
  binary.insert(binary.begin() + insertPos, decor, decor + 4);
}

// Patch the serialized SPIR-V binary so each `extern __shared__ T s[]` global
// (emitted by ASTToMLIR as `@__vc_dynshared_<name>` with a placeholder
// spirv.array<1 x T>) has its OpTypeArray length operand rewritten to reference
// the `__vc_wg_x` spec constant (SpecId 0 = block x). This is how the MLIR
// backend sizes dynamic shared memory to the runtime workgroup size, mirroring
// the GLSL backend's `shared T s[gl_WorkGroupSize.x]`.
//
// Why a binary patch: MLIR's spirv::ArrayType::get only accepts a constant
// `unsigned` count — there is no IR way to express a spec-constant-sized array.
// So ASTToMLIR emits a placeholder size of 1 and stamps the global with a
// recognizable `__vc_dynshared_<name>` symbol; we then rewrite the array's
// length operand here, post-serialization.
//
// Layout fix: MLIR emits the workgroup-size spec constants (`__vc_wg_x/y/z`,
// `gl_WorkGroupSize`) in the constants section, AFTER the type section. But an
// OpTypeArray's length operand requires its target id to already be defined
// (forward references are allowed only for annotations like OpDecorate, not for
// instruction operands). So we also relocate that contiguous spec-constant
// block to just before the first dynamic-shared OpTypeArray — the same layout
// glslc produces for `shared T s[gl_WorkGroupSize.x]`.
//
// Safety: rewriting the length operand changes no instruction's word count
// (OpTypeArray is always [result_id, element_type_id, length_id]), and the
// relocated spec-constant block only references itself, so no id renumbering is
// needed and all references stay valid. A static `__shared__ T s[1]` that
// happens to share the placeholder array type would also be rewritten — but
// that is benign: it only over-allocates (to workgroup-x elements) and accesses
// that were in-range stay in-range (spirv-val does not bounds-check array
// indexing). Real static shared arrays have a distinct type (size != 1).
//
// `literalBlockX > 0` selects the Vulkan SC shape: there the workgroup size is
// a literal baked into the pipeline, so instead of pointing the array length at
// a spec constant we synthesize an `OpConstant <int-type> <literalBlockX>` and
// relocate *that* before the first dynamic-shared array (same forward-reference
// rule). The int type is borrowed from the placeholder length constant the
// serializer already emitted, so no type synthesis is needed.
static void patchDynamicSharedArrays(SmallVectorImpl<uint32_t> &binary,
                                     int64_t literalBlockX = 0) {
  // Opcodes / word layouts we decode. For ops that have BOTH a result type and
  // a result id, SPIR-V lays out [wc|op, result_type_id, result_id, operands...]
  // (type first, then result). Type-defining ops (TypePointer/TypeArray) have
  // only a result id: [wc|op, result_id, operands...].
  constexpr uint32_t kOpName = 5;                  // [wc|op, target_id, name...]
  constexpr uint32_t kOpTypePointer = 32;          // [wc|op, result_id, SC, elem_id]
  constexpr uint32_t kOpTypeArray = 28;            // [wc|op, result_id, elem_id, len_id]
  constexpr uint32_t kOpVariable = 59;             // [wc|op, type_id, result_id, SC, ...]
  constexpr uint32_t kOpSpecConstant = 50;         // [wc|op, type_id, result_id, value...]
  constexpr uint32_t kOpSpecConstantComposite = 51; // [wc|op, type_id, result_id, constituents...]
  constexpr uint32_t kOpConstant = 43;             // [wc|op, type_id, result_id, value...]

  if (binary.size() < 5)
    return;

  auto readName = [&](size_t base, uint32_t wordCount, std::string &out) {
    out.clear();
    for (uint32_t w = 2; w < wordCount; ++w) {
      uint32_t val = binary[base + w];
      for (int b = 0; b < 4; ++b) {
        char c = static_cast<char>((val >> (8 * b)) & 0xFF);
        if (c == 0)
          return;
        out.push_back(c);
      }
    }
  };

  // Pass 1: collect ids, type-def offsets, variable result-types, and the
  // offsets of spec-constant instructions (by result id, which is at +2 for
  // these ops since they carry a result type first).
  std::optional<uint32_t> wgXId;
  uint32_t glWgSizeId = 0;
  llvm::SmallVector<uint32_t, 4> dynVarIds;
  struct TypeDef { size_t offset; uint32_t opcode; };
  llvm::DenseMap<uint32_t, TypeDef> typeDefs;
  llvm::DenseMap<uint32_t, uint32_t> varResultType; // var result_id -> type_id
  llvm::DenseMap<uint32_t, size_t> specConstOffsets; // result_id -> offset
  llvm::DenseMap<uint32_t, size_t> constOffsets;     // OpConstant result_id -> offset

  size_t i = 5;
  while (i < binary.size()) {
    uint32_t word = binary[i];
    uint32_t opcode = word & 0xFFFF;
    uint32_t wordCount = word >> 16;
    if (wordCount == 0 || i + wordCount > binary.size())
      break;
    if (opcode == kOpName && wordCount >= 3) {
      std::string name;
      readName(i, wordCount, name);
      uint32_t targetId = binary[i + 1];
      if (name == "__vc_wg_x")
        wgXId = targetId;
      else if (name == "gl_WorkGroupSize")
        glWgSizeId = targetId;
      else if (name.starts_with("__vc_dynshared_"))
        dynVarIds.push_back(targetId);
    } else if (opcode == kOpTypePointer && wordCount >= 4) {
      typeDefs[binary[i + 1]] = {i, opcode};
    } else if (opcode == kOpTypeArray && wordCount >= 4) {
      typeDefs[binary[i + 1]] = {i, opcode};
    } else if (opcode == kOpVariable && wordCount >= 4) {
      // [wc|op, type_id, result_id, storage_class, ...]
      varResultType[binary[i + 2]] = binary[i + 1];
    } else if ((opcode == kOpSpecConstant && wordCount >= 4) ||
               (opcode == kOpSpecConstantComposite && wordCount >= 3)) {
      // [wc|op, type_id, result_id, ...] — result_id at +2.
      specConstOffsets[binary[i + 2]] = i;
    } else if (opcode == kOpConstant && wordCount >= 4) {
      // [wc|op, type_id, result_id, value...] — result_id at +2. Recorded so
      // the SC literal path can borrow the placeholder length constant's type.
      constOffsets[binary[i + 2]] = i;
    }
    i += wordCount;
  }

  if (dynVarIds.empty())
    return; // nothing to patch
  if (!literalBlockX && !wgXId)
    return; // spec-constant mode needs __vc_wg_x to point the length at

  // Collect the candidate OpTypeArray offsets reachable from each dynshared
  // variable (var -> ptr -> first array in the element-type chain).
  auto resolveArrayOff = [&](uint32_t typeId) -> std::optional<size_t> {
    uint32_t cur = typeId;
    for (int guard = 0; guard < 16 && cur != 0; ++guard) {
      auto it = typeDefs.find(cur);
      if (it == typeDefs.end())
        break;
      size_t off = it->second.offset;
      uint32_t op = it->second.opcode;
      if (op == kOpTypePointer) {
        cur = binary[off + 3]; // element_type_id
        continue;
      }
      if (op == kOpTypeArray)
        return off;
      break;
    }
    return std::nullopt;
  };
  llvm::SmallVector<size_t, 4> arrayOffsets;
  for (uint32_t varId : dynVarIds) {
    auto it = varResultType.find(varId);
    if (it == varResultType.end())
      continue;
    if (auto off = resolveArrayOff(it->second))
      arrayOffsets.push_back(*off);
  }
  if (arrayOffsets.empty())
    return;
  size_t firstArray = *std::min_element(arrayOffsets.begin(), arrayOffsets.end());

  // Build the instruction block that defines the array's new length.
  //
  // Spec-constant mode: the contiguous spec-constant block starting at
  // __vc_wg_x and running through gl_WorkGroupSize. SC literal mode: a
  // synthesized `OpConstant <int-type> <literalBlockX>`, borrowing its integer
  // type from the placeholder length constant the serializer already emitted
  // for these arrays (so no type is invented, and the literal always fits — the
  // placeholder value is 1, and a block size of 0 is not a launch anyone makes).
  uint32_t lengthId = 0;
  llvm::SmallVector<uint32_t, 8> blockWords;
  size_t blkStart = 0, blkEnd = 0;
  if (literalBlockX) {
    auto lenIt = constOffsets.find(binary[arrayOffsets.front() + 3]);
    if (lenIt == constOffsets.end())
      return; // no placeholder constant to model the integer type on
    uint32_t lengthTypeId = binary[lenIt->second + 1];
    uint32_t newId = binary[3]; // SPIR-V bound word: the first unused result id
    binary[3] = newId + 1;
    lengthId = newId;
    blockWords = {(4u << 16) | kOpConstant, lengthTypeId, newId,
                  static_cast<uint32_t>(literalBlockX)};
  } else {
    // Locate the contiguous spec-constant block to relocate: starts at
    // __vc_wg_x and extends through the following SpecConstant/
    // SpecConstantComposite ops up to and including gl_WorkGroupSize.
    auto wxIt = specConstOffsets.find(*wgXId);
    if (wxIt == specConstOffsets.end())
      return;
    llvm::SmallVector<size_t, 4> specOffsets;
    specOffsets.push_back(wxIt->second);
    {
      size_t cur = wxIt->second;
      while (true) {
        uint32_t wc = binary[cur] >> 16;
        size_t next = cur + wc;
        if (next >= binary.size())
          break;
        uint32_t w = binary[next];
        uint32_t op = w & 0xFFFF;
        uint32_t nwc = w >> 16;
        if (nwc == 0 ||
            (op != kOpSpecConstant && op != kOpSpecConstantComposite))
          break;
        specOffsets.push_back(next);
        cur = next;
        if (binary[next + 2] == glWgSizeId) // result_id at +2
          break;
      }
    }
    std::sort(specOffsets.begin(), specOffsets.end());
    blkStart = specOffsets.front();
    blkEnd = blkStart;
    for (size_t off : specOffsets) {
      uint32_t wc = binary[off] >> 16;
      blkEnd = std::max(blkEnd, off + wc);
    }
    blockWords.assign(binary.begin() + blkStart, binary.begin() + blkEnd);
    lengthId = *wgXId;
  }

  if (!literalBlockX) {
    // If the block is already before the first array (no relocation needed),
    // just rewrite the lengths.
    if (blkEnd <= firstArray) {
      for (size_t off : arrayOffsets)
        binary[off + 3] = lengthId;
      return;
    }

    // Relocate: [0..firstArray) + [spec block] + [firstArray..blkStart) +
    // [blkEnd..end). This moves the spec-constant block to just before the
    // first dynamic-shared OpTypeArray.
    SmallVector<uint32_t, 0> out;
    out.reserve(binary.size());
    out.insert(out.end(), binary.begin(), binary.begin() + firstArray);
    out.insert(out.end(), binary.begin() + blkStart, binary.begin() + blkEnd);
    out.insert(out.end(), binary.begin() + firstArray, binary.begin() + blkStart);
    out.insert(out.end(), binary.begin() + blkEnd, binary.end());
    binary.swap(out);

    // Rewrite each candidate array's length operand. Offsets that were in
    // [firstArray, blkStart) shifted forward by the block length; those at/after
    // blkEnd shifted back. (firstArray itself is now the start of the relocated
    // block, so the arrays that were in [firstArray, blkStart) are now at
    // [firstArray + blkLen, blkStart + blkLen).)
    ptrdiff_t blkLen = (ptrdiff_t)(blkEnd - blkStart);
    for (size_t &off : arrayOffsets) {
      if (off >= blkEnd)
        off -= blkLen;
      else // off was in [firstArray, blkStart)
        off += blkLen;
    }
    for (size_t off : arrayOffsets) {
      if (off + 3 < binary.size())
        binary[off + 3] = lengthId;
    }
    return;
  }

  // SC literal mode: the block is new, so it is always inserted — and since
  // firstArray is the minimum offset, every array simply shifts forward by the
  // block length.
  SmallVector<uint32_t, 0> out;
  out.reserve(binary.size() + blockWords.size());
  out.insert(out.end(), binary.begin(), binary.begin() + firstArray);
  out.insert(out.end(), blockWords.begin(), blockWords.end());
  out.insert(out.end(), binary.begin() + firstArray, binary.end());
  binary.swap(out);
  const size_t blkLen = blockWords.size();
  for (size_t off : arrayOffsets) {
    if (off + blkLen + 3 < binary.size())
      binary[off + blkLen + 3] = lengthId;
  }
}

// Post-serialize patch: rewrite the storage-image carrier stubs emitted by
// ASTToMLIR's emitImageBuiltin into real OpImageRead / OpImageWrite.
//
// MLIR 18's spirv dialect has no spirv.ImageRead/Write ops and its spirv.Load
// verifier rejects storage-image result types, so imageRead/imageWrite are
// lowered to "carriers": a spirv.Store of the coord/texel vector into a
// per-call dummy Private global, named `__vc_imgread_<param>_<id>` /
// `__vc_imgwrite_{coord,texel}_<param>_<id>`, plus (for reads) a
// spirv.Undef %v4float placeholder for the result. The image variable itself
// is a normal spirv.GlobalVariable `__vc_img_<param>` (binds to a descriptor)
// and serializes natively.
//
// This pass: (1) locates each image variable + carrier by OpName, (2) decodes
// the carrier Store to recover the coord/texel operand id, (3) replaces each
// carrier with OpLoad(image var) -> image value, then OpImageRead (reusing the
// Undef's result id so existing uses resolve to the real read) or
// OpImageWrite, (4) drops the dummy carriers and the now-unused Undef.
static void patchImageOps(SmallVectorImpl<uint32_t> &binary) {
  constexpr uint32_t kOpName = 5;              // [wc|op, target_id, name...]
  constexpr uint32_t kOpTypeImage = 25;        // [wc|op, result_id, sampled_type_id, Dim, Depth, Arrayed, MS, Sampled, ImageFormat]
  constexpr uint32_t kOpTypeVector = 23;       // [wc|op, result_id, elem_type_id, ComponentCount]
  constexpr uint32_t kOpTypePointer = 32;      // [wc|op, result_id, SC, elem_id]
  constexpr uint32_t kOpVariable = 59;         // [wc|op, type_id, result_id, SC, ...]
  constexpr uint32_t kOpLoad = 61;             // [wc|op, type_id, result_id, ptr_id]
  constexpr uint32_t kOpStore = 62;            // [wc|op, ptr_id, value_id, (mem-ops)...]
  constexpr uint32_t kOpImageRead = 98;        // [wc|op, type_id, result_id, image_id, coord_id]
  constexpr uint32_t kOpImageWrite = 99;       // [wc|op, image_id, coord_id, texel_id]
  constexpr uint32_t kOpImageTexelPointer = 60; // [wc|op, type_id, result_id, Image, Coordinate, Sample]
  constexpr uint32_t kOpAtomicIAdd = 234;      // [wc|op, type_id, result_id, Pointer, Memory, Semantics, Value]
  constexpr uint32_t kOpAtomicISub = 235;
  constexpr uint32_t kOpAtomicSMin = 236;
  constexpr uint32_t kOpAtomicSMax = 238;
  constexpr uint32_t kOpAtomicAnd = 240;
  constexpr uint32_t kOpAtomicOr = 241;
  constexpr uint32_t kOpAtomicXor = 242;
  constexpr uint32_t kOpAtomicExchange = 229;
  constexpr uint32_t kOpAtomicCompareExchange = 230; // [wc|op, type_id, result_id, Pointer, Memory, Equal, Unequal, Value, Comparator]
  constexpr uint32_t kOpTypeInt = 21;          // [wc|op, result_id, width, signedness]
  constexpr uint32_t kOpConstant = 43;         // [wc|op, type_id, result_id, value...]
  constexpr uint32_t kScopeDevice = 1;         // Scope::Device
  constexpr uint32_t kSCUniformConstant = 0;   // StorageClass::UniformConstant
  constexpr uint32_t kSCImage = 11;            // StorageClass::Image (texel ptr)
  constexpr uint32_t kOpDecorate = 71;         // [wc|op, target_id, decoration, (extra operands)...]
  constexpr uint32_t kOpMemoryModel = 14;      // [wc|op, addressing, memory]
  constexpr uint32_t kDecorNonWritable = 24;   // OpDecorate <id> NonWritable
  constexpr uint32_t kDecorNonReadable = 25;   // OpDecorate <id> NonReadable

  if (binary.size() < 5)
    return;

  auto readName = [&](size_t base, uint32_t wordCount, std::string &out) {
    out.clear();
    for (uint32_t w = 2; w < wordCount; ++w) {
      uint32_t val = binary[base + w];
      for (int b = 0; b < 4; ++b) {
        char c = static_cast<char>((val >> (8 * b)) & 0xFF);
        if (c == 0)
          return;
        out.push_back(c);
      }
    }
  };

  // Pass 1: index the binary. Collect:
  //  - image variable name -> result id   (OpName "__vc_img_<param>" -> id)
  //  - var result id -> image type id     (OpVariable type_id -> ptr -> elem)
  //  - type id -> (offset, opcode)        (to walk ptr->image type chain)
  //  - dummy carrier var id -> (name)     (OpName "__vc_imgread_*" etc -> id)
  //  - dummy carrier var id -> store offset + stored value id (OpStore)
  //  - imageRead result placeholder: __vc_imgresult_<param>_<id> var id, and
  //    the OpLoad that reads it (reused as OpImageRead's result id).
  //  - OpUndef %v4float result id -> offset (legacy, kept for safety)
  llvm::DenseMap<uint32_t, uint32_t> varResultType; // var result_id -> type_id
  llvm::DenseMap<uint32_t, std::pair<size_t, uint32_t>> typeDefs; // id -> (off, op)
  llvm::StringMap<uint32_t> imageVarByName;          // param name -> var id
  struct Carrier {
    std::string name;     // full OpName string
    uint32_t dummyVarId;  // the carrier dummy variable's result id
  };
  llvm::SmallVector<Carrier, 8> carriers;
  // dummyVarId -> (storeOffset, valueId)
  llvm::DenseMap<uint32_t, std::pair<size_t, uint32_t>> carrierStore;
  // imageRead result placeholder variable name -> var id
  llvm::StringMap<uint32_t> resultVarByName;
  // result placeholder var id -> (loadOffset, loadResultId, resultTypeId)
  llvm::DenseMap<uint32_t, std::tuple<size_t, uint32_t, uint32_t>> resultLoad;
  // set of dummy carrier var ids (for filtering)
  llvm::DenseSet<uint32_t> carrierVarIds;
  // set of result placeholder var ids (for deletion)
  llvm::DenseSet<uint32_t> resultVarIds;

  // ---- image atomic carrier state (imageAtomicAdd/...) ----
  // Mirrors the read/write carrier machinery but splices OpImageTexelPointer +
  // OpAtomic* (MLIR 18 has spirv.Atomic* but no spirv.ImageTexelPointer).
  llvm::SmallVector<Carrier, 8> atomicCarriers;   // coord/value/cmp carriers
  llvm::SmallVector<Carrier, 8> atomicKindMarkers; // kind marker dummy vars
  llvm::DenseSet<uint32_t> atomicCarrierVarIds;    // all atomic dummy vars
  // dummyVarId -> (storeOffset, valueId) for coord/value/cmp carriers
  llvm::DenseMap<uint32_t, std::pair<size_t, uint32_t>> atomicCarrierStore;
  // atomic result placeholder var name -> var id
  llvm::StringMap<uint32_t> atomicResultVarByName;
  llvm::DenseSet<uint32_t> atomicResultVarIds;
  // result placeholder var id -> (loadOffset, loadResultId, resultTypeId=i32)
  llvm::DenseMap<uint32_t, std::tuple<size_t, uint32_t, uint32_t>>
      atomicResultLoad;
  // 32-bit OpTypeInt result ids (to reuse for the atomic value/result type)
  llvm::SmallVector<uint32_t, 4> intTypeIds;
  // Offset just past the last type-definition instruction (OpType*, OpConstant,
  // ...). New synthesized types/constants must be inserted HERE (after all
  // types they may reference, like %uint) so there are no forward references in
  // the type section. Tracked across all type-def and constant ops in Pass 1.
  size_t lastTypeDefEnd = 0;

  // image var id -> decoration to emit (NonReadable / NonWritable). Absent =>
  // read-write (no decoration). Populated from the __vc_imgro_/__vc_imgwo_
  // symbol prefix the ASTToMLIR emitter stamps on the image variable.
  llvm::DenseMap<uint32_t, uint32_t> imageAccess;
  // Offset just past the OpMemoryModel instruction — the earliest legal point
  // to insert OpDecorate (SPIR-V requires decorations to follow the memory
  // model). 0 if no OpMemoryModel was seen (shouldn't happen for a valid
  // module); inserts fall back to offset 5 in that case.
  size_t memoryModelEnd = 0;
  // Offset just past the last OpDecorate instruction — the end of the
  // annotation section. New OpDecorate inserts anchor here so they stay
  // contiguous with existing decorations and before the type section.
  size_t lastDecorateEnd = 0;

  size_t i = 5;
  while (i < binary.size()) {
    uint32_t word = binary[i];
    uint32_t opcode = word & 0xFFFF;
    uint32_t wordCount = word >> 16;
    if (wordCount == 0 || i + wordCount > binary.size())
      break;
    if (opcode == kOpName && wordCount >= 3) {
      std::string name;
      readName(i, wordCount, name);
      uint32_t targetId = binary[i + 1];
      if (name.starts_with("__vc_imgresult_")) {
        // imageRead result placeholder variable
        resultVarByName[name] = targetId;
        resultVarIds.insert(targetId);
      } else if (name.starts_with("__vc_imgatomic_result_")) {
        // image atomic result placeholder variable (scalar i32)
        atomicResultVarByName[name] = targetId;
        atomicResultVarIds.insert(targetId);
      } else if (name.starts_with("__vc_imgatomic_kind_")) {
        // kind marker: __vc_imgatomic_kind_<param>_<id>_<KIND>. No payload
        // store; the kind is encoded in the OpName suffix.
        atomicKindMarkers.push_back({name, targetId});
        atomicCarrierVarIds.insert(targetId); // delete the dummy var + OpName
      } else if (name.starts_with("__vc_imgatomic_coord_") ||
                 name.starts_with("__vc_imgatomic_value_") ||
                 name.starts_with("__vc_imgatomic_cmp_")) {
        atomicCarriers.push_back({name, targetId});
        atomicCarrierVarIds.insert(targetId);
      } else if (name.starts_with("__vc_imgread_") ||
                 name.starts_with("__vc_imgwrite_")) {
        carriers.push_back({name, targetId});
        carrierVarIds.insert(targetId);
      } else if (name.starts_with("__vc_imgro_")) {
        // read-only image variable: __vc_imgro_<param> -> decorate NonReadable
        std::string param = name.substr(strlen("__vc_imgro_"));
        imageVarByName[param] = targetId;
        imageAccess[targetId] = kDecorNonReadable;
      } else if (name.starts_with("__vc_imgwo_")) {
        // write-only image variable: __vc_imgwo_<param> -> decorate NonWritable
        std::string param = name.substr(strlen("__vc_imgwo_"));
        imageVarByName[param] = targetId;
        imageAccess[targetId] = kDecorNonWritable;
      } else if (name.starts_with("__vc_img_")) {
        // read-write image variable: __vc_img_<param> (no decoration)
        std::string param = name.substr(strlen("__vc_img_"));
        imageVarByName[param] = targetId;
      }
    } else if (opcode == kOpTypeImage && wordCount >= 9) {
      typeDefs[binary[i + 1]] = {i, opcode};
      lastTypeDefEnd = i + wordCount;
    } else if (opcode == kOpTypeVector && wordCount >= 4) {
      typeDefs[binary[i + 1]] = {i, opcode};
      lastTypeDefEnd = i + wordCount;
    } else if (opcode == kOpTypePointer && wordCount >= 4) {
      typeDefs[binary[i + 1]] = {i, opcode};
      lastTypeDefEnd = i + wordCount;
    } else if (opcode == kOpVariable && wordCount >= 4) {
      varResultType[binary[i + 2]] = binary[i + 1];
    } else if (opcode == kOpStore && wordCount >= 3) {
      // [wc|op, ptr_id, value_id, ...]
      uint32_t ptrId = binary[i + 1];
      uint32_t valId = binary[i + 2];
      if (carrierVarIds.contains(ptrId))
        carrierStore[ptrId] = {i, valId};
      else if (atomicCarrierVarIds.contains(ptrId))
        atomicCarrierStore[ptrId] = {i, valId};
    } else if (opcode == kOpLoad && wordCount >= 4) {
      // [wc|op, type_id, result_id, ptr_id]
      uint32_t ptrId = binary[i + 3];
      if (resultVarIds.contains(ptrId))
        resultLoad[ptrId] = {i, binary[i + 2], binary[i + 1]};
      else if (atomicResultVarIds.contains(ptrId))
        atomicResultLoad[ptrId] = {i, binary[i + 2], binary[i + 1]};
    } else if (opcode == kOpTypeInt && wordCount >= 4) {
      // Track 32-bit int type ids so the atomic patch can reuse an existing
      // OpTypeInt 32 rather than synthesizing a duplicate.
      if (binary[i + 2] == 32)
        intTypeIds.push_back(binary[i + 1]);
      typeDefs[binary[i + 1]] = {i, opcode};
      lastTypeDefEnd = i + wordCount;
    } else if (opcode == kOpConstant && wordCount >= 4) {
      // Constants live in the type/constant section; track the end so
      // synthesized types/constants can be appended without forward refs.
      lastTypeDefEnd = i + wordCount;
    } else if (opcode == kOpMemoryModel) {
      // OpDecorate must follow OpMemoryModel (SPIR-V layout: decorations come
      // after the memory model instruction). Record its end offset as the
      // anchor for the access-qualifier OpDecorate inserts.
      memoryModelEnd = i + wordCount;
    } else if (opcode == kOpDecorate) {
      // Track the end of the last existing OpDecorate so new OpDecorate
      // inserts land at the end of the annotation section (SPIR-V requires all
      // decorations to be contiguous, after EntryPoint/ExecutionMode and
      // before type definitions).
      lastDecorateEnd = i + wordCount;
    }
    i += wordCount;
  }

  if (carriers.empty() && atomicCarriers.empty() && atomicKindMarkers.empty())
    return;

  // Resolve image variable id -> image type id (walk var -> ptr -> image).
  auto resolveImageType = [&](uint32_t varId) -> uint32_t {
    auto it = varResultType.find(varId);
    if (it == varResultType.end())
      return 0;
    uint32_t cur = it->second;
    for (int guard = 0; guard < 16 && cur != 0; ++guard) {
      auto t = typeDefs.find(cur);
      if (t == typeDefs.end())
        return 0;
      size_t off = t->second.first;
      uint32_t op = t->second.second;
      if (op == kOpTypePointer) {
        cur = binary[off + 3]; // element_type_id
        continue;
      }
      if (op == kOpTypeImage)
        return cur;
      return 0;
    }
    return 0;
  };

  // Build the rewrite plan. For each carrier, classify read vs write and
  // extract (param, callId). Writes come in pairs (coord + texel) keyed by
  // (param, callId).
  struct ReadOp {
    uint32_t imageVarId;
    uint32_t coordId;
    size_t carrierStoreOff;  // offset of the carrier OpStore to delete
    size_t resultLoadOff;    // offset of the result OpLoad to rewrite
    uint32_t resultId;       // the Load's result id (reused for OpImageRead)
    uint32_t resultTypeId;   // v4float type id (OpImageRead result type)
  };
  struct WriteOp {
    uint32_t imageVarId;
    uint32_t coordId;
    uint32_t texelId;
    size_t coordStoreOff;
    size_t texelStoreOff;
  };
  llvm::SmallVector<ReadOp, 8> reads;
  llvm::SmallVector<WriteOp, 8> writes;

  // Map (param, callId) -> write carrier pieces.
  struct WritePieces {
    uint32_t imageVarId = 0;
    uint32_t coordId = 0;
    uint32_t texelId = 0;
    size_t coordStoreOff = 0;
    size_t texelStoreOff = 0;
    bool hasCoord = false, hasTexel = false;
  };
  llvm::StringMap<WritePieces> writeByKey;

  for (const Carrier &c : carriers) {
    // Parse name: __vc_imgread_<param>_<id> | __vc_imgwrite_{coord,texel}_<param>_<id>
    bool isRead = c.name.starts_with("__vc_imgread_");
    std::string rest = c.name.substr(isRead ? strlen("__vc_imgread_")
                                            : strlen("__vc_imgwrite_"));
    // For writes, strip the "coord_"/"texel_" prefix.
    bool isCoord = false, isTexel = false;
    if (!isRead) {
      if (rest.starts_with("coord_")) {
        isCoord = true;
        rest = rest.substr(strlen("coord_"));
      } else if (rest.starts_with("texel_")) {
        isTexel = true;
        rest = rest.substr(strlen("texel_"));
      }
    }
    // rest = "<param>_<id>"; split at last '_'
    auto lastUS = rest.rfind('_');
    if (lastUS == std::string::npos)
      continue;
    std::string param = rest.substr(0, lastUS);
    std::string callId = rest.substr(lastUS + 1);

    auto imgIt = imageVarByName.find(param);
    if (imgIt == imageVarByName.end())
      continue;
    uint32_t imageVarId = imgIt->second;
    auto stIt = carrierStore.find(c.dummyVarId);
    if (stIt == carrierStore.end())
      continue;
    auto [storeOff, valId] = stIt->second;

    if (isRead) {
      // Find this read's result placeholder Load by name: the result variable
      // is __vc_imgresult_<param>_<id>, and a single OpLoad reads it.
      std::string resName = "__vc_imgresult_" + param + "_" + callId;
      uint32_t resultId = 0, resultType = 0;
      size_t loadOff = 0;
      auto rvIt = resultVarByName.find(resName);
      if (rvIt != resultVarByName.end()) {
        auto rlIt = resultLoad.find(rvIt->second);
        if (rlIt != resultLoad.end()) {
          auto [off, rid, ty] = rlIt->second;
          loadOff = off;
          resultId = rid;
          resultType = ty;
        }
      }
      reads.push_back({imageVarId, valId, storeOff, loadOff,
                       resultId, resultType});
    } else {
      std::string key = param + "_" + callId;
      WritePieces &wp = writeByKey[key];
      wp.imageVarId = imageVarId;
      if (isCoord) {
        wp.coordId = valId;
        wp.coordStoreOff = storeOff;
        wp.hasCoord = true;
      } else if (isTexel) {
        wp.texelId = valId;
        wp.texelStoreOff = storeOff;
        wp.hasTexel = true;
      }
    }
  }
  for (auto &kv : writeByKey) {
    if (kv.second.hasCoord && kv.second.hasTexel) {
      const WritePieces &wp = kv.second;
      writes.push_back({wp.imageVarId, wp.coordId, wp.texelId,
                        wp.coordStoreOff, wp.texelStoreOff});
    }
  }

  // ---- image atomic plan ----
  // Each atomic call has: a coord carrier, a value carrier, an optional cmp
  // carrier (CompSwap), a kind marker (OpName suffix encodes the op), and a
  // result placeholder Load (scalar i32). Group carriers by (param, callId),
  // match the kind, and locate the result Load.
  struct AtomicOp {
    uint32_t imageVarId = 0;
    uint32_t coordId = 0;
    uint32_t valueId = 0;     // the data operand
    uint32_t cmpId = 0;       // comparator (CompSwap only); 0 otherwise
    bool isCompSwap = false;
    uint32_t opcode = 0;      // OpAtomic* opcode
    size_t resultLoadOff = 0; // offset of the result OpLoad to rewrite
    uint32_t resultId = 0;    // the Load's result id (reused for OpAtomic*)
    size_t coordStoreOff = 0; // for deletion
    size_t valueStoreOff = 0;
    size_t cmpStoreOff = 0;
    bool hasCmp = false;
  };

  // (param, callId) -> gathered pieces.
  struct AtomicPieces {
    uint32_t imageVarId = 0;
    uint32_t coordId = 0;  size_t coordStoreOff = 0; bool hasCoord = false;
    uint32_t valueId = 0;  size_t valueStoreOff = 0; bool hasValue = false;
    uint32_t cmpId = 0;    size_t cmpStoreOff = 0;   bool hasCmp = false;
    uint32_t opcode = 0;
    bool isCompSwap = false;
  };
  llvm::StringMap<AtomicPieces> atomicByKey;

  // Helper: parse "<param>_<id>" out of a carrier name whose prefix has been
  // stripped (returns param in `outParam`, id in `outId`).
  auto splitParamId = [](const std::string &s, std::string &outParam,
                         std::string &outId) {
    auto lastUS = s.rfind('_');
    if (lastUS == std::string::npos) return false;
    outParam = s.substr(0, lastUS);
    outId = s.substr(lastUS + 1);
    return true;
  };

  for (const Carrier &c : atomicCarriers) {
    // name: __vc_imgatomic_{coord,value,cmp}_<param>_<id>
    std::string rest;
    bool isCoord = false, isValue = false, isCmp = false;
    if (c.name.starts_with("__vc_imgatomic_coord_")) {
      isCoord = true; rest = c.name.substr(strlen("__vc_imgatomic_coord_"));
    } else if (c.name.starts_with("__vc_imgatomic_value_")) {
      isValue = true; rest = c.name.substr(strlen("__vc_imgatomic_value_"));
    } else if (c.name.starts_with("__vc_imgatomic_cmp_")) {
      isCmp = true; rest = c.name.substr(strlen("__vc_imgatomic_cmp_"));
    } else {
      continue;
    }
    std::string param, callId;
    if (!splitParamId(rest, param, callId)) continue;
    auto imgIt = imageVarByName.find(param);
    if (imgIt == imageVarByName.end()) continue;
    auto stIt = atomicCarrierStore.find(c.dummyVarId);
    if (stIt == atomicCarrierStore.end()) continue;
    auto [storeOff, valId] = stIt->second;
    std::string key = param + "_" + callId;
    AtomicPieces &ap = atomicByKey[key];
    ap.imageVarId = imgIt->second;
    if (isCoord) { ap.coordId = valId; ap.coordStoreOff = storeOff; ap.hasCoord = true; }
    else if (isValue) { ap.valueId = valId; ap.valueStoreOff = storeOff; ap.hasValue = true; }
    else if (isCmp) { ap.cmpId = valId; ap.cmpStoreOff = storeOff; ap.hasCmp = true; }
  }

  // Kind markers: __vc_imgatomic_kind_<param>_<id>_<KIND>
  static const llvm::StringMap<uint32_t> kKindOpcode = {
      {"add", kOpAtomicIAdd},     {"sub", kOpAtomicISub},
      {"min", kOpAtomicSMin},     {"max", kOpAtomicSMax},
      {"and", kOpAtomicAnd},      {"or", kOpAtomicOr},
      {"xor", kOpAtomicXor},      {"exch", kOpAtomicExchange},
      {"cswap", kOpAtomicCompareExchange},
  };
  for (const Carrier &c : atomicKindMarkers) {
    // rest = "<param>_<id>_<KIND>"
    std::string rest = c.name.substr(strlen("__vc_imgatomic_kind_"));
    // KIND is the suffix after the last '_'; (param,id) is before it.
    auto lastUS = rest.rfind('_');
    if (lastUS == std::string::npos) continue;
    std::string kindStr = rest.substr(lastUS + 1);
    std::string key = rest.substr(0, lastUS); // "<param>_<id>"
    auto kit = kKindOpcode.find(kindStr);
    if (kit == kKindOpcode.end()) continue;
    AtomicPieces &ap = atomicByKey[key];
    ap.opcode = kit->second;
    ap.isCompSwap = (kindStr == "cswap");
  }

  llvm::SmallVector<AtomicOp, 8> atomics;
  for (auto &kv : atomicByKey) {
    AtomicPieces &ap = kv.second;
    if (!ap.hasCoord || !ap.hasValue || ap.opcode == 0) continue;
    if (ap.isCompSwap && !ap.hasCmp) continue;
    std::string resName = "__vc_imgatomic_result_" + kv.first().str();
    AtomicOp op;
    op.imageVarId = ap.imageVarId;
    op.coordId = ap.coordId;
    op.valueId = ap.valueId;
    op.cmpId = ap.cmpId;
    op.isCompSwap = ap.isCompSwap;
    op.opcode = ap.opcode;
    op.coordStoreOff = ap.coordStoreOff;
    op.valueStoreOff = ap.valueStoreOff;
    op.cmpStoreOff = ap.cmpStoreOff;
    op.hasCmp = ap.hasCmp;
    auto rvIt = atomicResultVarByName.find(resName);
    if (rvIt != atomicResultVarByName.end()) {
      auto rlIt = atomicResultLoad.find(rvIt->second);
      if (rlIt != atomicResultLoad.end()) {
        auto [off, rid, ty] = rlIt->second;
        op.resultLoadOff = off;
        op.resultId = rid;
      }
    }
    atomics.push_back(op);
  }

  if (reads.empty() && writes.empty() && atomicKindMarkers.empty())
    return;

  // Determine the current Bound (header word at index 3). New ids are
  // allocated above Bound.
  uint32_t &bound = binary[3];

  // Build the set of offsets to DELETE (carrier stores, dummy OpVariables for
  // carriers, their OpName records, and the read Undef) and the list of
  // instruction sequences to INSERT (at the carrier store's offset, so the
  // new image ops land in the function body where the carriers lived).
  //
  // Insert sequences are emitted as raw word vectors; each is spliced in
  // place of its carrier store. Deletions of the dummy globals / OpNames /
  // Undef happen by filtering them out during the rebuild.
  llvm::DenseSet<size_t> deleteOffsets; // instruction-start offsets to drop

  // Mark carrier dummy OpVariables + their OpName records for deletion.
  // (We mark the OpStore offsets separately as splice points.)
  auto markNameAndVar = [&](uint32_t dummyVarId) {
    // find OpName targeting dummyVarId and the OpVariable defining it.
    size_t j = 5;
    while (j < binary.size()) {
      uint32_t w = binary[j];
      uint32_t op = w & 0xFFFF;
      uint32_t wc = w >> 16;
      if (wc == 0 || j + wc > binary.size())
        break;
      if (op == kOpName && wc >= 3 && binary[j + 1] == dummyVarId)
        deleteOffsets.insert(j);
      else if (op == kOpVariable && wc >= 4 && binary[j + 2] == dummyVarId)
        deleteOffsets.insert(j);
      j += wc;
    }
  };

  struct Insert {
    size_t at; // splice offset (replaces the instruction that starts here)
    SmallVector<uint32_t, 8> words;
  };
  llvm::SmallVector<Insert, 8> inserts;

  // For reads: emit OpLoad(image) + OpImageRead at the result Load's offset,
  // reusing the Load's result id for OpImageRead (so existing uses of the
  // placeholder resolve to the real read). The carrier coord Store and the
  // result placeholder Load/Variable are deleted.
  for (const ReadOp &r : reads) {
    uint32_t imgValId = bound++;
    // OpLoad's type_id is the POINTED-TO type (the image type), not the
    // pointer type. Resolve var -> ptr type -> element (image) type.
    uint32_t imgTypeId = 0;
    auto it = varResultType.find(r.imageVarId);
    if (it != varResultType.end()) {
      auto pt = typeDefs.find(it->second);
      if (pt != typeDefs.end() && pt->second.second == kOpTypePointer)
        imgTypeId = binary[pt->second.first + 3]; // element_type_id
    }
    SmallVector<uint32_t, 8> seq;
    // OpLoad [wc|op, type_id, result_id, ptr_id]
    seq.push_back((4u << 16) | kOpLoad);
    seq.push_back(imgTypeId);
    seq.push_back(imgValId);
    seq.push_back(r.imageVarId);
    // OpImageRead [wc|op, type_id, result_id, image_id, coord_id]
    seq.push_back((5u << 16) | kOpImageRead);
    seq.push_back(r.resultTypeId);
    seq.push_back(r.resultId);
    seq.push_back(imgValId);
    seq.push_back(r.coordId);
    inserts.push_back({r.resultLoadOff, std::move(seq)});
    // delete the carrier coord store and the result placeholder Load itself
    // (its slot is now the OpImageRead from the insert above).
    deleteOffsets.insert(r.carrierStoreOff);
    deleteOffsets.insert(r.resultLoadOff);
  }

  // For writes: emit OpLoad(image) + OpImageWrite at the coord carrier store,
  // and delete both carrier stores + dummy vars/names.
  for (const WriteOp &w : writes) {
    uint32_t imgValId = bound++;
    uint32_t imgTypeId = 0;
    auto it = varResultType.find(w.imageVarId);
    if (it != varResultType.end()) {
      auto pt = typeDefs.find(it->second);
      if (pt != typeDefs.end() && pt->second.second == kOpTypePointer)
        imgTypeId = binary[pt->second.first + 3];
    }
    SmallVector<uint32_t, 8> seq;
    seq.push_back((4u << 16) | kOpLoad);
    seq.push_back(imgTypeId);
    seq.push_back(imgValId);
    seq.push_back(w.imageVarId);
    seq.push_back((4u << 16) | kOpImageWrite);
    seq.push_back(imgValId);
    seq.push_back(w.coordId);
    seq.push_back(w.texelId);
    inserts.push_back({w.coordStoreOff, std::move(seq)});
    deleteOffsets.insert(w.coordStoreOff);
    deleteOffsets.insert(w.texelStoreOff);
  }

  // ---- image atomics: splice OpImageTexelPointer + OpAtomic* ----
  // The atomic's result reuses the result placeholder Load's result id (so
  // existing uses of the placeholder resolve to the real atomic result). The
  // carrier coord/value/cmp stores and the result placeholder Load/Variable
  // are deleted.
  //
  // Types needed (may not exist yet): OpTypeInt 32 (the atomic value/result
  // scalar) and OpTypePointer UniformConstant <i32> (the texel pointer's result
  // type), plus OpConstant 0/1 (Sample operand, scope, semantics — these are
  // Ids, not literals). All are inserted at `typeAnchor` = just past the last
  // existing type/constant def, so they come AFTER any type they reference
  // (e.g. %uint) and before the first function — no forward references. Ids are
  // allocated above Bound.
  if (!atomics.empty()) {
    size_t typeAnchor = lastTypeDefEnd ? lastTypeDefEnd
                        : (lastDecorateEnd ? lastDecorateEnd
                        : (memoryModelEnd ? memoryModelEnd : 5));

    // Find or create a 32-bit int type id.
    uint32_t i32Ty = 0;
    for (uint32_t id : intTypeIds) { i32Ty = id; break; }
    if (!i32Ty) {
      i32Ty = bound++;
      SmallVector<uint32_t> seq;
      // OpTypeInt [wc|op, result_id, width, signedness]  signedness=1 (signed)
      seq.push_back((4u << 16) | kOpTypeInt);
      seq.push_back(i32Ty);
      seq.push_back(32);
      seq.push_back(1);
      inserts.push_back({typeAnchor, std::move(seq)});
    }

    // Find or create OpTypePointer Image <i32> (the texel pointer result type
    // for OpImageTexelPointer — its Result Type MUST be a pointer with Storage
    // Class Image, NOT UniformConstant). Scan existing OpTypePointer defs for a
    // match; else synthesize one.
    uint32_t ptrI32Ty = 0;
    for (auto &kv : typeDefs) {
      if (kv.second.second != kOpTypePointer) continue;
      size_t off = kv.second.first;
      // [wc|op, result_id, SC, elem_id]
      if (binary[off + 2] == kSCImage && binary[off + 3] == i32Ty) {
        ptrI32Ty = kv.first;
        break;
      }
    }
    if (!ptrI32Ty) {
      ptrI32Ty = bound++;
      SmallVector<uint32_t> seq;
      seq.push_back((4u << 16) | kOpTypePointer);
      seq.push_back(ptrI32Ty);
      seq.push_back(kSCImage);
      seq.push_back(i32Ty);
      inserts.push_back({typeAnchor, std::move(seq)});
    }

    // OpImageTexelPointer's Sample operand is an Id (a scalar integer), NOT a
    // literal — it must be 0 for non-multisampled images, but it's still an Id
    // referencing a 0 constant. Find an existing OpConstant <i32> 0, or
    // synthesize one.
    uint32_t zeroI32Id = 0;
    for (size_t off = 5; off + 3 < binary.size();) {
      uint32_t w = binary[off];
      uint32_t op = w & 0xFFFF;
      uint32_t wc = w >> 16;
      if (wc == 0 || off + wc > binary.size()) break;
      // OpConstant [wc|op, type_id, result_id, value...]  (i32 value is 1 word)
      if (op == kOpConstant && wc == 4 && binary[off + 1] == i32Ty &&
          binary[off + 3] == 0) {
        zeroI32Id = binary[off + 2];
        break;
      }
      off += wc;
    }
    if (!zeroI32Id) {
      zeroI32Id = bound++;
      SmallVector<uint32_t> seq;
      // OpConstant [wc|op, type_id, result_id, value]
      seq.push_back((4u << 16) | kOpConstant);
      seq.push_back(i32Ty);
      seq.push_back(zeroI32Id);
      seq.push_back(0);
      inserts.push_back({typeAnchor, std::move(seq)});
    }

    // OpAtomic*'s Memory (scope) and Semantics operands are Ids referencing
    // constant integers, NOT literals. Scope=Device(1); Semantics=None(0) for
    // relaxed atomics. Find or synthesize a constant `1` for the Device scope.
    uint32_t oneI32Id = 0;
    for (size_t off = 5; off + 3 < binary.size();) {
      uint32_t w = binary[off];
      uint32_t op = w & 0xFFFF;
      uint32_t wc = w >> 16;
      if (wc == 0 || off + wc > binary.size()) break;
      if (op == kOpConstant && wc == 4 && binary[off + 1] == i32Ty &&
          binary[off + 3] == 1) {
        oneI32Id = binary[off + 2];
        break;
      }
      off += wc;
    }
    if (!oneI32Id) {
      oneI32Id = bound++;
      SmallVector<uint32_t> seq;
      seq.push_back((4u << 16) | kOpConstant);
      seq.push_back(i32Ty);
      seq.push_back(oneI32Id);
      seq.push_back(1);
      inserts.push_back({typeAnchor, std::move(seq)});
    }

    for (const AtomicOp &a : atomics) {
      // OpImageTexelPointer takes the image VARIABLE pointer directly as its
      // Image operand (NOT a loaded image value — this differs from
      // OpImageRead/Write, which take the loaded image). Verified against
      // glslc's lowering of `imageAtomicAdd`: the variable id is passed
      // straight to OpImageTexelPointer.
      SmallVector<uint32_t, 8> seq;

      // OpImageTexelPointer [wc|op, type_id(ptr<Image,i32>), result_id, Image, Coord, Sample]
      uint32_t texelPtrId = bound++;
      seq.push_back((6u << 16) | kOpImageTexelPointer);
      seq.push_back(ptrI32Ty);
      seq.push_back(texelPtrId);
      seq.push_back(a.imageVarId);  // the image variable pointer
      seq.push_back(a.coordId);
      seq.push_back(zeroI32Id);     // Sample Id = 0 (non-multisampled)

      // OpAtomic* [wc|op, type_id, result_id, Pointer, Memory, Semantics, Value]
      // OpAtomicCompareExchange adds Equal, Unequal, Comparator (no Semantics-
      // Value collapse): [wc|op, type_id, result_id, Pointer, Memory, Equal,
      // Unequal, Value, Comparator]
      if (a.isCompSwap) {
        seq.push_back((9u << 16) | kOpAtomicCompareExchange);
        seq.push_back(i32Ty);
        seq.push_back(a.resultId);
        seq.push_back(texelPtrId);
        seq.push_back(oneI32Id);    // Memory scope Id = Device(1)
        seq.push_back(zeroI32Id);   // Equal semantics Id = None(0)
        seq.push_back(zeroI32Id);   // Unequal semantics Id = None(0)
        seq.push_back(a.valueId);   // Value (data)
        seq.push_back(a.cmpId);     // Comparator
      } else {
        seq.push_back((7u << 16) | a.opcode);
        seq.push_back(i32Ty);
        seq.push_back(a.resultId);
        seq.push_back(texelPtrId);
        seq.push_back(oneI32Id);    // Memory scope Id = Device(1)
        seq.push_back(zeroI32Id);   // Semantics Id = None(0)
        seq.push_back(a.valueId);   // Value
      }
      inserts.push_back({a.resultLoadOff, std::move(seq)});
      // Delete the carrier stores (coord/value/cmp) and the result Load.
      deleteOffsets.insert(a.coordStoreOff);
      deleteOffsets.insert(a.valueStoreOff);
      if (a.hasCmp)
        deleteOffsets.insert(a.cmpStoreOff);
      deleteOffsets.insert(a.resultLoadOff);
    }
  }

  // Mark dummy carrier OpVariables + their OpNames for deletion.
  for (const Carrier &c : carriers)
    markNameAndVar(c.dummyVarId);
  // Mark read result placeholder OpVariables + their OpNames for deletion.
  for (const auto &kv : resultVarByName)
    markNameAndVar(kv.second);
  // Mark image-atomic carrier + kind-marker dummy OpVariables + OpNames.
  for (const Carrier &c : atomicCarriers)
    markNameAndVar(c.dummyVarId);
  for (const Carrier &c : atomicKindMarkers)
    markNameAndVar(c.dummyVarId);
  // Mark atomic result placeholder OpVariables + their OpNames.
  for (const auto &kv : atomicResultVarByName)
    markNameAndVar(kv.second);

  // Emit OpDecorate NonReadable / NonWritable for access-qualified image
  // variables (read_only / write_only). OpDecorate produces no result id, so
  // no Bound bump is needed. SPIR-V layout requires decorations to come AFTER
  // OpMemoryModel/OpEntryPoint/OpExecutionMode but BEFORE type definitions,
  // and all OpDecorate must be contiguous. Anchor just past the last existing
  // OpDecorate (the end of the annotation section); if none exist, fall back
  // to just past OpMemoryModel. spirv-val rejects decorations placed in the
  // wrong section.
  size_t decorAnchor = lastDecorateEnd ? lastDecorateEnd
                  : (memoryModelEnd ? memoryModelEnd : 5);
  for (const auto &kv : imageAccess) {
    uint32_t varId = kv.first;
    uint32_t decor = kv.second;
    SmallVector<uint32_t> seq;
    seq.push_back((3u << 16) | kOpDecorate); // WC=3: op, target, decoration
    seq.push_back(varId);
    seq.push_back(decor);
    inserts.push_back({decorAnchor, std::move(seq)});
  }

  // Rebuild: walk the binary; at each instruction, if it's an insert point,
  // emit the insert words (and skip the original instruction if also deleted);
  // else emit or skip per deleteOffsets.
  std::sort(inserts.begin(), inserts.end(),
            [](const Insert &a, const Insert &b) { return a.at < b.at; });

  SmallVector<uint32_t, 0> out;
  out.reserve(binary.size());
  // Copy header (5 words) verbatim.
  out.insert(out.end(), binary.begin(), binary.begin() + 5);

  size_t k = 5;
  size_t insIdx = 0;
  while (k < binary.size()) {
    uint32_t w = binary[k];
    uint32_t wc = w >> 16;
    if (wc == 0 || k + wc > binary.size())
      break;
    // Emit any inserts anchored at this offset first.
    while (insIdx < inserts.size() && inserts[insIdx].at == k) {
      out.insert(out.end(), inserts[insIdx].words.begin(),
                 inserts[insIdx].words.end());
      ++insIdx;
    }
    if (deleteOffsets.contains(k)) {
      k += wc;
      continue;
    }
    out.insert(out.end(), binary.begin() + k, binary.begin() + k + wc);
    k += wc;
  }
  // Trailing inserts beyond the last instruction (shouldn't happen, but safe).
  while (insIdx < inserts.size()) {
    out.insert(out.end(), inserts[insIdx].words.begin(),
               inserts[insIdx].words.end());
    ++insIdx;
  }

  binary.swap(out);
  // bound was already bumped in-place on the original header word, which we
  // copied into `out` via the header slice — so the new binary reflects it.
}

static bool loadSource(const std::string &path, llvm::SourceMgr &sm) {
  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code ec = buf.getError()) {
    llvm::errs() << "error: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  sm.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  return true;
}

// ---------------------------------------------------------------------------
// Vulkan SC emission (-emit=sc)
//
// Vulkan SC removed runtime pipeline compilation: every pipeline the
// application may ever create has to be compiled ahead of time by nvidia-pcc
// into a pipeline_cache.bin, and the application selects one at create time by a
// 16-byte UUID. Two consequences shape this emitter.
//
//  1. The workgroup size is part of a pipeline's identity, so a kernel launched
//     with two different block sizes needs two pipelines. Because the block size
//     is baked into the SPIR-V (there is no run time at which a specialization
//     constant could be supplied — see lowerWorkgroupSizeToLiteral), the whole
//     lowering pipeline runs once per distinct block size and each run produces
//     one binary.
//  2. There is no module in the process for the runtime to reflect, so the
//     descriptor-set layout and the push-constant range must be stated
//     explicitly — both in the pcc JSON and in the host descriptor handed to
//     vcLoadKernelOffline. Both are written from one table here, so they cannot
//     disagree; a disagreement would otherwise surface much later as a bare
//     VK_ERROR_VALIDATION_FAILED (-1000011001) from vkCreateComputePipelines
//     with no hint of which field differs.
//
// Layout of the output directory (-o):
//   <kernel>.b<X>_<Y>_<Z>.spv   block-size-baked SPIR-V for one kernel
//   <kernel>.b<X>_<Y>_<Z>.json  the nvidia-pcc descriptor that names it
//   <stem>_host.cpp             host C++ that loads them via
//                               vcLoadKernelOffline (no SPIR-V embedded)
//
// Feed the whole directory to one nvidia-pcc run (see tools/vksc/
// build_cache.sh): a partial run produces a cache whose other lookups fail with
// "no pipeline matches".
// ---------------------------------------------------------------------------

// Stable 16-byte offline-pipeline identifier for a (kernel, block) pair.
//
// pcc copies this UUID into the cache and the runtime looks the pipeline up by
// it, so it must be deterministic across compilations, distinct per block size
// (the same kernel at a different workgroup size is a different pipeline), and
// never all-zero (which the runtime read as "unset"). Two independent FNV-1a
// mixes of a canonical key give the 16 bytes.
// The 16-byte pipeline identifier, derived only from `key`.
//
// The identifier is what VkPipelineOfflineCreateInfo selects a pipeline with, so
// two distinct pipelines that hash the same would silently resolve to one
// another's binary. Callers therefore key it on the *final* SPIR-V bytes plus
// the entry point and block dims (see emitSc), never on the kernel name alone:
// two translation units can each define a kernel called `vadd`, and a name-only
// key would give them the same UUID. Hashing the content also keeps a rebuild of
// unchanged source byte-identical, so an existing pipeline_cache.bin stays valid
// — and it does not depend on the source path, so building the same file from a
// different directory does not invalidate the cache either.
static void scHashUuid(llvm::StringRef key, uint8_t out[16]) {
  auto fnv = [](llvm::StringRef s, uint64_t seed) {
    uint64_t h = seed;
    for (char ch : s) {
      h ^= static_cast<uint64_t>(static_cast<unsigned char>(ch));
      h *= 0x100000001b3ull;
    }
    return h;
  };
  static const char kPrefix[] = "vc-sc-v1|";
  uint64_t a = fnv(key, 0xcbf29ce484222325ull ^ fnv(kPrefix, 0xcbf29ce484222325ull));
  uint64_t b = fnv(key, 0x9e3779b97f4a7c15ull ^ fnv(kPrefix, 0x9e3779b97f4a7c15ull));
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<uint8_t>((a >> (8 * i)) & 0xFF);
    out[8 + i] = static_cast<uint8_t>((b >> (8 * i)) & 0xFF);
  }
  if (!a && !b) out[0] = 1; // never the all-zero "unset" identifier
}

// The nvidia-pcc pipeline descriptor for one (kernel, block) pair.
//
// The schema is NVIDIA's, not Vulkan's: the vendor samples under
// /ota/samples/vulkan-sc-samples/data/pipeline/<sample>/ are the reference, and
// this reproduces the shape of tools/vksc/vk_vadd_pipeline_1.json (the sample
// used to bring the cache up) with the fields a VC kernel changes:
//
//   * pName is the kernel's SPIR-V entry-point name. VC's MLIR backend emits one
//     OpEntryPoint per kernel, named after the mangled kernel symbol — NOT
//     "main" (that is the GLSL backend's convention). pcc compiles the entry
//     point named here, so naming the wrong one yields a cache entry the
//     application can never match.
//   * the bindings are one STORAGE_BUFFER per pointer parameter, in parameter
//     order, then one STORAGE_IMAGE per image parameter — the same order
//     VCToGPU::lowerScalarArgsToPushConstant and ASTToMLIR::materializeImage
//     Params assign binding numbers in, which is also the order
//     VCRuntime::buildLayout describes for an offline kernel.
//   * pSpecializationInfo is NULL: the workgroup size is baked into the SPIR-V,
//     so there is nothing left to specialize (and any specialization info the
//     application supplied would have to match what pcc cached exactly).
static std::string scPipelineJson(llvm::StringRef spvBasename,
                                  llvm::StringRef entryPoint,
                                  const uint8_t uuid[16], uint32_t ssboCount,
                                  uint32_t imageCount, uint32_t pcSize) {
  const uint32_t bindingCount = ssboCount + imageCount;
  std::string s;
  llvm::raw_string_ostream os(s);
  const char *kStage = "VK_SHADER_STAGE_COMPUTE_BIT";

  os << "{\n    \"ComputePipelineState\": {\n";
  os << "        \"YcbcrSamplers\": [],\n";
  os << "        \"ImmutableSamplers\": [],\n";
  os << "        \"DescriptorSetLayouts\": [\n            {\n";
  os << "                \"pSetLayouts_0\": {\n";
  os << "                    \"sType\": "
        "\"VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO\",\n";
  os << "                    \"pNext\": \"NULL\",\n";
  os << "                    \"flags\": \"0\",\n";
  os << "                    \"bindingCount\": " << bindingCount << ",\n";
  os << "                    \"pBindings\": [\n";
  for (uint32_t i = 0; i < bindingCount; ++i) {
    os << "                        {\n";
    os << "                            \"binding\": " << i << ",\n";
    os << "                            \"descriptorType\": \""
       << (i < ssboCount ? "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
                         : "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE")
       << "\",\n";
    os << "                            \"descriptorCount\": 1,\n";
    os << "                            \"stageFlags\": \"" << kStage << "\",\n";
    os << "                            \"pImmutableSamplers\": \"NULL\"\n";
    os << "                        }" << (i + 1 < bindingCount ? "," : "")
       << "\n";
  }
  os << "                    ]\n";
  os << "                }\n            }\n        ],\n";
  os << "        \"PipelineLayout\": {\n";
  os << "            \"sType\": "
        "\"VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO\",\n";
  os << "            \"pNext\": \"NULL\",\n";
  os << "            \"flags\": 0,\n";
  os << "            \"setLayoutCount\": 1,\n";
  os << "            \"pSetLayouts\": [\n                \"pSetLayouts_0\"\n"
        "            ],\n";
  if (pcSize) {
    os << "            \"pushConstantRangeCount\": 1,\n";
    os << "            \"pPushConstantRanges\": [\n                {\n";
    os << "                    \"sType\": "
          "\"VK_STRUCTURE_TYPE_PUSH_CONSTANT_RANGE\",\n";
    os << "                    \"stageFlags\": \"" << kStage << "\",\n";
    os << "                    \"offset\": 0,\n";
    os << "                    \"size\": " << pcSize << "\n";
    os << "                }\n            ]\n";
  } else {
    os << "            \"pushConstantRangeCount\": 0,\n";
    os << "            \"pPushConstantRanges\": []\n";
  }
  os << "        },\n";
  os << "        \"ComputePipeline\": {\n";
  os << "            \"sType\": "
        "\"VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO\",\n";
  os << "            \"pNext\": \"NULL\",\n";
  os << "            \"flags\": \"0\",\n";
  os << "            \"stage\": {\n";
  os << "                \"sType\": "
        "\"VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO\",\n";
  os << "                \"pNext\": \"NULL\",\n";
  os << "                \"flags\": \"0\",\n";
  os << "                \"stage\": \"" << kStage << "\",\n";
  os << "                \"module\": \"\",\n";
  os << "                \"pName\": \"" << entryPoint << "\",\n";
  os << "                \"pSpecializationInfo\": \"NULL\"\n";
  os << "            },\n";
  os << "            \"layout\": \"\",\n";
  os << "            \"basePipelineHandle\": \"\",\n";
  os << "            \"basePipelineIndex\": 0\n";
  os << "        },\n";
  os << "        \"ShaderFileNames\": [\n            {\n";
  os << "                \"stage\": \"" << kStage << "\",\n";
  os << "                \"filename\": \"" << spvBasename << "\"\n";
  os << "            }\n        ]\n";
  os << "    },\n";
  os << "    \"EnabledExtensions\": [],\n";
  os << "    \"PipelineUUID\": [\n";
  for (int i = 0; i < 16; ++i)
    os << "        " << (unsigned)uuid[i] << (i < 15 ? "," : "") << "\n";
  os << "    ]\n}\n";
  return s;
}

static bool scWriteText(llvm::StringRef path, llvm::StringRef text) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "vc: cannot write " << path << ": " << ec.message() << "\n";
    return false;
  }
  out << text;
  out.close();
  return true;
}

// One offline pipeline: a kernel at one block size, plus everything pcc and the
// runtime need to agree on.
struct ScPipelineInfo {
  std::string kernel;     // mangled device symbol = the OpEntryPoint name
  int64_t bx = 1, by = 1, bz = 1;
  uint8_t uuid[16] = {};
  uint32_t ssboCount = 0, imageCount = 0, pcSize = 0;
};

// The `-emit=sc` driver. Returns a process exit code.
static int emitSc(const TranslationUnit &tu, MLIRContext &ctx,
                  bool warningsAsErrors, bool allowF64MathF32,
                  const std::string &outDir, const std::string &inputFilename) {
  using namespace mlir;

  // --- kernels: device symbol -> decl -------------------------------------
  llvm::StringMap<const FunctionDecl *> kernelBySymbol;
  std::function<void(const std::vector<NodePtr> &)> collect =
      [&](const std::vector<NodePtr> &decls) {
        for (const auto &d : decls) {
          if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
            collect(static_cast<const NamespaceDecl *>(d.get())->decls);
            continue;
          }
          if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
          auto *fn = static_cast<const FunctionDecl *>(d.get());
          if (fn->deviceAttr != DeviceAttr::Global) continue;
          kernelBySymbol[deviceMangledName(fn)] = fn;
        }
      };
  collect(tu.decls);
  if (kernelBySymbol.empty()) {
    llvm::errs() << "vc: no __global__ kernel found in " << inputFilename
                 << "\n";
    return 1;
  }

  // The launch callee's key: a bare name, or the `::`-scope chain mangled the
  // same way the host backend's handle key is (they must agree — the handle the
  // host loads is the one the runtime reads). Mirrors ASTToHost's
  // launchHandleName.
  auto calleeKey = [](const ASTNode *callee) -> std::string {
    if (!callee) return {};
    if (callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      return static_cast<const DeclRefExpr *>(callee)->name.str();
    if (callee->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
      std::vector<std::string> parts;
      const ASTNode *cur = callee;
      while (cur && cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
        auto *ma = static_cast<const MemberAccessExpr *>(cur);
        parts.push_back(ma->member.str());
        cur = ma->base.get();
      }
      if (cur && cur->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
        parts.push_back(static_cast<const DeclRefExpr *>(cur)->name.str());
      std::reverse(parts.begin(), parts.end());
      std::string chain;
      for (auto &p : parts) {
        if (!chain.empty()) chain += "::";
        chain += p;
      }
      return mangleScopeName(chain);
    }
    return {};
  };

  // --- launch sites -> one pipeline per (kernel, block size) ---------------
  //
  // A block size that is not a compile-time constant is fatal here, and it is
  // worth saying why rather than silently picking one: the workgroup size is
  // baked into the offline pipeline, so a launch whose size is only known at run
  // time has no pipeline that could ever match it. The kernel would fail at
  // dispatch with VK_ERROR_NO_PIPELINE_MATCH; refusing to compile says so early.
  std::vector<ScPipelineInfo> pipelines;
  std::set<std::tuple<int64_t, int64_t, int64_t>> blockSizes;
  std::set<std::pair<std::string, std::tuple<int64_t, int64_t, int64_t>>> seen;
  unsigned nonConst = 0;
  for (const LaunchExpr *l : tu.launchSites) {
    std::string key = calleeKey(l->callee.get());
    auto kit = kernelBySymbol.find(key);
    if (kit == kernelBySymbol.end()) continue; // not a kernel we know
    if (!l->blockIsConst) {
      llvm::errs() << "vc: kernel '" << key
                   << "': block size must be constant under Vulkan SC (the "
                      "workgroup size is baked into the precompiled pipeline)\n";
      ++nonConst;
      continue;
    }
    std::tuple<int64_t, int64_t, int64_t> bs{l->blockX, l->blockY, l->blockZ};
    std::string sym = deviceMangledName(kit->second);
    if (!seen.insert({sym, bs}).second) continue;

    ScPipelineInfo p;
    p.kernel = sym;
    std::tie(p.bx, p.by, p.bz) = bs;
    // p.uuid is filled in once the SPIR-V exists (see the block-size loop).
    const FunctionDecl *fn = kit->second;
    for (const ParamDecl *pd : fn->params) {
      if (!pd->type) continue;
      if (pd->type->getKind() == TypeKind::Image) ++p.imageCount;
      else if (pd->type->getKind() == TypeKind::Pointer) ++p.ssboCount;
    }
    blockSizes.insert(bs);
    pipelines.push_back(std::move(p));
  }
  if (nonConst) return 1;
  if (pipelines.empty()) {
    llvm::errs() << "vc: no kernel launch found in " << inputFilename
                 << " (nothing to precompile for Vulkan SC)\n";
    return 1;
  }

  std::error_code ec;
  if (std::error_code mkec = llvm::sys::fs::create_directories(outDir)) {
    llvm::errs() << "vc: cannot create " << outDir << ": " << mkec.message()
                 << "\n";
    return 1;
  }

  // --- one lowering run per distinct block size ---------------------------
  //
  // Each run re-translates the AST: the lowering pipeline consumes the module
  // (it erases the VC ops and repacks them into a gpu.module), so a fresh module
  // is the clean way to get a second LocalSize out of the same source.
  for (const auto &bs : blockSizes) {
    int64_t bx = std::get<0>(bs), by = std::get<1>(bs), bz = std::get<2>(bs);
    auto module =
        codegen::translateASTToMLIR(tu, ctx, warningsAsErrors, allowF64MathF32);
    if (!module) {
      llvm::errs() << "vc: codegen failed\n";
      return 1;
    }
    // Set *after* translateASTToMLIR: it installs its own module attributes as
    // its last act, so an attribute set before the call would be overwritten.
    // (`mod` is a ModuleOp value: attribute access goes through Op's operator->
    // to the underlying Operation, which is where setAttr/getAttrs live.)
    ModuleOp mod = *module;
    SmallVector<Attribute, 3> dims = {IntegerAttr::get(IntegerType::get(&ctx, 32), bx),
                                      IntegerAttr::get(IntegerType::get(&ctx, 32), by),
                                      IntegerAttr::get(IntegerType::get(&ctx, 32), bz)};
    mod->setAttr("vc.sc.block_dims", ArrayAttr::get(&ctx, dims));

    // Cooperative matrix / wmma has no place in an SC pipeline: the runtime
    // cannot enable the feature (the SC header has no
    // VkPhysicalDeviceCooperativeMatrixFeaturesKHR and no extension list on
    // this path), so a pipeline built from wmma SPIR-V would have no device
    // capability behind it and could only fail at create or dispatch time.
    // Refusing here — where the source is still in hand, and with the flag the
    // ASTToMLIR set — is the only point at which the user can act on it.
    if (mod->hasAttr("vc.uses_coopmatrix")) {
      llvm::errs() << "vc: Vulkan SC: this translation unit uses cooperative "
                      "matrix (wmma::), which is not available under Vulkan SC\n";
      return 1;
    }

    if (failed(codegen::runLoweringPipeline(*module))) return 1;

    // The push-constant size packKernels recorded (see
    // lowerScalarArgsToPushConstant) — the only way to know it without a module
    // in the process at run time.
    for (NamedAttribute na : mod->getAttrs()) {
      llvm::StringRef name = na.getName().strref();
      if (!name.consume_front("vc.sc.pc_size.")) continue;
      auto size = na.getValue().dyn_cast<IntegerAttr>();
      if (!size) continue;
      for (ScPipelineInfo &p : pipelines)
        if (p.kernel == name && p.bx == bx && p.by == by && p.bz == bz)
          p.pcSize = (uint32_t)size.getInt();
    }

    SmallVector<uint32_t, 0> binary;
    mlir::spirv::ModuleOp spirvModule;
    module->walk([&](mlir::spirv::ModuleOp m) {
      if (!spirvModule) spirvModule = m;
    });
    if (!spirvModule || failed(mlir::spirv::serialize(spirvModule, binary))) {
      llvm::errs() << "vc: spirv translation failed\n";
      return 1;
    }
    // No patchWorkgroupSizeBuiltin: it decorates a gl_WorkGroupSize
    // spec-constant composite that only exists on the spec-constant path. The
    // dynamic-shared patch still applies, but with the literal block size
    // instead of a spec-constant reference.
    patchDynamicSharedArrays(binary, bx);
    patchImageOps(binary);

    // Identifier, now that the final bytes exist: content + entry point + block
    // dims. The bytes are the whole module (every kernel in the file), so the
    // entry point and the block dims are what separate the pipelines that share
    // one .spv. Every pipeline is read here — `pipelines` and `blockSizes` are
    // built from the same launch-site walk.
    {
      llvm::StringRef bytes(reinterpret_cast<const char *>(binary.data()),
                            binary.size() * sizeof(uint32_t));
      for (ScPipelineInfo &p : pipelines) {
        if (p.bx != bx || p.by != by || p.bz != bz) continue;
        std::string key = (bytes + "|" + p.kernel + "|" +
                           std::to_string(bx) + "x" + std::to_string(by) + "x" +
                           std::to_string(bz))
                              .str();
        scHashUuid(key, p.uuid);
      }
    }
    for (const ScPipelineInfo &p : pipelines) {
      if (p.bx != bx || p.by != by || p.bz != bz) continue;
      std::string stem = p.kernel + ".b" + std::to_string(bx) + "_" +
                         std::to_string(by) + "_" + std::to_string(bz);
      SmallString<256> spvPath(outDir);
      llvm::sys::path::append(spvPath, stem + ".spv");
      std::error_code wec;
      llvm::raw_fd_ostream out(spvPath, wec, llvm::sys::fs::OF_None);
      if (wec) {
        llvm::errs() << "vc: cannot write " << spvPath << ": "
                     << wec.message() << "\n";
        return 1;
      }
      out.write(reinterpret_cast<const char *>(binary.data()),
                binary.size() * sizeof(uint32_t));
      out.close();

      SmallString<256> jsonPath(outDir);
      llvm::sys::path::append(jsonPath, stem + ".json");
      if (!scWriteText(jsonPath,
                       scPipelineJson(stem + ".spv", p.kernel, p.uuid,
                                      p.ssboCount, p.imageCount, p.pcSize)))
        return 1;
    }
  }

  // --- host C++ ------------------------------------------------------------
  //
  // The host descriptor table carries the same numbers the JSONs do, and the
  // launcher loads each pipeline by UUID. No SPIR-V is embedded: on SC the
  // binary is in pipeline_cache.bin, not in the executable.
  std::vector<host::HostScPipeline> scTable;
  for (const ScPipelineInfo &p : pipelines) {
    host::HostScPipeline h;
    h.kernelName = p.kernel;
    h.entryPoint = p.kernel; // MLIR names each OpEntryPoint after the symbol
    std::memcpy(h.uuid, p.uuid, 16);
    h.blockX = (uint32_t)p.bx;
    h.blockY = (uint32_t)p.by;
    h.blockZ = (uint32_t)p.bz;
    h.storageBufferBindingCount = p.ssboCount;
    h.imageBindingCount = p.imageCount;
    h.pushConstantSize = p.pcSize;
    scTable.push_back(std::move(h));
  }

  SmallString<256> hostPath(outDir);
  llvm::sys::path::append(
      hostPath, llvm::sys::path::stem(llvm::sys::path::filename(inputFilename)) +
                    "_host.cpp");
  {
    std::string hostSrc;
    llvm::raw_string_ostream os(hostSrc);
    if (!host::translateASTToHost(tu, /*modules=*/{}, os, &scTable)) {
      llvm::errs() << "vc: no host main() found in " << inputFilename << "\n";
      return 1;
    }
    if (!scWriteText(hostPath, hostSrc)) return 1;
  }

  llvm::errs() << "vc: Vulkan SC: " << pipelines.size() << " offline pipeline"
               << (pipelines.size() == 1 ? "" : "s") << " into '" << outDir
               << "' (" << blockSizes.size() << " block "
               << (blockSizes.size() == 1 ? "size" : "sizes") << "); host "
               << hostPath << "\n";
  llvm::errs()
      << "vc: feed the whole directory to one nvidia-pcc run, then patch the "
         "cache header:\n"
      << "      tools/vksc/build_cache.sh " << outDir << "\n"
      << "vc: the reservation must cover every pipeline; the runtime defaults are "
         "generous,\n"
      << "    but to pin them:\n"
      << "      VcScReservation r{}; vcScReservationDefaults(&r);\n"
      << "      r.computePipelines = r.pipelineLayouts = r.descriptorSetLayouts = "
      << pipelines.size() << ";\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input .vc file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> emit("emit",
      llvm::cl::desc("output kind (ast|mlir|spirv|host|full|sc)"),
      llvm::cl::init("mlir"));
  llvm::cl::opt<std::string> outputFilename("o",
      llvm::cl::desc("output filename"), llvm::cl::init("-"));
  llvm::cl::opt<bool> warningsAsErrors("Werror",
      llvm::cl::desc("treat warnings as errors"));
  llvm::cl::opt<bool> syntaxOnly("fsyntax-only",
      llvm::cl::desc("lex, parse and type-check only; emit no output"));
  llvm::cl::opt<bool> allowF64MathF32(
      "fallow-f64-math-f32",
      llvm::cl::desc("compute double transcendentals (sin/pow/...) in float "
                     "instead of rejecting them"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "VC compiler\n");

  EmitKind kind = EmitKind::MLIR;
  if (emit == "ast") kind = EmitKind::AST;
  else if (emit == "spirv") kind = EmitKind::SPIRV;
  else if (emit == "host") kind = EmitKind::Host;
  else if (emit == "full") kind = EmitKind::Full;
  else if (emit == "sc") kind = EmitKind::SC;
  else if (emit != "mlir") {
    llvm::errs() << "unknown -emit=" << emit << "\n";
    return 1;
  }

  llvm::SourceMgr sm;
  if (!loadSource(inputFilename, sm)) return 1;

  int mainBuf = sm.getMainFileID();
  Lexer lex(sm, mainBuf);
  SourceLocation start;
  TranslationUnit tu{start};
  Parser parser(lex, sm, tu);
  if (!parser.parseTranslationUnit()) {
    llvm::errs() << "parse failed\n";
    return 1;
  }
  Sema sema(tu, sm);
  sema.setWarningsAsErrors(warningsAsErrors);
  if (!sema.analyze()) {
    llvm::errs() << "sema: aborting due to errors\n";
    return 1;
  }

  // SPIR-V forbids recursion; rewrite bounded linear self-recursion in
  // __device__ functions into iterative loops before codegen. A recursive
  // function whose shape the pass can't handle is a hard error here.
  if (!unrollDeviceRecursion(tu)) {
    llvm::errs() << "recursion: aborting due to errors\n";
    return 1;
  }

  if (syntaxOnly) return 0;

  if (kind == EmitKind::AST) {
    dumpAST(tu, llvm::outs());
    return 0;
  }

  // Vulkan SC: emits its own outputs (per-block-size SPIR-V + pcc JSON + host
  // C++) into a directory, so it owns the whole pipeline run and never reaches
  // the single-module path below. It is dispatched here, after the AST-level
  // passes above (notably unrollDeviceRecursion) have run, and before the
  // context setup, since it re-translates the AST once per block size.
  if (kind == EmitKind::SC) {
    MLIRContext scCtx;
    DialectRegistry scRegistry;
    scRegistry.insert<vc::VCDialect, mlir::spirv::SPIRVDialect,
                      mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::index::IndexDialect>();
    scCtx.appendDialectRegistry(scRegistry);
    std::string outDir = outputFilename.getValue();
    if (outDir.empty() || outDir == "-") outDir = ".";
    return emitSc(tu, scCtx, warningsAsErrors, allowF64MathF32, outDir,
                  inputFilename.getValue());
  }

  // MLIR / SPIRV
  MLIRContext ctx;
  DialectRegistry registry;
  registry.insert<vc::VCDialect, mlir::spirv::SPIRVDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::index::IndexDialect>();
  ctx.appendDialectRegistry(registry);

  // MLIR's default diagnostic path prints Errors only: with no handler
  // registered, DiagnosticEngineImpl::emit returns as soon as the severity is
  // not Error (mlir/lib/IR/Diagnostics.cpp), so a warning emitted by codegen
  // would vanish without a trace. Report the non-error severities here, in the
  // same `location: severity: message` shape MLIR uses for errors — a notice
  // the user cannot see is not a notice.
  ctx.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
      return mlir::failure(); // Leave errors to MLIR's own printer.
    llvm::raw_ostream &os = llvm::errs();
    if (!llvm::isa<mlir::UnknownLoc>(diag.getLocation()))
      os << diag.getLocation() << ": ";
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Warning:
      os << "warning: ";
      break;
    case mlir::DiagnosticSeverity::Remark:
      os << "remark: ";
      break;
    default:
      os << "note: ";
      break;
    }
    os << diag.str() << '\n';
    os.flush();
    return mlir::success();
  });

  auto module = codegen::translateASTToMLIR(tu, ctx, warningsAsErrors,
                                            allowF64MathF32);
  if (!module) {
    llvm::errs() << "codegen failed\n";
    return 1;
  }

  if (kind == EmitKind::MLIR) {
    module->print(llvm::outs());
    return 0;
  }

  // SPIRV: run lowering then translate to binary.
  if (failed(codegen::runLoweringPipeline(*module)))
    return 1;

  // Serialize the spirv.module to a SPIR-V binary.
  SmallVector<uint32_t, 0> binary;
  mlir::spirv::ModuleOp spirvModule;
  module->walk([&](mlir::spirv::ModuleOp m) {
    if (!spirvModule)
      spirvModule = m;
  });
  if (!spirvModule || failed(mlir::spirv::serialize(spirvModule, binary))) {
    llvm::errs() << "spirv translation failed\n";
    return 1;
  }

  // Decorate the gl_WorkGroupSize spec-composite BuiltIn WorkgroupSize so the
  // runtime's specialization constants (SpecId 0/1/2) drive the hardware
  // workgroup size. See patchWorkgroupSizeBuiltin for why this is a binary
  // patch rather than an IR-level decoration.
  patchWorkgroupSizeBuiltin(binary);

  // Rewrite each `extern __shared__ T s[]` global's placeholder array size to
  // the workgroup-x spec constant, sizing dynamic shared memory to the runtime
  // block size. See patchDynamicSharedArrays for why this is a binary patch.
  patchDynamicSharedArrays(binary);

  // Rewrite storage-image carrier stubs (imageRead/imageWrite) into real
  // OpImageRead/OpImageWrite. MLIR 18's spirv dialect has no ImageRead/Write
  // ops and its spirv.Load verifier rejects storage-image result types, so the
  // ops are emitted as named carrier stores and finalized here. See
  // patchImageOps.
  patchImageOps(binary);

  // -emit=spirv: write the single SPIR-V binary and stop. (A SPIR-V file
  // can't concatenate multiple entry points across kernels, so like the GLSL
  // driver we only write the first kernel's module — but MLIR emits one
  // binary with one OpEntryPoint per kernel, so the whole binary is written.)
  if (kind == EmitKind::SPIRV) {
    std::error_code ec;
    auto out = std::make_unique<llvm::ToolOutputFile>(
        outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "cannot open " << outputFilename << ": " << ec.message()
                   << "\n";
      return 1;
    }
    out->os().write(reinterpret_cast<const char *>(binary.data()),
                    binary.size() * sizeof(uint32_t));
    out->keep();
    return 0;
  }

  // -emit=host / -emit=full: build one HostSpirvModule per __global__ kernel.
  // MLIR emits a single SPIR-V binary carrying one OpEntryPoint per kernel
  // named after the kernel symbol, so every module shares the same `binary`
  // (its lifetime outlives linkHostExecutable) and sets `entryPoint` to the
  // kernel name — vcLoadKernel then loads that entry point out of the shared
  // binary. (The GLSL backend instead emits one .spv per kernel with entry
  // "main"; see ASTToHost/HostLink.)
  //
  // The kernel symbol is the MANGLED device name (`ns::kernel` -> `ns_kernel`)
  // because the MLIR backend's buildFunction registers the func.func under that
  // mangled name (mirroring the host launchHandleName), and the serialized
  // OpEntryPoint carries the same symbol. A bare `fn->name` would mismatch the
  // spirv entry point and the host's launch handle. Kernels nested in
  // namespaces are found by recursing into NamespaceDecl bodies.
  std::vector<host::HostSpirvModule> hostModules;
  std::function<void(const std::vector<NodePtr> &)> collectKernels =
      [&](const std::vector<NodePtr> &decls) {
        for (const auto &d : decls) {
          if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
            collectKernels(static_cast<const NamespaceDecl *>(d.get())->decls);
            continue;
          }
          if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
          auto *fn = static_cast<FunctionDecl *>(d.get());
          if (fn->deviceAttr != DeviceAttr::Global)
            continue;
          std::string sym = deviceMangledName(fn);
          hostModules.push_back({sym, binary.data(), binary.size(),
                                 /*entryPoint=*/sym});
        }
      };
  collectKernels(tu.decls);
  if (hostModules.empty()) {
    llvm::errs() << "no __global__ kernel found in " << inputFilename << "\n";
    return 1;
  }

  return host::linkHostExecutable(tu, hostModules, outputFilename,
                                  kind == EmitKind::Host);
}
