#!/usr/bin/env python3
"""Print the capabilities a SPIR-V binary declares, one line each.

Usage: spirv_capabilities.py <kernel.spv>

Output shape (consumed by FileCheck in test/MLIR):

  capability Shader
  capability Float64

`vc -emit=mlir` dumps the VC-dialect IR, which is built before packKernels
creates the gpu.module and its spirv.target_env, so the capability list that
VCToGPU derives from the kernels' types is not visible there. The serialized
module is: it carries one OpCapability per entry of that list, and it is the
same list vkCreateShaderModule validates against the device's enabled features
(see checkSpirvCapabilities in VCRuntime.cpp).
"""

import struct
import sys

MAGIC = 0x07230203
OP_CAPABILITY = 17
# Header is magic, version, generator, id bound, reserved.
FIRST_INSTRUCTION = 5
# Capability ids from mlir/Dialect/SPIRV/IR/SPIRVBase.td, restricted to the ones
# a VC kernel can declare; anything else prints as cap<id>.
NAMES = {
    1: "Shader",
    9: "Float16",
    10: "Float64",
    11: "Int64",
    61: "GroupNonUniform",
    62: "GroupNonUniformVote",
    63: "GroupNonUniformArithmetic",
    64: "GroupNonUniformBallot",
    65: "GroupNonUniformShuffle",
    66: "GroupNonUniformShuffleRelative",
    4433: "StorageBuffer16BitAccess",
    4435: "StoragePushConstant16",
    6022: "CooperativeMatrixKHR",
}


def capabilities(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 4 * FIRST_INSTRUCTION or struct.unpack("<I", data[:4])[0] != MAGIC:
        raise ValueError(f"{path} is not a SPIR-V binary (no 0x{MAGIC:08x} magic)")
    words = struct.unpack("<%dI" % (len(data) // 4), data)

    caps = []
    i = FIRST_INSTRUCTION
    while i < len(words):
        word_count, opcode = words[i] >> 16, words[i] & 0xFFFF
        if word_count == 0 or i + word_count > len(words):
            raise ValueError(f"malformed instruction at word {i} of {path}")
        # OpCapability is always two words: the header and one operand.
        if opcode == OP_CAPABILITY:
            caps.append(words[i + 1])
        i += word_count
    return caps


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        caps = capabilities(argv[1])
    except (OSError, ValueError) as err:
        print(f"spirv_capabilities: {err}", file=sys.stderr)
        return 1
    if not caps:
        print("spirv_capabilities: no OpCapability found", file=sys.stderr)
        return 1
    for cap in caps:
        print(f"capability {NAMES.get(cap, 'cap' + str(cap))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
