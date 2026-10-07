#!/usr/bin/env python3
"""Print the capabilities a SPIR-V binary declares, one line each.

Usage: spirv_capabilities.py <kernel.spv>

Output shape (consumed by FileCheck in test/MLIR):

  memory-model Logical Vulkan
  capability Shader
  capability Float64

`vc -emit=mlir` dumps the VC-dialect IR, which is built before packKernels
creates the gpu.module and its spirv.target_env, so the capability list that
VCToGPU derives from the kernels' types is not visible there. The serialized
module is: it carries one OpCapability per entry of that list, and it is the
same list vkCreateShaderModule validates against the device's enabled features
(see checkSpirvCapabilities in VCRuntime.cpp).

The OpMemoryModel line prints alongside because a capability and the memory
model must agree: SPV_KHR_cooperative_matrix requires a module that declares
CooperativeMatrixKHR to also declare VulkanMemoryModel *and* use that model, and
the two are set in different places in the lowering (see
useVulkanMemoryModelForCooperativeMatrix in LoweringPasses.cpp).
"""

import struct
import sys

MAGIC = 0x07230203
OP_CAPABILITY = 17
OP_MEMORY_MODEL = 14
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
    5345: "VulkanMemoryModel",
    6022: "CooperativeMatrixKHR",
}
# spirv.hpp: AddressingModel / MemoryModel literals. Unknown values print as the
# bare number.
ADDRESSING_MODELS = {0: "Logical", 1: "Physical32", 2: "Physical64", 3: "PhysicalStorageBuffer64"}
MEMORY_MODELS = {0: "Simple", 1: "GLSL450", 2: "OpenCL", 3: "Vulkan"}


def decode(path):
    """Return (capabilities, memory-model pair or None) from a SPIR-V binary."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 4 * FIRST_INSTRUCTION or struct.unpack("<I", data[:4])[0] != MAGIC:
        raise ValueError(f"{path} is not a SPIR-V binary (no 0x{MAGIC:08x} magic)")
    words = struct.unpack("<%dI" % (len(data) // 4), data)

    caps = []
    memory_model = None
    i = FIRST_INSTRUCTION
    while i < len(words):
        word_count, opcode = words[i] >> 16, words[i] & 0xFFFF
        if word_count == 0 or i + word_count > len(words):
            raise ValueError(f"malformed instruction at word {i} of {path}")
        # OpCapability is always two words: the header and one operand.
        if opcode == OP_CAPABILITY:
            caps.append(words[i + 1])
        elif opcode == OP_MEMORY_MODEL and word_count >= 3:
            memory_model = (words[i + 1], words[i + 2])
        i += word_count
    return caps, memory_model


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        caps, memory_model = decode(argv[1])
    except (OSError, ValueError) as err:
        print(f"spirv_capabilities: {err}", file=sys.stderr)
        return 1
    if not caps:
        print("spirv_capabilities: no OpCapability found", file=sys.stderr)
        return 1
    if memory_model:
        addr, model = memory_model
        print(f"memory-model {ADDRESSING_MODELS.get(addr, addr)} "
              f"{MEMORY_MODELS.get(model, model)}")
    for cap in caps:
        print(f"capability {NAMES.get(cap, 'cap' + str(cap))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
