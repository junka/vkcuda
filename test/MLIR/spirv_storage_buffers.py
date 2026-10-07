#!/usr/bin/env python3
"""Print how a SPIR-V binary lays out its descriptor storage buffers.

Usage: spirv_storage_buffers.py <kernel.spv>

Output shape (consumed by FileCheck in test/MLIR):

  storage-buffer f16Scale_arg_0 binding=0 stride=2 element=half
  pointer-bitcasts 0

`vc -emit=mlir` dumps the VC-dialect IR, which predates the memref -> SPIR-V
conversion and the f16-SSBO narrowing in LoweringPasses.cpp, so neither the
interface variable's ArrayStride nor the absence of a pointer bitcast is
observable there. Both are host/device contracts rather than validation
details:

  * ArrayStride is how the device indexes the array. The MemRef -> SPIR-V
    converter widens sub-4-byte StorageBuffer elements to 4 bytes, while the
    host mallocs a __half array packed at 2 bytes per element, so element i
    lands at 4*i instead of 2*i and the kernel reads and writes the wrong
    memory — valid SPIR-V computing on a layout the host never allocated.
  * OpBitcast may not take or yield a logical pointer (spirv-val rejects it and
    MoltenVK fails vkCreateComputePipelines). It is what spirv-lower-abi-attrs
    emits to bridge a kernel argument whose type disagrees with the interface
    variable built from the function signature — the same stride mismatch seen
    from the other side.
"""

import struct
import sys

MAGIC = 0x07230203
# Header is magic, version, generator, id bound, reserved.
FIRST_INSTRUCTION = 5

OP_NAME = 5
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_ARRAY = 28
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_VARIABLE = 59
OP_DECORATE = 71
OP_BITCAST = 124

DEC_ARRAY_STRIDE = 6
DEC_BINDING = 33
STORAGE_BUFFER = 12
UNIFORM = 2


def literal_string(words):
    """Decode a SPIR-V literal string (UTF-8, NUL-padded to a whole word)."""
    raw = struct.pack("<%dI" % len(words), *words)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def decode(words, path):
    names, strides, bindings = {}, {}, {}
    scalars = {}    # id -> "half" / "float" / "int32" / ...
    arrays = {}     # id -> element id (OpTypeArray, OpTypeRuntimeArray)
    structs = {}    # id -> [member ids]
    pointers = {}   # id -> (storage class, pointee id)
    variables = []  # (result type id, result id, storage class)
    bitcasts = []   # (result type id, operand type id)

    i = FIRST_INSTRUCTION
    while i < len(words):
        word_count, opcode = words[i] >> 16, words[i] & 0xFFFF
        if word_count == 0 or i + word_count > len(words):
            raise ValueError(f"malformed instruction at word {i} of {path}")
        body = words[i + 1 : i + word_count]
        if opcode == OP_NAME and len(body) >= 2:
            names[body[0]] = literal_string(body[1:])
        elif opcode == OP_DECORATE and len(body) >= 3:
            if body[1] == DEC_ARRAY_STRIDE:
                strides[body[0]] = body[2]
            elif body[1] == DEC_BINDING:
                bindings[body[0]] = body[2]
        elif opcode == OP_TYPE_INT and len(body) >= 3:
            scalars[body[0]] = ("u" if body[2] == 0 else "i") + str(body[1])
        elif opcode == OP_TYPE_FLOAT and len(body) >= 2:
            scalars[body[0]] = {16: "half", 32: "float", 64: "double"}.get(
                body[1], f"float{body[1]}"
            )
        elif opcode in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY) and len(body) >= 2:
            arrays[body[0]] = body[1]
        elif opcode == OP_TYPE_STRUCT and len(body) >= 1:
            structs[body[0]] = list(body[1:])
        elif opcode == OP_TYPE_POINTER and len(body) >= 3:
            pointers[body[0]] = (body[1], body[2])
        elif opcode == OP_VARIABLE and len(body) >= 3:
            variables.append((body[0], body[1], body[2]))
        elif opcode == OP_BITCAST and len(body) >= 3:
            bitcasts.append((body[0], body[2]))
        i += word_count

    def element(ty, seen):
        """Unwrap descriptor structs until the runtime array's element type."""
        if ty in seen:
            return None, None
        seen = seen | {ty}
        if ty in arrays:
            inner, _ = element(arrays[ty], seen)
            if inner is None:
                inner = scalars.get(arrays[ty], f"type{arrays[ty]}")
            return inner, strides.get(ty)
        if ty in structs:
            for member in structs[ty]:
                inner, stride = element(member, seen)
                if inner is not None:
                    return inner, stride
            return None, None
        return scalars.get(ty), None

    lines = []
    for result_type, var_id, storage_class in variables:
        kind = {STORAGE_BUFFER: "storage-buffer", UNIFORM: "uniform"}.get(
            storage_class
        )
        if kind is None or result_type not in pointers:
            continue
        elem, stride = element(pointers[result_type][1], set())
        lines.append(
            f"{kind} {names.get(var_id, '?')} binding={bindings.get(var_id, '?')} "
            f"stride={stride} element={elem}"
        )

    # Only logical pointers are illegal to bitcast; a pointer-typed operand means
    # the bridge over a layout disagreement, not a numeric reinterpretation.
    pointer_bitcasts = sum(
        1
        for result_type, operand_type in bitcasts
        if result_type in pointers or operand_type in pointers
    )
    lines.append(f"pointer-bitcasts {pointer_bitcasts}")
    return lines


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path = argv[1]
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if (
            len(data) < 4 * FIRST_INSTRUCTION
            or struct.unpack("<I", data[:4])[0] != MAGIC
        ):
            raise ValueError(f"{path} is not a SPIR-V binary (no 0x{MAGIC:08x} magic)")
        lines = decode(struct.unpack("<%dI" % (len(data) // 4), data), path)
    except (OSError, ValueError) as err:
        print(f"spirv_storage_buffers: {err}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
