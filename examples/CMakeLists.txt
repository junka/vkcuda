# All demos are single-file `.vc` sources (kernel + host main + launch in one
# file), compiled by the `vcc` driver at build time. There are no hand-written
# host .cpp files anymore. The demos under test/ double as the test corpus; the
# hand-written ones in this directory are the ones meant to be read. Either way
# vcc lowers the device subset to SPIR-V, the host subset to C++ embedding it,
# and drives g++ to produce a self-contained exe.
#
# `vcc` bakes the include dir / runtime lib / vulkan flags at its own link, so
# each custom command only needs the input and output path.

set(VC_SINGLEFILE_DEMOS vadd reduce matmul struct features features2 async_overlap atomics vectors sync warp enum half constant vote bool sizeof const_local comma default_arg define async_copy namespace class multi_kernel)

foreach(demo ${VC_SINGLEFILE_DEMOS})
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/${demo}_sf
    COMMAND vcc ${CMAKE_SOURCE_DIR}/test/${demo}.vc
            -o ${CMAKE_BINARY_DIR}/${demo}_sf
    DEPENDS vcc ${CMAKE_SOURCE_DIR}/test/${demo}.vc
    COMMENT "vcc compiling test/${demo}.vc (single-file CUDA-style)"
    VERBATIM)
  add_custom_target(${demo}_sf ALL
    SOURCES ${CMAKE_SOURCE_DIR}/test/${demo}.vc
    DEPENDS ${CMAKE_BINARY_DIR}/${demo}_sf)
endforeach()

# The hand-written examples. These live here rather than under test/ because
# they are written to be read: singlefile is the canonical minimal program,
# flash_attn the worked FlashAttention-3-style kernel (also cross-checked
# against PyTorch by compare_flash_attn.py in this directory).
foreach(example singlefile flash_attn)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/${example}
    COMMAND vcc ${CMAKE_CURRENT_SOURCE_DIR}/${example}.vc
            -o ${CMAKE_BINARY_DIR}/${example}
    DEPENDS vcc ${CMAKE_CURRENT_SOURCE_DIR}/${example}.vc
    COMMENT "vcc compiling ${example}.vc (single-file CUDA-style)"
    VERBATIM)
  add_custom_target(${example} ALL
    SOURCES ${example}.vc
    DEPENDS ${CMAKE_BINARY_DIR}/${example})
endforeach()
