# All demos are single-file `.vc` sources (kernel + host main + launch in one
# file), compiled by the `vcc` driver at build time. There are no hand-written
# host .cpp files anymore. Each demo lives under test/ (which doubles as the
# single-file demo directory); vcc lowers the device subset to SPIR-V, the host
# subset to C++ embedding it, and drives g++ to produce a self-contained exe.
#
# `vcc` bakes the include dir / runtime lib / vulkan flags at its own link, so
# each custom command only needs the input and output path.

set(VC_SINGLEFILE_DEMOS vadd reduce matmul struct features features2 async_overlap atomics vectors sync warp)

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

# The original canonical single-file example (kept as its own target for
# documentation/reference alongside the test/ demos).
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/singlefile
  COMMAND vcc ${CMAKE_CURRENT_SOURCE_DIR}/singlefile.vc
          -o ${CMAKE_BINARY_DIR}/singlefile
  DEPENDS vcc ${CMAKE_CURRENT_SOURCE_DIR}/singlefile.vc
  COMMENT "vcc compiling singlefile.vc (single-file CUDA-style)"
  VERBATIM)
add_custom_target(singlefile ALL
  SOURCES singlefile.vc
  DEPENDS ${CMAKE_BINARY_DIR}/singlefile)
