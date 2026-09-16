# Frontend library: pure C++ (LLVM support libs only, no MLIR dependency).
# Also hosts the GLSL codegen backend (AST -> GLSL), which likewise needs no
# MLIR and lets the compiler produce SPIR-V via glslc without libmlir.
add_library(VCFrontend STATIC
  Lexer.cpp
  Parser.cpp
  ASTHelpers.cpp
  ASTDumper.cpp
  BuiltinRegistry.cpp
  Mangle.cpp
  Sema.cpp
  ../Codegen/ASTToGLSL.cpp
  ../Codegen/ASTToHost.cpp
  ../Codegen/HostLink.cpp
)

target_include_directories(VCFrontend PUBLIC
  ${VC_INCLUDE_DIR}
)
# LLVM headers are upstream; SYSTEM suppresses their build warnings.
target_include_directories(VCFrontend SYSTEM PUBLIC
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(VCFrontend PUBLIC cxx_std_20)

# LLVM is built without RTTI; match it.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(VCFrontend PUBLIC -fno-rtti)
endif()

# Frontend only needs LLVM support headers (ADT, SourceMgr, raw_ostream).
# Link the single shared libLLVM (the `LLVM` imported target) rather than the
# static libLLVMSupport.a / libLLVMCore.a: when the MLIR backend is also built,
# MLIR's imported targets pull in libLLVM.so transitively, and mixing the
# static support lib with the shared one causes a duplicate registration of
# LLVM command-line options (e.g. `debug-counter`) at startup -> fatal error.
# The shared library exports every symbol the frontend needs.
target_link_libraries(VCFrontend PUBLIC LLVM)

# HostLink.cpp (the shared host-C++ + g++ linking stage) needs the baked-in
# VCRuntime/header/Vulkan locations to invoke g++ at runtime; the driver
# binaries inherit these via VCCodegen/VCFrontend too. VCRuntime is always
# built (not MLIR-gated), so $<TARGET_FILE:VCRuntime> resolves in every config.
file(TO_CMAKE_PATH "${VC_INCLUDE_DIR}" VC_INCLUDE_DIR_ABS)
set(VC_RUNTIME_LIB_PATH "$<TARGET_FILE:VCRuntime>")
set(VC_VULKAN_LIB_FLAG "-lvulkan")
target_compile_definitions(VCFrontend PRIVATE
  VC_INCLUDE_DIR="${VC_INCLUDE_DIR_ABS}"
  VC_RUNTIME_LIB="${VC_RUNTIME_LIB_PATH}"
  VC_VULKAN_LIBS="${VC_VULKAN_LIB_FLAG}"
)
