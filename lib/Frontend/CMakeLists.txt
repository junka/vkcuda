# Frontend library: pure C++ (LLVM support libs only, no MLIR dependency).
# Also hosts the GLSL codegen backend (AST -> GLSL), which likewise needs no
# MLIR and lets the compiler produce SPIR-V via glslc without libmlir.
add_library(VCFrontend STATIC
  Lexer.cpp
  Parser.cpp
  ASTHelpers.cpp
  ASTDumper.cpp
  Sema.cpp
  ../Codegen/ASTToGLSL.cpp
)

target_include_directories(VCFrontend PUBLIC
  ${VC_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIRS}
)

target_compile_features(VCFrontend PUBLIC cxx_std_20)

# LLVM is built without RTTI; match it.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(VCFrontend PUBLIC -fno-rtti)
endif()

# Frontend only needs LLVM support headers (ADT, SourceMgr, raw_ostream),
# which are header-only or in LLVM's support library.
target_link_libraries(VCFrontend PUBLIC LLVMCore LLVMSupport)
