# Frontend-only smoke test (no MLIR dependency). Parses a .vc file and
# dumps its AST. Useful to validate the lexer/parser without the full
# codegen pipeline.
add_executable(vc-dump-ast vc-dump-ast.cpp)
target_include_directories(vc-dump-ast PRIVATE
  ${VC_INCLUDE_DIR}
)
# LLVM headers are upstream; SYSTEM suppresses their build warnings.
target_include_directories(vc-dump-ast SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
)
target_compile_features(vc-dump-ast PRIVATE cxx_std_20)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(vc-dump-ast PRIVATE -fno-rtti)
endif()
target_link_libraries(vc-dump-ast PRIVATE VCFrontend LLVM)
