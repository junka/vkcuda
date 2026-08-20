add_executable(vector_add vector_add.cpp)

target_include_directories(vector_add PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(vector_add PRIVATE cxx_std_20)
target_link_libraries(vector_add PRIVATE VCRuntime)

add_executable(block_reduce block_reduce.cpp)

target_include_directories(block_reduce PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(block_reduce PRIVATE cxx_std_20)
target_link_libraries(block_reduce PRIVATE VCRuntime)

add_executable(matmul matmul.cpp)

target_include_directories(matmul PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(matmul PRIVATE cxx_std_20)
target_link_libraries(matmul PRIVATE VCRuntime)
