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

add_executable(async_overlap async_overlap.cpp)

target_include_directories(async_overlap PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(async_overlap PRIVATE cxx_std_20)
target_link_libraries(async_overlap PRIVATE VCRuntime)

add_executable(features_demo features_demo.cpp)

target_include_directories(features_demo PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(features_demo PRIVATE cxx_std_20)
target_link_libraries(features_demo PRIVATE VCRuntime)

add_executable(struct_demo struct_demo.cpp)

target_include_directories(struct_demo PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(struct_demo PRIVATE cxx_std_20)
target_link_libraries(struct_demo PRIVATE VCRuntime)

add_executable(features2_demo features2_demo.cpp)

target_include_directories(features2_demo PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(features2_demo PRIVATE cxx_std_20)
target_link_libraries(features2_demo PRIVATE VCRuntime)
