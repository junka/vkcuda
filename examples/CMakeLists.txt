add_executable(vector_add vector_add.cpp)

target_include_directories(vector_add PRIVATE ${VC_INCLUDE_DIR})
target_compile_features(vector_add PRIVATE cxx_std_20)
target_link_libraries(vector_add PRIVATE VCRuntime)
