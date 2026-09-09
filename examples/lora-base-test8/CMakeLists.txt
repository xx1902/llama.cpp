set(TARGET llama-lora-base-test8)

# test8 直接编译带完整注释的综合运行时副本，不改动 test4-test7 的实验目标。
add_executable(${TARGET} lora-base-test8.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_LORA_BASE_TEST8=1)
