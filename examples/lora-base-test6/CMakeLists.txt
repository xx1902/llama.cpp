set(TARGET llama-lora-base-test6)

# Test6 is an experiment suite around the system-v2 runtime. Reuse the tested
# runtime source instead of maintaining a diverging copy of the cache engine.
add_executable(${TARGET} ../lora-base-test4/lora-base-test4.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_LORA_BASE_TEST6=1)
