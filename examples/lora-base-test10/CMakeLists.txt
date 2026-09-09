set(TARGET llama-lora-base-test10-cuda)

# Personal experiment target. It measures scheduling behavior only and does not
# modify llama.cpp inference code.
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(TEST10_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures for the personal test10 benchmark")

    add_executable(${TARGET} cuda_parallel_benchmark.cu)
    target_link_libraries(${TARGET} PRIVATE CUDA::cudart ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
    set_target_properties(${TARGET} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${TEST10_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION OFF)

    install(TARGETS ${TARGET} RUNTIME)
else()
    message(STATUS "Skipping ${TARGET}: GGML_CUDA is disabled")
endif()
