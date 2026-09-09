# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(batched)
    add_subdirectory(debug)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(idle)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(lora-base)
    add_subdirectory(lora-base-test)
    add_subdirectory(lora-base-test2)
    add_subdirectory(lora-base-test3)
    add_subdirectory(lora-base-test4)
    add_subdirectory(lora-base-test5)
    add_subdirectory(lora-base-test6)
    add_subdirectory(lora-base-test7)
    add_subdirectory(lora-base-test8)
    add_subdirectory(lora-base-test10)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(simple-lora)
    add_subdirectory(simple-lora-kv)
    add_subdirectory(simple-lora-memory)
    add_subdirectory(simple-lora-sys)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)
    add_subdirectory(diffusion)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
