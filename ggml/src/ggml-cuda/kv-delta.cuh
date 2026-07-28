#pragma once

#include "ggml-cuda.h"

void * ggml_backend_cuda_kv_delta_submit(
        const ggml_backend_cuda_kv_delta_input * inputs,
        size_t n_inputs);
bool ggml_backend_cuda_kv_delta_finish(void * job);
void ggml_backend_cuda_kv_delta_cancel(void * job);
