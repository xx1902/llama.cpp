#include "kv-delta.cuh"

#include "common.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

struct kv_delta_device_input {
    const char * anchor;
    const char * child;
    const uint32_t * anchor_cells;
    const uint32_t * child_cells;
    int8_t * q8;
    float * scales;
    uint64_t anchor_token_stride;
    uint64_t anchor_col_stride;
    uint64_t child_token_stride;
    uint64_t child_col_stride;
    int32_t n_tokens;
    int32_t n_embd;
    int32_t type;
};

struct kv_delta_host_output {
    int8_t * q8;
    float * scales;
    size_t q8_offset;
    size_t scale_offset;
    size_t q8_count;
    size_t scale_count;
};

struct kv_delta_cuda_job {
    int device = -1;
    cudaStream_t stream = nullptr;
    kv_delta_device_input * device_inputs = nullptr;
    uint32_t * device_cells = nullptr;
    int8_t * device_q8 = nullptr;
    float * device_scales = nullptr;
    int8_t * host_q8 = nullptr;
    float * host_scales = nullptr;
    size_t q8_count = 0;
    size_t scale_count = 0;
    std::vector<kv_delta_host_output> outputs;
};

static __device__ __forceinline__ float kv_delta_read(
        const char * base, uint64_t offset, int32_t type) {
    if (type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float *>(base + offset);
    }
    if (type == GGML_TYPE_F16) {
        return __half2float(*reinterpret_cast<const half *>(base + offset));
    }
    if (type == GGML_TYPE_BF16) {
        const uint32_t bits =
                static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(base + offset)) << 16;
        return __uint_as_float(bits);
    }
    return 0.0f;
}

static __global__ void kv_delta_q8_fused(
        const kv_delta_device_input * inputs,
        int32_t n_inputs) {
    const int32_t input_index = blockIdx.y;
    const int32_t token_index = blockIdx.z;
    if (input_index >= n_inputs) {
        return;
    }

    const kv_delta_device_input input = inputs[input_index];
    if (token_index >= input.n_tokens) {
        return;
    }

    const uint32_t anchor_cell = input.anchor_cells[token_index];
    const uint32_t child_cell = input.child_cells[token_index];
    float local_max = 0.0f;

    for (int32_t col = threadIdx.x; col < input.n_embd; col += blockDim.x) {
        const float anchor = kv_delta_read(
                input.anchor,
                static_cast<uint64_t>(anchor_cell) * input.anchor_token_stride +
                        static_cast<uint64_t>(col) * input.anchor_col_stride,
                input.type);
        const float child = kv_delta_read(
                input.child,
                static_cast<uint64_t>(child_cell) * input.child_token_stride +
                        static_cast<uint64_t>(col) * input.child_col_stride,
                input.type);
        local_max = fmaxf(local_max, fabsf(child - anchor));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] = fmaxf(
                    reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    const float scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) {
        input.scales[token_index] = scale;
    }
    __syncthreads();

    const size_t q8_base = static_cast<size_t>(token_index) * input.n_embd;
    for (int32_t col = threadIdx.x; col < input.n_embd; col += blockDim.x) {
        const float anchor = kv_delta_read(
                input.anchor,
                static_cast<uint64_t>(anchor_cell) * input.anchor_token_stride +
                        static_cast<uint64_t>(col) * input.anchor_col_stride,
                input.type);
        const float child = kv_delta_read(
                input.child,
                static_cast<uint64_t>(child_cell) * input.child_token_stride +
                        static_cast<uint64_t>(col) * input.child_col_stride,
                input.type);
        const int value = static_cast<int>(nearbyintf((child - anchor) / scale));
        input.q8[q8_base + col] = static_cast<int8_t>(max(-127, min(127, value)));
    }
}

static void kv_delta_job_release(kv_delta_cuda_job * job) {
    if (job == nullptr) {
        return;
    }
    ggml_cuda_set_device(job->device);
    if (job->device_inputs != nullptr) cudaFree(job->device_inputs);
    if (job->device_cells != nullptr) cudaFree(job->device_cells);
    if (job->device_q8 != nullptr) cudaFree(job->device_q8);
    if (job->device_scales != nullptr) cudaFree(job->device_scales);
    if (job->host_q8 != nullptr) cudaFreeHost(job->host_q8);
    if (job->host_scales != nullptr) cudaFreeHost(job->host_scales);
    if (job->stream != nullptr) cudaStreamDestroy(job->stream);
    delete job;
}

static bool kv_delta_cuda_ok(cudaError_t error) {
    return error == cudaSuccess;
}

static cudaError_t kv_delta_create_background_stream(cudaStream_t * stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
#else
    int least_priority = 0;
    int greatest_priority = 0;
    const cudaError_t priority_status = cudaDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority);
    if (priority_status != cudaSuccess) {
        return priority_status;
    }
    return cudaStreamCreateWithPriority(
            stream, cudaStreamNonBlocking, least_priority);
#endif
}

} // namespace

void * ggml_backend_cuda_kv_delta_submit(
        const ggml_backend_cuda_kv_delta_input * inputs,
        size_t n_inputs) {
    if (inputs == nullptr || n_inputs == 0 || n_inputs > 65535) {
        return nullptr;
    }

    auto * job = new kv_delta_cuda_job;
    std::vector<kv_delta_device_input> host_inputs(n_inputs);
    size_t cell_count = 0;
    int32_t max_tokens = 0;

    for (size_t i = 0; i < n_inputs; ++i) {
        const auto & input = inputs[i];
        if (input.anchor == nullptr || input.child == nullptr ||
                input.anchor->buffer == nullptr || input.child->buffer == nullptr ||
                input.anchor->data == nullptr || input.child->data == nullptr ||
                input.anchor->type != input.child->type ||
                (input.anchor->type != GGML_TYPE_F16 &&
                 input.anchor->type != GGML_TYPE_F32 &&
                 input.anchor->type != GGML_TYPE_BF16) ||
                input.anchor_cells == nullptr || input.child_cells == nullptr ||
                input.q8 == nullptr || input.scales == nullptr ||
                input.n_tokens <= 0 || input.n_embd <= 0) {
            delete job;
            return nullptr;
        }

        cudaPointerAttributes attributes = {};
        if (!kv_delta_cuda_ok(cudaPointerGetAttributes(&attributes, input.anchor->data))) {
            delete job;
            return nullptr;
        }
        const int device = attributes.device;
        if (job->device < 0) {
            job->device = device;
        } else if (job->device != device) {
            delete job;
            return nullptr;
        }

        job->outputs.push_back({
                input.q8,
                input.scales,
                job->q8_count,
                job->scale_count,
                static_cast<size_t>(input.n_tokens) * input.n_embd,
                static_cast<size_t>(input.n_tokens)});
        job->q8_count += static_cast<size_t>(input.n_tokens) * input.n_embd;
        job->scale_count += static_cast<size_t>(input.n_tokens);
        cell_count += static_cast<size_t>(input.n_tokens) * 2;
        max_tokens = std::max(max_tokens, input.n_tokens);
    }

    ggml_cuda_set_device(job->device);
    if (!kv_delta_cuda_ok(kv_delta_create_background_stream(&job->stream)) ||
            !kv_delta_cuda_ok(cudaMalloc(
                    reinterpret_cast<void **>(&job->device_inputs),
                    n_inputs * sizeof(kv_delta_device_input))) ||
            !kv_delta_cuda_ok(cudaMalloc(
                    reinterpret_cast<void **>(&job->device_cells),
                    cell_count * sizeof(uint32_t))) ||
            !kv_delta_cuda_ok(cudaMalloc(
                    reinterpret_cast<void **>(&job->device_q8),
                    job->q8_count * sizeof(int8_t))) ||
            !kv_delta_cuda_ok(cudaMalloc(
                    reinterpret_cast<void **>(&job->device_scales),
                    job->scale_count * sizeof(float))) ||
            !kv_delta_cuda_ok(cudaMallocHost(
                    reinterpret_cast<void **>(&job->host_q8),
                    job->q8_count * sizeof(int8_t))) ||
            !kv_delta_cuda_ok(cudaMallocHost(
                    reinterpret_cast<void **>(&job->host_scales),
                    job->scale_count * sizeof(float)))) {
        kv_delta_job_release(job);
        return nullptr;
    }

    size_t cell_offset = 0;
    for (size_t i = 0; i < n_inputs; ++i) {
        const auto & input = inputs[i];
        auto & device_input = host_inputs[i];
        device_input.anchor = static_cast<const char *>(input.anchor->data);
        device_input.child = static_cast<const char *>(input.child->data);
        device_input.anchor_cells = job->device_cells + cell_offset;
        device_input.child_cells = job->device_cells + cell_offset + input.n_tokens;
        device_input.q8 = job->device_q8 + job->outputs[i].q8_offset;
        device_input.scales = job->device_scales + job->outputs[i].scale_offset;
        device_input.n_tokens = input.n_tokens;
        device_input.n_embd = input.n_embd;
        device_input.type = input.anchor->type;

        const size_t type_size = ggml_type_size(input.anchor->type);
        if (input.transposed) {
            device_input.anchor_token_stride = type_size;
            device_input.child_token_stride = type_size;
            device_input.anchor_col_stride = static_cast<uint64_t>(input.anchor_stride) * type_size;
            device_input.child_col_stride = static_cast<uint64_t>(input.child_stride) * type_size;
        } else {
            device_input.anchor_token_stride = input.anchor->nb[1];
            device_input.child_token_stride = input.child->nb[1];
            device_input.anchor_col_stride = input.anchor->nb[0];
            device_input.child_col_stride = input.child->nb[0];
        }

        if (!kv_delta_cuda_ok(cudaMemcpyAsync(
                    job->device_cells + cell_offset,
                    input.anchor_cells,
                    input.n_tokens * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    job->stream)) ||
                !kv_delta_cuda_ok(cudaMemcpyAsync(
                    job->device_cells + cell_offset + input.n_tokens,
                    input.child_cells,
                    input.n_tokens * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    job->stream))) {
            cudaStreamSynchronize(job->stream);
            kv_delta_job_release(job);
            return nullptr;
        }
        cell_offset += static_cast<size_t>(input.n_tokens) * 2;
    }

    if (!kv_delta_cuda_ok(cudaMemcpyAsync(
                job->device_inputs,
                host_inputs.data(),
                n_inputs * sizeof(kv_delta_device_input),
                cudaMemcpyHostToDevice,
                job->stream))) {
        cudaStreamSynchronize(job->stream);
        kv_delta_job_release(job);
        return nullptr;
    }

    const dim3 block(256, 1, 1);
    const dim3 grid(1, static_cast<unsigned int>(n_inputs), static_cast<unsigned int>(max_tokens));
    kv_delta_q8_fused<<<grid, block, 0, job->stream>>>(
            job->device_inputs, static_cast<int32_t>(n_inputs));
    if (!kv_delta_cuda_ok(cudaGetLastError()) ||
            !kv_delta_cuda_ok(cudaMemcpyAsync(
                job->host_q8,
                job->device_q8,
                job->q8_count * sizeof(int8_t),
                cudaMemcpyDeviceToHost,
                job->stream)) ||
            !kv_delta_cuda_ok(cudaMemcpyAsync(
                job->host_scales,
                job->device_scales,
                job->scale_count * sizeof(float),
                cudaMemcpyDeviceToHost,
                job->stream))) {
        cudaStreamSynchronize(job->stream);
        kv_delta_job_release(job);
        return nullptr;
    }

    return job;
}

bool ggml_backend_cuda_kv_delta_finish(void * opaque_job) {
    auto * job = static_cast<kv_delta_cuda_job *>(opaque_job);
    if (job == nullptr) {
        return false;
    }
    ggml_cuda_set_device(job->device);
    const bool ok = kv_delta_cuda_ok(cudaStreamSynchronize(job->stream));
    if (ok) {
        for (const auto & output : job->outputs) {
            std::memcpy(
                    output.q8,
                    job->host_q8 + output.q8_offset,
                    output.q8_count * sizeof(int8_t));
            std::memcpy(
                    output.scales,
                    job->host_scales + output.scale_offset,
                    output.scale_count * sizeof(float));
        }
    }
    kv_delta_job_release(job);
    return ok;
}

void ggml_backend_cuda_kv_delta_cancel(void * opaque_job) {
    auto * job = static_cast<kv_delta_cuda_job *>(opaque_job);
    if (job == nullptr) {
        return;
    }
    ggml_cuda_set_device(job->device);
    cudaStreamSynchronize(job->stream);
    kv_delta_job_release(job);
}
