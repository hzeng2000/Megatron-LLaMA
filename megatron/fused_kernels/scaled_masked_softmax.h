/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <c10/macros/Macros.h>

namespace {

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

/*
 * Extended softmax (from native aten pytorch) with following additional features
 * 1) input scaling
 * 2) Explicit masking
 */	
template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_masked_softmax_warp_forward(
    output_t *dst, 
    const input_t *src,
    const uint8_t *mask, 
    const acc_t scale, 
    int micro_batch_size, 
    int element_count,
    int pad_batches) 
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and 
    // warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, )
    // gridDim/blockIdx = (seq_len, attn_heads, batches) 
    int first_batch = (blockDim.y * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))+ threadIdx.y) * WARP_BATCH;
    int pad_first_batch = 0;
    if (pad_batches != 1) { // bert style
    	pad_first_batch = (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) * WARP_BATCH;
    } else { // gpt2 style
        pad_first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    }

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * element_count + local_idx;
    dst += first_batch * element_count + local_idx;
    mask += pad_first_batch * element_count + local_idx;

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;

	#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            int itr_idx = i*element_count+it*WARP_SIZE;

            if (element_index < batch_element_count) {
	        if (mask[itr_idx] != 1) {
		    elements[i][it] = (acc_t)src[itr_idx] * scale;
		} else {
                    elements[i][it] = -10000.0;
		} 
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            elements[i][it] = std::exp((elements[i][it] - max_value[i]));
            sum[i] += elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                dst[i*element_count+it*WARP_SIZE] = (output_t)(elements[i][it] / sum[i]);
            } else {
                break;
            } 
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_masked_softmax_warp_backward(
    output_t *gradInput, 
    input_t *grad, 
    const input_t *output,
    acc_t scale, 
    int micro_batch_size, 
    int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and 
    // warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, )
    // gridDim/blockIdx = (seq_len, attn_heads, batches) 
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    
    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    // the first element to process by the current thread
    int thread_offset = first_batch * element_count + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS] { 0.0f };
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
	    if (element_index < batch_element_count) {
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
	    } else {
                output_reg[i][it] = acc_t(0);
            }
        }

       #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
	    if (element_index < batch_element_count) {
                grad_reg[i][it] = (acc_t)grad[i*element_count+it*WARP_SIZE] * output_reg[i][it];
	    } else {
                grad_reg[i][it] = acc_t(0);
	    }
        }
    }
   
    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                gradInput[i*element_count+it*WARP_SIZE] = (output_t)(scale * (grad_reg[i][it] - output_reg[i][it] * sum[i]));
            } 
        }
    }
}

} // end of anonymous namespace

template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_forward(
    output_t *dst, 
    const input_t *src, 
    const uint8_t *mask,
    const input_t scale, 
    int query_seq_len, 
    int key_seq_len, 
    int batches,
    int attn_heads,
    int pad_batches)
{
    TORCH_INTERNAL_ASSERT(key_seq_len >= 0 && key_seq_len <= 2048 );
    if (key_seq_len == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(key_seq_len);
        const int next_power_of_two = 1 << log2_elements;
        int batch_count = batches * attn_heads * query_seq_len;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
	    int batches_per_block = warps_per_block * batches_per_warp;
	    TORCH_INTERNAL_ASSERT(query_seq_len%batches_per_block == 0);
        dim3 blocks(query_seq_len/batches_per_block, attn_heads, batches);
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 0>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 1: // 2
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 1>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 2: // 4
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 2>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 3: // 8
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 3>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 4: // 16
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 4>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 5: // 32
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 5>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 6: // 64
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 6>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 7: // 128
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 7>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 8: // 256
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 8>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 9: // 512
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 9>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 10: // 1024
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 10>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            case 11: // 2048
                scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 11>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
                break;
            default:
                break;
        }
    }
}

template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_backward(
    output_t *grad_input, 
    input_t *grad, 
    const input_t *output, 
    const acc_t scale, 
    int query_seq_len, 
    int key_seq_len, 
    int batches,
    int attn_heads)
{
    TORCH_INTERNAL_ASSERT( key_seq_len >= 0 && key_seq_len <= 2048 );
    if (key_seq_len == 0) {
       return;
    } else {
        int log2_elements = log2_ceil(key_seq_len);
        const int next_power_of_two = 1 << log2_elements;
        int batch_count = batches *  attn_heads * query_seq_len;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
	int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = batch_count/batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 0>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 1: // 2
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 1>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 2: // 4
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 2>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 3: // 8
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 3>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 4: // 16
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 4>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 5: // 32
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 5>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 6: // 64
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 6>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 7: // 128
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 7>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 8: // 256
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 8>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 9: // 512
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 9>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 10: // 1024
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 10>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            case 11: // 2048
                scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 11>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, batch_count, key_seq_len);
                break;
            default:
                break;
        }
    }
}