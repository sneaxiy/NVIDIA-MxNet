/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file nhwc_batch_norm_kernel.h
 * \brief CUDA NHWC Batch Normalization code
 * \author Shankara Rao Thejaswi Nanditale, Dick Carter, Maxim Milakov,
 *         Evgeni Krimer, Kartikeya Goyal
 *         */
#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_
#include <stdint.h>
#include <string>
#include <algorithm>
#include "nhwc_batch_norm_common.h"

// CTA margin used by cooperative launch. Can be overridden by env var NHWC_BATCHNORM_LAUNCH_MARGIN.
#define NHWC_BATCHNORM_LAUNCH_MARGIN_MIN     3
#define NHWC_BATCHNORM_LAUNCH_MARGIN_DEFAULT NHWC_BATCHNORM_LAUNCH_MARGIN_MIN

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int div_up(int m, int n) {
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormFwdInferenceParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dst, *gmem_src1;
    // the final mean and variance as calculated during the training process
    float *gmem_mean, *gmem_var;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // The dimensions.
    int nhw, c;
    // epsilon
    float var_eps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormFwdParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dst, *gmem_src1;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // running mean/var (refer BN API from cudnn doc)
    float *gmem_running_mean, *gmem_running_var;
    // saved mean/var (refer BN API from cudnn doc)
    float *gmem_saved_mean, *gmem_saved_var;
    // ReLU bitmask
    unsigned int *gmem_relu_bitmask;
    // The dimensions.
    int nhw, c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    float svar_inv_count;
    // factor to scale sum of squared errors to get running variance. Should be 1/nhw or 1/(nhw-1).
    float rvar_inv_count;
    // The buffer to do the reduction for mean, stddev and count.
    float *gmem_sums;
    // The buffer to count items in the different CTAs.
    int *gmem_counts;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // The epsilon to apply to the computation of the variance.
    float var_eps;
    // outer loop count
    int outer_loops;
    // exponential average factor
    float exp_avg_factor;
    // number of CTAs along .x dimension
    int c_blks;
    // pixels_per_thread_in_smem
    int pixels_per_thread_in_smem;

    void* my_data;
    void* pair_data[MAX_SYNC_ITERS];
    int sync_iters;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormBwdParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dy, *gmem_dst, *gmem_dst1;
    // dscale/dbias
    float *gmem_dscale, *gmem_dbias;
    // The scale and bias.
    float *gmem_scale, *gmem_bias;
    // The mean/inv-var saved from fwd pass
    float *gmem_saved_mean, *gmem_saved_var;
    // ReLU bitmask
    unsigned int *gmem_relu_bitmask;
    // The dimensions.
    int nhw, c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    float svar_inv_count;
    // The buffer to do the reduction for dscale and dbias
    float *gmem_sums;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // outer loop count
    int outer_loops;
    // number of CTAs along .x dimension
    int c_blks;
    // pixels_per_thread_in_smem
    int pixels_per_thread_in_smem;

    void* my_data;
    void* pair_data[MAX_SYNC_ITERS];
    int sync_iters;
    float wgrad_coeff;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class NhwcBatchNormCode {
 public:
    static const char nhwc_batch_norm_common[];
    static const char nhwc_batch_norm_fwd_inference[];
    static const char nhwc_batch_norm_bwd[];
    static const char nhwc_batch_norm_fwd[];
    static const char nhwc_batch_norm_bwd_relu[];
    static const char nhwc_batch_norm_bwd_add_relu[];
    static bool replace(std::string* str, const std::string& from, const std::string& to) {
        size_t start_pos = str->find(from);
        if (start_pos == std::string::npos)
            return false;
        str->replace(start_pos, from.length(), to);
        return true;
    }
    static std::string get_nhwc_batch_norm_kernel(const std::string kernel,
                                               const std::string old_name,
                                               const std::string new_name) {
        std::string ret = kernel;
        replace(&ret, old_name, new_name);
        return ret;
    }
    static std::string get_bhwc_batch_norm_params(int threads_per_pixel, int c_elements_per_cta,
            int pixels_per_thread_in_reg, int pixels_per_thread_in_smem,
            int elements_per_ldg, int use_online_approach,
            int outer_loops, int sync_iters,
            bool use_relu, bool use_add_relu,
            int occupancy) {
        int outer_loops_ = outer_loops ==1 ? 1 : 0;
        std::string sync_iter_s = sync_iters > 0 ? "true" :"false";
        std::string code =
                           "typedef unsigned short uint16_t;\n"
                           "typedef unsigned int uint32_t;\n"
                           "typedef unsigned int uint;\n"
                           "using Storage =uint16_t;\n"
                           "constexpr int THREADS_PER_CTA = 512;\n"
                           + std::string("constexpr int THREADS_PER_PIXEL = ")
                           + std::to_string(threads_per_pixel) + ";\n"
                           + std::string("constexpr int C_ELEMENTS_PER_CTA = ")
                           + std::to_string(c_elements_per_cta) + ";\n"
                           + std::string(R"code(
                              // The size of a warp.
                              constexpr int THREADS_PER_WARP = 32;
                              // The number of reducing ops, each uses its own space : mean, var, dscale, dbias
                              constexpr int REDUCE_OPS = 4;
                              // Maximum block.y supported - limited due to buffer allocation
                              constexpr int MAX_BLOCK_Y = 256;
                              constexpr int MAX_OFFSET = REDUCE_OPS * MAX_BLOCK_Y;
                              constexpr int BYTES_PER_ELEM = 4;
                              // Maximum number of synchronization steps
                              // in group batchnorm. Corresponds to 16 GPUs
                              constexpr int MAX_SYNC_ITERS = 4;
                              // Buffer size per sync step - for each reduction op in block.y, each thread
                              // stores elements_per_ldg times an fp32 value along with an int32 flag
                              constexpr int SINGLE_SYNC_BUFFER_BYTES =
                                        MAX_OFFSET*C_ELEMENTS_PER_CTA*2*BYTES_PER_ELEM;
                              // Size of the synchronization buffer
                              // in group batchnorm
                              constexpr int SYNC_BUFFER_SIZE = MAX_SYNC_ITERS * SINGLE_SYNC_BUFFER_BYTES;
                           )code") + "\n"
                           + std::string("constexpr int PIXELS_PER_THREAD_IN_REGISTERS = ")
                           + std::to_string(pixels_per_thread_in_reg) + ";\n"
                           + std::string("constexpr int PIXELS_PER_THREAD_IN_SMEM = ")
                           + std::to_string(pixels_per_thread_in_smem) + ";\n"
                           + std::string("constexpr int ELEMENTS_PER_LDG = ")
                           + std::to_string(elements_per_ldg) + ";\n"
                           + std::string("constexpr int USE_ONLINE_APPROACH = ")
                           + std::to_string(use_online_approach) + ";\n"
                           + std::string("constexpr int OUTER_LOOPS_ = ")
                           + std::to_string(outer_loops_) + ";\n"
                           + std::string("constexpr bool SYNC_ITERS = ")
                           + sync_iter_s + ";\n"
                           + std::string("constexpr bool USE_RELU = ")
                           + std::string(use_relu?"true":"false") + ";\n"
                           + std::string("constexpr bool USE_ADD_RELU = ")
                           + std::string(use_add_relu?"true":"false") + ";\n"
                           + std::string("constexpr int DESIRED_OCCUPANCY = ")
                           + std::to_string(occupancy) + ";\n";
        return code;
    }
};

#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_
