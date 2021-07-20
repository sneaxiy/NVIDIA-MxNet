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
 * \file nhwc_batch_norm_common.h
 * \brief CUDA NHWC Batch Normalization code
 * \author Shankara Rao Thejaswi Nanditale, Dick Carter, Maxim Milakov, Evgeni Krimer, Kartikeya Goyal
*/
# pragma once
#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_COMMON_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_COMMON_H_

#include <vector>

namespace {

// The size of a warp.
constexpr int THREADS_PER_WARP = 32;
// The number of reducing ops, each uses its own space : mean, var, dscale, dbias
constexpr int REDUCE_OPS = 4;
// Maximum block.y supported - limited due to buffer allocation
constexpr int MAX_BLOCK_Y = 256;
constexpr int MAX_OFFSET = REDUCE_OPS*MAX_BLOCK_Y;
constexpr int BYTES_PER_ELEM = 4;
// Buffer size per sync step - for each reduction op in block.y, each thread
// stores elements_per_ldg times an fp32 value along with an int32 flag
constexpr int SINGLE_SYNC_BUFFER_BYTES =
              MAX_OFFSET*64*2*BYTES_PER_ELEM;

// Maximum number of synchronization steps
// in group batchnorm. Corresponds to 16 GPUs
constexpr int MAX_SYNC_ITERS = 4;

// Size of the synchronization buffer
// in group batchnorm
constexpr int SYNC_BUFFER_SIZE = MAX_SYNC_ITERS * SINGLE_SYNC_BUFFER_BYTES;

#ifndef ROUND_UP_TO_MULTIPLE
#define ROUND_UP_TO_MULTIPLE(x, m) ((((x) + (m) - 1) / (m)) * (m))
#endif
// Set the function's dynamic smem limit to the max to accomodate the needs of the desired kernel
// launch.  This is a macro because of the difficulty of getting g++ to be happy with casting
// kernel function addresses to the void* argument of cudaFuncSetAttribute.  This macro is designed
// to be used once inline for each different kernel function (or instantiated template), and
// not with a variable func argument.  This macro may allow multiple threads controlling the
// same device to perform the cudaFuncSetAttribute action, but such is idempotent and thread-safe.
#define ENSURE_MAX_DYN_SMEM_LIMIT(func, desired_dyn_smem, static_smem, device_id, func_str)        \
  do {                                                                                             \
    static std::unordered_map<decltype(func), int> set_dyn_limit;                                  \
    if (set_dyn_limit.find(func) == set_dyn_limit.end()) {                                         \
      int32_t max_smem_limit = MaxSharedMemoryPerBlockOptin(device_id);                            \
      int32_t max_dyn_limit = max_smem_limit - ROUND_UP_TO_MULTIPLE(static_smem, 256);             \
      auto err =                                                                                   \
      cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_dyn_limit);    \
      /* Work-around if the limit setting attempted is too large for some reason */                \
      while (err != CUDA_SUCCESS) {                                                                \
        max_dyn_limit -= 1024;                                                                     \
        if (max_dyn_limit < 0)                                                                     \
          break;                                                                                   \
        err = cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,            \
                max_dyn_limit);                                                                    \
      }                                                                                            \
      if (err != CUDA_SUCCESS) {                                                                   \
          const char* error_string;                                                                \
          cuGetErrorString(err, &error_string);                                                    \
          LOG(FATAL) << "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES failed: "                \
          << error_string;                                                                         \
      }                                                                                            \
      set_dyn_limit.insert({func, max_dyn_limit});                                                 \
    }                                                                                              \
    auto it = set_dyn_limit.find(func);                                                            \
    CHECK_LE(desired_dyn_smem, it->second)                                                         \
        << func_str << " needs " << desired_dyn_smem  << " dynamic smem bytes/block, but only "    \
        << it->second << " are available";                                           \
  } while (0)

}  // namespace

#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_COMMON_H_
