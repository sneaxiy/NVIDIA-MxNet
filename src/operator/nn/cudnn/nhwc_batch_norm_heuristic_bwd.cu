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
* \file nhwc_batch_norm_neuristic_bwd.cu
* \brief CUDA NHWC Batch Normalization code with fused addition
* \author Dick Carter, Kartikeya Goyal
*/
#include "nhwc_batch_norm.h"
void NhwcBatchNorm::useHeuristicBwd(size_t m, int c, int device_id, bool gbn) {
    int use_hardcode_heuristic = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD", 1));
    int verbose = static_cast<int>(dmlc::GetEnv(
            "MXNET_CUDNN_NHWC_BN_VERBOSE", 0));
    if (gbn || ( ((c % 8) != 0) && (c % 4 == 0)) || device_id < 0) {
        THREADS_PER_PIXEL_BWD = 16;
        C_ELEMENTS_PER_CTA_BWD = 64;
        BN_MAX_OCCUPANCY_BWD = 2;
        PIXELS_PER_THREAD_IN_REGISTERS_BWD = 3;
    } else if (use_hardcode_heuristic > 0 && c == 32) {
        THREADS_PER_PIXEL_BWD = 4;
        C_ELEMENTS_PER_CTA_BWD = 32;
        BN_MAX_OCCUPANCY_BWD = 1;
        PIXELS_PER_THREAD_IN_REGISTERS_BWD = 6;
    } else if (use_hardcode_heuristic > 0 && c % 8 == 0) {
        std::vector<int> c_elements_per_cta_v {32, 64, 128, 256};
        std::vector<int> pixels_in_reg_v {1, 2, 3, 4, 5, 6, 7, 8, 12};  // sorted
        // max pixels in regs before compiler spills
        //  Output of heuristic
        std::map<int, float> time_map = {{32, FLT_MAX}, {64, FLT_MAX},
            {128, FLT_MAX}, {256, FLT_MAX}};
        // TODO(kgoyal): fix to correct values after using RTC
        std::map<int, int> reg_map = {{32, 6}, {64, 6}, {128, 6}, {256, 4}};
        useHeuristicHelper(true, false, m, c, device_id, 0.85, 2048,
                c_elements_per_cta_v, pixels_in_reg_v,
                &time_map, &reg_map, &BN_MAX_OCCUPANCY_BWD, &C_ELEMENTS_PER_CTA_BWD,
                &THREADS_PER_PIXEL_BWD, &PIXELS_PER_THREAD_IN_REGISTERS_BWD);
    } else {
        THREADS_PER_PIXEL_BWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_THREADS_PER_PIXEL_BWD", 16));
        C_ELEMENTS_PER_CTA_BWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_C_ELEMENTS_PER_CTA_BWD", 64));
        BN_MAX_OCCUPANCY_BWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_MAX_OCCUPANCY_BWD", 2));
        PIXELS_PER_THREAD_IN_REGISTERS_BWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_PIXELS_PER_THREAD_IN_REGISTERS_BWD", 3));
        if (C_ELEMENTS_PER_CTA_BWD > c) {
            C_ELEMENTS_PER_CTA_BWD = 64;
            THREADS_PER_PIXEL_BWD = 16;
            BN_MAX_OCCUPANCY_BWD = 2;
            PIXELS_PER_THREAD_IN_REGISTERS_BWD = 3;
        }
    }  // end if
    if (verbose > 0) {
        LOG(INFO) << "-useHeuristicBwd- " << use_hardcode_heuristic << " " << m
        << " " << c << " " << device_id << " " << gbn << "\n"
        << "THREADS_PER_PIXEL_BWD = " << THREADS_PER_PIXEL_BWD
        << " C_ELEMENTS_PER_CTA_BWD = " << C_ELEMENTS_PER_CTA_BWD
        << " BN_MAX_OCCUPANCY_BWD = " << BN_MAX_OCCUPANCY_BWD
        << " PIXELS_PER_THREAD_IN_REGISTERS_BWD = " << PIXELS_PER_THREAD_IN_REGISTERS_BWD
        << "\n";
    }
}
