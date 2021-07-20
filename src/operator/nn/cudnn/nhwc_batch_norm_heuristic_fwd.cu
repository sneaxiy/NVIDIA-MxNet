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
* \file nhwc_batch_norm_neuristic_fwd.h
* \brief CUDA NHWC Batch Normalization code with fused addition
* \author Dick Carter, Kartikeya Goyal
*/
#include "nhwc_batch_norm.h"
void NhwcBatchNorm::useHeuristicHelper(bool is_bwd, bool is_add, size_t m, int c,
                                       const int& device_id, const float bw_decay,
                                       const float THRESHOLD_KB,
                                       const std::vector<int>& c_elements_per_cta_v,
                                       const std::vector<int>& pixels_in_reg_v,
                                       std::map<int, float>* time_map,
                                       std::map<int, int>* reg_map,
                                       int* BN_MAX_OCCUPANCY,
                                       int* C_ELEMENTS_PER_CTA,
                                       int* THREADS_PER_PIXEL,
                                       int* PIXELS_PER_THREAD_IN_REGISTERS) {
    int verbose = static_cast<int>(dmlc::GetEnv(
            "MXNET_CUDNN_NHWC_BN_VERBOSE", 0));
    int phase_mult = 1;  // represents num NHWC tensors, x for fprop and x, dy for dgrad
    if (is_bwd) phase_mult = 2;
    const int ELT_SIZE = 2;  // 2B
    if (is_bwd) {
        *BN_MAX_OCCUPANCY = 1;
    } else {
        *BN_MAX_OCCUPANCY = 1;
    }
    const int elements_per_ldg = 8;
    const int THREADS_PER_CTA = 512;
    int add = 0;  // does an extra add
    if (is_add) add = 1;
    m = m * phase_mult;
    const float max_footprint_kb = ELT_SIZE * m * c;

    // This is the heuristic
    // It takes into account partial persistence,
    // exponetially decaying B/W (with stride) and quantization
    for (auto i : c_elements_per_cta_v) {
        if (i > c) {
            continue;
        }
        int j = (*reg_map)[i] * phase_mult;
        int threads_per_pixel = i / elements_per_ldg;
        // can't use get_max_smem_fwd()
        int static_smem_size = threads_per_pixel * (THREADS_PER_CTA / 32) *
                                elements_per_ldg * sizeof(float);
        int max_pixels_in_smem_per_cta = (MaxSharedMemoryPerBlockOptin(device_id)
                                            - static_smem_size) /
                                            (THREADS_PER_CTA * elements_per_ldg * ELT_SIZE);
        max_pixels_in_smem_per_cta -= 1;
        int pixels_per_thread = j + max_pixels_in_smem_per_cta;
        int pixels_per_cta = pixels_per_thread * (THREADS_PER_CTA / threads_per_pixel);
        int min_pixels_per_cta = THREADS_PER_CTA / threads_per_pixel;
        int grid_x = div_up(m, pixels_per_cta);
        int max_grid_x = div_up(m, min_pixels_per_cta);
        int sm_limit = MultiprocessorCount(device_id) - launch_margin(
                SMArch(device_id), MultiprocessorCount(device_id), is_bwd);
        int grid_y = 1;
        // Here we try to maximize X
        if (grid_x < sm_limit) {
            // Now we start using gridy : for instance for n=17 h,w=7x7 c=256
            // max_grid_x = 7 for i=32, which means we will launch 256 / 32 = 8 grid y's
            // for total grid size of 56 --
            // this is better than using C=256 as the syncs with overlap
            grid_y = max(1, min(static_cast<int>(sm_limit / grid_x), div_up(c, i)));
        }
        float footprint = ELT_SIZE * m * (i * grid_y) / 1024;  // KB
        float pp = min(grid_x, sm_limit) * i * pixels_per_cta * ELT_SIZE / 1024
                    + L2CacheSize(device_id) / 1024;  // KB
        float dram_traffic = (footprint * (2 + add) + max(0.0f, footprint - pp))
                                * div_up(c , i * grid_y);  // KB
        float exp = log2(static_cast<float>(static_cast<float>(min(c, 256)) / (i * grid_y)));
        float bw = pow(bw_decay, exp);
        int grid = min(static_cast<int>(sm_limit / grid_y), max_grid_x) * grid_y;
        int regs = div_up((div_up(m, grid) * grid_y), (THREADS_PER_CTA / threads_per_pixel));
        bw = bw * static_cast<float>(static_cast<float>(grid) / sm_limit);
        if (verbose > 1) {
            LOG(INFO) << "c_elemnts_per_cta = " << i << "\n"
                         "\tthreads_per_pixel = " << threads_per_pixel << "\n"
                         "\tmax_pixels_in_smem_per_cta = " << max_pixels_in_smem_per_cta << "\n"
                         "\tpixels_per_thread = " << pixels_per_thread << "\n"
                         "\tpixels_per_cta = " << pixels_per_cta << "\n"
                         "\tgrid_x = " << grid_x << "\n"
                         "\tmax_grid_x = " << max_grid_x << "\n"
                         "\tsm_limit = " << sm_limit << "\n"
                         "\tgrid_y = " << grid_y << "\n"
                         "\tfootprint = " << footprint << "\n"
                         "\tL2CacheSize = KB" << L2CacheSize(device_id) / 1024 << "\n"
                         "\tpp = " << pp << "\n"
                         "\tdram_traffic = " << dram_traffic << "\n"
                         "\texp =" << exp << "\n"
                         "\tgrid =" << grid << "\n"
                         "\tregs =" << regs << "\n"
                         "\tbw =" << bw << "\n";
        }
        auto it = time_map->find(i);
        if (it != time_map->end()) it->second = dram_traffic / bw;

        // Assign regs_map
        j = j / phase_mult;
        if (regs >= pixels_per_thread) {
            regs = j;  // will use outer-loops
        } else {
            if (regs > max_pixels_in_smem_per_cta) {
                regs -= max_pixels_in_smem_per_cta;
                regs = max(1, regs / phase_mult);
            } else {
                if (regs >= j) {
                    // say regs = 9 and smem_pixels = 19 and j = 8
                    // Then we would want smem_pixels = 1 and regs = 8
                    regs = j;
                } else {
                    // say regs = 7 and smem_pixels = 19 and j = 8
                    // Then we would want smem_pixels = 0 and regs = 7
                    regs = regs;
                    regs = max(1, regs / phase_mult);
                }
            }
        }
        CHECK(regs > 0) << "regs = " << regs << "reg < 1\n";
        auto itr = reg_map->find(i);
        if (itr != reg_map->end()) itr->second = regs;
    }

    // Debug Prints
    if (verbose > 1) {
        std::string tmp_s ="input = " + std::to_string(m) + " c = " + std::to_string(c)+ " ";
        for (const auto& i : *time_map) {
            tmp_s += std::to_string(i.first) + " = " + std::to_string(i.second) + "  ";
        }
        tmp_s += "\n";
        LOG(INFO) << tmp_s;
    }

    // Emperically is data is "small"
    if (max_footprint_kb < THRESHOLD_KB) {
        // Always with a few exceptions
        // (where we want 256_8 for B/W) want to use 32_8 to maximize grid.y
        // to have multiple syncs in parallel
        *THREADS_PER_PIXEL = 4;
        *C_ELEMENTS_PER_CTA = 32;
        *BN_MAX_OCCUPANCY = 1;
    } else {
        *BN_MAX_OCCUPANCY = 1;
        float first_min_value = FLT_MAX;
        for (const auto& k : *time_map) {
            if (k.second <  first_min_value) {
                first_min_value = k.second;
                *C_ELEMENTS_PER_CTA = k.first;
            }
        }
        *THREADS_PER_PIXEL = *C_ELEMENTS_PER_CTA / elements_per_ldg;
    }

    auto it = reg_map->find(*C_ELEMENTS_PER_CTA);
    if (it != reg_map->end()) {
        int closet_value_ge = it->second;
        for (auto j : pixels_in_reg_v) {
            if (j >= closet_value_ge) {
                closet_value_ge = j;
                break;
            }
        }
        *PIXELS_PER_THREAD_IN_REGISTERS = closet_value_ge;
    } else {
        LOG(FATAL) << "C_ELEMENTS_PER_CTA = " << *C_ELEMENTS_PER_CTA
            << "not found in reg_map\n";
    }
}

void NhwcBatchNorm::useHeuristicFwd(size_t m, int c, int device_id, bool gbn) {
    int use_hardcode_heuristic = static_cast<int>(dmlc::GetEnv(
            "MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD", 1));
    int verbose = static_cast<int>(dmlc::GetEnv(
            "MXNET_CUDNN_NHWC_BN_VERBOSE", 0));

    if (gbn || ( ((c % 8) != 0) && (c % 4 == 0)) || device_id < 0) {
        THREADS_PER_PIXEL_FWD = 16;
        C_ELEMENTS_PER_CTA_FWD = 64;
        BN_MAX_OCCUPANCY_FWD = 2;
        PIXELS_PER_THREAD_IN_REGISTERS_FWD = 5;

    } else if (use_hardcode_heuristic > 0 && c == 32) {
        THREADS_PER_PIXEL_FWD = 4;
        C_ELEMENTS_PER_CTA_FWD = 32;
        BN_MAX_OCCUPANCY_FWD = 1;
        PIXELS_PER_THREAD_IN_REGISTERS_FWD = 12;
    } else if (use_hardcode_heuristic > 0 && c % 8 == 0) {
        std::vector<int> c_elements_per_cta_v {32, 64, 128, 256};
        std::vector<int> pixels_in_reg_v {1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15};  // sorted
        // Output of heuristic
        std::map<int, float> time_map = {{32, FLT_MAX}, {64, FLT_MAX}, {128, FLT_MAX},
                                         {256, FLT_MAX}};
        // max pixels in regs before compiler spills
        std::map<int, int> reg_map = {{32, 12}, {64, 14}, {128, 14}, {256, 8}};

        useHeuristicHelper(false, false, m, c, device_id, 0.85, 2048,
                c_elements_per_cta_v, pixels_in_reg_v,
                &time_map, &reg_map, &BN_MAX_OCCUPANCY_FWD, &C_ELEMENTS_PER_CTA_FWD,
                &THREADS_PER_PIXEL_FWD, &PIXELS_PER_THREAD_IN_REGISTERS_FWD);
    } else {
        THREADS_PER_PIXEL_FWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_THREADS_PER_PIXEL_FWD", 16));
        C_ELEMENTS_PER_CTA_FWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_C_ELEMENTS_PER_CTA_FWD", 64));
        BN_MAX_OCCUPANCY_FWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_MAX_OCCUPANCY_FWD", 2));
        PIXELS_PER_THREAD_IN_REGISTERS_FWD = static_cast<int>(dmlc::GetEnv(
                "MXNET_CUDNN_NHWC_BN_PIXELS_PER_THREAD_IN_REGISTERS_FWD", 5));
        if (C_ELEMENTS_PER_CTA_FWD > c) {
            C_ELEMENTS_PER_CTA_FWD = 64;
            THREADS_PER_PIXEL_FWD = 16;
            BN_MAX_OCCUPANCY_FWD = 2;
            PIXELS_PER_THREAD_IN_REGISTERS_FWD = 5;
        }
    }  // end if
    if (verbose > 0) {
        LOG(INFO) << "-useHeuristicFwd- " << use_hardcode_heuristic << " " << m
        << " " << c << " " << device_id << " " << gbn << "\n"
        << "THREADS_PER_PIXEL_FWD = "  <<  THREADS_PER_PIXEL_FWD
        << " C_ELEMENTS_PER_CTA_FWD = " <<  C_ELEMENTS_PER_CTA_FWD
        << " BN_MAX_OCCUPANCY_FWD = " <<  BN_MAX_OCCUPANCY_FWD
        << " PIXELS_PER_THREAD_IN_REGISTERS_FWD = " <<  PIXELS_PER_THREAD_IN_REGISTERS_FWD
        << "\n";
    }
}
