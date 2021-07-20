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
 * \file nhwc_batch_norm_add_relu.h
 * \brief CUDA NHWC Batch Normalization code with fused addition
 * \author Shankara Rao Thejaswi Nanditale, Dick Carter, Maxim Milakov, Kartikeya Goyal
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_ADD_RELU_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_ADD_RELU_H_

#include <cudnn.h>

#include <algorithm>
#include <vector>
#include <string>
#include <map>

#include "../../../common/cuda/rtc.h"

#include "nhwc_batch_norm_kernel.h"

#include "../../../common/cuda/utils.h"


#define VERBOSE_DEFAULT false

class NhwcBatchNormAddRelu {
 public:
  int pixels_per_thread_in_smem_fwd;
  int pixels_per_thread_in_smem_bwd;

  int pixels_per_thread_fwd;
  int pixels_per_thread_bwd;

  size_t static_smem_size_fwd;
  size_t static_smem_size_bwd;
  size_t dynamic_smem_size_fwd;
  size_t dynamic_smem_size_bwd;

  int pixels_per_cta_fwd;
  int pixels_per_cta_bwd;

  int THREADS_PER_PIXEL_FWD;
  int C_ELEMENTS_PER_CTA_FWD;
  int ELEMENTS_PER_LDG_FWD;
  int PIXELS_PER_THREAD_IN_REGISTERS_FWD;
  int BN_MAX_OCCUPANCY_FWD;
  int THREADS_PER_PIXEL_BWD;
  int C_ELEMENTS_PER_CTA_BWD;
  int ELEMENTS_PER_LDG_BWD;
  int PIXELS_PER_THREAD_IN_REGISTERS_BWD;
  int BN_MAX_OCCUPANCY_BWD;

  int PIXELS_PER_LDG_FWD;
  int PIXELS_PER_LDG_BWD;
  int MIN_PIXELS_PER_CTA_FWD;
  int MIN_PIXELS_PER_CTA_BWD;

  const int PIXELS_PER_CTA_FWD_INFERENCE;

  NhwcBatchNormAddRelu() = delete;  // will never be generated
  explicit NhwcBatchNormAddRelu(int gbn):
  PIXELS_PER_CTA_FWD_INFERENCE(THREADS_PER_CTA / THREADS_PER_PIXEL_FWD_INFERENCE *
        PIXELS_PER_THREAD_FWD_INFERENCE) {
    name_ = "nhwc_batchnormaddrelu";
    useHeuristic(0, 0, -1, gbn > 1);  // set heuristic values to defualt values
    // Initialize with 0 smem.
    // This allows max workspace size allocated in case we increase smem later
    init_fwd_params(0);
    init_bwd_params(0);
  }

  void die() {
    std::cerr << "batchnormaddrelu not initialized" << std::endl;
    exit(-1);
  }

  inline void fwd(cudaStream_t stream, int device_id);
  inline void dgrad(cudaStream_t stream, int device_id);
  template <typename gpu>
  inline void fwdInference(mshadow::Stream<gpu> *s, int device_id);
  inline void init_fwd_params(int new_pixels_per_thread_in_smem_fwd);
  inline void init_bwd_params(int new_pixels_per_thread_in_smem_bwd);
  inline dim3 calc_fwd_grid(int device_id, int *loop, bool use_group_bn);
  inline dim3 calc_bwd_grid(int device_id, int *loop, bool use_group_bn);
  void useHeuristicFwdBwdHelper(int m, int c, int device_id, bool is_bwd, bool is_add,
                                const float bw_decay, const float THRESHOLD_KB,
                                const std::vector<int>& c_elements_per_cta_v,
                                const std::vector<int>& pixels_in_reg_v,
                                std::map<int, float>* time_map,
                                std::map<int, int>* reg_map,
                                int* THREADS_PER_PIXEL, int* C_ELEMENTS_PER_CTA,
                                int* BN_MAX_OCCUPANCY, int* PIXELS_PER_THREAD_IN_REGISTERS,
                                bool force);
  void useHeuristicFwdBwd(size_t m, int c, int device_id, bool gbn);

  void useHeuristic(size_t m, int c, int device_id, bool gbn) {
    useHeuristicFwdBwd(m, c, device_id, gbn);
    ELEMENTS_PER_LDG_FWD = C_ELEMENTS_PER_CTA_FWD / THREADS_PER_PIXEL_FWD;
    ELEMENTS_PER_LDG_BWD = C_ELEMENTS_PER_CTA_BWD / THREADS_PER_PIXEL_BWD;

    PIXELS_PER_LDG_FWD = THREADS_PER_CTA / THREADS_PER_PIXEL_FWD;
    PIXELS_PER_LDG_BWD = THREADS_PER_CTA / THREADS_PER_PIXEL_BWD;
    MIN_PIXELS_PER_CTA_FWD = THREADS_PER_CTA / THREADS_PER_PIXEL_FWD *
                               PIXELS_PER_THREAD_IN_REGISTERS_FWD;
    MIN_PIXELS_PER_CTA_BWD = THREADS_PER_CTA / THREADS_PER_PIXEL_BWD *
                               PIXELS_PER_THREAD_IN_REGISTERS_BWD;
  }

  void setInputOutputParams(size_t input_size, int c, int device_id, int bn_group) {
    m_ = input_size / c;
    int m_bn_adjusted = m_ * bn_group;
    c_ = c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    svar_inv_count_ = 1.f / m_bn_adjusted;
    // factor to scale sum of squared errors to get running variance. Should be 1/nhw or 1/(nhw-1).
    bool unbiased_running_variance = dmlc::GetEnv("MXNET_CUDNN_NHWC_BN_UNBIASED_RUNNING_VAR", true);
    int divisor = unbiased_running_variance ? (m_bn_adjusted - 1) : m_bn_adjusted;
    // nhw == 1 is unlikely, but by setting the rvar_inv_count_ == 1.f, we avoid running var infs.
    rvar_inv_count_ = divisor == 0 ? 1.f : 1.f / divisor;
    useHeuristic(m_, c, device_id, bn_group > 1);
  }

  inline const std::vector<size_t> numWorkspaceBytes(const bool& is_fwd) const;

  inline void setWorkspacePointers(
      const std::vector<void*>&  workspace,
      const std::vector<size_t>& num_workspace_bytes);

  void setInputOutputPointers(void* X, void* dX, void* Y, void *dY, void* addend, void* dAddend) {
    X_ = X;
    dX_  = dX;
    Y_   = Y;
    dY_  = dY;
    addend_   = addend;
    dAddend_  = dAddend;
  }

  // Sets the pointers for the scale and weight (in that order) data and derivative buffers.
  void setWeightPointers(const std::vector<void*>& weight_pointers,
                                 const std::vector<void*>& deriv_pointers) {
    assert(weight_pointers.size() == 2);
    assert(deriv_pointers.size()  == 2);
    scale_  = static_cast<float*>(weight_pointers[0]);
    bias_   = static_cast<float*>(weight_pointers[1]);
    dscale_ = static_cast<float*>(deriv_pointers[0]);
    dbias_  = static_cast<float*>(deriv_pointers[1]);
  }

  // Sets the pointers for the population mean and variance buffers, in that order.
  void setParameterPointers(const std::vector<void*>& param_pointers) {
    assert(param_pointers.size() == 2);
    population_mean_     = static_cast<float*>(param_pointers[0]);
    population_variance_ = static_cast<float*>(param_pointers[1]);
  }

  void setConstants(const double exp_avg_factor, const double eps) {
    exp_avg_factor_ = exp_avg_factor;
    eps_ = eps;
  }

  void setBNGroup(const int bn_group, void** xbuf_ptr, const int device_id) {
     bn_group_ = bn_group;
     xbuf_ptr_ = xbuf_ptr;
  }

  void processCudnnStatus(const cudnnStatus_t& status,
                          const std::string& string = std::string(),
                          bool verbose = VERBOSE_DEFAULT) {
    if (status != CUDNN_STATUS_SUCCESS)
      LOG(FATAL) << string << " " << cudnnGetErrorString(status);
    else if (verbose)
      LOG(INFO) << string << " " << cudnnGetErrorString(status);
  }

  void checkCudaStatus(const std::string& string = std::string(),
                       bool verbose = VERBOSE_DEFAULT) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
      LOG(FATAL) << string << " " << cudaGetErrorString(status);
    else if (verbose)
      LOG(INFO) << string << " " << cudaGetErrorString(status);
  }

  size_t size_retired_ctas(int grid_y) const {
    // Note that the value of max_grid_y to handle known GPUs is about 160.
    const int max_grid_y = 1024;
    if (grid_y > max_grid_y)
      LOG(INFO) << "GPU capabilities exceeds assumptions.";
    const int retired_cta_bytes = max_grid_y * 2 * sizeof(int);
    // Since the region will be initialized once and used for many kernels,
    // the idea is to return an ample size that will cover all uses.
    return retired_cta_bytes;
  }

  void*  X_ = nullptr;
  void* dX_ = nullptr;
  void*  Y_ = nullptr;
  void* dY_ = nullptr;
  void*  addend_ = nullptr;
  void* dAddend_ = nullptr;

  // Learned scale and bias weights.
  float* scale_  = nullptr;
  float* dscale_ = nullptr;
  float* bias_   = nullptr;
  float* dbias_  = nullptr;

  // Computed population mean and variance parameters.
  float* population_mean_     = nullptr;
  float* population_variance_ = nullptr;

  // Workspace buffers for minibatch mean and variance (computed in fwd, needed by bwd).
  float* minibatch_mean_     = nullptr;
  float* minibatch_variance_ = nullptr;

  int m_ = 0;  // Number of values per channel that BN is normalizing.
  int c_ = 0;  // Number of channels over which BN is normalizing.

  float svar_inv_count_ = 0.f;  // factor to scale sum of squared errors to get saved variance
  float rvar_inv_count_ = 0.f;  // factor to scale sum of squared errors to get running variance

  double exp_avg_factor_ = 0.;
  double eps_            = 0.;
  std::string name_;

  int bn_group_ = 1;
  void** xbuf_ptr_ = nullptr;

 private:
  void setTensorDescriptor(cudnnTensorDescriptor_t descriptor,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t     data_type,
                           int n, int c, int h, int w) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    status = cudnnSetTensor4dDescriptor(descriptor, format, data_type, n, c, h, w);
    processCudnnStatus(status, "set tensor descriptor");
  }

  void createTensorDescriptor(cudnnTensorDescriptor_t *descriptor) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    status = cudnnCreateTensorDescriptor(descriptor);
    processCudnnStatus(status, "create tensor_descriptor");
  }

  void destroyTensorDescriptor(cudnnTensorDescriptor_t descriptor) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    status = cudnnDestroyTensorDescriptor(descriptor);
    processCudnnStatus(status, "destroy tensor_descriptor");
  }

 protected:
  float *partial_sums_ = nullptr;
  int *partial_counts_ = nullptr;
  int *retired_ctas_   = nullptr;
  unsigned int *relu_bitmask_ = nullptr;

  inline void _setFwdParams(NhwcBatchNormFwdParams *params) const;
  inline void _setFwdInferenceParams(NhwcBatchNormFwdInferenceParams *params) const;
  inline void _setBwdParams(NhwcBatchNormBwdParams *params) const;

  // @todo: ability to configure these?
  // Kernel params
  static const int USE_ONLINE_APPROACH = 1;
  static const int THREADS_PER_CTA = 512;
  static const int MAX_SMEM_WITHOUT_OPT_IN = 48 * 1024;

  typedef uint16_t StorageType;
  static const int PIXELS_PER_THREAD_FWD_INFERENCE = 4;
  static const int THREADS_PER_PIXEL_FWD_INFERENCE = 16;
  static const int ELEMENTS_PER_LDG_FWD_INFERENCE = 4;
  static const int C_ELEMENTS_PER_CTA_INFERENCE = 64;

  // max grid.y in case of group bn is limited by exchange buffer size
  static const int MAX_GBN_BLOCK_Y = 256;

  // Helper function to launch the forward kernel.

  // We calculate (based on smem usage) the achievable occupancy and make sure we run a kernel
  // version that was compiled with that occupancy in its launch bounds.  This way, we avoid
  // needless register spills.

  // We also increase max dynamic shared memory when opt in needed. If something wrong happens,
  // either Cooperative launch or cudaFuncSetAttribute will show error correctly.
  void _fwdKernelLauncher(cudaStream_t stream, NhwcBatchNormFwdParams params,
                                dim3 grid_dim, int outer_loops, int device_id, int occupancy) {
    int outer_loops_ = outer_loops ==1 ? 1 : 0;
    std::vector<const void*> args;
    args.emplace_back(&params);
    std::string code_params = NhwcBatchNormCode::get_bhwc_batch_norm_params(THREADS_PER_PIXEL_FWD,
            C_ELEMENTS_PER_CTA_FWD,
            PIXELS_PER_THREAD_IN_REGISTERS_FWD,
            pixels_per_thread_in_smem_fwd,
            ELEMENTS_PER_LDG_FWD, USE_ONLINE_APPROACH,
            outer_loops, params.sync_iters,
            false, true, occupancy);
    std::string old_name = "nhwc_batch_norm_fwd";
    std::string new_name = "nhwc_batch_norm_fwd_add_relu_" +
                           std::to_string(THREADS_PER_PIXEL_FWD) + "_" +
                           std::to_string(ELEMENTS_PER_LDG_FWD) + "_" +
                           std::to_string(pixels_per_thread_in_smem_fwd) + "_" +
                           std::to_string(PIXELS_PER_THREAD_IN_REGISTERS_FWD) + "_" +
                           std::to_string(outer_loops_) + "_" +
                           std::to_string(occupancy);
    std::string code_s = std::string(NhwcBatchNormCode::nhwc_batch_norm_common) +
                         std::string(NhwcBatchNormCode::nhwc_batch_norm_fwd);
    auto kernel = mxnet::common::cuda::rtc::get_function(code_params,
                                new_name,
                                NhwcBatchNormCode::get_nhwc_batch_norm_kernel(
                                    code_s,
                                    old_name,
                                    new_name),
                                device_id);
    cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
    checkCudaStatus(name_ + " fwd add relu ser coop kernel (cudaFuncSetAttribute carveout)");
    ENSURE_MAX_DYN_SMEM_LIMIT(kernel, dynamic_smem_size_fwd, static_smem_size_fwd,
                                  device_id, name_ + " fwd add relu coop serial kernel");
    void *params_ptr = static_cast<void*>(&params);
    auto err = cuLaunchCooperativeKernel(kernel, grid_dim.x, grid_dim.y, grid_dim.z,
            THREADS_PER_CTA, 1, 1,
            dynamic_smem_size_fwd, stream,  const_cast<void**>(args.data()));
    checkCudaStatus(name_ + " fwd add relu ser coop kernel");
    if (err != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(err, &error_string);
        LOG(FATAL) << "cuLaunchCooperativeKernel failed: "
               << err << " " << error_string << ": "
               << new_name << " "
               << "(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ") "
               << dynamic_smem_size_fwd << " "
               << args.size();
    }
  }

  // Helper function to launch the backward kernel.

  void _bwdKernelLauncher(cudaStream_t stream, NhwcBatchNormBwdParams params,
                                dim3 grid_dim, int outer_loops, int device_id) {
    std::vector<const void*> args;
    args.emplace_back(&params);
    int outer_loops_ = outer_loops ==1 ? 1 : 0;
    int occupancy = smem_driven_bwd_occupancy(device_id) >= 2 ? 2 : 1;
    std::string code_params = NhwcBatchNormCode::get_bhwc_batch_norm_params(THREADS_PER_PIXEL_BWD,
            C_ELEMENTS_PER_CTA_BWD,
            PIXELS_PER_THREAD_IN_REGISTERS_BWD, pixels_per_thread_in_smem_bwd,
            ELEMENTS_PER_LDG_BWD, USE_ONLINE_APPROACH, outer_loops, params.sync_iters,
            false, true, occupancy);
    std::string old_name = "nhwc_batch_norm_bwd_add_relu";
    std::string new_name = "nhwc_batch_norm_bwd_add_relu_" +
               std::to_string(THREADS_PER_PIXEL_BWD) + "_" +
               std::to_string(ELEMENTS_PER_LDG_BWD) + "_" +
               std::to_string(pixels_per_thread_in_smem_bwd) + "_" +
               std::to_string(PIXELS_PER_THREAD_IN_REGISTERS_BWD) + "_" +
               std::to_string(outer_loops_) + "_" +
               std::to_string(occupancy);
    std::string code_s = std::string(NhwcBatchNormCode::nhwc_batch_norm_common) +
                         std::string(NhwcBatchNormCode::nhwc_batch_norm_bwd_add_relu);
    auto kernel = mxnet::common::cuda::rtc::get_function(code_params,
                               new_name,
                               NhwcBatchNormCode::get_nhwc_batch_norm_kernel(
                                   code_s,
                                   old_name,
                                    new_name),
                               device_id);
    cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
    checkCudaStatus(name_ + new_name +
            " bwd add relu ser coop kernel (cudaFuncSetAttribute carveout)");
    ENSURE_MAX_DYN_SMEM_LIMIT(kernel, dynamic_smem_size_bwd, static_smem_size_bwd,
                                  device_id, name_ + new_name + " bwd_add_relu coop serial kernel");
    void *params_ptr = static_cast<void*>(&params);
    auto err = cuLaunchCooperativeKernel(kernel, grid_dim.x, grid_dim.y, grid_dim.z,
            THREADS_PER_CTA, 1, 1,
            dynamic_smem_size_bwd, stream,  const_cast<void**>(args.data()));
    checkCudaStatus(name_ + new_name + " bwd_add_relu ser coop kernel");
    if (err != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(err, &error_string);
        LOG(FATAL) << "cuLaunchCooperativeKernel failed: "
               << err << " " << error_string << ": "
               << new_name << " "
               << "(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ") "
               << dynamic_smem_size_bwd << " "
               << args.size();
    }
  }

 private:
  // Calculate the headroom for cooperative kernel launch based on chip arch and full-grid capacity.
  size_t launch_margin(int sm_arch, size_t full_gpu_grid_x, bool is_bwd) {
    static bool warning_logged = false;
    static int lm_env = is_bwd ? dmlc::GetEnv("NHWC_BATCHNORM_LAUNCH_MARGIN", -1) : -1;
    int lm_default = (sm_arch >= 70 && sm_arch < 80) ? NHWC_BATCHNORM_LAUNCH_MARGIN_DEFAULT : 0;
    size_t lm_min = (sm_arch >= 70 && sm_arch < 80) ? NHWC_BATCHNORM_LAUNCH_MARGIN_MIN : 0;
    size_t lm = static_cast<size_t>(lm_env >= 0 ? lm_env : lm_default);
    lm = std::max(lm_min, lm);
    if (lm >= full_gpu_grid_x) {
      lm = full_gpu_grid_x - 1;  // Let at least 1 block run
      if (!warning_logged) {
        warning_logged = true;
        LOG(INFO) << "Specified launch margin is too large on device of arch " << sm_arch
                  << ", using NHWC_BATCHNORM_LAUNCH_MARGIN=" << lm;
      }
    }
    return lm;
  }

  int get_max_smem_fwd(int device_id) {
    using namespace mxnet::common::cuda;
    int sm_arch = SMArch(device_id);
    uint32_t smem_per_sm = MaxSharedMemoryPerBlockOptin(device_id);
    int static_smem_size_fwd = THREADS_PER_PIXEL_FWD * (THREADS_PER_CTA / 32) *
      ELEMENTS_PER_LDG_FWD * sizeof(float);
    int pixels_per_thread_in_smem_fwd = 1;
    int occ_start = BN_MAX_OCCUPANCY_FWD;
    for (int occ = occ_start; occ >=1; occ--) {
        float dynamic_smem_size_budget = smem_per_sm / occ - static_smem_size_fwd;
        if (dynamic_smem_size_budget > 0) {
            pixels_per_thread_in_smem_fwd = static_cast<int>(((dynamic_smem_size_budget
                                           / THREADS_PER_CTA)
                                           / ELEMENTS_PER_LDG_FWD)
                                           / sizeof(uint16_t));
            if (pixels_per_thread_in_smem_fwd > 0) {
                break;
            }
        }
    }
    return static_cast<int>(pixels_per_thread_in_smem_fwd);
  }

  int get_max_smem_bwd(int device_id) {
    using namespace mxnet::common::cuda;
    int sm_arch = SMArch(device_id);
    uint32_t smem_per_sm = MaxSharedMemoryPerBlockOptin(device_id);
    int static_smem_size_bwd = THREADS_PER_PIXEL_BWD * (THREADS_PER_CTA / 32) *
      ELEMENTS_PER_LDG_BWD * sizeof(float);
    int pixels_per_thread_in_smem_bwd = 1;
    int occ_start = BN_MAX_OCCUPANCY_BWD;
    for (int occ = occ_start; occ >=1; occ--) {
        float dynamic_smem_size_budget = smem_per_sm / occ - static_smem_size_bwd;
        pixels_per_thread_in_smem_bwd = static_cast<int>(((((dynamic_smem_size_budget
                                        / THREADS_PER_CTA)
                                        / ELEMENTS_PER_LDG_BWD)
                                        / sizeof(uint16_t))
                                        / 2));
        if (pixels_per_thread_in_smem_bwd > 0) {
            break;
        }
    }
    return static_cast<int>(pixels_per_thread_in_smem_bwd);
  }

  // Calculate the max number of CTAs allowed in the grid for the fwd kernel.
  size_t max_fwd_grid_x(int device_id) {
    using namespace mxnet::common::cuda;
    size_t full_gpu_grid_x = static_cast<size_t>(MultiprocessorCount(device_id) *
                                                 smem_driven_fwd_occupancy(device_id));
    return full_gpu_grid_x - launch_margin(SMArch(device_id), full_gpu_grid_x, false);
  }

  // Calculate the max number of CTAs allowed in the grid for the bwd kernel.
  size_t max_bwd_grid_x(int device_id) {
    using namespace mxnet::common::cuda;
    size_t full_gpu_grid_x = static_cast<size_t>(MultiprocessorCount(device_id) *
                                                 smem_driven_bwd_occupancy(device_id));
    return full_gpu_grid_x - launch_margin(SMArch(device_id), full_gpu_grid_x, true);
  }

  // Loads env variable to determine allowed max CTA per SM occupancy
  int get_occupancy_cap(const int BN_MAX_OCCUPANCY) const {
    static int cta_cap_env = dmlc::GetEnv("NHWC_BATCHNORM_MAX_CTA_PER_SM", 2);
    int cta_cap = std::min(cta_cap_env, BN_MAX_OCCUPANCY);
    if (cta_cap < 1) {
      cta_cap = 1;
      LOG(INFO) << "Unable to set max cta per SM lower than 1 "
                << ", using NHWC_BATCHNORM_MAX_CTA_PER_SM =" << cta_cap;
    }
    return cta_cap;
  }

  // Calculates capped CTA per SM occupancy
  int capped_occupancy(const int occupancy, const int BN_MAX_OCCUPANCY) const {
    int cap = get_occupancy_cap(BN_MAX_OCCUPANCY);
    return std::min(occupancy, cap);
  }

  // Calculate the expected fwd kernel occupancy, as dictated by shared memory usage.
  int smem_driven_fwd_occupancy(int device_id) {
    using namespace mxnet::common::cuda;
    int total_smem_size_fwd = dynamic_smem_size_fwd + static_smem_size_fwd;
    int occupancy = MaxSharedMemoryPerBlockOptin(device_id) / total_smem_size_fwd;
    return capped_occupancy(occupancy, BN_MAX_OCCUPANCY_FWD);
  }

  // Calculate the expected bwd kernel occupancy, as dictated by shared memory usage.
  int smem_driven_bwd_occupancy(int device_id) {
    using namespace mxnet::common::cuda;
    int total_smem_size_bwd = dynamic_smem_size_bwd + static_smem_size_bwd;
    int occupancy = MaxSharedMemoryPerBlockOptin(device_id) / total_smem_size_bwd;
    return capped_occupancy(occupancy, BN_MAX_OCCUPANCY_BWD);
  }
};

const std::vector<size_t> NhwcBatchNormAddRelu::numWorkspaceBytes(const bool& is_fwd) const {
  assert(c_ > 0);

  // choose the max memory required between fwd/bwd passes
  int grid_x_fwd = div_up(m_, MIN_PIXELS_PER_CTA_FWD);
  int grid_x_bwd = div_up(m_, MIN_PIXELS_PER_CTA_BWD);
  int grid_x = max(grid_x_fwd, grid_x_bwd);

  const size_t num_mean_bytes     = c_ * sizeof(float);
  const size_t num_variance_bytes = num_mean_bytes;
  if (is_fwd) {
    int grid_y = div_up(c_, C_ELEMENTS_PER_CTA_FWD);
    const int MAX_C_ELEMENTS_PER_CTA_FWD = 256;
    int elems_per_group = ((m_ + 31) & ~31) * (MAX_C_ELEMENTS_PER_CTA_FWD / 32);
    int group_count = div_up(c_, MAX_C_ELEMENTS_PER_CTA_FWD);
    const size_t bitmask_bytes = elems_per_group * group_count * sizeof(unsigned int);

    const size_t size_sums          = grid_y * grid_x * THREADS_PER_PIXEL_FWD *
      ELEMENTS_PER_LDG_FWD * 2 * sizeof(float);
    const size_t size_counts        = grid_y * grid_x * sizeof(int);
    return {num_mean_bytes, num_variance_bytes, bitmask_bytes,
          size_retired_ctas(grid_y), size_sums, size_counts};
  } else {
    int grid_y = div_up(c_, C_ELEMENTS_PER_CTA_BWD);
    const int MAX_C_ELEMENTS_PER_CTA_BWD = 256;
    int elems_per_group = ((m_ + 31) & ~31) * (MAX_C_ELEMENTS_PER_CTA_BWD / 32);
    int group_count = div_up(c_, MAX_C_ELEMENTS_PER_CTA_BWD);
    const size_t bitmask_bytes = elems_per_group * group_count * sizeof(unsigned int);

    const size_t size_sums          = grid_y * grid_x * THREADS_PER_PIXEL_BWD *
      ELEMENTS_PER_LDG_BWD * 2 * sizeof(float);
    const size_t size_counts        = grid_y * grid_x * sizeof(int);
    return {num_mean_bytes, num_variance_bytes, bitmask_bytes,
          size_retired_ctas(grid_y), size_sums, size_counts};
  }
}

void NhwcBatchNormAddRelu::setWorkspacePointers(
      const std::vector<void*>& workspace,
      const std::vector<size_t>& num_workspace_bytes) {
  assert(workspace.size() == 6);
  assert(num_workspace_bytes.size() == 6);

  minibatch_mean_     = static_cast<float*>(workspace[0]);
  minibatch_variance_ = static_cast<float*>(workspace[1]);
  relu_bitmask_       = static_cast<unsigned int*>(workspace[2]);
  retired_ctas_       = static_cast<int*>(workspace[3]);
  partial_sums_       = static_cast<float*>(workspace[4]);
  partial_counts_     = static_cast<int*>(workspace[5]);
}

void NhwcBatchNormAddRelu::_setFwdParams(NhwcBatchNormFwdParams *params) const {
  params->gmem_src          = static_cast<uint16_t*>(X_);
  params->gmem_dst          = static_cast<uint16_t*>(Y_);
  params->gmem_src1         = static_cast<uint16_t*>(addend_);
  params->gmem_bias         = bias_;
  params->gmem_scale        = scale_;
  params->gmem_running_mean = population_mean_;
  params->gmem_running_var  = population_variance_;
  params->gmem_saved_mean   = minibatch_mean_;
  params->gmem_saved_var    = minibatch_variance_;
  params->gmem_relu_bitmask = relu_bitmask_;
  params->nhw               = m_;
  params->c                 = c_;
  params->svar_inv_count    = svar_inv_count_;
  params->rvar_inv_count    = rvar_inv_count_;
  params->gmem_sums         = partial_sums_;
  params->gmem_counts       = partial_counts_;
  params->gmem_retired_ctas = retired_ctas_;
  params->var_eps           = eps_;
  params->outer_loops       = 0;
  params->exp_avg_factor    = static_cast<float>(exp_avg_factor_);
  params->c_blks            = div_up(c_, C_ELEMENTS_PER_CTA_FWD);
  params->pixels_per_thread_in_smem = pixels_per_thread_in_smem_fwd;
}

void NhwcBatchNormAddRelu::_setFwdInferenceParams(NhwcBatchNormFwdInferenceParams
                                                        *params) const {
  params->gmem_src   = static_cast<uint16_t*>(X_);
  params->gmem_dst   = static_cast<uint16_t*>(Y_);
  params->gmem_src1  = static_cast<uint16_t*>(addend_);
  params->gmem_bias  = bias_;
  params->gmem_scale = scale_;
  params->gmem_mean  = population_mean_;
  params->gmem_var   = population_variance_;
  params->nhw        = m_;
  params->c          = c_;
  params->var_eps    = eps_;
}

void NhwcBatchNormAddRelu::_setBwdParams(NhwcBatchNormBwdParams *params) const {
  params->gmem_src          = static_cast<uint16_t*>(X_);
  params->gmem_dy           = static_cast<uint16_t*>(dY_);
  params->gmem_dst          = static_cast<uint16_t*>(dX_);
  params->gmem_dst1         = static_cast<uint16_t*>(dAddend_);
  params->gmem_relu_bitmask = relu_bitmask_;
  params->gmem_dscale       = dscale_;
  params->gmem_dbias        = dbias_;
  params->gmem_scale        = scale_;
  params->gmem_bias         = bias_;
  params->gmem_saved_mean   = minibatch_mean_;
  params->gmem_saved_var    = minibatch_variance_;
  params->nhw               = m_;
  params->c                 = c_;
  params->svar_inv_count    = svar_inv_count_;
  params->gmem_sums         = partial_sums_;
  params->gmem_retired_ctas = retired_ctas_;
  params->outer_loops       = 0;
  params->c_blks            = div_up(c_, C_ELEMENTS_PER_CTA_BWD);
  params->pixels_per_thread_in_smem = pixels_per_thread_in_smem_bwd;
}

template <typename gpu>
void NhwcBatchNormAddRelu::fwdInference(mshadow::Stream<gpu> *s, int device_id) {
  bool ptrs_are_set =
      scale_ != nullptr
      && bias_ != nullptr
      //      && minibatch_mean_ != nullptr
      //      && minibatch_variance_ != nullptr
      && population_mean_ != nullptr
      && population_variance_ != nullptr
      && X_ != nullptr
      //      && dX_ != nullptr
      && Y_ != nullptr
      && addend_ != nullptr
      //      && dY_ != nullptr
      //      && dscale_ != nullptr
      //      && dbias_ != nullptr
      && partial_sums_   != nullptr
      && partial_counts_ != nullptr;

  if (!ptrs_are_set)
    die();

  dim3 grid_dim;
  grid_dim.x = div_up(m_, PIXELS_PER_CTA_FWD_INFERENCE);
  grid_dim.y = div_up(c_, C_ELEMENTS_PER_CTA_INFERENCE);

  // @todo: maybe just move this inside initialize routine?
  NhwcBatchNormFwdInferenceParams params;
  _setFwdInferenceParams(&params);

  std::vector<const void*> args;
  args.emplace_back(&params);
  std::string code_params = NhwcBatchNormCode::get_bhwc_batch_norm_params(
                                        THREADS_PER_PIXEL_FWD_INFERENCE,
                                        C_ELEMENTS_PER_CTA_INFERENCE, 0, 0,
                                        ELEMENTS_PER_LDG_FWD_INFERENCE, USE_ONLINE_APPROACH, 0, 0,
                                        false, true, 1);
  auto kernel = mxnet::common::cuda::rtc::get_function(code_params,
                             "nhwc_batch_norm_fwd_inference",
                             std::string(NhwcBatchNormCode::nhwc_batch_norm_common) +
                             std::string(NhwcBatchNormCode::nhwc_batch_norm_fwd_inference),
                             device_id);
  mxnet::common::cuda::rtc::launch(kernel, grid_dim, {512, 1, 1}, 0, s, &args);
}

void NhwcBatchNormAddRelu::init_fwd_params(int new_pixels_per_thread_in_smem_fwd) {
  pixels_per_thread_in_smem_fwd = new_pixels_per_thread_in_smem_fwd;
  pixels_per_thread_fwd = PIXELS_PER_THREAD_IN_REGISTERS_FWD +
      pixels_per_thread_in_smem_fwd;
  static_smem_size_fwd = THREADS_PER_PIXEL_FWD * (THREADS_PER_CTA / 32) *
      ELEMENTS_PER_LDG_FWD * sizeof(float);
  dynamic_smem_size_fwd = pixels_per_thread_in_smem_fwd * THREADS_PER_CTA *
      ELEMENTS_PER_LDG_FWD * sizeof(uint16_t);
  pixels_per_cta_fwd = THREADS_PER_CTA / THREADS_PER_PIXEL_FWD *
      pixels_per_thread_fwd;
}

void NhwcBatchNormAddRelu::init_bwd_params(int new_pixels_per_thread_in_smem_bwd) {
  pixels_per_thread_in_smem_bwd = new_pixels_per_thread_in_smem_bwd;
  pixels_per_thread_bwd = PIXELS_PER_THREAD_IN_REGISTERS_BWD +
      pixels_per_thread_in_smem_bwd;
  static_smem_size_bwd = THREADS_PER_PIXEL_BWD * (THREADS_PER_CTA / 32) *
      ELEMENTS_PER_LDG_BWD * sizeof(float);
  dynamic_smem_size_bwd = pixels_per_thread_in_smem_bwd * THREADS_PER_CTA *
      ELEMENTS_PER_LDG_BWD * 2 * sizeof(uint16_t);
  pixels_per_cta_bwd = THREADS_PER_CTA / THREADS_PER_PIXEL_BWD *
      pixels_per_thread_bwd;
}

dim3 NhwcBatchNormAddRelu::calc_fwd_grid(int device_id, int *loop, bool use_group_bn) {
  // set smem config to max for current device
  init_fwd_params(get_max_smem_fwd(device_id));

  dim3 grid_dim;
  grid_dim.x = div_up(m_, pixels_per_cta_fwd);
  int c_blks = div_up(c_, C_ELEMENTS_PER_CTA_FWD);
  unsigned int max_grid_x = max_fwd_grid_x(device_id);
  static bool reduce_quantization = dmlc::GetEnv("MXNET_CUDNN_NHWC_BN_QUANT_RED", true);
  if (grid_dim.x <= max_grid_x) {
    *loop = 1;
    grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));

    // we try to increase CTAs to max if it is not
    // Increase help in general but not always, due to margin
    if (reduce_quantization && grid_dim.x*grid_dim.y < max_grid_x) {
      // calculate minimal smem pixel not turn into loop
      int new_pixels_per_thread_in_smem_fwd = max(
        div_up(div_up(m_, (THREADS_PER_CTA / THREADS_PER_PIXEL_FWD)), (max_grid_x/grid_dim.y))
        - PIXELS_PER_THREAD_IN_REGISTERS_FWD, 0);
      // re-do block assignment
      init_fwd_params(new_pixels_per_thread_in_smem_fwd);
      grid_dim.x = div_up(m_, pixels_per_cta_fwd);

      grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));
    }
    assert(!use_group_bn || (grid_dim.y < MAX_GBN_BLOCK_Y));   // FIXME: turn into a loop
  } else {
    grid_dim.x = max_grid_x;
    grid_dim.y = 1;
    int nhw_in_regs = m_ - pixels_per_thread_in_smem_fwd * PIXELS_PER_LDG_FWD * grid_dim.x;
    int pixels_per_iteration = PIXELS_PER_THREAD_IN_REGISTERS_FWD * PIXELS_PER_LDG_FWD * grid_dim.x;
    *loop = div_up(nhw_in_regs, pixels_per_iteration);
  }
  return grid_dim;
}

dim3 NhwcBatchNormAddRelu::calc_bwd_grid(int device_id, int *loop, bool use_group_bn) {
  // set smem config to max for current device
  // need to store 2 in backward
  init_bwd_params(get_max_smem_bwd(device_id));

  dim3 grid_dim;
  grid_dim.x = div_up(m_, pixels_per_cta_bwd);
  int c_blks = div_up(c_, C_ELEMENTS_PER_CTA_BWD);
  unsigned int max_grid_x = max_bwd_grid_x(device_id);

  static bool reduce_quantization = dmlc::GetEnv("MXNET_CUDNN_NHWC_BN_QUANT_RED", true);
  if (grid_dim.x <= max_grid_x) {
    *loop = 1;
    grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));

    // we try to increase CTAs to max if it is not
    // Increase help in general but not always, due to margin
    if (reduce_quantization && grid_dim.x*grid_dim.y < max_grid_x) {
      // calculate minimal smem pixel not turn into loop
      int new_pixels_per_thread_in_smem_bwd = max(
        div_up(div_up(m_, (THREADS_PER_CTA / THREADS_PER_PIXEL_BWD)), (max_grid_x/grid_dim.y))
        - PIXELS_PER_THREAD_IN_REGISTERS_BWD, 0);
      // re-do block assignment
      init_bwd_params(new_pixels_per_thread_in_smem_bwd);
      grid_dim.x = div_up(m_, pixels_per_cta_bwd);
      grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));
    }
    assert(!use_group_bn || (grid_dim.y < MAX_GBN_BLOCK_Y));   // FIXME: turn into a loop
  } else {
    grid_dim.x = max_grid_x;
    grid_dim.y = 1;
    int nhw_in_regs = m_ - pixels_per_thread_in_smem_bwd * PIXELS_PER_LDG_BWD * grid_dim.x;
    int pixels_per_iteration = PIXELS_PER_THREAD_IN_REGISTERS_BWD *
                               PIXELS_PER_LDG_BWD * grid_dim.x;
    *loop = div_up(nhw_in_regs, pixels_per_iteration);
  }
  return grid_dim;
}

void NhwcBatchNormAddRelu::fwd(cudaStream_t stream, int device_id) {
  bool ptrs_are_set =
      scale_ != nullptr
      && bias_ != nullptr
      && minibatch_mean_ != nullptr
      && minibatch_variance_ != nullptr
      && relu_bitmask_ != nullptr
      && population_mean_ != nullptr
      && population_variance_ != nullptr
      && X_ != nullptr
      //      && dX_ != nullptr
      && Y_ != nullptr
      && addend_ != nullptr
      //      && dY_ != nullptr
      //      && dscale_ != nullptr
      //      && dbias_ != nullptr
      && partial_sums_   != nullptr
      && partial_counts_ != nullptr
      && retired_ctas_   != nullptr;

  if (!ptrs_are_set)
    die();

  // reset of retired_cta_count no longer needed

  NhwcBatchNormFwdParams params;
  _setFwdParams(&params);

  // while external argument is bn_group, defining the number of gpus cooperating in a group,
  // the parameter passed down to the kernel is sync_iters, defining the number of synchronization
  // iterations to be performed. Number of iterations is log2 of group size, since at each
  // iteration pairs exchange information "across additional dimension". i.e.
  // iter 0:  0 <-> 1   2 <-> 3   4 <-> 5  6 <-> 7
  // iter 1:  0 <-> 2   1 <-> 3   4 <-> 6  5 <-> 7
  // currently supporting only up to bn_group=16 (sync_iters=4)
  // to avoid any "grouping" sync_iters should be ==0
  params.my_data = nullptr;
  for (int i = 0; i < MAX_SYNC_ITERS; ++i) {
    params.pair_data[i] = nullptr;
  }
  params.sync_iters = 0;
  for (int g = 2; g <= bn_group_; g *= 2) {
    params.my_data = xbuf_ptr_[device_id];
    params.pair_data[params.sync_iters] = xbuf_ptr_[device_id^(g / 2)];
    params.sync_iters++;
  }
  CHECK_LE(params.sync_iters, MAX_SYNC_ITERS) << "Value passed to bn_group parameter is too high";

  dim3 grid_dim = calc_fwd_grid(device_id, &params.outer_loops, (bn_group_ > 1));
  params.pixels_per_thread_in_smem = pixels_per_thread_in_smem_fwd;
  int occupancy = smem_driven_fwd_occupancy(device_id) >= 2 ? 2 : 1;
  _fwdKernelLauncher(stream, params, grid_dim, params.outer_loops, device_id, occupancy);
}

void NhwcBatchNormAddRelu::dgrad(cudaStream_t stream, int device_id) {
  bool ptrs_are_set =
      scale_ != nullptr
      && bias_ != nullptr
      && minibatch_mean_ != nullptr
      && minibatch_variance_ != nullptr
      && relu_bitmask_ != nullptr
      //      && population_mean_ != nullptr
      //      && population_variance_ != nullptr
      && X_ != nullptr
      && dX_ != nullptr
      //      && Y_ != nullptr
      && dY_ != nullptr
      && dAddend_ != nullptr
      && dscale_ != nullptr
      && dbias_ != nullptr
      && retired_ctas_   != nullptr;

  if (!ptrs_are_set)
    die();

  // reset of retired_cta_count no longer needed

  NhwcBatchNormBwdParams params;
  _setBwdParams(&params);

  params.my_data = nullptr;
  for (int i = 0; i < MAX_SYNC_ITERS; ++i) {
    params.pair_data[i] = nullptr;
  }
  params.wgrad_coeff = 1.0;
  params.sync_iters = 0;
  for (int g = 2; g <= bn_group_; g *= 2) {
    params.my_data = xbuf_ptr_[device_id];
    params.pair_data[params.sync_iters] = xbuf_ptr_[device_id^(g / 2)];
    params.sync_iters++;
  }
  CHECK_LE(params.sync_iters, MAX_SYNC_ITERS) << "Value passed to bn_group parameter is too high";
  if (bn_group_ > 0) {
    params.wgrad_coeff = 1.0 / bn_group_;  // dscale and dbiad will be adjusted using this
                                           // factor, so that after reduction gradient will
                                           // match a case of a single GPU running with
                                           // local batch that equals group span batch
  }

  dim3 grid_dim = calc_bwd_grid(device_id, &params.outer_loops, (bn_group_ > 1));
  params.pixels_per_thread_in_smem = pixels_per_thread_in_smem_bwd;
  _bwdKernelLauncher(stream, params, grid_dim, params.outer_loops, device_id);
}
#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_ADD_RELU_H_
