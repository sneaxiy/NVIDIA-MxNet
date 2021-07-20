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
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_

#include <mxnet/storage.h>
#include <dmlc/parameter.h>
#include <mxnet/tuple.h>
#if MXNET_USE_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <vector>
#include <set>
#include <mutex>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "../spatial_parallel_convolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../../../common/cuda/utils.h"
#include "../cublas_fully_connected-inl.h"
#include "../../contrib/spatial_parallel_support.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && MXNET_USE_NCCL == 1

// Equivalent algo performance threshhold (e.g. 1.01 == 1% performance difference)
// Used to prune Tensor Core algos with no appreciable performance benefit.
#define ALGO_PERF_THRESHOLD 1.01

namespace {

template <typename DType>
__global__ void ExtractHaloFilter(DType* out, const DType* in, const index_t slice,
                                  const index_t stride, const index_t dim,
                                  const index_t total_elements) {
  const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t base_x = tid % stride;
  const index_t base_y = tid / stride;
  const index_t base = base_x + (slice + base_y * dim) * stride;
  if (base < total_elements) {
    out[tid] = in[base];
  }
}

template <typename DType>
__global__ void AddHaloWGrad(const DType* in, DType* out, const index_t slice,
                             const index_t stride, const index_t dim,
                             const index_t total_elements) {
  const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t base_x = tid % stride;
  const index_t base_y = tid / stride;
  const index_t base = base_x + (slice + base_y * dim) * stride;
  if (base < total_elements) {
    out[base] += in[tid];
  }
}

}  // namespace

/*!
 * \brief The Operator used to perform convolution using cuDNN kernels.
 */
template<typename DType>
class CuDNNSPConvolutionOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

 public:
  CuDNNSPConvolutionOp() : back_bias_get_workspace_performed_(false) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&halo_in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&halo_out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&main_filter_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&halo_filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&main_forward_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&halo_forward_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&main_back_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&halo_back_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&main_back_conv_desc_w_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&halo_back_conv_desc_w_));
    parallelize_backward_kernels_ = Context::GetGPUStreamsPerWorker() >= 2;
    CHECK(parallelize_backward_kernels_)
      << "Running with only 1 stream, NCCL will be exposed. ERROR (for now)!!!";
  }

  void Init(const SpatialParallelConvolutionParam& param,
            int forward_compute_type,
            int backward_compute_type,
            const mxnet::ShapeVector& in_shape,
            const mxnet::ShapeVector& out_shape,
            const RunContext& rctx,
            bool add_to_weight) {
    using namespace mshadow;
    this->param_ = param;
    backward_compute_type_ = backward_compute_type;
    // If no local setting for TensorCore use policy, look to global policy.
    if (!param_.cudnn_tensor_core.has_value())
      param_.cudnn_tensor_core = GetEnvAllowTensorCore();
    this->add_to_weight_ = add_to_weight;
    InitBufferForParam();
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    dtype_ = DataType<DType>::kCudnnFlag;

    auto effective_layout = param_.layout.value();
    auto halo_layout = param_.layout.value();
    switch (effective_layout) {
      case mshadow::kNCW:
      case mshadow::kNWC:
      case mshadow::kCWN:
        LOG(FATAL) << "1-D convolutions are not supported!";
      default: break;
    }

    switch (effective_layout) {
      case mshadow::kNHWC: halo_layout = mshadow::kNWC; break;
      case mshadow::kNDHWC: halo_layout = mshadow::kNHWC; break;
      default: LOG(FATAL) << "Only NDHWC and NHWC supported for now.";
    }

    MSHADOW_LAYOUT_SWITCH(effective_layout, Layout, {
      main_format_ = LayoutType<Layout>::kCudnnFlag;
    });
    MSHADOW_LAYOUT_SWITCH(halo_layout, Layout, {
      halo_format_ = LayoutType<Layout>::kCudnnFlag;
    });

    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type, rctx.ctx.dev_id))
      LOG(FATAL) << "Convolution parameters not supported by cuDNN implementation.";

    InitDescriptors(in_shape, out_shape,
                    cudnn_forward_compute_type, cudnn_backward_compute_type);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }

    dgrad_as_gemm_debug_ = dmlc::GetEnv("MXNET_CONV_DGRAD_AS_GEMM_DEBUG", false);
    dgrad_as_gemm_ = dmlc::GetEnv("MXNET_CONV_DGRAD_AS_GEMM", false) &&
                     param_.kernel.ndim() == 2 &&
                     param_.stride == mxnet::TShape {1, 1} &&
                     param_.kernel == mxnet::TShape {1, 1} &&
                     effective_layout == mshadow::kNHWC;

    // In cuDNN_v6, dilated convolution descriptors are compatible with only a
    // single convolution algorithm.  Despite this, we go through the algorithm
    // selection process, which will return the only algorithm supported.  This
    // approach keeps the treatment of convolution cases uniform and will
    // naturally respond to more algorithms supporting dilated convolutions in
    // future cuDNN releases.
    SelectMainAlgo(rctx, in_shape, out_shape,
                   cudnn_forward_compute_type, cudnn_backward_compute_type);
    mxnet::ShapeVector halo_in_shape(2);
    halo_in_shape[0] = RemoveFirstSpatialDim(in_shape[0]);
    halo_in_shape[1] = RemoveFirstSpatialDim(in_shape[1]);
    mxnet::ShapeVector halo_out_shape(1);
    halo_out_shape[0] = RemoveFirstSpatialDim(out_shape[0]);

    SelectHaloAlgo(rctx, halo_in_shape, halo_out_shape,
                   cudnn_forward_compute_type, cudnn_backward_compute_type);
    GetTempSize(rctx);
    NCCLCommContainer::Param p = {param_.num_gpus,
                                  param_.rank,
                                  param_.nccl_unique_id};
    NCCLCommContainer::Init(p);
  }

  ~CuDNNSPConvolutionOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(halo_in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(halo_out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(main_filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(main_forward_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(main_back_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(main_back_conv_desc_w_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(halo_back_conv_desc_w_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(halo_filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(halo_forward_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(halo_back_conv_desc_));
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 3 : 4;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    auto dshape = in_data[0].shape_;
    auto oshape = out_data[0].shape_;
    auto halo_dshape = RemoveFirstSpatialDim(dshape);
    size_t halo_size = halo_dshape.Size() * sizeof(DType);
    mxnet::TShape wshape = in_data[spconv::kWeight].shape_;
    mxnet::TShape halo_wshape = RemoveFirstSpatialDim(wshape);
    size_t halo_filter_size = halo_wshape.Size() * sizeof(DType);
    auto workspaces_stages = AllocateTempWorkspaces(ctx, {{halo_size, halo_size,
                                                           forward_workspace_byte_},
                                                          {halo_size, halo_size,
                                                           halo_forward_workspace_byte_,
                                                           halo_filter_size,
                                                           halo_forward_workspace_byte_,
                                                           halo_filter_size}});
    auto& forward_conv_workspace = workspaces_stages[0][2];
    size_t workspace_size = TensorSizeBytes(forward_conv_workspace);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = GetNdPtr(in_data[spconv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *out_ptr = GetNdPtr(out_data[spconv::kOut], param_.kernel.ndim() + 2, s);

    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    typename DataType<DType>::ScaleType beta_add = 1.0f;

    index_t first_index = param_.rank * dshape[1];
    index_t last_index = (param_.rank + 1) * dshape[1] - 1;
    const bool needs_recv_previous_halo = param_.rank > 0 &&
                                          (first_index % param_.stride[0] == 0);
    const bool needs_recv_next_halo = (param_.rank < param_.num_gpus - 1) &&
                                      (last_index % param_.stride[0] == 0);
    const bool needs_send_first_halo = param_.rank > 0 &&
                                       ((first_index - 1) % param_.stride[0] == 0);
    const bool needs_send_last_halo = (param_.rank < param_.num_gpus - 1) &&
                                      ((last_index + 1) % param_.stride[0] == 0);
    const bool needs_communicate = needs_recv_next_halo || needs_recv_previous_halo ||
                                   needs_send_first_halo || needs_send_last_halo;

    if (needs_communicate) {
      // They stay the same between stages
      auto& previous_halo_data = workspaces_stages[0][0];
      auto& next_halo_data = workspaces_stages[0][1];
      {
        // First phase: Fwd + NCCL
        auto& workspaces = workspaces_stages[0];
        SyncedGPUAuxStream s_nccl = ctx.get_gpu_aux_stream();
        CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                        &alpha,
                        in_desc_,
                        data_ptr,
                        main_filter_desc_,
                        wmat_ptr,
                        main_forward_conv_desc_,
                        main_forward_algo_.AlgoNumber(),
                        forward_conv_workspace.dptr_,
                        workspace_size,
                        req[spconv::kOut] == kAddTo? &beta_add : &beta,
                        out_desc_,
                        out_ptr));
        {
          cudaStream_t nccl_stream = Stream<gpu>::GetStream(s_nccl.GetStream());
          std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
          ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param_.num_gpus));
          ncclGroupStart();
          if (needs_recv_previous_halo) {
            ncclRecv(previous_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_recv_next_halo) {
            ncclRecv(next_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          if (needs_send_first_halo) {
            ncclSend(data_ptr, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_send_last_halo) {
            ncclSend(data_ptr + halo_dshape.Size() * (dshape[1] - 1),
                     halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          ncclGroupEnd();
        }
      }
      {
        // Second phase - halo convolutions
        auto& workspaces = workspaces_stages[1];
        SyncedGPUAuxStream s_aux = ctx.get_gpu_aux_stream();
        if (needs_recv_previous_halo) {
          auto& workspace = workspaces[2];
          auto& filter = workspaces[3];
          index_t n_elements = halo_wshape.Size();
          const int threads = 512;
          const index_t blocks = common::div_round(n_elements, threads);
          cudaStream_t stream = Stream<gpu>::GetStream(s);
          index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
          ExtractHaloFilter<<<blocks, threads, 0, stream>>>(filter.dptr_, wmat_ptr, 0,
                                                            stride, wshape[1], wshape.Size());
          CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                             &alpha,
                                             halo_in_desc_,
                                             previous_halo_data.dptr_,
                                             halo_filter_desc_,
                                             filter.dptr_,
                                             halo_forward_conv_desc_,
                                             halo_forward_algo_.AlgoNumber(),
                                             workspace.dptr_,
                                             halo_forward_workspace_byte_,
                                             &beta_add,
                                             halo_out_desc_,
                                             out_ptr));
        }
        if (needs_recv_next_halo) {
          auto& workspace = workspaces[4];
          auto& filter = workspaces[5];
          index_t n_elements = halo_wshape.Size();
          const int threads = 512;
          const index_t blocks = common::div_round(n_elements, threads);
          cudaStream_t stream = Stream<gpu>::GetStream(s_aux.GetStream());
          index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
          ExtractHaloFilter<<<blocks, threads, 0, stream>>>(filter.dptr_, wmat_ptr, wshape[1] - 1,
                                                            stride, wshape[1], wshape.Size());
          CUDNN_CALL(cudnnConvolutionForward(s_aux.GetStream()->dnn_handle_,
                                             &alpha,
                                             halo_in_desc_,
                                             next_halo_data.dptr_,
                                             halo_filter_desc_,
                                             filter.dptr_,
                                             halo_forward_conv_desc_,
                                             halo_forward_algo_.AlgoNumber(),
                                             workspace.dptr_,
                                             halo_forward_workspace_byte_,
                                             &beta_add,
                                             halo_out_desc_,
                                             out_ptr + RemoveFirstSpatialDim(oshape).Size() *
                                                       (oshape[1] - 1)));
        }
      }
    } else {
      CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                      &alpha,
                      in_desc_,
                      data_ptr,
                      main_filter_desc_,
                      wmat_ptr,
                      main_forward_conv_desc_,
                      main_forward_algo_.AlgoNumber(),
                      forward_conv_workspace.dptr_,
                      workspace_size,
                      req[spconv::kOut] == kAddTo? &beta_add : &beta,
                      out_desc_,
                        out_ptr));
    }

    bool perform_forward_bias = !param_.no_bias;
    bool perform_cuda_forward_bias = perform_forward_bias &&
                                     FeaturesLastLayout() &&
                                     dmlc::GetEnv("MXNET_CONV_CUDA_FORWARD_BIAS", false);
    if (perform_forward_bias) {
      if (perform_cuda_forward_bias) {
        int output_features = static_cast<int>(Features(out_data[spconv::kOut].shape_));
        Tensor<gpu, 1, DType> bias =
          in_data[spconv::kBias].get_with_shape<gpu, 1, DType>(Shape1(output_features), s);
        Tensor<gpu, 2, DType> out = FlattenAs2DHead<gpu, DType>(out_data[spconv::kOut], ctx);
        auto &data = out;  // Only data.shape_[0] is used by AddBias()
        AddBias(bias, data, out, s);
      } else {
        Tensor<gpu, 1, DType> bias = in_data[spconv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                                  &alpha,
                                  bias_desc_,
                                  bias.dptr_,
                                  &beta_add,
                                  out_desc_,
                                  out_ptr));
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 4 : 3;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req[spconv::kDummy], kNullOp) << "Expects gradient for the dummy parameter!";
    Stream<gpu> *s = ctx.get_stream<gpu>();
    const auto& dshape = out_grad[spconv::kOut].shape_;
    const auto& halo_dshape = RemoveFirstSpatialDim(dshape);
    const auto& oshape = in_grad[spconv::kData].shape_;
    const size_t halo_size = halo_dshape.Size() * sizeof(DType);
    mxnet::TShape wshape = in_data[spconv::kWeight].shape_;
    mxnet::TShape halo_wshape = RemoveFirstSpatialDim(wshape);
    size_t halo_filter_size = halo_wshape.Size() * sizeof(DType);

    // I/O's should have 2 more dims than the kernel dim
    DType *grad_ptr = GetNdPtr(out_grad[spconv::kOut], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *gwmat_ptr = GetNdPtr(in_grad[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *data_ptr = GetNdPtr(in_data[spconv::kData], param_.kernel.ndim() + 2, s);
    DType *gdata_ptr = GetNdPtr(in_grad[spconv::kData], param_.kernel.ndim() + 2, s);

    bool perform_backward_bias = !param_.no_bias && (req[spconv::kBias] != kNullOp);
    bool perform_cuda_backward_bias = perform_backward_bias &&
                                      FeaturesLastLayout() &&
                                      dmlc::GetEnv("MXNET_CONV_CUDA_BACKWARD_BIAS", true);
    if (perform_cuda_backward_bias && !back_bias_get_workspace_performed_) {
      auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
      int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
      auto back_workspace_byte_bias = AddBiasGradWorkspaceSizeBytes(in_grad[spconv::kBias],
                                                                     y_grad,
                                                                     req[spconv::kBias],
                                                                     output_features, ctx);
      back_workspace_byte_wgrad_ = std::max(back_workspace_byte_wgrad_,
                                            static_cast<size_t>(back_workspace_byte_bias));
      back_bias_get_workspace_performed_ = true;
    }

    size_t backward_workspace_byte =
        parallelize_backward_kernels_ ? back_workspace_byte_dgrad_ + back_workspace_byte_wgrad_
                                      : std::max(back_workspace_byte_dgrad_,
                                                 back_workspace_byte_wgrad_);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte);
    // First stage: [dgrad] + [NCCL]
    // Second stage: [wgrad, bgrad] + [halo_dgrad, halo_dgrad]
    const auto& workspaces_stages = AllocateTempWorkspaces(ctx,
                                                           {{halo_size, halo_size,
                                                             back_workspace_byte_dgrad_,
                                                             back_workspace_byte_wgrad_},
                                                            {halo_size, halo_size,
                                                             back_workspace_byte_wgrad_,
                                                             halo_filter_size,
                                                             halo_filter_size,
                                                             halo_back_workspace_byte_dgrad_}});

    index_t first_index = param_.rank * dshape[1];
    index_t last_index = (param_.rank + 1) * dshape[1] - 1;
    const bool needs_recv_previous_halo = param_.rank > 0 &&
                                          (param_.stride[0] == 1 ||
                                           first_index % param_.stride[0] != 0);
    const bool needs_recv_next_halo = (param_.rank < param_.num_gpus - 1) &&
                                      (param_.stride[0] == 1 ||
                                       last_index % param_.stride[0] != 0);
    const bool needs_send_first_halo = param_.rank > 0 &&
                                       (param_.stride[0] == 1 ||
                                        (first_index - 1) % param_.stride[0] != 0);
    const bool needs_send_last_halo = (param_.rank < param_.num_gpus - 1) &&
                                      (param_.stride[0] == 1 ||
                                       (last_index + 1) % param_.stride[0] != 0);
    bool needs_communicate = needs_recv_next_halo || needs_recv_previous_halo ||
                             needs_send_first_halo || needs_send_last_halo;
    needs_communicate = needs_communicate;

    if (needs_communicate) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      // They stay the same between stages
      auto& previous_halo_data = workspaces_stages[0][0];
      auto& next_halo_data = workspaces_stages[0][1];
      {
        auto& workspaces = workspaces_stages[0];
        SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
        // First stage: [dgrad, wgrad] + [NCCL]
        if (req[spconv::kData] != kNullOp) {
          CUDNN_CALL(cudnnConvolutionBackwardData(s->dnn_handle_,
              &alpha,
              main_filter_desc_,
              wmat_ptr,
              out_desc_,
              grad_ptr,
              main_back_conv_desc_,
              main_back_algo_.AlgoNumber(),
              workspaces[2].dptr_,
              back_workspace_byte_dgrad_,
              req[spconv::kData] == kAddTo? &beta_add : &beta,
              in_desc_,
              gdata_ptr));
        }
        if (req[spconv::kWeight] != kNullOp) {
          CHECK_EQ(add_to_weight_, req[spconv::kWeight] == kAddTo);
          CUDNN_CALL(
            cudnnConvolutionBackwardFilter(s->dnn_handle_,
                                           &alpha,
                                           in_desc_,
                                           data_ptr,
                                           out_desc_,
                                           grad_ptr,
                                           main_back_conv_desc_w_,
                                           main_back_algo_w_.AlgoNumber(),
                                           workspaces[3].dptr_,
                                           back_workspace_byte_wgrad_,
                                           req[spconv::kWeight] == kAddTo? &beta_add : &beta,
                                           main_filter_desc_,
                                           gwmat_ptr));
        }
        {
          cudaStream_t nccl_stream = Stream<gpu>::GetStream(aux_stream.GetStream());
          std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
          ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param_.num_gpus));
          ncclGroupStart();
          if (needs_recv_previous_halo) {
            ncclRecv(previous_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_recv_next_halo) {
            ncclRecv(next_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          if (needs_send_first_halo) {
            ncclSend(grad_ptr, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_send_last_halo) {
            ncclSend(grad_ptr + halo_dshape.Size() * (dshape[1] - 1),
                     halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          ncclGroupEnd();
        }
      }
      {
        // Second stage: [halo_wgrad, halo_wgrad, bgrad] + [halo_dgrad, halo_dgrad]
        SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
        auto& workspaces = workspaces_stages[1];
        if (req[spconv::kWeight] != kNullOp) {
          if (needs_recv_previous_halo) {
            auto& dfilter = workspaces[3];
            CUDNN_CALL(
              cudnnConvolutionBackwardFilter(s->dnn_handle_,
                                             &alpha,
                                             halo_in_desc_,
                                             data_ptr,
                                             halo_out_desc_,
                                             previous_halo_data.dptr_,
                                             halo_back_conv_desc_w_,
                                             halo_back_algo_w_.AlgoNumber(),
                                             workspaces[2].dptr_,
                                             back_workspace_byte_wgrad_,
                                             &beta,
                                             halo_filter_desc_,
                                             dfilter.dptr_));
            index_t n_elements = halo_wshape.Size();
            const int threads = 512;
            const index_t blocks = common::div_round(n_elements, threads);
            cudaStream_t stream = Stream<gpu>::GetStream(s);
            index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
            AddHaloWGrad<<<blocks, threads, 0, stream>>>(dfilter.dptr_, gwmat_ptr, wshape[1] - 1,
                                                         stride, wshape[1], wshape.Size());
          }
          if (needs_recv_next_halo) {
            auto& dfilter = workspaces[3];
            CUDNN_CALL(
              cudnnConvolutionBackwardFilter(s->dnn_handle_,
                                             &alpha,
                                             halo_in_desc_,
                                             data_ptr +
                                             RemoveFirstSpatialDim(oshape).Size() * (oshape[1] - 1),
                                             halo_out_desc_,
                                             next_halo_data.dptr_,
                                             halo_back_conv_desc_w_,
                                             halo_back_algo_w_.AlgoNumber(),
                                             workspaces[2].dptr_,
                                             back_workspace_byte_wgrad_,
                                             &beta,
                                             halo_filter_desc_,
                                             dfilter.dptr_));
            index_t n_elements = halo_wshape.Size();
            const int threads = 512;
            const index_t blocks = common::div_round(n_elements, threads);
            cudaStream_t stream = Stream<gpu>::GetStream(s);
            index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
            AddHaloWGrad<<<blocks, threads, 0, stream>>>(dfilter.dptr_, gwmat_ptr, 0,
                                                         stride, wshape[1], wshape.Size());
          }
        }
        if (perform_backward_bias) {
          if (perform_cuda_backward_bias) {
            auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
            int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
            Tensor<gpu, 1, uint8_t> workspace(reinterpret_cast<uint8_t*>(workspaces[2].dptr_),
                                              Shape1(workspaces[2].shape_.Size() * sizeof(DType)));
            AddBiasGrad(in_grad[spconv::kBias], y_grad, req[spconv::kBias], output_features,
                        ctx, spconv::kTempSpace, &workspace);
          } else {
            Tensor<gpu, 1, DType> gbias = in_grad[spconv::kBias].get<gpu, 1, DType>(s);
            CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                                &alpha,
                                                out_desc_,
                                                grad_ptr,
                                                req[spconv::kBias] == kAddTo ? &beta_add : &beta,
                                                bias_desc_,
                                                gbias.dptr_));
          }
        }
        if (needs_recv_previous_halo && req[spconv::kData] != kNullOp) {
          auto& workspace = workspaces[5];
          auto& filter = workspaces[4];
          index_t n_elements = halo_wshape.Size();
          const int threads = 512;
          const index_t blocks = common::div_round(n_elements, threads);
          cudaStream_t stream = Stream<gpu>::GetStream(aux_stream.GetStream());
          index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
          ExtractHaloFilter<<<blocks, threads, 0, stream>>>(filter.dptr_, wmat_ptr, wshape[1] - 1,
                                                            stride, wshape[1], wshape.Size());
          CUDNN_CALL(cudnnConvolutionBackwardData(aux_stream.GetStream()->dnn_handle_,
                                                  &alpha,
                                                  halo_filter_desc_,
                                                  filter.dptr_,
                                                  halo_out_desc_,
                                                  previous_halo_data.dptr_,
                                                  halo_back_conv_desc_,
                                                  halo_back_algo_.AlgoNumber(),
                                                  workspace.dptr_,
                                                  halo_back_workspace_byte_dgrad_,
                                                  &beta_add,
                                                  halo_in_desc_,
                                                  gdata_ptr));
        }
        if (needs_recv_next_halo && req[spconv::kData] != kNullOp) {
          auto& workspace = workspaces[5];
          auto& filter = workspaces[4];
          index_t n_elements = halo_wshape.Size();
          const int threads = 512;
          const index_t blocks = common::div_round(n_elements, threads);
          cudaStream_t stream = Stream<gpu>::GetStream(aux_stream.GetStream());
          index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
          ExtractHaloFilter<<<blocks, threads, 0, stream>>>(filter.dptr_, wmat_ptr, 0,
                                                            stride, wshape[1], wshape.Size());
          CUDNN_CALL(cudnnConvolutionBackwardData(aux_stream.GetStream()->dnn_handle_,
                                                  &alpha,
                                                  halo_filter_desc_,
                                                  filter.dptr_,
                                                  halo_out_desc_,
                                                  next_halo_data.dptr_,
                                                  halo_back_conv_desc_,
                                                  halo_back_algo_.AlgoNumber(),
                                                  workspace.dptr_,
                                                  halo_back_workspace_byte_dgrad_,
                                                  &beta_add,
                                                  halo_in_desc_,
                                                  gdata_ptr +
                                                  RemoveFirstSpatialDim(oshape).Size() *
                                                  (oshape[1] - 1)));
        }
      }
    } else {
      SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
      size_t workspace_size = TensorSizeBytes(workspace);
      DType *workspace_dptr_wgrad = workspace.dptr_;
      DType *workspace_dptr_dgrad = workspace.dptr_;
      if (parallelize_backward_kernels_) {
        CHECK_LE(back_workspace_byte_dgrad_ + back_workspace_byte_wgrad_, workspace_size);
        // Large allocations at some point will be given their own page.  Pass this alignment on to
        // the larger of the two separate dgrad/wgrad workspaces.  This probably doesn't matter, but
        // corresponds more closely to the workspace alignments used during cudnnFind. However,
        // if the cuda backward bias kernel is employed, make sure the backward filter workspace
        // pointer (shared with the cuda backward bias kernel) is not the 'stacked' workspace.
        if (back_workspace_byte_dgrad_ > back_workspace_byte_wgrad_ && !perform_cuda_backward_bias)
          workspace_dptr_wgrad = workspace.dptr_ + back_workspace_byte_dgrad_ / sizeof(DType);
        else
          workspace_dptr_dgrad = workspace.dptr_ + back_workspace_byte_wgrad_ / sizeof(DType);
      } else {
        CHECK_LE(back_workspace_byte_dgrad_, workspace_size);
        CHECK_LE(back_workspace_byte_wgrad_, workspace_size);
      }
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      if (req[spconv::kData] != kNullOp) {
          CUDNN_CALL(cudnnConvolutionBackwardData(aux_stream.GetStream()->dnn_handle_,
              &alpha,
              main_filter_desc_,
              wmat_ptr,
              out_desc_,
              grad_ptr,
              main_back_conv_desc_,
              main_back_algo_.AlgoNumber(),
              workspace_dptr_dgrad,
              back_workspace_byte_dgrad_,
              req[spconv::kData] == kAddTo? &beta_add : &beta,
              in_desc_,
              gdata_ptr));
      }
      if (req[spconv::kWeight] != kNullOp) {
          CHECK_EQ(add_to_weight_, req[spconv::kWeight] == kAddTo);
          CUDNN_CALL(cudnnConvolutionBackwardFilter(s->dnn_handle_,
              &alpha,
              in_desc_,
              data_ptr,
              out_desc_,
              grad_ptr,
              main_back_conv_desc_w_,
              main_back_algo_w_.AlgoNumber(),
              workspace_dptr_wgrad,
              back_workspace_byte_wgrad_,
              req[spconv::kWeight] == kAddTo? &beta_add : &beta,
              main_filter_desc_,
              gwmat_ptr));
      }
      if (perform_backward_bias) {
        if (perform_cuda_backward_bias) {
          auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
          int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
          AddBiasGrad(in_grad[spconv::kBias], y_grad, req[spconv::kBias], output_features,
                      ctx, spconv::kTempSpace);
        } else {
          Tensor<gpu, 1, DType> gbias = in_grad[spconv::kBias].get<gpu, 1, DType>(s);
          CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad_ptr,
                                              req[spconv::kBias] == kAddTo ? &beta_add : &beta,
                                              bias_desc_,
                                              gbias.dptr_));
        }
      }
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the convolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.  Dilation only enabled after v6.0.20.
 */
  static bool Supports(SpatialParallelConvolutionParam param,
                       int forward_compute_type,
                       int backward_compute_type,
                       int dev_id) {
    using namespace mshadow;

    // NDHWC, NHWC, NHC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (true_fp16 &&
        (layout_val == kNDHWC || layout_val == kNHWC || layout_val == kNWC))
      return false;

    // Permits graceful fallback to pseudo-fp16 on heterogenous systems
    if (!SupportsFloat16Compute(dev_id) &&
        (forward_compute_type == kFloat16 || backward_compute_type == kFloat16)) {
      return false;
    }

    return true;
  }

 private:
/*!
 * \brief Translate an mxnet datatype to the corresponding cudnnDataType_t.
 */
  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  void InitDescriptors(const mxnet::ShapeVector& in_shape,
                       const mxnet::ShapeVector& out_shape,
                       cudnnDataType_t cudnn_forward_compute_type,
                       cudnnDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 3 : 4;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    mxnet::TShape dshape = in_shape[spconv::kData];
    mxnet::TShape wshape = in_shape[spconv::kWeight];
    mxnet::TShape oshape = out_shape[spconv::kOut];
    mxnet::TShape dstride, ostride;
    CHECK(param_.layout.value() == kNDHWC ||
          param_.layout.value() == kNHWC) << "Supports only NHWC and NDHWC for now.";
    LayoutFlag halo_layout = param_.layout.value() == kNDHWC ? kNHWC : kNWC;

    mxnet::TShape halo_wshape = RemoveFirstSpatialDim(wshape);
    mxnet::TShape halo_dshape = RemoveFirstSpatialDim(dshape);
    mxnet::TShape halo_oshape = RemoveFirstSpatialDim(oshape);

    mxnet::TShape halo_dstride, halo_ostride;

    if (param_.kernel.ndim() == 1 || param_.kernel.ndim() == 2) {
      LOG(FATAL) << "1D and 2D not supported for now!";
      /*
      // 1d or 2d conv
      auto pad = param_.kernel.ndim() == 2 ?
        param_.pad : mxnet::TShape({0, param_.pad[0]});
      auto stride = param_.kernel.ndim() == 2 ?
        param_.stride : mxnet::TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ?
        param_.dilate : mxnet::TShape({1, param_.dilate[0]});
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(main_forward_conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_forward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(main_back_conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_backward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(main_back_conv_desc_w_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_backward_compute_type));
      if (param_.kernel.ndim() == 2) {
        wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), param_.layout.value(), kNCHW);
        dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);
        oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
      } else {
        wshape = ConvertLayout(wshape.get<3>(), param_.layout.value(), kNCW);
        wshape = mxnet::TShape({wshape[0], wshape[1], 1, wshape[2]});
        dstride = ConvertLayout(Strides<3>(dshape), param_.layout.value(), kNCW);
        dstride = mxnet::TShape({dstride[0], dstride[1], dstride[1], dstride[2]});
        dshape = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        dshape = mxnet::TShape({dshape[0], dshape[1], 1, dshape[2]});
        ostride = ConvertLayout(Strides<3>(oshape), param_.layout.value(), kNCW);
        ostride = mxnet::TShape({ostride[0], ostride[1], ostride[1], ostride[2]});
        oshape = ConvertLayout(oshape.get<3>(), param_.layout.value(), kNCW);
        oshape = mxnet::TShape({oshape[0], oshape[1], 1, oshape[2]});
      }
      CUDNN_CALL(cudnnSetFilter4dDescriptor(main_filter_desc_,
                                            dtype_,
                                            main_format_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
      auto kernel_h = wshape[2];
      auto kernel_w = wshape[3];
      // The 5x5 non-fused Winograd kernel is fast, but because of its reduced numerical
      // accuracy compared to other algos, users must opt-in to its use.
      bool exclude_nonfused_winograd_5x5 =
        !dmlc::GetEnv("MXNET_CUDNN_ENABLE_WINOGRAD_NONFUSED_5X5", false);
      if (exclude_nonfused_winograd_5x5 && kernel_h == 5 && kernel_w == 5) {
        excluded_forward_algos_.insert(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
        excluded_back_algos_.insert(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
        excluded_back_algos_w_.insert(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
      }
      */
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      std::vector<int> wshape_buffer(wshape.ndim());
      std::vector<int> halo_wshape_buffer(wshape.ndim() - 1);
      wshape = ConvertLayout(wshape.get<5>(), param_.layout.value(), kNCDHW);
      halo_wshape = ConvertLayout(halo_wshape.get<4>(), halo_layout, kNCHW);

      CUDNN_CALL(cudnnSetFilterNdDescriptor(main_filter_desc_,
                                            dtype_,
                                            main_format_,
                                            static_cast<int>(wshape.ndim()),
                                            CastTShapeToIntPtr(wshape, &wshape_buffer)));
      CUDNN_CALL(cudnnSetFilterNdDescriptor(halo_filter_desc_,
                                            dtype_,
                                            halo_format_,
                                            static_cast<int>(wshape.ndim() - 1),
                                            CastTShapeToIntPtr(halo_wshape, &halo_wshape_buffer)));
      // main convolution descriptors
      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(main_forward_conv_desc_,
                                                 3,
                                                 param_pad_.data(),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_forward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(main_back_conv_desc_,
                                                 3,
                                                 param_pad_.data(),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(main_back_conv_desc_w_,
                                                 3,
                                                 param_pad_.data(),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));
      // halo convolution descriptors
      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(halo_forward_conv_desc_,
                                                 2,
                                                 param_pad_.data() + 1,
                                                 param_stride_.data() + 1,
                                                 param_dilate_.data() + 1,
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_forward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(halo_back_conv_desc_,
                                                 2,
                                                 param_pad_.data() + 1,
                                                 param_stride_.data() + 1,
                                                 param_dilate_.data() + 1,
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(halo_back_conv_desc_w_,
                                                 2,
                                                 param_pad_.data() + 1,
                                                 param_stride_.data() + 1,
                                                 param_dilate_.data() + 1,
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));

      dstride = ConvertLayout(Strides<5>(dshape), param_.layout.value(), kNCDHW);
      dshape = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
      halo_dstride = ConvertLayout(Strides<4>(halo_dshape), halo_layout, kNCHW);
      halo_dshape = ConvertLayout(halo_dshape.get<4>(), halo_layout, kNCHW);

      ostride = ConvertLayout(Strides<5>(oshape), param_.layout.value(), kNCDHW);
      oshape = ConvertLayout(oshape.get<5>(), param_.layout.value(), kNCDHW);
      halo_ostride = ConvertLayout(Strides<4>(halo_oshape), halo_layout, kNCHW);
      halo_oshape = ConvertLayout(halo_oshape.get<4>(), halo_layout, kNCHW);
    }
    // Set "allow tensor core" flag in convolution descriptors, if available.
    cudnnMathType_t math_type = param_.cudnn_tensor_core.value() ? CUDNN_TENSOR_OP_MATH
                                                                 : CUDNN_DEFAULT_MATH;
#if CUDNN_VERSION >= 7200
    if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
        (DataType<DType>::kFlag != kFloat16))
      math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#endif
    CUDNN_CALL(cudnnSetConvolutionMathType(main_forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(main_back_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(main_back_conv_desc_w_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(halo_forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(halo_back_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(halo_back_conv_desc_w_, math_type));
    CHECK_EQ(param_.num_group, 1) << "Does not support grouped convolutions for now.";
    CUDNN_CALL(cudnnSetConvolutionGroupCount(main_forward_conv_desc_, param_.num_group));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(main_back_conv_desc_, param_.num_group));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(main_back_conv_desc_w_, param_.num_group));

    SetTensorDescriptor(in_desc_, dtype_, dshape, dstride);
    SetTensorDescriptor(out_desc_, dtype_, oshape, ostride);
    SetTensorDescriptor(halo_in_desc_, dtype_, halo_dshape, halo_dstride);
    SetTensorDescriptor(halo_out_desc_, dtype_, halo_oshape, halo_ostride);


    if (!param_.no_bias) {
      mxnet::TShape bias = in_shape[spconv::kBias];
      int bias_dim = static_cast<int>(bias[0]);
      std::vector<int> bias_shape = {1,
                                     bias_dim,
                                     1, 1};
      std::vector<int> bias_stride = {bias_dim, 1, bias_dim, bias_dim};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(bias_dim);
      }
      CUDNN_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                          dtype_,
                                          static_cast<int>(bias_shape.size()),
                                          &bias_shape[0],
                                          &bias_stride[0]));
    }
  }

  void CuDNNAlgoSetter(const RunContext& rctx,
                  const mxnet::ShapeVector& in_shape,
                  const mxnet::ShapeVector& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type,
                  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
    // Not in algo registry, must determine via *Get*() or *Find*()
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));

    // Since the function signature of *Get*_v7() matches that of *Find*(),
    // we can unify the find-vs-get logic by using function pointers.

    // Forward Algorithm Find/Get() v7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(s->dnn_handle_));
    int actual_fwd_algos = 0;
    auto fwd_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionForwardAlgorithm_v7
                                              : cudnnFindConvolutionForwardAlgorithm;
    CUDNN_CALL((*fwd_algo_discoverer)(s->dnn_handle_,
                                      in_desc_,
                                      main_filter_desc_,
                                      main_forward_conv_desc_,
                                      out_desc_,
                                      fwd_results.size(),
                                      &actual_fwd_algos,
                                      fwd_results.data()));
    fwd_results.resize(actual_fwd_algos);
    AlgoFinalSelect<cudnnConvolutionFwdAlgoPerf_t,
                    cudnnConvolutionFwdAlgo_t>(fwd_results, "forward",
                                               param_.cudnn_algo_fwd, workspace_byte,
                                               fwd, excluded_forward_algos_);

    // Backprop-to-Filter Algorithm Find/Get() v7
    auto max_bwd_filt_algos = MaxBackwardFilterAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filt_results(max_bwd_filt_algos);
    int actual_bwd_filter_algos = 0;
    // In cudnn v7.1.4, find() returned wgrad algos that could fail for large c if we
    // were summing into the output (i.e. beta != 0).  Get() returned OK algos though.
    auto bwd_filter_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionBackwardFilterAlgorithm_v7
                                              : cudnnFindConvolutionBackwardFilterAlgorithm;
    CUDNN_CALL((*bwd_filter_algo_discoverer)(s->dnn_handle_,
                                             in_desc_,
                                             out_desc_,
                                             main_back_conv_desc_w_,
                                             main_filter_desc_,
                                             bwd_filt_results.size(),
                                             &actual_bwd_filter_algos,
                                             bwd_filt_results.data()));
    bwd_filt_results.resize(actual_bwd_filter_algos);
    AlgoFinalSelect<cudnnConvolutionBwdFilterAlgoPerf_t,
                    cudnnConvolutionBwdFilterAlgo_t>(bwd_filt_results, "backprop-to-filter",
                                                     param_.cudnn_algo_bwd_filter, workspace_byte,
                                                     flt, excluded_back_algos_w_);

    // Backprop-to-Data Algorithm Find/Get() v7
    auto max_bwd_data_algos = MaxBackwardDataAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_results(max_bwd_data_algos);
    int actual_bwd_data_algos = 0;
    auto bwd_data_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionBackwardDataAlgorithm_v7
                                              : cudnnFindConvolutionBackwardDataAlgorithm;
    CUDNN_CALL((*bwd_data_algo_discoverer)(s->dnn_handle_,
                                           main_filter_desc_,
                                           out_desc_,
                                           main_back_conv_desc_,
                                           in_desc_,
                                           bwd_data_results.size(),
                                           &actual_bwd_data_algos,
                                           bwd_data_results.data()));
    bwd_data_results.resize(actual_bwd_data_algos);
    AlgoFinalSelect<cudnnConvolutionBwdDataAlgoPerf_t,
                    cudnnConvolutionBwdDataAlgo_t>(bwd_data_results, "backprop-to-data",
                                                   param_.cudnn_algo_bwd_data, workspace_byte,
                                                   bwd, excluded_back_algos_);

    // Fix for issue #11241
    int cudnn_find_issue_max_features = 64 * 1024;
    if (add_to_weight_ && Features(in_shape[spconv::kData]) >= cudnn_find_issue_max_features) {
      flt->Set(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, true);
    }

    if (param_.cudnn_algo_verbose) {
      std::string key = CuDNNSPConvAlgoReg::Get()->ToString(param_, in_shape, out_shape, dtype_,
                                                            cudnn_forward_compute_type,
                                                            cudnn_backward_compute_type,
                                                            SMArch(rctx.ctx.dev_id));
      LOG(INFO) << "Algo selection for convolution: " << key;
      LOG(INFO) << "            forward : " << fwd->AlgoNumber() <<
                TensorCoreStr(rctx.ctx, fwd->IsTensorCoreAlgo());
      LOG(INFO) << "    backprop-to-data: " << bwd->AlgoNumber() <<
                TensorCoreStr(rctx.ctx, bwd->IsTensorCoreAlgo());
      LOG(INFO) << "  backprop-to-filter: " << flt->AlgoNumber() <<
                TensorCoreStr(rctx.ctx, flt->IsTensorCoreAlgo());
      LOG(INFO) << "";
    }
  }

  void SelectMainAlgo(const RunContext& rctx,
                      const mxnet::ShapeVector& in_shape,
                      const mxnet::ShapeVector& out_shape,
                      cudnnDataType_t cudnn_forward_compute_type,
                      cudnnDataType_t cudnn_backward_compute_type) {
    SelectAlgo(rctx, in_shape, out_shape, cudnn_forward_compute_type,
               cudnn_backward_compute_type, param_, dtype_, add_to_weight_,
               &main_forward_algo_, &main_back_algo_, &main_back_algo_w_,
               main_forward_conv_desc_, main_back_conv_desc_, main_back_conv_desc_w_);
  }

  void SelectHaloAlgo(const RunContext& rctx,
                      const mxnet::ShapeVector& in_shape,
                      const mxnet::ShapeVector& out_shape,
                      cudnnDataType_t cudnn_forward_compute_type,
                      cudnnDataType_t cudnn_backward_compute_type) {
    SpatialParallelConvolutionParam halo_param = param_;
    halo_param.kernel = mxnet::TShape(param_.kernel.begin() + 1, param_.kernel.end());
    halo_param.stride = mxnet::TShape(param_.stride.begin() + 1, param_.stride.end());
    halo_param.dilate = mxnet::TShape(param_.dilate.begin() + 1, param_.dilate.end());
    halo_param.pad = mxnet::TShape(param_.pad.begin() + 1, param_.pad.end());
    SelectAlgo(rctx, in_shape, out_shape, cudnn_forward_compute_type,
               cudnn_backward_compute_type, halo_param, dtype_, add_to_weight_,
               &halo_forward_algo_, &halo_back_algo_, &halo_back_algo_w_,
               halo_forward_conv_desc_, halo_back_conv_desc_, halo_back_conv_desc_w_);
  }

  void SelectAlgo(const RunContext& rctx,
                  const mxnet::ShapeVector& in_shape,
                  const mxnet::ShapeVector& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type,
                  const SpatialParallelConvolutionParam& param,
                  cudnnDataType_t dtype,
                  bool add_to_weight,
                  CuDNNAlgo<cudnnConvolutionFwdAlgo_t>* forward_algo,
                  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t>* back_algo,
                  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t>* back_algo_w,
                  cudnnConvolutionDescriptor_t forward_conv_desc,
                  cudnnConvolutionDescriptor_t back_conv_desc,
                  cudnnConvolutionDescriptor_t back_conv_desc_w) {
    auto algo_setter = [&](CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                           CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                           CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
      if (param_.cudnn_tune.value() == spconv::kOff) {
        // The routine will only be calling cudnnGet, so no need to grab the Storage lock.
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      } else {
        // One potential problem is that cudnnFind() uses cudaMalloc() to directly allocate
        // I/O and workspace areas, and these allocations may result in an out-of-memory
        // error even though the StorageMangager free pool is not empty.  Ideally, cudnnFind
        // would use MXNet's storage allocator for its I/O and workspace areas, instead of using
        // the area carved out by MXNET_GPU_MEM_POOL_RESERVE.
        // To get somewhat the same effect as this, we can pre-allocate the areas needed for the
        // I/Os (possibly triggering a desirable StorageManager::ReleaseAll()), followed by a
        // DirectFree(), which makes these areas available for cudnn's subsequent cudaMalloc().

        // Allocate for x (or dx), w (or dw) and y (or dy).
        ReserveElements({in_shape[spconv::kData].Size(),
                         in_shape[spconv::kWeight].Size(),
                         out_shape[spconv::kOut].Size()});

        // We're about to call cudnnFind so we need to quiet the system by grabbing
        // the Storage lock.  Concurrent cudaMalloc's can disrupt the accurate timing
        // measurements of the algos, and can prevent the cuda driver's proper freeing
        // of cudnnFind's internal temporary allocations.  Grabbing the lock might also
        // impede other threads from launching work on the GPU.
        std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      }
    };

    CuDNNSPConvAlgoReg::Get()->FindOrElseRegister(param, in_shape, out_shape, dtype,
                                                  cudnn_forward_compute_type,
                                                  cudnn_backward_compute_type,
                                                  SMArch(rctx.ctx.dev_id), add_to_weight,
                                                  forward_algo, back_algo,
                                                  back_algo_w, algo_setter);

    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc, forward_algo->MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc, back_algo->MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w, back_algo_w->MathType()));
  }

  // Convert the `is_tensor_core_algo` flag to a string for verbose-mode output
  std::string TensorCoreStr(const Context& ctx, bool is_tensor_core_algo) {
    // GPU's before Volta (sm_70) would not be expected to run Tensor Core algos,
    // so we don't report whether the algo is/is-not Tensor Core in that case.
    if (!SupportsTensorCore(ctx.dev_id))
      return std::string("");
    else if (is_tensor_core_algo)
      return std::string(" (Tensor Core)");
    else
      return std::string(" (not Tensor Core)");
  }

  std::string FixedFormat(float f, int width, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << std::setw(width) << f;
    return ss.str();
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints and a possible user algo preference.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results, std::string kernel_name,
                       int32_t algo_preference, size_t workspace_byte, CuDNNAlgo<AlgoType> *algo,
                       const std::set<AlgoType> &excluded_algos) {
    // Determine the fastest acceptable algo that matches the algo_preference (-1 = any),
    // regardless of mathType.
    auto mode = param_.cudnn_tune.value() == spconv::kOff ? " get " : " find ";
    if (param_.cudnn_algo_verbose && dmlc::GetEnv("MXNET_CUDNN_ALGO_VERBOSE_LEVEL", 1) >= 2) {
      LOG(INFO) << "Full results of algo" << mode << kernel_name << ":";
      for (const auto &result : perf_results) {
        auto math_type_str = "-";
          if (result.mathType == CUDNN_TENSOR_OP_MATH)
            math_type_str = "+";
        LOG(INFO) << "    algo: " << result.algo <<
                     ", TC" << math_type_str <<
                     ", time: " << FixedFormat(result.time, 7, 3) << "ms" <<
                     ", wksp = " << result.memory <<
                     ", status = " << result.status;
      }
    }

    bool enforce_determinism = dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false);
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      bool algo_exclusion =
          param_.cudnn_tensor_core_only && !algo_is_tensor_core ||
          (result.algo != algo_preference) && (excluded_algos.count(result.algo) != 0);
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (!enforce_determinism || result.determinism == cudnnDeterminism_t::CUDNN_DETERMINISTIC) &&
          (param_.cudnn_tune.value() == spconv::kFastest || result.memory <= workspace_byte) &&
          (algo_preference == -1 || algo_preference == result.algo) &&
          !algo_exclusion) {
        // Fix for a current cuDNNv7 behavior where algos are reported twice
        // with equivalent performance (both as Tensor Core and not Tensor Core).
        if ((result.mathType == CUDNN_TENSOR_OP_MATH) &&
             (i != perf_results.size() - 1) &&
             !param_.cudnn_tensor_core_only) {
          const auto &next_result = perf_results[i+1];
          if (next_result.status == CUDNN_STATUS_SUCCESS &&
              next_result.algo == result.algo &&
              next_result.memory == result.memory &&
              next_result.mathType != CUDNN_TENSOR_OP_MATH &&
              next_result.time < ALGO_PERF_THRESHOLD * result.time) {
              // Skip over this result- it's not really a Tensor Core algo.
              // Prefer instead the next equivalent non-Tensor Core algo.
                continue;
          }
        }
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    if (algo_preference != -1)
      LOG(FATAL) << "Failed to" << mode << kernel_name
                 << " convolution algorithm " << algo_preference
                 << " with workspace size of " << workspace_byte << " bytes,"
                 << " please consider reducing batch/model size or increasing the workspace size";
    else
      LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " convolution algorithm"
                 << " with workspace size of " << workspace_byte << " bytes,"
                 << " please consider reducing batch/model size or increasing the workspace size";
  }


  void GetTempSize(const RunContext& rctx) {
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
    if (!dgrad_as_gemm_) {
        CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               main_filter_desc_,
               out_desc_,
               main_back_conv_desc_,
               in_desc_,
               main_back_algo_.AlgoNumber(),
               &back_workspace_byte_dgrad_));
        CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               halo_filter_desc_,
               halo_out_desc_,
               halo_back_conv_desc_,
               halo_in_desc_,
               halo_back_algo_.AlgoNumber(),
               &halo_back_workspace_byte_dgrad_));
    } else {
        back_workspace_byte_dgrad_ = 0;
    }
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               main_back_conv_desc_w_,
               main_filter_desc_,
               main_back_algo_w_.AlgoNumber(),
               &back_workspace_byte_wgrad_));
    size_t halo_back_workspace_byte_wgrad;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               halo_in_desc_,
               halo_out_desc_,
               halo_back_conv_desc_w_,
               halo_filter_desc_,
               halo_back_algo_w_.AlgoNumber(),
               &halo_back_workspace_byte_wgrad));
    back_workspace_byte_wgrad_ = std::max(back_workspace_byte_wgrad_,
                                          halo_back_workspace_byte_wgrad);
    // cudaMalloc returns addresses that are aligned for large accesses (e.g. to 512 bytes).
    // Since we only make one allocation and divide it into two parts when we parallelize
    // the dgrad and wgrad kernels, we round the sizes up to this alignment size so the
    // dptrs respect this alignment, even if the separate areas are stacked.
    const size_t dptr_alignment = 512;
    back_workspace_byte_dgrad_ = RoundToMultiple(back_workspace_byte_dgrad_, dptr_alignment);
    back_workspace_byte_wgrad_ = RoundToMultiple(back_workspace_byte_wgrad_, dptr_alignment);

    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               main_filter_desc_,
               main_forward_conv_desc_,
               out_desc_,
               main_forward_algo_.AlgoNumber(),
               &forward_workspace_byte_));
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               halo_in_desc_,
               halo_filter_desc_,
               halo_forward_conv_desc_,
               halo_out_desc_,
               halo_forward_algo_.AlgoNumber(),
               &halo_forward_workspace_byte_));
  }

  int *CastTShapeToIntPtr(const mxnet::TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = nullptr;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Converts a mxnet::TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const mxnet::TShape &s) {
    int ndim = s.ndim();
    mxnet::TShape strides(ndim, -1);
    for (int i = 0; i != ndim; ++i)
      strides[i] = s.ProdShape(i+1, ndim);
    return strides.get<dim>();
  }

  void InitBufferForParam() {
    CastTShapeToIntPtr(param_.stride, &param_stride_);
    CastTShapeToIntPtr(param_.dilate, &param_dilate_);
    CastTShapeToIntPtr(param_.pad, &param_pad_);
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  static size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words =
      std::max<size_t>(1, RoundToMultiple(size_bytes, sizeof(DType)) / sizeof(DType));
    return ctx.requested[spconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(size_words), s);
  }

  std::vector<std::vector<mshadow::Tensor<gpu, 1, DType>>>
  AllocateTempWorkspaces(const OpContext &ctx,
                         const std::vector<std::vector<size_t>> &sizes_bytes_stages) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t max_total_size = 0;
    std::vector<std::vector<mshadow::Tensor<gpu, 1, DType>>> ret;
    std::vector<std::vector<size_t>> rounded_sizes_in_words_stages;
    // Get maximum size
    for (const auto& sizes_bytes : sizes_bytes_stages) {
      rounded_sizes_in_words_stages.emplace_back();
      auto& rounded_sizes_in_words = rounded_sizes_in_words_stages.back();
      std::transform(sizes_bytes.cbegin(), sizes_bytes.cend(),
                     std::back_inserter(rounded_sizes_in_words),
                     [](const size_t s) -> std::size_t {
                       const size_t dptr_alignment = 512 / sizeof(DType);
                       size_t size_in_words = std::max<size_t>(1,
                                                RoundToMultiple(s, sizeof(DType)) /
                                                sizeof(DType));
                       size_t aligned = RoundToMultiple(size_in_words, dptr_alignment);
                       return aligned;
                     });
      size_t total_size = 0;
      for (const size_t s : rounded_sizes_in_words) {
        total_size += s;
      }
      max_total_size = std::max(max_total_size, total_size);
    }
    auto total_storage = ctx.requested[spconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(max_total_size), s);
    for (const auto& rounded_sizes_in_words : rounded_sizes_in_words_stages) {
      ret.emplace_back();
      auto& current = ret.back();
      DType *ptr = total_storage.dptr_;
      std::transform(rounded_sizes_in_words.cbegin(), rounded_sizes_in_words.cend(),
                     std::back_inserter(current),
                     [&ptr](const size_t s) -> mshadow::Tensor<gpu, 1, DType> {
                       mshadow::Tensor<gpu, 1, DType> t(ptr, mshadow::Shape1(s));
                       ptr += s;
                       return t;
                     });
    }
    return ret;
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }

  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const mxnet::TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3: c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1]; break;
      case 4: c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1]; break;
      case 5: c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1]; break;
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Does the operation's data layout have the features dimension 'c' last?
  bool FeaturesLastLayout() {
    return param_.layout.value() == kNWC ||
           param_.layout.value() == kNHWC ||
           param_.layout.value() == kNDHWC;
  }

    // Give a tensor shape of this operation, return the N * H * W
  int64_t GetNHW(const TShape &dshape) {
    int nhw = 0;
    switch (dshape.ndim()) {
      case 3:
      {
         auto tmp = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
         nhw = tmp[0] * tmp[2];
         break;
      }
      case 4:
      {
        auto tmp = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        nhw = tmp[0] * tmp[2] * tmp[3];
        break;
      }
      case 5:
      {
        auto tmp = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
        nhw = tmp[0] * tmp[3] * tmp[4];
        break;
      }
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return nhw;
  }

  // Make a number of allocations and directly free them, ensuring room for an equivalent set of
  // cudaMalloc() calls by (say) cudnnFind().  `elements` spec the alloc size in DTypes, not bytes.
  void ReserveElements(const std::vector<size_t> &elements) {
    std::vector<Storage::Handle> handles;
    for (size_t alloc_element : elements)
        handles.push_back(Storage::Get()->Alloc(alloc_element * sizeof(DType), Context::GPU()));
    for (auto &handle : handles)
        Storage::Get()->DirectFree(handle);
  }

  // Log that no suitable algo was found that met the workspace constraints, then exit.
  void LogNoSuitableAlgoAndExit(int num_algos_tried, size_t min_memory_needs,
                                size_t workspace_byte, std::string algo_kind) {
    LOG(FATAL) << num_algos_tried << " " << algo_kind << " with minimum memory requirement "
               << min_memory_needs << " bytes have been tried. Workspace size is set to "
               << workspace_byte << " bytes, please consider reducing the batch/model size, "
               << "or increasing workspace size.";
  }

  void SetTensorDescriptor(cudnnTensorDescriptor_t desc,
                           cudnnDataType_t dtype,
                           const mxnet::TShape& shape,
                           const mxnet::TShape& strides) {
    std::vector<int> shape_buffer(shape.ndim());
    nnvm::ShapeTypeCast(shape.begin(), shape.end(), shape_buffer.data());
    std::vector<int> stride_buffer(strides.ndim());
    nnvm::ShapeTypeCast(strides.begin(), strides.end(), stride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(desc,
                                          dtype,
                                          static_cast<int>(shape.ndim()),
                                          shape_buffer.data(),
                                          stride_buffer.data()));
  }

  mxnet::TShape RemoveFirstSpatialDim(const mxnet::TShape& shape) {
    mxnet::TShape ret(shape.ndim() - 1, -1);

    for (int i = 0, count = 0; i < shape.ndim(); ++i) {
      // Remove D or H
      if (i != 1) {
        ret[count] = shape[i];
        ++count;
      }
    }

    return ret;
  }

  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;
  std::vector<int> param_pad_;

  // Temp workspace size in bytes needed for Forward() operation.
  size_t forward_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() dgrad (data gradient) operation.
  size_t back_workspace_byte_dgrad_;
  // Temp workspace size in bytes needed for Backward() wgrad (weight gradient) operation.
  size_t back_workspace_byte_wgrad_;
  // Temp workspace size in bytes needed for Forward() operation for halo.
  size_t halo_forward_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() dgrad (data gradient) operation for halo.
  size_t halo_back_workspace_byte_dgrad_;
  // If backward bias is needed, has the workspace size been requested
  bool back_bias_get_workspace_performed_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t halo_in_desc_;
  cudnnTensorDescriptor_t halo_out_desc_;
  cudnnFilterDescriptor_t main_filter_desc_;
  cudnnFilterDescriptor_t halo_filter_desc_;
  // Convolution descriptor for forward inference operation
  cudnnConvolutionDescriptor_t main_forward_conv_desc_;
  cudnnConvolutionDescriptor_t halo_forward_conv_desc_;
  // Convolution descriptor for back-prop operations to the data
  cudnnConvolutionDescriptor_t main_back_conv_desc_;
  cudnnConvolutionDescriptor_t halo_back_conv_desc_;
  // Convolution descriptor for back-prop operations to the weights
  cudnnConvolutionDescriptor_t main_back_conv_desc_w_;
  cudnnConvolutionDescriptor_t halo_back_conv_desc_w_;
  // Should dgrad and wgrad be launched into separate streams
  bool parallelize_backward_kernels_;
  // Algorithm for the forward inference operation
  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> main_forward_algo_;
  // Algorithm for the back-prop operation to the data
  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> main_back_algo_;
  // Algorithm for the back-prop operation to the weights
  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> main_back_algo_w_;
  // Algorithm for the forward inference operation for halos
  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> halo_forward_algo_;
  // Algorithm for the back-prop operation to the data for halos
  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> halo_back_algo_;
  // Algorithm for the back-prop operation to the weights for halos
  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> halo_back_algo_w_;
  // compute precision of backward operation
  int backward_compute_type_;
  cudnnTensorFormat_t main_format_;
  cudnnTensorFormat_t halo_format_;
  SpatialParallelConvolutionParam param_;
  // Is req[kWeight] == spconv::kAddTo ?
  bool add_to_weight_;
  // forward algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionFwdAlgo_t> excluded_forward_algos_;
  // dgrad algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionBwdDataAlgo_t> excluded_back_algos_;
  // wgrad algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionBwdFilterAlgo_t> excluded_back_algos_w_;
  bool dgrad_as_gemm_;
  bool dgrad_as_gemm_debug_;
};

#endif  // CUDNN && NCCL
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
