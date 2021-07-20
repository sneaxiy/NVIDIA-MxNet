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
 * \file nhwc_batch_norm-inl.h
 * \brief
 * \author Dick Carter and Junyuan Xie
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
#include <mxnet/storage.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include <algorithm>
#include "../batch_norm-inl.h"
#include "nhwc_batch_norm.h"
#include "../../../common/cuda/utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
namespace nhwcbatchnorm {
enum NhwcBatchNormOpInputs {kData, kGamma, kBeta};
enum NhwcBatchNormOpOutputs {kOut, kMean, kInvVar};
enum NhwcBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
enum NhwcBatchNormOpResource {kTempSpace};
}  // namespace nhwcbatchnorm

#if defined(__CUDACC__)
template<typename DType>
class NhwcBatchNormOp {
 public:
  typedef typename mshadow::DataType<DType>::ScaleType ScaleType;
  NhwcBatchNormOp() {
    using namespace mshadow;
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;
  }

  void Init(const BatchNormParam &param) {
    if (param.act_type.has_value())
      CHECK_EQ(param.act_type.value(), activation::kReLU) <<
        "Only ReLU activation fusion supported.";
    if (!nhwc_layer_)
      nhwc_layer_.reset(new NhwcBatchNorm(param.bn_group));

    this->param_ = param;

    workspace_total_bytes_ = 0;
  }

  ~NhwcBatchNormOp() {
    if (retired_ctas_hdl_.dptr != nullptr) {
      // We're probably at program exit, bypass the conventional approach of
      //     mxnet::Storage::Get()->DirectFree(retired_ctas_hdl_);
      CUDA_CALL(cudaFree(retired_ctas_hdl_.dptr));
      retired_ctas_hdl_.dptr = nullptr;
    }
  }

  void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);

    // This routine may be used by an InstanceNorm, whose definition does
    // not include a moving_mean and moving_var state machine.  This case
    // is identified here by detecting a 0-length aux_states TBlob vector.
    // InstanceNorm always normalizes based on the actual input statistics, as
    // performed by the fwd_training kernel, independent of the ctx.is_train state.

    bool invoke_fwd_training_kernel = ctx.is_train || aux_states.size() == 0;

    // Both fwd() and fwdInference() kernels require the moving stats tensors.
    // The InstanceNorm use-case will not supply them, and does not care how
    // they are modified.  Generate a dummy tensor as a bit-bucket in this case.
    bool supply_moving_stats_tensor = aux_states.size() == 0;

    CHECK(aux_states.size() == 0U || aux_states.size() == 2U);
    if (invoke_fwd_training_kernel) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
    }
    // InstanceNorm::Forward() in an inference model may call this routine with kWriteInplace.
    if (!(req[nhwcbatchnorm::kOut] == kWriteTo ||
          req[nhwcbatchnorm::kOut] == kWriteInplace))
      LOG(FATAL) << "Unexpected req value for output: " << req[nhwcbatchnorm::kOut];
    CHECK_GE(in_data[nhwcbatchnorm::kData].ndim(), 2);
    CHECK_LE(in_data[nhwcbatchnorm::kData].ndim(), 5);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Init(in_data[nhwcbatchnorm::kData].shape_, ctx.run_ctx.ctx.dev_id,
            true, invoke_fwd_training_kernel);
    int features = in_data[nhwcbatchnorm::kData].shape_[param_.axis];

    int dim = in_data[nhwcbatchnorm::kData].ndim();
    DType *nhwc_X = GetNdPtr(in_data[nhwcbatchnorm::kData], dim, s);
    DType *nhwc_dX = nullptr;
    DType *nhwc_Y = GetNdPtr(out_data[nhwcbatchnorm::kOut], dim, s);
    DType *nhwc_dY = nullptr;

    nhwc_layer_->setInputOutputPointers(nhwc_X, nhwc_dX, nhwc_Y, nhwc_dY);

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[nhwcbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> beta =
        in_data[nhwcbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      std::vector<void *> weights = {gamma.dptr_, beta.dptr_};
      std::vector<void *> dweights = {nullptr, nullptr};
      nhwc_layer_->setWeightPointers(weights, dweights);

      // No more workspace_total_bytes_ additions, so now allocate the workspace
      mshadow::Tensor<gpu, 1, ScaleType> temp_space =
          this->AllocateTempWorkspace(ctx, workspace_total_bytes_);

      if (supply_moving_stats_tensor) {
        // One dummy allocation is fine as a bit-bucket for both moving_mean and moving_inv_std
        void *dptr = reinterpret_cast<char *>(temp_space.dptr_) + dummy_moving_stats_offset_;
        nhwc_layer_->setParameterPointers({dptr, dptr});
        param_.momentum = 0.0f;
      } else {
        // moving_mean and moving_inv_std tensors are supplied by the model
        Tensor<gpu, 1, DTypeParam> moving_mean =
          aux_states[nhwcbatchnorm::kMovingMean]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        Tensor<gpu, 1, DTypeParam> moving_inv_var =
          aux_states[nhwcbatchnorm::kMovingInvVar]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        nhwc_layer_->setParameterPointers({moving_mean.dptr_, moving_inv_var.dptr_});
      }

      if (param_.fix_gamma) gamma = 1.f;

      // MXNet already has allocated Tensors for the minibatch mean and variance.
      // These are the first two sizes returned by NhwcBatchNorm::numWorkspaceBytes
      // The other sizes must be allocated explicitly as temp workspace tensors
      std::vector<void*> workspace;

      if (invoke_fwd_training_kernel) {
        Tensor<gpu, 1, DTypeParam> save_mean =
          out_data[nhwcbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        Tensor<gpu, 1, DTypeParam> save_inv_var =
          out_data[nhwcbatchnorm::kInvVar]
            .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        workspace.push_back(save_mean.dptr_);
        workspace.push_back(save_inv_var.dptr_);
        workspace.push_back(GetRetiredCtasDptr(s));
        nhwc_layer_->setBNGroup(param_.bn_group, reinterpret_cast<void**>(param_.xbuf_ptr),
                                ctx.run_ctx.ctx.dev_id);
      } else {
        workspace.push_back(nullptr);
        workspace.push_back(nullptr);
        workspace.push_back(nullptr);
      }

      for (auto offset : workspace_byte_offsets_) {
        void *dptr = reinterpret_cast<char *>(temp_space.dptr_) + offset;
        workspace.push_back(dptr);
      }

      nhwc_layer_->setWorkspacePointers(workspace);

      bool fuse_relu = param_.act_type.has_value() &&
        (param_.act_type.value() == activation::kReLU);
      if (invoke_fwd_training_kernel) {
        nhwc_layer_->fwd(Stream<gpu>::GetStream(s), fuse_relu, ctx.run_ctx.ctx.dev_id);
      } else {
        nhwc_layer_->fwdInference(s, fuse_relu, ctx.run_ctx.ctx.dev_id);
      }
    })
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs,
                size_t inuse_tempspace_bytes = 0U) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // Some invocations of Backward() may include the moving_mean and moving_in_var
    // as the last 2 of 8 inputs, but these are not used by this impl, so not required.
    CHECK_GE(inputs.size(), 6U);
    CHECK_EQ(outputs.size(), 3U);

    // Rename the inputs and outputs.
    const TBlob &out_grad = inputs[0];
    const TBlob &out_mean = inputs[1];
    const TBlob &out_var = inputs[2];
    const TBlob &in_data = inputs[3];
    const TBlob &in_gamma = inputs[4];
    const TBlob &in_beta = inputs[5];   // not sure if this is right @TODO check
    const std::vector<TBlob> &in_grad = outputs;

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Init(in_data.shape_, ctx.run_ctx.ctx.dev_id, false);

//    CHECK(ctx.is_train && !param_.use_global_stats)
//        << "use global statistics is not yet supported in CuDNNBatchNorm";

    int dim = in_data.ndim();
    DType *nhwc_X = GetNdPtr(in_data, dim, s);
    DType *nhwc_dX = GetNdPtr(in_grad[nhwcbatchnorm::kData], dim, s);
    DType *nhwc_Y = nullptr;
    DType *nhwc_dY = GetNdPtr(out_grad, dim, s);
    nhwc_layer_->setInputOutputPointers(nhwc_X, nhwc_dX, nhwc_Y, nhwc_dY);

    int features = in_data.shape_[param_.axis];

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_gamma.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      Tensor<gpu, 1, DTypeParam> beta =
        in_beta.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      Tensor<gpu, 1, DTypeParam> dbeta =
        in_grad[nhwcbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> dgamma =
        in_grad[nhwcbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_mean =
        out_mean.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_inv_var =
        out_var.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      // beta.dptr_ only needed if fused relu.  OK to set always though.
      std::vector<void *> weights = {gamma.dptr_, beta.dptr_};
      std::vector<void *> dweights = {dgamma.dptr_, dbeta.dptr_};
      nhwc_layer_->setWeightPointers(weights, dweights);

      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);

      // MXNet already has allocated Tensors for the minibatch mean and variance.
      // These are the first two sizes returned by NhwcBatchNormSerialLayer::numWorkspaceBytes
      // The other sizes must be allocated explicitly as temp workspace tensors

      std::vector<void*> workspace;
      workspace.push_back(save_mean.dptr_);
      workspace.push_back(save_inv_var.dptr_);
      workspace.push_back(GetRetiredCtasDptr(s));

      mshadow::Tensor<gpu, 1, ScaleType> temp_space =
          this->AllocateTempWorkspace(ctx, workspace_total_bytes_ + inuse_tempspace_bytes);

      for (auto offset : workspace_byte_offsets_) {
        void *dptr = reinterpret_cast<char *>(temp_space.dptr_) + inuse_tempspace_bytes + offset;
        workspace.push_back(dptr);
      }
      nhwc_layer_->setWorkspacePointers(workspace);

      if (param_.fix_gamma) gamma = 1.f;
      bool fuse_relu = param_.act_type.has_value() &&
        (param_.act_type.value() == activation::kReLU);
      nhwc_layer_->setBNGroup(param_.bn_group, reinterpret_cast<void**>(param_.xbuf_ptr),
                              ctx.run_ctx.ctx.dev_id);
      nhwc_layer_->dgrad(Stream<gpu>::GetStream(s), fuse_relu, ctx.run_ctx.ctx.dev_id);
      if (param_.fix_gamma) dgamma = 0.f;
    })
  }

/*!
 * \brief Returns whether the nhwc kernel supports the batchnorm
 * operation described by `param`.
 */
  static bool Supports(const BatchNormParam &param, int dtype, const TShape& shape,
                       const Context& ctx) {
    // Axis parameters are in the range [0,shape.ndim()-1].
    using namespace mxnet::common::cuda;
    int dim = shape.ndim();
    bool retVal = !param.use_global_stats &&
           (dim >= 3 && dim <= 5) &&
           dtype == mshadow::kFloat16 &&
           (param.axis == dim - 1 || param.axis == -1) &&
           shape[dim - 1] % 4 == 0 &&
           SupportsCooperativeLaunch(ctx.dev_id);
    return retVal;
  }

  // Return the maximum workspace size in bytes needed by Forward() (if is_fwd == true), or
  // needed by Backward() (if is_fwd == false).
  // Note: this routine alters the object state.
  size_t GetWorkspaceSizeBytes(const TShape &in_shape, mshadow::Stream<gpu> *s,
                               bool is_fwd, bool is_train = true) {
    // Make sure Init(param) has been called
    CHECK(nhwc_layer_ != nullptr);

    Init(in_shape, is_fwd, is_train);
    return workspace_total_bytes_;
  }

 private:
  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, mshadow::Stream<gpu> *s) {
    DType *data_ptr = NULL;
    if (dim == 2) {
      mshadow::Tensor<gpu, 2, DType> data = tb.get<gpu, 2, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 3) {
      mshadow::Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      mshadow::Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      mshadow::Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 2, 3, 4 or 5.";
    }
    return data_ptr;
  }

  void Init(const TShape &in_shape, int device_id, bool is_fwd, bool is_train = true) {
      int dim = in_shape.ndim();
      CHECK(dim >= 3 && dim <= 5) << "NHWC BatchNorm Op data dim " << dim
                                  << " not in supported range [3,5].";
      param_.axis = batchnorm::GetRealAxis(in_shape, param_.axis);
      auto supported_axis = dim - 1;
      CHECK_EQ(param_.axis, supported_axis)
        << "axis param must be " << supported_axis << ", found: " << param_.axis;

      int c = in_shape[supported_axis];
      double exp_avg_factor = 1.0 - param_.momentum;
      double eps = param_.eps;
      int bn_group = is_train ? param_.bn_group : 1;

      nhwc_layer_->setInputOutputParams(in_shape.Size(), c, bn_group, device_id);

      // memory allocation - handled by MXNet

      // layer buffer setup.  Taken from nhwc_batchnorm_test.h
      nhwc_layer_->setConstants(exp_avg_factor, eps);

      // Check that first two returned values from numWorkspaceBytes are as expected
      std::vector<size_t> workspace_bytes = nhwc_layer_->numWorkspaceBytes(is_fwd);
      auto expected_bytes = c * sizeof(ScaleType);
      CHECK_EQ(workspace_bytes[0], expected_bytes) << "Unexpected save_mean workspace size.";
      CHECK_EQ(workspace_bytes[1], expected_bytes) << "Unexpected save_variance workspace size.";

      // The next value from numWorkspaceBytes is the retired_ctas_ workspace.  This is not
      // a temp workspace either, because it needs a one-time initialization to 0.

      retired_cta_bytes_ = workspace_bytes[2];

      // Remainder of workspace_bytes are temp areas to be allocated.
      // This is done as one chunk, then carved up via separately offsetted pointers.
      int start_index = 3;
      int offset_alignment = 512;
      workspace_byte_offsets_.clear();
      workspace_total_bytes_ = 0;
      for (std::vector<size_t>::size_type i = start_index; i != workspace_bytes.size(); ++i) {
         // First record the byte offset, which we desire to match default cudaMalloc alignment
         // of 512 bytes.
        workspace_total_bytes_ = round_up_to_multiple(workspace_total_bytes_, offset_alignment);
        workspace_byte_offsets_.push_back(workspace_total_bytes_);
        // Now increase the total bytes by this allocation
        auto alloc_bytes = workspace_bytes[i];
        workspace_total_bytes_ += alloc_bytes;
      }

      // If this Op is used by InstanceNorm::Forward(), we may need to supply
      // a small dummy moving mean/var tensor, so for simplicity always leave room.
      dummy_moving_stats_offset_ = round_up_to_multiple(workspace_total_bytes_, offset_alignment);
      workspace_total_bytes_ = dummy_moving_stats_offset_ +
                               c * mshadow::mshadow_sizeof(dtype_param_);
  }

  void *GetRetiredCtasDptr(mshadow::Stream<gpu> *s) {
    CHECK_GT(retired_cta_bytes_, 0);
    if (retired_ctas_hdl_.dptr == nullptr) {
      retired_ctas_hdl_ = mxnet::Storage::Get()->Alloc(retired_cta_bytes_, Context::GPU());
      // Zero the retired_ctas_ area once.  Kernels use this area to synchronize the kernel
      // blocks, which then leave the value as 0 for the next kernel.  This region can
      // be shared by all kernels launched by this thread, because they are launched into
      // the same stream and hence are serialized.
      CUDA_CALL(cudaMemsetAsync(retired_ctas_hdl_.dptr, 0,
                                retired_cta_bytes_, mshadow::Stream<gpu>::GetStream(s)));
    }
    return retired_ctas_hdl_.dptr;
  }

  // Increase a value until it is a multiple of `multiple`.
  size_t round_up_to_multiple(size_t x, int multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
  }

  // Allocates a 1D Tensor of ScaleType words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, ScaleType> AllocateTempWorkspace(const OpContext &ctx,
                                                           size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words = max(static_cast<size_t>(1),
                            (size_bytes + sizeof(ScaleType) - 1) / sizeof(ScaleType));
    auto retval = ctx.requested[nhwcbatchnorm::kTempSpace].get_space_typed<gpu, 1, ScaleType>(
        mshadow::Shape1(size_words), s);
    return retval;
  }

  // The captive NhwcBatchNorm object wrapped by this NhwcBatchNormOp
  std::shared_ptr<NhwcBatchNorm> nhwc_layer_;
  // The cudnn dtype of the primary data I/O of the BatchNorm
  cudnnDataType_t dtype_;
  // The dtype of the parameters (e.g. gamma, beta, moving_* and saved_*)
  int dtype_param_;
  // A copy of the python-supplied params of the Op
  BatchNormParam param_;
  // Maximum size of the temp workspace requirements for both Forward() and Backward()
  size_t workspace_total_bytes_;
  // Offset within the workspace of a dummy moving_mean/moving_var tensor (if needed)
  size_t dummy_moving_stats_offset_;
  // Offset within the workspace of the various temp tensors needed
  std::vector<int> workspace_byte_offsets_;
  // The size of the privately-allocated cta-sync tensor (needs only 1-time init)
  int retired_cta_bytes_ = 0;
  // Storage handle for the privately-allocated cta-sync tensor (needs only 1-time init)
  mxnet::Storage::Handle retired_ctas_hdl_;
};

// Note: Definition in ../batch_norm.cu
template<typename DType>
NhwcBatchNormOp<DType> &GetNhwcOp(const BatchNormParam& param);
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
