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
 * \file instance_norm.cu
 * \brief Implements Ba et. al, Instance Normalization (https://arxiv.org/abs/1607.06450).
*/
#include "./instance_norm_v2-inl.h"
#include "./batch_norm-inl.h"

#if MXNET_USE_CUDNN == 1
#include "./cudnn/nhwc_batch_norm-inl.h"
#endif

using namespace mshadow::cuda;

namespace mxnet {
namespace op {

// Not sure yet whether axis=-1 impl based on BatchNorm will be part of this operator
// or a different one.  This is where the impl 'switch' might appear.

// Create the BatchNormParam that would be used to implement the InstanceNorm
BatchNormParam
ToBatchnormParam(const InstanceNormParamV2 &in_param) {
    BatchNormParam bn_param;

    // Copy commonly held data members
    bn_param.eps = in_param.eps;
    bn_param.axis = in_param.axis;
    // Supply additional data members of BatchNormParam
    bn_param.momentum = 0.f;
    bn_param.fix_gamma = false;
    bn_param.use_global_stats = false;
    bn_param.output_mean_var = true;
    bn_param.cudnn_off = false;
    bn_param.act_type = in_param.act_type;
    bn_param.bn_group = in_param.xbuf_group;
    bn_param.xbuf_ptr = in_param.xbuf_ptr;
    // Leave optional bn_param data members unset - {min,max}_calib_range

    return bn_param;
}

// Create a view into a TBlob of one element of the batch.  Keep same shape dimensionality.
TBlob Slice(const TBlob &in, int i) {
  // Start with a copy of input TBlob, since only shape_ and dptr_ will change
  TBlob out(in);

  // Correct output blob shape_
  out.shape_[0] = 1;

  // Correct dptr_
  auto n = in.shape_[0];
  auto elements_per_slice = in.Size() / n;
  int dtype = in.type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    out.dptr_ = out.dptr<DType>() + elements_per_slice * i;
  });

  return out;
}

template<typename DType>
void InstanceNormComputeViaBatchnorm(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  InstanceNormParamV2 inst_param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  BatchNormParam bn_param = ToBatchnormParam(inst_param);
  mxnet::TShape shape = inputs[0].shape_;
  int n = shape[0];
  auto &bn_op = GetNhwcOp<DType>(bn_param);
  // Don't pass in moving_mean or moving_var- Batchnorm Op will supply tensors as needed
  const std::vector<TBlob> aux_states{};

  // Invoke Batchnorm Forward over each element of the batch separately
  for (int i = 0; i != n; ++i) {
    const std::vector<TBlob> bn_inputs{Slice(inputs[instancenorm::kData], i),
                                       inputs[instancenorm::kGamma],
                                       inputs[instancenorm::kBeta]};
    const std::vector<TBlob> bn_outputs{Slice(outputs[instancenorm::kOut], i),
                                        Slice(outputs[instancenorm::kMean], i),
                                        Slice(outputs[instancenorm::kStd], i)};

    bn_op.Forward(ctx, bn_inputs, req, bn_outputs, aux_states);
  }
}

// Increase a value until it is a multiple of `multiple`.
size_t round_up_to_multiple(size_t x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

// Allocates a 1D Tensor of DType words with size in bytes >= `size_bytes`.
// Always allocates at least one word.
template<typename DType>
mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx,
                                                           size_t size_bytes) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  size_t size_words = max(static_cast<size_t>(1),
                          (size_bytes + sizeof(DType) - 1) / sizeof(DType));
  auto retval = ctx.requested[nhwcbatchnorm::kTempSpace].get_space_typed<gpu, 1, DType>(
      mshadow::Shape1(size_words), s);
  return retval;
}

template<typename DType>
void InstanceNormGradComputeViaBatchnorm(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  InstanceNormParamV2 inst_param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  BatchNormParam bn_param = ToBatchnormParam(inst_param);
  mxnet::TShape data_shape = inputs[0].shape_;
  int n = data_shape[0];
  int c = outputs[instancenorm::kGamma].Size();
  auto &bn_op = GetNhwcOp<DType>(bn_param);

  // bn_op::Backward() will be called once for each element of the minibatch, but if this
  // is done multiple times (i.e. n > 1) the individual dgamma and dbeta tensors must be
  // summed together into a final dgamma and dbeta result.  This requires an enlarged
  // workspace, with the actual BatchNorm workspace offset into this space.

  bool reduce_dgamma_dbeta = n > 1;
  size_t bn_workspace_bytes = bn_op.GetWorkspaceSizeBytes(data_shape, s, false);
  size_t in_workspace_bytes = bn_workspace_bytes;
  size_t workspace_offset = 0;

  TBlob temp_dgamma;
  TBlob temp_dbeta;
  auto dgamma_dtype = outputs[instancenorm::kGamma].type_flag_;

  if (reduce_dgamma_dbeta) {
    auto temp_dgamma_shape = Shape2(n, c);
    auto temp_dgamma_bytes = round_up_to_multiple(temp_dgamma_shape.Size() *
                                                     mshadow_sizeof(dgamma_dtype), 512);
    auto temp_dgamma_padded_elements = temp_dgamma_bytes / mshadow_sizeof(dgamma_dtype);
    workspace_offset = 2 * temp_dgamma_bytes;
    in_workspace_bytes += workspace_offset;

    MSHADOW_REAL_TYPE_SWITCH(dgamma_dtype, DgammaDType, {
      temp_dgamma = AllocateTempWorkspace<DgammaDType>(ctx, in_workspace_bytes);
      temp_dgamma.shape_ = temp_dgamma_shape;
      temp_dbeta = temp_dgamma;
      temp_dbeta.dptr_ = temp_dbeta.dptr<DgammaDType>() + temp_dgamma_padded_elements;
    });
  }

  const TBlob INograd = inputs[0];
  const TBlob INdata = inputs[1];
  const TBlob INgamma = inputs[2];
  const TBlob INbeta = inputs[3];
  const TBlob INmean = inputs[4];
  const TBlob INstd = inputs[5];

  // Invoke Batchnorm Backward over each element of the batch separately
  // The Batchnorm gradient node I/O's are:
  // inputs - grad_out, saved_mean, saved_var, data_in, moving_mean, moving_var
  // outputs - grad_in, grad_gamma, grad_beta
  for (int i = 0; i != n; ++i) {
    const std::vector<TBlob> bn_inputs{Slice(INograd, i),
                                       Slice(INmean, i),
                                       Slice(INstd, i),
                                       Slice(INdata, i),
                                       INgamma,
                                       INbeta};

    auto dgamma_out = reduce_dgamma_dbeta ? Slice(temp_dgamma, i).reshape(Shape1(c))
                                          : outputs[instancenorm::kGamma];
    auto dbeta_out = reduce_dgamma_dbeta ? Slice(temp_dbeta, i).reshape(Shape1(c))
                                         : outputs[instancenorm::kBeta];

    const std::vector<TBlob> bn_outputs{Slice(outputs[instancenorm::kData], i),
                                        dgamma_out,
                                        dbeta_out};
    // If we're recucing dgamma and dbeta, tell bn_op to stack
    // its workspace on top of `workspace_offset` bytes.
    bn_op.Backward(ctx, bn_inputs, req, bn_outputs, workspace_offset);
  }

  if (reduce_dgamma_dbeta) {
    // If bn_op.Backward stayed within its advertised workspace, then the workspace
    // pointer should not have changed.  This is important to check though, since
    // if the tempspace was reallocated, then bn_op.Backward() wrongly put its temp
    // dgamma and dbeta results in a now-freed and possibly corrupted location.
    TBlob post_backward_dgamma;
    MSHADOW_REAL_TYPE_SWITCH(dgamma_dtype, DgammaDType, {
      post_backward_dgamma = AllocateTempWorkspace<DgammaDType>(ctx, in_workspace_bytes);
    });
    CHECK_EQ(temp_dgamma.dptr_, post_backward_dgamma.dptr_) <<
      "Unexpected reallocation of Tempspace invalidates InstanceNorm handling";

    // Reduce the 2D unreduced_dgamma and unreduced_dbeta tensors into the final 1D output tensors

    MSHADOW_REAL_TYPE_SWITCH(dgamma_dtype, DgammaDType, {
      Tensor<gpu, 2, DgammaDType> unreduced_dgamma =
        temp_dgamma.get_with_shape<gpu, 2, DgammaDType>(Shape2(n, c), s);
      Tensor<gpu, 1, DgammaDType> dgamma =
        outputs[instancenorm::kGamma].get_with_shape<gpu, 1, DgammaDType>(Shape1(c), s);
      Assign(dgamma, req[instancenorm::kGamma], sumall_except_dim<1>(unreduced_dgamma));

      Tensor<gpu, 2, DgammaDType> unreduced_dbeta =
        temp_dbeta.get_with_shape<gpu, 2, DgammaDType>(Shape2(n, c), s);
      Tensor<gpu, 1, DgammaDType> dbeta =
        outputs[instancenorm::kBeta].get_with_shape<gpu, 1, DgammaDType>(Shape1(c), s);
      Assign(dbeta, req[instancenorm::kBeta], sumall_except_dim<1>(unreduced_dbeta));
    });
  }
}

template<>
void InstanceNormCompute<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  InstanceNormParamV2 inst_param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  int dtype = inputs[0].type_flag_;
  mxnet::TShape shape = inputs[0].shape_;
  // If we use the NhwcBatchNormOp, it will be in a loop with n == 1
  shape[0] = 1;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (NhwcBatchNormOp<DType>::Supports(ToBatchnormParam(inst_param),
                                         dtype, shape, ctx.run_ctx.ctx)) {
      InstanceNormComputeViaBatchnorm<DType>(attrs, ctx, inputs, req, outputs);
    } else {
      InstanceNormComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
    }
  });
}

template<>
void InstanceNormGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  InstanceNormParamV2 inst_param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  int dtype = inputs[0].type_flag_;
  mxnet::TShape shape = inputs[0].shape_;
  // If we use the NhwcBatchNormOp, it will be in a loop with n == 1
  shape[0] = 1;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (NhwcBatchNormOp<DType>::Supports(ToBatchnormParam(inst_param),
                                         dtype, shape, ctx.run_ctx.ctx)) {
      InstanceNormGradComputeViaBatchnorm<DType>(attrs, ctx, inputs, req, outputs);
    } else {
      InstanceNormGradComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
    }
  });
}

NNVM_REGISTER_OP(InstanceNormV2)
.set_attr<FCompute>("FCompute<gpu>", InstanceNormCompute<gpu>);

NNVM_REGISTER_OP(_backward_InstanceNormV2)
.set_attr<FCompute>("FCompute<gpu>", InstanceNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
