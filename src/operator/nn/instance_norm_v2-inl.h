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
 * \file instance_norm_v2-inl.h
 * \brief Implements Ba et. al, Instance Normalization (https://arxiv.org/abs/1607.06450).
*/
#ifndef MXNET_OPERATOR_NN_INSTANCE_NORM_V2_INL_H_
#define MXNET_OPERATOR_NN_INSTANCE_NORM_V2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "mxnet/tuple.h"
#include "activation-inl.h"

namespace mxnet {
namespace op {

namespace instancenorm {
  enum InstanceNormOpInputs {kData, kGamma, kBeta};  // kGamma: scaling params, kBeta: shift biases
  enum InstanceNormOpOutputs {kOut, kMean, kStd,
    kNumOutputs  // Not an I/O! Leave this at the end
  };  // req, out_data
}  // namespace instancenorm

struct InstanceNormParamV2 : public dmlc::Parameter<InstanceNormParamV2> {
  int axis;
  float eps;
  bool output_mean_var;

  dmlc::optional<int> act_type;
  int xbuf_group;
  uint64_t xbuf_ptr;

  DMLC_DECLARE_PARAMETER(InstanceNormParamV2) {
    DMLC_DECLARE_FIELD(axis).set_default(1)
      .describe("The axis to perform instance normalization. "
                "Usually, this should be be axis of the channel dimension. "
                "Negative values means indexing from right to left.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
      .describe("An `epsilon` parameter to prevent division by 0.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
      .describe("Output the mean and std calculated along the given axis.");
    // Should track the definition in activation-inl.h
    DMLC_DECLARE_FIELD(act_type)
        .add_enum("relu", activation::kReLU)
        .add_enum("sigmoid", activation::kSigmoid)
        .add_enum("tanh", activation::kTanh)
        .add_enum("softrelu", activation::kSoftReLU)
        .set_default(dmlc::optional<int>())
        .describe("Fused activation function to be applied.");
    DMLC_DECLARE_FIELD(xbuf_group).set_default(1)
    .describe("exchange buffer group size");
    DMLC_DECLARE_FIELD(xbuf_ptr).set_default(0)
    .describe("exchange buffer pointer");
  }
};

static int GetRealAxis(int axis, int ndim) {
  return axis < 0 ? (axis + ndim) : axis;
}

template<typename xpu>
void InstanceNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs);

template<typename xpu>
void InstanceNormComputeGeneral(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx, const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const InstanceNormParamV2& param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  CHECK_EQ(param.xbuf_group, 1) << "group InstanceNorm not supported by this impl";
  CHECK(!param.act_type.has_value()) << "integrated "
                                     << param.act_type.value()
                                     << " activation not supported by this InstanceNorm impl";
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  int axis = GetRealAxis(param.axis, inputs[0].ndim());
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  CHECK_EQ(inputs.size(), 3U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Reshape gamma and beta to be broadcastable
  mxnet::TShape new_param_shape(inputs[0].shape_.begin(), inputs[0].shape_.end());
  for (int i = 0; i < inputs[0].ndim(); i++) {
    if (i != axis) {
      new_param_shape[i] = 1;
    }
  }
  const TBlob gamma = inputs[1].reshape(new_param_shape);
  const TBlob beta = inputs[2].reshape(new_param_shape);
  // Compute necessary data for the reduce operation.
  mxnet::TShape red_src_shape, red_dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, outputs[instancenorm::kMean].shape_,
                              &red_src_shape, &red_dst_shape);
  const TBlob in_data = inputs[0].reshape(red_src_shape);
  const TBlob mean_data = outputs[instancenorm::kMean].reshape(red_dst_shape);
  const TBlob std_data = outputs[instancenorm::kStd].reshape(red_dst_shape);
  int channel_size = red_src_shape.Size() / red_dst_shape.Size();
  // Initialize the workspace
  Tensor<xpu, 1, char> workspace;
  size_t workspace_size = 0;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
    workspace_size =
      broadcast::ReduceWorkspaceSize(s, mean_data.shape_, req[0],
                                     in_data.shape_, sizeof(AType));
  });
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  if constexpr (std::is_same<xpu, mshadow::cpu>::value) {
    if (inputs[0].type_flag_ == mshadow::kFloat16)
      LOG(FATAL) << "InstanceNorm Op with float16 data not supported on cpu";
    // Like with BatchNorm, acc_type is float32 if DType is float16, else DType
    bool safe_acc = inputs[0].type_flag_ == mshadow::kFloat16;

    // Calculate mean
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, mean_data, req[0], workspace, in_data);
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, mean_data, req[0], workspace, in_data);
        }
        Tensor<xpu, 1, AType> mean_data_tensor = mean_data.FlatTo1D<xpu, AType>(s);
        mean_data_tensor /= scalar<AType>(channel_size);
      });
    });


    // Calculate data = data - mean
    BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                       {inputs[0], outputs[instancenorm::kMean]},
                                                       {kWriteTo}, {outputs[0]});
    // Calculate std
    const TBlob centered_out = outputs[0].reshape(red_src_shape);
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::square, true>(
            s, std_data, req[0], workspace, centered_out);
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::square, false>(
            s, std_data, req[0], workspace, centered_out);
        }
        Tensor<xpu, 1, AType> std_data_tensor = std_data.FlatTo1D<xpu, AType>(s);
        std_data_tensor = F<mshadow_op::square_root>(std_data_tensor / scalar<AType>(channel_size)
                          + scalar<AType>(param.eps));
      });
    });
    // Calculate data = data / std
    BinaryBroadcastCompute<xpu, mshadow_op::div>(attrs, ctx,
                                                 {outputs[0], outputs[instancenorm::kStd]},
                                                 {kWriteTo}, {outputs[0]});
    // Calculate data = data * gamma
    BinaryBroadcastCompute<xpu, mshadow_op::mul>(attrs, ctx,
                                                 {outputs[0], gamma},
                                                 {kWriteTo}, {outputs[0]});
    // Calculate data = data + beta
    BinaryBroadcastCompute<xpu, mshadow_op::plus>(attrs, ctx,
                                                  {outputs[0], beta},
                                                  {kWriteTo}, {outputs[0]});
  } else {
#if MXNET_USE_CUDA
    // Calculate mean
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, mean_data, req[0], workspace, in_data,
                           "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
        Tensor<xpu, 1, AType> mean_data_tensor = mean_data.FlatTo1D<xpu, AType>(s);
        mean_data_tensor /= scalar<AType>(channel_size);
    });
    // Calculate data = data - mean
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                      {inputs[0], outputs[instancenorm::kMean]},
                                      {kWriteTo}, {outputs[0]});
    // Calculate std
    const TBlob centered_out = outputs[0].reshape(red_src_shape);
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, std_data, req[0], workspace, centered_out,
                           "red::sum{}", NDim, "square");
    });
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      Tensor<xpu, 1, AType> std_data_tensor = std_data.FlatTo1D<xpu, AType>(s);
      std_data_tensor = F<mshadow_op::square_root>(std_data_tensor / scalar<AType>(channel_size)
                        + scalar<AType>(param.eps));
    });
    // Calculate data = data / std
    BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                      {outputs[0], outputs[instancenorm::kStd]},
                                      {kWriteTo}, {outputs[0]});
    // Calculate data = data * gamma
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {outputs[0], gamma},
                                      {kWriteTo}, {outputs[0]});
    // Calculate data = data + beta
    BinaryBroadcastRTCCompute {"add"}(attrs, ctx,
                                      {outputs[0], beta},
                                      {kWriteTo}, {outputs[0]});
#endif
  }
}

template<typename xpu>
void InstanceNormGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs);

template<typename xpu>
void InstanceNormGradComputeGeneralImpl(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const TBlob& ograd,
                                     const TBlob& data,
                                     const TBlob& gamma,
                                     const TBlob& mean,
                                     const TBlob& std,
                                     const TBlob& normalized_data,
                                     const TBlob& ograd_mult,
                                     const TBlob& red_out,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs,
                                     const mshadow::Tensor<xpu, 1, char>& workspace,
                                     const mxnet::TShape& red_dst_shape,
                                     const mxnet::TShape& red_src_shape,
                                     const mxnet::TShape& red_exclude_dst_shape,
                                     const mxnet::TShape& red_exclude_src_shape,
                                     const int channel_size);

#ifndef __CUDACC__
template <>
void InstanceNormGradComputeGeneralImpl<cpu>(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const TBlob& ograd,
                                          const TBlob& data,
                                          const TBlob& gamma,
                                          const TBlob& mean,
                                          const TBlob& std,
                                          const TBlob& normalized_data,
                                          const TBlob& ograd_mult,
                                          const TBlob& red_out,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<TBlob>& outputs,
                                          const mshadow::Tensor<cpu, 1, char>& workspace,
                                          const mxnet::TShape& red_dst_shape,
                                          const mxnet::TShape& red_src_shape,
                                          const mxnet::TShape& red_exclude_dst_shape,
                                          const mxnet::TShape& red_exclude_src_shape,
                                          const int channel_size) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  // Compute normalized_data = (data - mean) / std
  BinaryBroadcastCompute<cpu, mshadow_op::minus>(attrs, ctx,
                                                 {data, mean},
                                                 {kWriteTo}, {normalized_data});
  BinaryBroadcastCompute<cpu, mshadow_op::div>(attrs, ctx,
                                               {normalized_data, std},
                                               {kWriteTo}, {normalized_data});
  // Like with BatchNorm, acc_type is float32 if DType is float16, else DType
  bool safe_acc = outputs[0].type_flag_ == mshadow::kFloat16;

  // Calculate grad_beta
  if (req[2] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[2].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, AType, mshadow_op::identity, true>(
            s, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
            ograd.reshape(red_exclude_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, AType, mshadow_op::identity, false>(
            s, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
            ograd.reshape(red_exclude_src_shape));
        }
      });
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
  ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>(attrs, ctx, {normalized_data, ograd},
                                                      {kWriteTo}, {ograd_mult});
  if (req[1] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[1].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, AType, mshadow_op::identity, true>(
            s, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
            ograd_mult.reshape(red_exclude_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, AType, mshadow_op::identity, false>(
            s, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
            ograd_mult.reshape(red_exclude_src_shape));
        }
      });
    });
  }

  // Calculate grad_data:
  //   ograd_mult = ograd * gamma / std
  //   grad_data = ograd_mult - mean(ograd_mult, axis)
  //               + normalized_data * (-mean(normalized_data * ograd_mult, axis))
  if (req[0] != kNullOp) {
    BinaryBroadcastCompute<cpu, op::mshadow_op::mul>(attrs, ctx,
                                                    {ograd, gamma},
                                                    {kWriteTo}, {ograd_mult});
    BinaryBroadcastCompute<cpu, op::mshadow_op::div>(attrs, ctx,
                                                    {ograd_mult, std},
                                                    {kWriteTo}, {ograd_mult});
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        }
      });
      Tensor<cpu, 1, AType> red_out_tensor = red_out.FlatTo1D<cpu, AType>(s);
      red_out_tensor /= scalar<AType>(channel_size);
    });
    BinaryBroadcastCompute<cpu, op::mshadow_op::minus>(attrs, ctx,
                                                      {ograd_mult, red_out},
                                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>(attrs, ctx, {ograd_mult, normalized_data},
                                                        {kWriteTo}, {ograd_mult});
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        }
      });
      Tensor<cpu, 1, AType> red_out_tensor = red_out.FlatTo1D<cpu, AType>(s);
      red_out_tensor /=  scalar<AType>(- channel_size);
    });
    BinaryBroadcastCompute<cpu, mshadow_op::mul>(attrs, ctx,
                                                 {normalized_data, red_out},
                                                 {kAddTo}, {outputs[0]});
  }
}

#else

template <>
void InstanceNormGradComputeGeneralImpl<gpu>(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const TBlob& ograd,
                                          const TBlob& data,
                                          const TBlob& gamma,
                                          const TBlob& mean,
                                          const TBlob& std,
                                          const TBlob& normalized_data,
                                          const TBlob& ograd_mult,
                                          const TBlob& red_out,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<TBlob>& outputs,
                                          const mshadow::Tensor<gpu, 1, char>& workspace,
                                          const mxnet::TShape& red_dst_shape,
                                          const mxnet::TShape& red_src_shape,
                                          const mxnet::TShape& red_exclude_dst_shape,
                                          const mxnet::TShape& red_exclude_src_shape,
                                          const int channel_size) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  // Compute normalized_data = (data - mean) / std
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {data, mean},
                                    {kWriteTo}, {normalized_data});
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {normalized_data, std},
                                    {kWriteTo}, {normalized_data});
  // Calculate grad_beta
  if (req[2] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
                           ograd.reshape(red_exclude_src_shape), "red::sum{}", NDim, "identity");
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
  ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {normalized_data, ograd},
                                   {kWriteTo}, {ograd_mult});
  if (req[1] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
                           ograd_mult.reshape(red_exclude_src_shape), "red::sum{}", NDim,
                           "identity");
    });
  }
  // Calculate grad_data:
  //   ograd_mult = ograd * gamma / std
  //   grad_data = ograd_mult - mean(ograd_mult, axis)
  //               + normalized_data * (-mean(normalized_data * ograd_mult, axis))
  if (req[0] != kNullOp) {
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {ograd, gamma},
                                      {kWriteTo}, {ograd_mult});
    BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                      {ograd_mult, std},
                                      {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                             ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });

    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      Tensor<gpu, 1, AType> red_out_tensor = red_out.FlatTo1D<gpu, AType>(s);
      red_out_tensor /= scalar<AType>(channel_size);
    });
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                      {ograd_mult, red_out},
                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {ograd_mult, normalized_data},
                                     {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                           ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
      Tensor<gpu, 1, AType> red_out_tensor = red_out.FlatTo1D<gpu, AType>(s);
      red_out_tensor /=  scalar<AType>(- channel_size);
    });
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {normalized_data, red_out},
                                      {kAddTo}, {outputs[0]});
  }
}
#endif

/*
Calculate the gradient of instance normalization.
We have the following gradient for gamma, beta and x:

\bar{x} = (x - mean) / std
w = og * r / std
grad_gamma = sum(\bar{x} og, exclude_axis)
grad_beta = sum(og, exclude_axis)
grad_x = w - mean(w, axis) - \bar{x} * mean(w * \bar{x}, axis)
*/
template<typename xpu>
void InstanceNormGradComputeGeneral(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx, const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 6U);
  const InstanceNormParamV2& param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  CHECK_EQ(param.xbuf_group, 1) << "group InstanceNorm not supported by this impl";
  CHECK(!param.act_type.has_value()) << "integrated "
                                     << param.act_type.value()
                                     << " activation not supported by this InstanceNorm impl";
  int axis = param.axis;
  if (axis < 0) {
    axis += inputs[0].ndim();
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Reshape gamma to be broadcastable
  mxnet::TShape new_param_shape(inputs[0].shape_.begin(), inputs[0].shape_.end());
  for (int i = 0; i < inputs[0].ndim(); i++) {
    if (i != axis) {
      new_param_shape[i] = 1;
    }
  }
  const TBlob ograd = inputs[0];
  const TBlob data = inputs[1];
  const TBlob gamma = inputs[2].reshape(new_param_shape);
  const TBlob beta = inputs[3].reshape(new_param_shape);
  const TBlob mean = inputs[4];
  const TBlob std = inputs[5];
  // Prepare the necessary shapes for reduction
  mxnet::TShape red_src_shape, red_dst_shape, red_exclude_src_shape, red_exclude_dst_shape;
  BroadcastReduceShapeCompact(ograd.shape_, mean.shape_, &red_src_shape, &red_dst_shape);
  BroadcastReduceShapeCompact(ograd.shape_, gamma.shape_,
                              &red_exclude_src_shape, &red_exclude_dst_shape);
  int channel_size = red_src_shape.Size() / red_dst_shape.Size();
  // Initialize the workspace + Construct the temporary TBlobs
  Tensor<xpu, 1, char> workspace;
  size_t reduce_workspace_size = 0;
  size_t data_size = 0;
  size_t red_out_size = 0;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, DType, AType, {
    data_size = sizeof(DType) * data.Size();
    red_out_size = sizeof(AType) * mean.Size();
    // There are two types of reduction workloads: reduce over axis and reduce exclude axis
    // We take the maximum of the workspace sizes required by these workloads.
    // Also, we explicitly set the req_type=kAddto in case we want to use it.
    reduce_workspace_size =
      std::max(reduce_workspace_size,
               broadcast::ReduceWorkspaceSize(s, red_dst_shape,
                                              kAddTo, red_src_shape,
                                              sizeof(AType)));
    reduce_workspace_size =
      std::max(reduce_workspace_size,
               broadcast::ReduceWorkspaceSize(s, red_exclude_dst_shape, kAddTo,
                                              red_exclude_src_shape,
                                              sizeof(AType)));
  });
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
    Shape1(reduce_workspace_size + data_size * 2 + red_out_size), s);
  const TBlob normalized_data = TBlob(workspace.dptr_ + reduce_workspace_size,
                                      data.shape_, data.dev_mask(), data.type_flag_, data.dev_id());
  const TBlob ograd_mult = TBlob(workspace.dptr_ + reduce_workspace_size + data_size,
                                 ograd.shape_, ograd.dev_mask(), ograd.type_flag_, ograd.dev_id());
  const TBlob red_out = TBlob(workspace.dptr_ + reduce_workspace_size + data_size * 2,
                              mean.shape_, mean.dev_mask(), mean.type_flag_, mean.dev_id());

  InstanceNormGradComputeGeneralImpl(attrs, ctx, ograd, data, gamma, mean, std, normalized_data,
                                  ograd_mult, red_out, req, outputs, workspace, red_dst_shape,
                                  red_src_shape, red_exclude_dst_shape, red_exclude_src_shape,
                                  channel_size);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_INSTANCE_NORM_V2_INL_H_
