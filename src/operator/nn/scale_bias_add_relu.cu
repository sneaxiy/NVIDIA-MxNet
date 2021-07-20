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
 * Copyright (c) 2017 by Contributors
 * \file scale_bias_add_relu.cu
 * \brief
 * \author Kartikeya Goyal
*/

#include "./scale_bias_add_relu-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_scale_bias_add_relu-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNScaleBiasAddReluOp<DType>& GetCuDNNScaleBiasAddReluOp(
                                                 const ScaleBiasAddReluParam& param,
                                                 const std::vector<mxnet::TShape>& in_shape,
                                                 const std::vector<mxnet::TShape>& out_shape,
                                                 const OpContext& ctx) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<ScaleBiasAddReluSignature,
                                         std::shared_ptr<CuDNNScaleBiasAddReluOp<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<ScaleBiasAddReluSignature,
                                            std::shared_ptr<CuDNNScaleBiasAddReluOp<DType> >,
                                            OpHash> ops;
#endif
  ScaleBiasAddReluSignature key(param);
  size_t ndim = 0;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(
    ndim /* for in and out shapes */ +
    1 /* for dev_id */);
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(ctx.run_ctx.ctx.dev_id);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNScaleBiasAddReluOp<DType>> op(
        new CuDNNScaleBiasAddReluOp<DType>());
    auto ins_ret = ops.insert(std::pair<ScaleBiasAddReluSignature,
        std::shared_ptr<CuDNNScaleBiasAddReluOp<DType>>>(key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, in_shape, out_shape, ctx);
  }
  return *it->second;
}
#endif

template<>
void ScaleBiasAddReluCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ScaleBiasAddReluParam& param = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  int dtype;
  mxnet::TShape in_data_shape;

  if (param.dual_scale_bias) {
    dtype = inputs[scale_bias_add_relu::kxDSBARData].type_flag_;
    in_data_shape = inputs[scale_bias_add_relu::kxDSBARData].shape_;
  } else {
    if (param.fused_add) {
        dtype = inputs[scale_bias_add_relu::kxSBARData].type_flag_;
        in_data_shape = inputs[scale_bias_add_relu::kxSBARData].shape_;
    } else {
        dtype = inputs[scale_bias_add_relu::kxSBRData].type_flag_;
        in_data_shape = inputs[scale_bias_add_relu::kxSBRData].shape_;
    }
  }
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (!CuDNNScaleBiasAddReluOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This ScaleBiasAddRelu is not supported by cudnn"
                   << ", MXNET ScaleBiasAddRelu is applied.";
      ScaleBiasAddReluOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    } else {
      std::vector<mxnet::TShape> in_shape(inputs.size());
      std::vector<mxnet::TShape> out_shape(outputs.size());

      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = inputs[i].shape_;
      for (size_t i = 0; i < out_shape.size(); i++)
        out_shape[i] = outputs[i].shape_;

      CuDNNScaleBiasAddReluOp<DType> &op = GetCuDNNScaleBiasAddReluOp<DType>(param,
                in_shape, out_shape, ctx);
        op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ScaleBiasAddReluOp<gpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

template<>
void ScaleBiasAddReluGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ScaleBiasAddReluParam& param = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);

  size_t num_fwd_inputs = scale_bias_add_relu::NumInputs(param.dual_scale_bias, param.fused_add);
  size_t num_fwd_ios = inputs.size();
  size_t num_fwd_outputs = num_fwd_ios - num_fwd_inputs;
  std::vector<TBlob> fwd_out_data(inputs.begin(), inputs.begin() + num_fwd_outputs);
  std::vector<TBlob> fwd_in_data(inputs.begin() + num_fwd_outputs, inputs.end());

  // Remember, for fwd_out_data[kOut], we've swapped in the gradient for the output itself.
  const TBlob &out_grad = fwd_out_data[scale_bias_add_relu::kOut];
  const TBlob &out_bitmask = fwd_out_data[scale_bias_add_relu::kBitMask];

  // Gradient types will be the same as the corresponding output
  int dtype = out_grad.type_flag_;

  mxnet::TShape in_data_shape;
  if (param.dual_scale_bias) {
    in_data_shape = inputs[scale_bias_add_relu::kxDSBARData].shape_;
  } else {
    if (param.fused_add) {
        in_data_shape = inputs[scale_bias_add_relu::kxSBARData].shape_;
    } else {
        in_data_shape = inputs[scale_bias_add_relu::kxSBRData].shape_;
    }
  }

#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (!CuDNNScaleBiasAddReluOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This ScaleBiasAddRelu is not supported by cudnn"
                   << ", MXNET ScaleBiasAddRelu is applied.";
      ScaleBiasAddReluOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, fwd_in_data, req, outputs);
    } else {
      std::vector<mxnet::TShape> in_shape(fwd_in_data.size());
      std::vector<mxnet::TShape> out_shape(fwd_out_data.size());
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = fwd_in_data[i].shape_;
      for (size_t i = 0; i < out_shape.size(); i++)
        out_shape[i] = fwd_out_data[i].shape_;
      CuDNNScaleBiasAddReluOp<DType> &op = GetCuDNNScaleBiasAddReluOp<DType>(param,
        in_shape, out_shape, ctx);
      op.Backward(ctx, std::vector<TBlob>{out_grad, out_bitmask}, fwd_in_data, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ScaleBiasAddReluOp<gpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad, out_bitmask}, fwd_in_data, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

NNVM_REGISTER_OP(ScaleBiasAddRelu)
.set_attr<FCompute>("FCompute<gpu>", ScaleBiasAddReluCompute<gpu>);

NNVM_REGISTER_OP(_backward_ScaleBiasAddRelu)
.set_attr<FCompute>("FCompute<gpu>", ScaleBiasAddReluGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
