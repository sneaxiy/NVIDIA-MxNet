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
 * \file instance_norm.cc
 * \brief Implements Ba et. al, Instance Normalization (https://arxiv.org/abs/1607.06450).
*/

#include "instance_norm_v2-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InstanceNormParamV2);

static bool InstanceNormShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  const InstanceNormParamV2& param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const mxnet::TShape &dshape = in_shape->at(instancenorm::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  int axis = GetRealAxis(param.axis, dshape.ndim());
  CHECK(axis >= 0 && axis < dshape.ndim())
    << "Channel axis out of range: axis=" << param.axis;

  const index_t channelCount = dshape[axis];

  SHAPE_ASSIGN_CHECK(*in_shape,
                     instancenorm::kGamma,
                     mxnet::TShape(Shape1(channelCount)));
  SHAPE_ASSIGN_CHECK(*in_shape,
                     instancenorm::kBeta,
                     mxnet::TShape(Shape1(channelCount)));
  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  mxnet::TShape moments_shape(dshape.begin(), dshape.end());
  for (int i = 0; i != moments_shape.ndim(); ++i)
    if (i != 0 && i != axis)
      moments_shape[i] = 1;
  out_shape->push_back(moments_shape);  // kMean
  out_shape->push_back(moments_shape);  // kInvstd
  return true;
}

static bool InstanceNormType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type,
                          std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_GE(in_type->size(), 1U);
  const size_t n_out = 3;
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param;
  int dtype = (*in_type)[0];
  if (type_is_none(dtype)) {
    // Input type is undefined, we try backward inference
     if (out_type->size() == 0 || type_is_none((*out_type)[0])) {
       // Neither the input nor the output are defined,
       // types cannot be infered for this op
       return false;
     } else {
       // Input type is undefined but output type is: backward inference
       dtype = (*out_type)[0];
       (*in_type)[0] = dtype;
       MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
         dtype_param = mshadow::DataType<AccRealX>::kFlag; });
     }
  } else {
    // Input type is defined but output type is not: forward inference
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    out_type->clear();
    out_type->push_back(dtype);
    for (size_t i = 1; i < n_out; ++i) {
      out_type->push_back(dtype_param);
    }
  }
  std::vector<std::string> args{"data", "gamma", "beta"};
  CHECK_LE(in_type->size(), args.size());
  for (size_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  return true;
}

template<>
void InstanceNormCompute<cpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  return InstanceNormComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

template<>
void InstanceNormGradCompute<cpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  return InstanceNormGradComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(InstanceNormV2)
.add_alias("_npx_instance_norm")
.describe(R"code(Applies instance normalization to the n-dimensional input array.

This operator takes an n-dimensional input array where (n>2) and normalizes
the input using the following formula:

.. math::

  out = \frac{x - mean[data]}{ \sqrt{Var[data] + \epsilon}} * gamma + beta

This layer is similar to batch normalization layer (`BatchNorm`)
with two differences: first, the normalization is
carried out per example (instance), not over a batch. Second, the
same normalization is applied both at test and train time. This
operation is also known as `contrast normalization`.

If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
`gamma` and `beta` parameters must be vectors of shape [channel].

This implementation is based on this paper [1]_

.. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
   D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).

Examples::

  // Input of shape (2,1,2)
  x = [[[ 1.1,  2.2]],
       [[ 3.3,  4.4]]]

  // gamma parameter of length 1
  gamma = [1.5]

  // beta parameter of length 1
  beta = [0.5]

  // Instance normalization is calculated with the above formula
  InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
                                [[-0.99752653,  1.99752724]]]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<InstanceNormParamV2>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "std"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const InstanceNormParamV2& param = nnvm::get<InstanceNormParamV2>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<mxnet::FInferShape>("FInferShape", InstanceNormShape)
.set_attr<nnvm::FInferType>("FInferType", InstanceNormType)
.set_attr<FCompute>("FCompute<cpu>", InstanceNormCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::ObjectPtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(ograds[0]);  // ograd
  heads.push_back(n->inputs[0]);  // data
  heads.push_back(n->inputs[1]);  // gamma
  heads.push_back(n->inputs[2]);  // beta, needed when act_type='relu'
  heads.emplace_back(n, 1, 0);  // mean
  heads.emplace_back(n, 2, 0);  // std
  return MakeGradNode("_backward_InstanceNormV2", n, heads, n->attrs.dict);
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data to instance normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(InstanceNormParamV2::__FIELDS__());


NNVM_REGISTER_OP(_backward_InstanceNormV2)
.set_num_inputs(6)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InstanceNormParamV2>)
.set_attr<FCompute>("FCompute<cpu>", InstanceNormGradCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
