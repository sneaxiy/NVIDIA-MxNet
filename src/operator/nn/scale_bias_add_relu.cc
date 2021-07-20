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
 * \file scale_bias_add_relu.cc
 * \brief
 * \author Kartikeya Goyal, Dick Carter
*/

#include "./scale_bias_add_relu-inl.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#if MXNET_USE_NNPACK == 1
#include "../nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ScaleBiasAddReluParam);

static inline std::vector<std::string> ListArguments(const ScaleBiasAddReluParam& param_) {
  if (param_.dual_scale_bias) {
    return {"dataX", "dataZ", "x_equiv_scale", "x_equiv_bias",
     "z_equiv_scale", "z_equiv_bias",
     "x_gamma", "x_beta", "x_mean", "x_invvar",
     "z_gamma", "z_beta", "z_mean", "z_invvar"
     };
  } else {
    return {"dataX", "dataZ", "x_equiv_scale", "x_equiv_bias",
      "x_gamma", "x_beta", "x_mean", "x_invvar"
    };
  }
}

static bool ScaleBiasAddReluShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  using namespace mshadow;
  const ScaleBiasAddReluParam& param_ = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);

  if ( !param_.dual_scale_bias ) {
      if ( param_.fused_add ) {
        CHECK_EQ(in_shape->size(),
        static_cast<size_t>(scale_bias_add_relu::NumInputs(
          param_.dual_scale_bias,
          param_.fused_add)))
        << "Input:[dataX, dataZ, equiv_scaleX, equiv_biasX]";
      } else {
        CHECK_EQ(in_shape->size(),
        static_cast<size_t>(scale_bias_add_relu::NumInputs(
          param_.dual_scale_bias,
          param_.fused_add)))
        << "Input:[dataX, equiv_scaleX, equiv_biasX]";
      }
  } else {
      CHECK_EQ(in_shape->size(),
      static_cast<size_t>(scale_bias_add_relu::NumInputs(
        param_.dual_scale_bias,
        param_.fused_add)))
      << "Input:[dataX, dataZ, equiv_scaleX, equiv_biasX, equiv_scaleZ, equiv_biasZ]";
  }

  out_shape->resize(
    scale_bias_add_relu::NumOutputs(
      param_.dual_scale_bias,
      param_.fused_add),
    TShape());

  TShape &dshp = (*in_shape)[0];
  if ( param_.dual_scale_bias ) {
         dshp = (*in_shape)[scale_bias_add_relu::kxDSBARData];
  } else {
        if ( param_.fused_add ) {
                dshp = (*in_shape)[scale_bias_add_relu::kxSBARData];
        } else {
                dshp = (*in_shape)[scale_bias_add_relu::kxSBRData];
        }
  }

  Shape<4> oshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNHWC);
  SHAPE_ASSIGN_CHECK(*out_shape, scale_bias_add_relu::kOut,
    ConvertLayout(oshape, kNHWC, param_.layout.value()));

  const TShape &oshp = (*out_shape)[scale_bias_add_relu::kOut];
  Shape<4> tmpshape = ConvertLayout(oshp.get<4>(), param_.layout.value(), kNCHW);
  int C = tmpshape[1];
  int64_t N = tmpshape[0];
  int64_t H = tmpshape[2];
  int64_t W = tmpshape[3];
  int64_t NDHW = N * H * W;
  int32_t C_int32Elems = ((C + 63) & ~63) / 32;
  int32_t NDHW_int32Elems = (NDHW + 31) & ~31;

  SHAPE_ASSIGN_CHECK(*out_shape, scale_bias_add_relu::kBitMask,
                     Shape1(C_int32Elems * NDHW_int32Elems));
  return true;
}

static bool ScaleBiasAddReluType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const ScaleBiasAddReluParam& param_ = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  CHECK_GE(in_type->size(),
   static_cast<size_t>(scale_bias_add_relu::NumInputs(
    param_.dual_scale_bias,
    param_.fused_add)));
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";

  int dtype_statistics;

  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_statistics = mshadow::DataType<AccRealX>::kFlag; });

  // Default expected input dtype matches that of the 1st (i.e. data) input.
  std::vector<int> input_types(in_type->size(), dtype);
  // However the 'mean', 'var', 'gamma' and 'beta' inputs, if present,
  // may have increased precision.
  if ( param_.dual_scale_bias ) {
    input_types[scale_bias_add_relu::kxDSBARGamma] = dtype_statistics;
    input_types[scale_bias_add_relu::kxDSBARBeta] = dtype_statistics;
    input_types[scale_bias_add_relu::kxDSBARMean] = dtype_statistics;
    input_types[scale_bias_add_relu::kxDSBARInvVar] = dtype_statistics;
    input_types[scale_bias_add_relu::kzDSBARGamma] = dtype_statistics;
    input_types[scale_bias_add_relu::kzDSBARBeta] = dtype_statistics;
    input_types[scale_bias_add_relu::kzDSBARMean] = dtype_statistics;
    input_types[scale_bias_add_relu::kzDSBARInvVar] = dtype_statistics;
  } else {
    if ( param_.fused_add ) {
      input_types[scale_bias_add_relu::kxSBARGamma] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBARBeta] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBARMean] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBARInvVar] = dtype_statistics;
    } else {
      input_types[scale_bias_add_relu::kxSBRGamma] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBRBeta] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBRMean] = dtype_statistics;
      input_types[scale_bias_add_relu::kxSBRInvVar] = dtype_statistics;
    }
  }

  for (size_t i = 0; i < in_type->size(); ++i) {
    int expected_type = input_types[i];
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = expected_type;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], expected_type, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  // 1st data output is of type 'dtype', rest are of greater precision 'dtype_statistics'
  out_type->push_back(dtype);
  out_type->push_back(mshadow::kInt32);

  while (out_type->size() < static_cast<size_t>(scale_bias_add_relu::kNumOutputs))
    out_type->push_back(dtype_statistics);
  return true;
}

void ScaleBiasAddReluParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  ScaleBiasAddReluParam param_;
  try {
    param_.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param_);
}

struct ScaleBiasAddReluGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const ScaleBiasAddReluParam& param_ =
        nnvm::get<ScaleBiasAddReluParam>(n->attrs.parsed);
    size_t num_fwd_inputs = n->inputs.size();
    size_t num_fwd_outputs = n->num_outputs();
    size_t expected_inputs = static_cast<size_t>(scale_bias_add_relu::NumInputs
                             (param_.dual_scale_bias, param_.fused_add));
    if ( !param_.dual_scale_bias ) {
      if ( param_.fused_add ) {
              CHECK_EQ(num_fwd_inputs, expected_inputs)
                << "Input:[dataX, dataZ, equiv_scaleX, equiv_biasX]";
      } else {
        CHECK_EQ(num_fwd_inputs, expected_inputs)
                << "Input:[dataX, equiv_scaleX, equiv_biasX]";
      }
    } else {
      CHECK_EQ(num_fwd_inputs, expected_inputs)
        << "Input:[dataX, dataZ, equiv_scaleX, equiv_biasX, equiv_scaleZ, equiv_biasZ]";
    }

    std::vector<nnvm::NodeEntry> heads;
    // We copy the outputs and the inputs of the forward node to the inputs of the backward node,
    // with the one *important* exception that the first backward input is the gradient of the first
    // output, not the output itself.  The benefit is that vectors of the forward node output-
    // and input-shapes are easily obtained, as is useful for operator instance lookup and init.

    std::vector<nnvm::NodeEntry> out_data(num_fwd_outputs);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
      out_data[i] = nnvm::NodeEntry{n, i, 0};
    }

    // The one data output gradient, the remainder of the outputs, and all forward node inputs
    // are inputs of the backward node.
    heads.reserve(num_fwd_outputs + num_fwd_inputs);
    if (param_.dual_scale_bias) {
          CHECK_GT(ograds.size(), scale_bias_add_relu::kxDSBARData)
                << "Not enough gradients of ScaleBiasAddRelu node.";
    } else {
      if (param_.fused_add) {
              CHECK_GT(ograds.size(), scale_bias_add_relu::kxSBARData)
                << "Not enough gradients of ScaleBiasAddRelu node.";
      } else {
              CHECK_GT(ograds.size(), scale_bias_add_relu::kxSBRData)
                << "Not enough gradients of ScaleBiasAddRelu node.";
      }
    }
    // Copy all outputs of forward node to the backward node, but use the gradient of the primary
    // output, instead of the output itself.  Rest are copied to have shape info readily available.
    for (uint32_t i = 0; i < num_fwd_outputs; ++i) {
      heads.push_back((i == scale_bias_add_relu::kOut) ? ograds[i] : out_data[i]);
    }
    // Copy all inputs of forward node to backward node
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      heads.push_back(n->inputs[i]);
    }

    nnvm::ObjectPtr gnode = nnvm::Node::Create();
    gnode->inputs = std::move(heads);
    gnode->control_deps.emplace_back(n);
    gnode->attrs = n->attrs;
    gnode->attrs.op = nnvm::Op::Get("_backward_ScaleBiasAddRelu");
    gnode->attrs.name = n->attrs.name + "_backward";

    std::vector<nnvm::NodeEntry> in_grad(num_fwd_inputs);
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      in_grad[i] = nnvm::NodeEntry{gnode, i, 0};
    }

    return in_grad;
  }
};

NNVM_REGISTER_OP(ScaleBiasAddRelu)
.describe(R"code(Compute ScaleBiasAddRelu.
)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ScaleBiasAddReluParam& params = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  return scale_bias_add_relu::NumInputs(params.dual_scale_bias, params.fused_add);
})
.set_num_outputs(scale_bias_add_relu::kNumOutputs)
.set_attr_parser(ScaleBiasAddReluParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const ScaleBiasAddReluParam& params = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  if ( !params.dual_scale_bias ) {
    if (params.fused_add) {
      return std::vector<std::string>{"dataX", "dataZ", "x_equiv_scale", "x_equiv_bias",
        "x_gamma", "x_beta", "x_mean", "x_invvar"};
    } else {
      return std::vector<std::string>{"dataX", "x_equiv_scale", "x_equiv_bias",
        "x_gamma", "x_beta", "x_mean", "x_invvar"};
    }
  } else {
    return std::vector<std::string>{"dataX", "dataZ", "x_equiv_scale", "x_equiv_bias",
     "z_equiv_scale", "z_equiv_bias",
     "x_gamma", "x_beta", "x_mean", "x_invvar",
     "z_gamma", "z_beta", "z_mean", "z_invvar"};
  }
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "bitmask"};
})
.set_attr<mxnet::FInferShape>("FInferShape", ScaleBiasAddReluShape)
.set_attr<nnvm::FInferType>("FInferType", ScaleBiasAddReluType)
.set_attr<FCompute>("FCompute<cpu>", ScaleBiasAddReluCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                           ScaleBiasAddReluGrad{"_backward_ScaleBiasAddRelu"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("dataX", "NDArray-or-Symbol", "Input data to the ScaleBiasAddReluOp.")
.add_argument("dataZ", "NDArray-or-Symbol", "Input data to the ScaleBiasAddReluOp.")
.add_argument("x_equiv_scale", "NDArray-or-Symbol", "equivalent scale array")
.add_argument("x_equiv_bias", "NDArray-or-Symbol", "equivalent bias array")
.add_argument("z_equiv_scale", "NDArray-or-Symbol", "equivalent scale array")
.add_argument("z_equiv_bias", "NDArray-or-Symbol", "equivalent bias array")
.add_argument("x_gamma", "NDArray-or-Symbol", "")
.add_argument("x_beta", "NDArray-or-Symbol", "")
.add_argument("x_mean", "NDArray-or-Symbol", "")
.add_argument("x_invvar", "NDArray-or-Symbol", "")
.add_argument("z_gamma", "NDArray-or-Symbol", "")
.add_argument("z_beta", "NDArray-or-Symbol", "")
.add_argument("z_mean", "NDArray-or-Symbol", "")
.add_argument("z_invvar", "NDArray-or-Symbol", "")
.add_arguments(ScaleBiasAddReluParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_ScaleBiasAddRelu)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ScaleBiasAddReluParam& params = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  return scale_bias_add_relu::NumInputs(
    params.dual_scale_bias, params.fused_add) + scale_bias_add_relu::kNumOutputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const ScaleBiasAddReluParam& params = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  // The outputs of the backward node are the fwd-node input gradients, so one per fwd node input.
  return scale_bias_add_relu::NumInputs(params.dual_scale_bias, params.fused_add);
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ScaleBiasAddReluParamParser)
.set_attr<FCompute>("FCompute<cpu>", ScaleBiasAddReluGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
