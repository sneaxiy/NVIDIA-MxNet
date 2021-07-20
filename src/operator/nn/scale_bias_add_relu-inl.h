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
 * \file scale_bias_add_relu-inl.h
 * \brief
 *  There are three modes this op can be used in
 *  Mode1 DBAR : dual_scale_bias_add_relu :
 *          Takes two tensors (X,Y) of dtype
 *          scale(sx) and bias(bx) for X of type fp32
 *          scale(sy) and bias(by) for Y of type fp32
 *          output, bitmask = relu(sx * X + bx + sy * Y + by)
 *
 *  Mode 2 SBAR : scale_bias_add_relu :
 *           Takes two tensors (X,Y) of dtype
 *           scale(sx) and bias(bx) for X of type fp32
 *           output, bitmask = relu((sx * X + bx) + Y)
 *
 * Mode 3 SBR : scale_bias_relu :
 *          Take one tensor X of dtype
 *          scale(sx) and bias(bx) for X of type fp32
 *          output = relu((sx * X +bx))
 *
 * \ref:
 * \author Kartikeya Goyal, Dick Carter
*/

#ifndef MXNET_OPERATOR_NN_SCALE_BIAS_ADD_RELU_INL_H_
#define MXNET_OPERATOR_NN_SCALE_BIAS_ADD_RELU_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "./im2col.h"
#include "activation-inl.h"

namespace mxnet {
namespace op {

namespace scale_bias_add_relu {
  enum ScaleBiasAddReluOpInputs {kxSBARData, kzSBARData, kxSBARScale, kxSBARBias,
    kxSBARGamma, kxSBARBeta, kxSBARMean, kxSBARInvVar,
    kNumInputsScaleBiasAddRelu  // Not an I/O! Leave this at the end
  };
  enum DualScaleBiasAddReluOpInputs {kxDSBARData, kzDSBARData, kxDSBARScale, kxDSBARBias,
    kzDSBARScale, kzDSBARBias,
    kxDSBARGamma, kxDSBARBeta, kxDSBARMean, kxDSBARInvVar,
    kzDSBARGamma, kzDSBARBeta, kzDSBARMean, kzDSBARInvVar,
    kNumInputsDualScaleBias  // Not an I/O! Leave this at the end
  };
  enum ScaleBiasReluOpInputs {kxSBRData, kxSBRScale, kxSBRBias,
      kxSBRGamma, kxSBRBeta, kxSBRMean, kxSBRInvVar,
      kNumInputsScaleBiasRelu  // Not an I/O! Leave this at the end
  };
  enum ScaleBiasAddReluOpOutputs {kOut, kBitMask,
       kNumOutputs  // Not an I/O! Leave this at the end};
  };
  enum ScaleBiasAddReluOpResource {kTempSpace};

  /*! \brief Default channel axis if none specified in the params */
  constexpr int DEFAULT_AXIS = 1;

  inline int NumOutputs(bool dual_scale_bias, bool fused_add) {
    return static_cast<int>(kNumOutputs);
  }

  inline int NumInputs(bool dual_scale_bias, bool fused_add) {
    if (dual_scale_bias) {
      return static_cast<int>(kNumInputsDualScaleBias);
    } else {
      if (fused_add) {
        return static_cast<int>(kNumInputsScaleBiasAddRelu);
      }
    }
    return kNumInputsScaleBiasRelu;
  }
}  // namespace scale_bias_add_relu

struct ScaleBiasAddReluParam : public dmlc::Parameter<ScaleBiasAddReluParam> {
  // --> bn(a) -->  add --> relu ----> output
  //                 ^
  //                 |
  // --> bn(a) -------
  // bn(a) :

  double eps;
  bool dual_scale_bias;
  bool fused_add;
  bool fused_relu;

  dmlc::optional<int> layout;
  dmlc::optional<int> act_type;

  DMLC_DECLARE_PARAMETER(ScaleBiasAddReluParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be no less than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(dual_scale_bias).set_default(true)
    .describe("dual scale bias");
    DMLC_DECLARE_FIELD(fused_add).set_default(true)
    .describe("dual scale bias");
    DMLC_DECLARE_FIELD(fused_relu).set_default(true)
    .describe("dual scale bias");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NHWC", mshadow::kNHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input and output"
              "NHWC is only supported on GPU.");
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .set_default(dmlc::optional<int>())
    .describe("Fused activation function to be applied.");
  }

  bool operator==(const ScaleBiasAddReluParam& other) const {
    return this->eps == other.eps &&
           this->dual_scale_bias == other.dual_scale_bias &&
           this->fused_add == other.fused_add &&
           this->fused_relu == other.fused_relu &&
           this->layout == other.layout &&
           this->act_type == other.act_type;
  }
};

void ScaleBiasAddReluParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<ScaleBiasAddReluParam> ScaleBiasAddReluSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::ScaleBiasAddReluParam> {
  size_t operator()(const mxnet::op::ScaleBiasAddReluParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.eps);
    ret = dmlc::HashCombine(ret, val.dual_scale_bias);
    ret = dmlc::HashCombine(ret, val.fused_add);
    ret = dmlc::HashCombine(ret, val.fused_relu);
    ret = dmlc::HashCombine(ret, val.layout);
    ret = dmlc::HashCombine(ret, val.act_type ? val.act_type.value() : -1);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ScaleBiasAddReluOp {
 public:
  void Init(ScaleBiasAddReluParam p) {
    LOG(FATAL) << "Only cudnn-based ScaleBiasAddRelu supported.";
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    CHECK(param_.layout.value() == mshadow::kNCHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    LOG(FATAL) << "Only cudnn-based ScaleBiasAddRelu supported.";
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    LOG(FATAL) << "Only cudnn-based ScaleBiasAddRelu supported.";
  }

 private:
  ScaleBiasAddReluParam param_;
};  // class ScaleBiasAddReluOp

template<typename xpu>
void ScaleBiasAddReluCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const ScaleBiasAddReluParam& param = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  if (param.dual_scale_bias) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[scale_bias_add_relu::kxDSBARData].type_flag_, DType, {
      ScaleBiasAddReluOp<xpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    });
  } else {
      if (param.fused_add) {
        MSHADOW_REAL_TYPE_SWITCH(inputs[scale_bias_add_relu::kxSBARData].type_flag_, DType, {
          ScaleBiasAddReluOp<xpu, DType> op;
          op.Init(param);
          op.Forward(ctx, inputs, req, outputs);
        });
      } else {
        MSHADOW_REAL_TYPE_SWITCH(inputs[scale_bias_add_relu::kxSBRData].type_flag_, DType, {
          ScaleBiasAddReluOp<xpu, DType> op;
          op.Init(param);
          op.Forward(ctx, inputs, req, outputs);
        });
      }
  }
}

template<typename xpu>
void ScaleBiasAddReluGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const ScaleBiasAddReluParam& param = nnvm::get<ScaleBiasAddReluParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    ScaleBiasAddReluOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_SCALE_BIAS_ADD_RELU_INL_H_
