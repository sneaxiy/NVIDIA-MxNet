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
 * \file instance_norm-inl.h
 * \brief Reproducing paper Instance Normalization: The Missing Ingredient for
 * Fast Stylization, D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_INSTANCE_NORM_INL_H_
#define MXNET_OPERATOR_INSTANCE_NORM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace instance_norm {
enum InstanceNormInputs { kData, kGamma, kBeta };
enum InstanceNormOutputs { kOut, kMean, kVar };
enum InstanceNormBackResource { kTempSpace };
}  // namespace instance_norm

struct InstanceNormParam : public dmlc::Parameter<InstanceNormParam> {
  float eps;
  DMLC_DECLARE_PARAMETER(InstanceNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f).describe(
        "An `epsilon` parameter to prevent division by 0.");
  }
};  // struct InstanceNormParam

template <typename xpu, typename DType>
class InstanceNormOp : public Operator {
 public:
  explicit InstanceNormOp(InstanceNormParam param) { param_ = param; }
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using AType = typename mxnet_op::AccType<DType>::type;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);

    CHECK_GE(in_data[instance_norm::kData].ndim(), 3)
        << "InstanceNorm only supports input tensors of rank >= 3.";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[instance_norm::kData].size(0);
    int c = in_data[instance_norm::kData].size(1);
    int rest_dim =
        static_cast<int>(in_data[instance_norm::kData].Size() / n / c);
    Shape<2> s2 = Shape2(n * c, rest_dim);
    const AType scale = static_cast<AType>(1.0 / rest_dim);
    const AType eps = static_cast<AType>(param_.eps);
    // Get Inputs
    Tensor<xpu, 2, DType> data =
        in_data[instance_norm::kData].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 1, AType> gamma =
        in_data[instance_norm::kGamma].get<xpu, 1, AType>(s);
    Tensor<xpu, 1, AType> beta = in_data[instance_norm::kBeta].get<xpu, 1, AType>(s);
    // Get Outputs
    Tensor<xpu, 2, DType> out =
        out_data[instance_norm::kOut].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 1, AType> var = out_data[instance_norm::kVar].FlatTo1D<xpu, AType>(s);
    Tensor<xpu, 1, AType> mean =
        out_data[instance_norm::kMean].FlatTo1D<xpu, AType>(s);
    // Calculate mean + var
    mean = scale * sumall_except_dim<0>(tcast<AType>(data));
    var = scale * sumall_except_dim<0>(F<mshadow_op::square>(
                  tcast<AType>(data) - broadcast<0>(mean, data.shape_)));
    Assign(out, req[instance_norm::kOut],
           tcast<DType>(
                        broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), out.shape_) *
                        (tcast<AType>(data) - broadcast<0>(mean, data.shape_)) /
                        F<mshadow_op::square_root>(broadcast<0>(var + eps, data.shape_)) +
                        broadcast<0>(reshape(repmat(beta, n), Shape1(n * c)), out.shape_)));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using AType = typename mxnet_op::AccType<DType>::type;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);

    CHECK_GE(in_data[instance_norm::kData].ndim(), 3)
        << "InstanceNorm only supports input tensors of rank > 2.";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[instance_norm::kData].size(0);
    int c = in_data[instance_norm::kData].size(1);
    int rest_dim =
        static_cast<int>(in_data[instance_norm::kData].Size() / n / c);
    Shape<2> s2 = Shape2(n * c, rest_dim);
    Shape<3> s3 = Shape3(n, c, rest_dim);
    const AType scale = static_cast<AType>(1.0 / rest_dim);
    const AType scale2 = static_cast<AType>(2.0 / rest_dim);
    const AType eps = static_cast<AType>(param_.eps);
    // Get Inputs
    Tensor<xpu, 2, DType> data =
        in_data[instance_norm::kData].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> gdata =
        in_grad[instance_norm::kData].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 1, AType> gamma =
        in_data[instance_norm::kGamma].get<xpu, 1, AType>(s);
    Tensor<xpu, 1, AType> ggamma =
        in_grad[instance_norm::kGamma].get<xpu, 1, AType>(s);
    Tensor<xpu, 1, AType> gbeta = in_grad[instance_norm::kBeta].get<xpu, 1, AType>(s);
    // Get Outputs
    Tensor<xpu, 2, DType> gout =
        out_grad[instance_norm::kOut].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 1, AType> var = out_data[instance_norm::kVar].FlatTo1D<xpu, AType>(s);
    Tensor<xpu, 1, AType> mean =
        out_data[instance_norm::kMean].FlatTo1D<xpu, AType>(s);
    // Get temp space
    Tensor<xpu, 2, AType> workspace =
        ctx.requested[instance_norm::kTempSpace].get_space_typed<xpu, 2, AType>(
            mshadow::Shape2(3, mean.shape_[0]), s);
    Tensor<xpu, 1, AType> gmean = workspace[0];
    Tensor<xpu, 1, AType> gvar = workspace[1];
    Tensor<xpu, 1, AType> tmp = workspace[2];

    // calculate temps

    gvar = sumall_except_dim<0>(
        (tcast<AType>(gout) *
         broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), data.shape_)) *
        (tcast<AType>(data) - broadcast<0>(mean, data.shape_)) * scalar<AType>(-0.5f) *
         F<mshadow_op::power>(broadcast<0>(var + eps, data.shape_),
                              scalar<AType>(-1.5f)));

    gmean = sumall_except_dim<0>(
        tcast<AType>(gout) *
        broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), data.shape_));
    gmean *= scalar<AType>(-1.0f) / F<mshadow_op::square_root>(var + eps);
    tmp = scale * sumall_except_dim<0>(
                  scalar<AType>(-2.0f) * (tcast<AType>(data) - broadcast<0>(mean, data.shape_)));
    tmp *= gvar;
    gmean += tmp;

    // Calculate grads
    Assign(gbeta, req[instance_norm::kBeta],
           sumall_except_dim<0>(swapaxis<1, 0>(reshape(tcast<AType>(gout), s3))));
    Assign(ggamma, req[instance_norm::kGamma],
           sumall_except_dim<0>(swapaxis<1, 0>(
             reshape(tcast<AType>(gout) * (tcast<AType>(data) - broadcast<0>(mean, data.shape_)) /
                     F<mshadow_op::square_root>(broadcast<0>(var + eps, data.shape_)),
                       s3))));

    Assign(gdata, req[instance_norm::kData], tcast<DType>(
        tcast<AType>(gout) *
        broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), data.shape_) *
        broadcast<0>(scalar<AType>(1.0f) / F<mshadow_op::square_root>(var + eps), data.shape_) +
        broadcast<0>(gvar, data.shape_) * scale2 *
        (tcast<AType>(data) - broadcast<0>(mean, data.shape_)) +
        broadcast<0>(gmean, data.shape_) * scale));
  }

 private:
  InstanceNormParam param_;
};  // class InstanceNormOp

template <typename xpu>
Operator *CreateOp(InstanceNormParam param, int dtype);

#if DMLC_USE_CXX11
class InstanceNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;

    in_shape->at(1) = mxnet::TShape(Shape1(dshape[1]));
    in_shape->at(2) = mxnet::TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";

    // If we want InstanceNorm to behave like BatchNorm, we'll use this:

    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;

    for (size_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (size_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype_param);
    int n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (int i = 1; i < n_out; ++i ) out_type->push_back(dtype_param);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new InstanceNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "InstanceNorm"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[instance_norm::kOut], out_data[instance_norm::kMean],
            out_data[instance_norm::kVar], in_data[instance_norm::kData],
            in_data[instance_norm::kGamma]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 3; }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator *CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  InstanceNormParam param_;
};      // InstanceNormProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_INSTANCE_NORM_INL_H_
