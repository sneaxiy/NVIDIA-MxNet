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
 * \file softmax-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_NN_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_SOFTMAX_INL_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {
namespace mxnet_op {

struct softmax_fwd {
  template<typename AType>
  MSHADOW_XINLINE static AType Map(float a, AType b) {
    return AType(expf(a)/b);
  }

  template<typename AType>
  MSHADOW_XINLINE static AType Map(double a, AType b) {
    return AType(exp(a)/b);
  }
};


struct log_softmax_fwd {
  template<typename DType>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return a - logf(b);
  }

  template<typename DType>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return a - log(b);
  }
};


template<typename OP, bool negate, typename AType, typename DType, typename OType,
         typename IType, int ndim>
inline void Softmax(Stream<cpu> *s, DType *in, OType *out, IType *length,
                    Shape<ndim> shape, int axis, const DType temperature) {
  index_t M = shape[axis];
  if (M == 0) return;
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  if (length == nullptr) {
    #pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);

      DType mmax = negate ? -in[base] : in[base];
      DType val;
      for (index_t j = 1; j < M; ++j) {
        val = negate ? -in[base + j*sa] : in[base + j*sa];
        if (mmax < val) mmax = val;
      }

      AType sum = AType(0);
      DType in_val;
      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          sum += std::exp(in_val - mmax);
        }

        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          out[base + j*sa] = OP::Map(in_val - mmax, sum);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          sum += std::exp((in_val - mmax)/temperature);
        }

        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          out[base + j*sa] = OP::Map((in_val - mmax)/temperature, sum);
        }
      }
    }
  } else {
    #pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t len = static_cast<index_t>(length[i]);
      index_t base = unravel_dot(i, sshape, stride);

      DType mmax = negate ? -in[base] : in[base];
      DType val;
      for (index_t j = 1; j < len; ++j) {
        val = negate ? -in[base + j*sa] : in[base + j*sa];
        if (mmax < val) mmax = val;
      }
      for (index_t j = len; j < M; ++j) {
        out[base + j*sa] = OType(0.0f);
      }

      AType sum = AType(0);
      DType in_val;
      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      if (temperature == 1.0) {
        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          sum += std::exp(in_val - mmax);
        }

        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          out[base + j*sa] = OP::Map(in_val - mmax, sum);
        }
      } else {
        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          sum += std::exp((in_val - mmax)/temperature);
        }

        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j*sa] : in[base + j*sa];
          out[base + j*sa] = OP::Map((in_val - mmax)/temperature, sum);
        }
      }
    }
  }
}

struct softmax_bwd {
  template<typename DType, typename AType>
  MSHADOW_XINLINE static AType Map(DType ograd, DType out, AType sum) {
    return AType(out * (ograd - sum));
  }
};


struct log_softmax_bwd {
  template<typename AType>
  MSHADOW_XINLINE static AType Map(float ograd, float out, AType sum) {
    return AType(ograd - expf(out)*sum);
  }

  template<typename AType>
  MSHADOW_XINLINE static AType Map(double ograd, double out, AType sum) {
    return AType(ograd - exp(out)*sum);
  }
};

template<typename OP1, typename OP2, int Req, bool negate,
         typename AType, typename DType, typename OType, typename IType, int ndim>
inline void SoftmaxGrad(Stream<cpu> *s, OType *out, OType *ograd,
                        DType *igrad, IType *length, Shape<ndim> shape,
                        int axis, const DType temperature) {
  index_t M = shape[axis];
  if (M == 0) return;
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  if (length != nullptr) {
    #pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);
      index_t len = static_cast<index_t>(length[i]);

      AType sum = AType(0);
      for (index_t j = 0; j < len; ++j) {
        sum += OP1::Map(ograd[base + j*sa], out[base + j*sa]);
      }

      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      DType final_result;
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ?
                         -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) :
                         OP2::Map(ograd[base + j*sa], out[base + j*sa], sum);
          final_result = (j < len) ? final_result : DType(0.0f);
          KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ?
                         -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature :
                         OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature;
          final_result = (j < len) ? final_result : DType(0.0f);
          KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
        }
      }
    }
  } else {
    #pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);

      AType sum = AType(0);
      for (index_t j = 0; j < M; ++j) {
        sum += OP1::Map(ograd[base + j*sa], out[base + j*sa]);
      }

      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      DType final_result;
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ?
                         -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) :
                         OP2::Map(ograd[base + j*sa], out[base + j*sa], sum);
          KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ?
                         -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature :
                         OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature;
          KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
        }
      }
    }
  }
}

}  // namespace mxnet_op


struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  int axis;
  dmlc::optional<double> temperature;
  dmlc::optional<int> dtype;
  dmlc::optional<bool> use_length;
  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
    .describe("The axis along which to compute softmax.");
    DMLC_DECLARE_FIELD(temperature).set_default(dmlc::optional<double>())
    .describe("Temperature parameter in softmax");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(dmlc::optional<int>())
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to the same as input's dtype if not defined (dtype=None).");
    DMLC_DECLARE_FIELD(use_length)
    .set_default(dmlc::optional<bool>(false))
    .describe("Whether to use the length input as a mask over the data input.");
  }

  bool operator==(const SoftmaxParam& other) const {
    return this->axis == other.axis &&
           this->temperature == other.temperature &&
           this->dtype == other.dtype &&
           this->use_length == other.use_length;
  }
};

static inline bool softmax_has_dtype_override(const nnvm::NodeAttrs& attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  return param.dtype.has_value() && param.dtype.value() != -1;
}

static inline bool softmax_use_length(const nnvm::NodeAttrs& attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  return param.use_length.value();
}

static inline bool SoftmaxOpType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), softmax_use_length(attrs) ? 2U : 1U);

  if (softmax_has_dtype_override(attrs)) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
    type_assign(&(*in_attrs)[0], (*out_attrs)[0]);
    return true;
  } else {
    std::vector<int> tmp = {in_attrs->at(0)};
    return ElemwiseType<1, 1>(attrs, &tmp, out_attrs);
  }
}

static inline bool SoftmaxOpShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector *in_attrs,
                                  mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.use_length.value() ? 2U : 1U);

  if (param.use_length.value()) {
    mxnet::TShape& dshape = in_attrs->at(0);
    mxnet::TShape tmp_shape((dshape.ndim() == 1) ? 1U : dshape.ndim() - 1, 1);
    int j = 0;
    int axis = param.axis != -1 ? param.axis : dshape.ndim() - 1;
    for (int i = 0; i < dshape.ndim(); ++i) {
      if (i != axis) {
        tmp_shape[j++] = dshape[i];
      }
    }
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);
  }
  mxnet::ShapeVector tmp = {in_attrs->at(0)};
  return ElemwiseShape<1, 1>(attrs, &tmp, out_attrs);
}

static inline bool SoftmaxGradOpShape(const nnvm::NodeAttrs& attrs,
                                      mxnet::ShapeVector *in_attrs,
                                      mxnet::ShapeVector *out_attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      mxnet::ShapeVector ins = {in_attrs->at(0), in_attrs->at(1), in_attrs->at(3)};
      mxnet::ShapeVector dgrad = {out_attrs->at(0)};
      bool res = ElemwiseShape<3, 1>(attrs, &ins, &dgrad);
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, ins[0]);
      SHAPE_ASSIGN_CHECK(*in_attrs, 1, ins[1]);
      SHAPE_ASSIGN_CHECK(*in_attrs, 3, ins[2]);
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, dgrad[0]);
      mxnet::ShapeVector length = {in_attrs->at(2)};
      mxnet::ShapeVector lgrad = {out_attrs->at(1)};
      res = (res && ElemwiseShape<1, 1>(attrs, &length, &lgrad));
      SHAPE_ASSIGN_CHECK(*in_attrs, 2, length[0]);
      SHAPE_ASSIGN_CHECK(*out_attrs, 1, lgrad[0]);
      return res;
    } else {
      return ElemwiseShape<3, 1>(attrs, in_attrs, out_attrs);
    }
  } else {
    return ElemwiseShape<2, 1>(attrs, in_attrs, out_attrs);
  }
}

static inline bool SoftmaxGradOpType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), softmax_use_length(attrs) ? 2U : 1U);
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    CHECK_EQ(in_attrs->size(), softmax_use_length(attrs) ? 4U : 3U);
    int in_dtype = (*in_attrs)[1];
    int out_dtype = (*in_attrs)[softmax_use_length(attrs) ? 3 : 2];
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_dtype);
    if (softmax_use_length(attrs)) {
      TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
    }

    bool inferred_io = (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1;
    bool inferred_length = !softmax_use_length(attrs) ||
                           ((*out_attrs)[1] != -1 && (*in_attrs)[1] != -1);

    return inferred_io && inferred_length;
  } else {
    CHECK_EQ(in_attrs->size(), 2U);
    int out_dtype = (*in_attrs)[1];
    TYPE_ASSIGN_CHECK(*out_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);

    return (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1;
  }
}

static inline std::vector<std::pair<int, int> >
SoftmaxGradOpInplaceOption(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}, {2, 1}, {3, 0}};
    } else {
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}, {2, 0}};
    }
  } else {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  }
}

static inline uint32_t SoftmaxGradOpNumInputs(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    return softmax_use_length(attrs) ? 4 : 3;
  }
  return 2;
}

static inline std::vector<std::string> SoftmaxGradOpInputNames(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      return std::vector<std::string>{"ograd", "data", "length", "output"};
    } else {
      return std::vector<std::string>{"ograd", "data", "output"};
    }
  } else {
    return std::vector<std::string>{"ograd", "output"};
  }
}

struct SoftmaxFGradient {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    if (softmax_has_dtype_override(n->attrs) || softmax_use_length(n->attrs)) {
      return ElemwiseGradUseInOut {op_name}(n, ograds);
    } else {
      return ElemwiseGradUseOut {op_name}(n, ograds);
    }
  }
};

template<typename xpu, typename OP, bool negate = false>
void SoftmaxCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp || inputs[0].Size() == 0U) return;
  CHECK_NE(req[0], kAddTo);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature = param.temperature.has_value() ?
    param.temperature.value() : 1.0;
  mxnet::TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", false);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for softmax with float16 inputs. "
                    "See https://mxnet.apache.org/api/faq/env_var "
                    "for more details.");
  }

  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, DType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      int type = kInt32;
      if (param.use_length.value()) {
        CHECK(inputs.size() > 1)
          << "Mask needs to be provided when using softmax with use_length=True.";
        type = inputs[1].type_flag_;
      }
      MXNET_INT32_INT64_TYPE_SWITCH(type, IType, {
        IType* mask_ptr = nullptr;
        if (param.use_length.value()) {
          mask_ptr = inputs[1].dptr<IType>();
        }
        if (safe_acc) {
          if (shape.ndim() == 2) {
            Softmax<OP, negate, AType>(
              ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<OType>(), mask_ptr, shape.get<2>(),
              axis, static_cast<DType>(temperature));
          } else {
            Softmax<OP, negate, AType>(
              ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<OType>(), mask_ptr, shape.get<3>(),
              axis, static_cast<DType>(temperature));
          }
        } else {
          if (shape.ndim() == 2) {
            Softmax<OP, negate, DType>(
              ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<OType>(), mask_ptr, shape.get<2>(),
              axis, static_cast<DType>(temperature));
          } else {
            Softmax<OP, negate, DType>(
              ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<OType>(), mask_ptr, shape.get<3>(),
              axis, static_cast<DType>(temperature));
          }
        }
      });
    });
  });
}

#if MXNET_USE_CUDA

struct SoftmaxRTCCompute {
  std::string OP;
  bool negate = false;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

struct SoftmaxRTCGradCompute {
  std::string OP1;
  std::string OP2;
  bool negate = false;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif

template<typename xpu, typename OP1, typename OP2, bool negate = false>
void SoftmaxGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (softmax_use_length(attrs)) {
    MXNET_INT32_INT64_TYPE_SWITCH(inputs[2].type_flag_, IType, {
      if (req[1] != kNullOp) {
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(
          ctx.get_stream<xpu>(), outputs[1].Size(), outputs[1].dptr<IType>());
      }
    });
  }
  if (req[0] == kNullOp) return;
  const int itype = softmax_use_length(attrs) ? inputs[2].type_flag_ : kInt32;
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature = param.temperature.has_value() ?
    param.temperature.value() : 1.0;
  mxnet::TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);

  int out_idx = softmax_has_dtype_override(attrs) ? 2 : 1;
  out_idx = softmax_use_length(attrs) ? 3 : out_idx;
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", false);

  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, OType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MXNET_INT32_INT64_TYPE_SWITCH(itype, IType, {
          IType * length_ptr = nullptr;
          if (softmax_use_length(attrs)) {
            length_ptr = inputs[2].dptr<IType>();
          }
          if (safe_acc) {
            if (shape.ndim() == 2) {
              SoftmaxGrad<OP1, OP2, Req, negate, AType>(
                  ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
                  inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
                  length_ptr, shape.get<2>(), axis,
                  static_cast<DType>(temperature));
            } else {
              SoftmaxGrad<OP1, OP2, Req, negate, AType>(
                  ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
                  inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
                  length_ptr, shape.get<3>(), axis,
                  static_cast<DType>(temperature));
            }
          } else {
            if (shape.ndim() == 2) {
              SoftmaxGrad<OP1, OP2, Req, negate, DType>(
                  ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
                  inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
                  length_ptr, shape.get<2>(), axis,
                  static_cast<DType>(temperature));
            } else {
              SoftmaxGrad<OP1, OP2, Req, negate, DType>(
                  ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
                  inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
                  length_ptr, shape.get<3>(), axis,
                  static_cast<DType>(temperature));
            }
          }
        });
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::SoftmaxParam> {
  size_t operator()(const mxnet::op::SoftmaxParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.axis);
    ret = dmlc::HashCombine(ret, val.temperature);
    ret = dmlc::HashCombine(ret, val.dtype);
    ret = dmlc::HashCombine(ret, val.use_length);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_NN_SOFTMAX_INL_H_
