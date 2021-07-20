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
 * Copyright (c) 2019 by Contributors
 * \file cudnn_scale_bias_add_relu-inl.h
 * \brief
 * \author Kartikeya Goyal
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_SCALE_BIAS_ADD_RELU_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_SCALE_BIAS_ADD_RELU_INL_H_

#include <mxnet/storage.h>
#include <algorithm>
#include <vector>
#include <set>
#include <mutex>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include "../scale_bias_add_relu-inl.h"
#include "../../../common/cuda/utils.h"
#include "nhwc_batch_norm-inl.h"
#include "nhwc_batch_norm_add_relu-inl.h"
#include "cudnn_common_op.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

/*!
 * \brief The Operator used to perform dual scale bias add relu using cuDNN kernels.
 */
template<typename DType>
class CuDNNScaleBiasAddReluOp {
 public:
  CuDNNScaleBiasAddReluOp()
#if CUDNN_VERSION >= 7600
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK)
#endif
  {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_x_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_x_bn_eq_bias_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_x_bn_eq_scale_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&equiv_z_scale_bias_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_z_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_z_bn_eq_bias_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_z_bn_eq_scale_));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_relu_bitmask_desc_));
  }

  ~CuDNNScaleBiasAddReluOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_x_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_x_bn_eq_bias_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_x_bn_eq_scale_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(equiv_z_scale_bias_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_z_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_z_bn_eq_bias_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_z_bn_eq_scale_));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_relu_bitmask_desc_));
  }

  void Init(const ScaleBiasAddReluParam& param,
            const std::vector<TShape>& in_shape,
            const std::vector<TShape>& out_shape,
            const OpContext& ctx) {
#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    using namespace mshadow;
    this->param_ = param;

    auto cudnn_fwd_compute_type = convertToCuDNNDataType(mshadow::kFloat32);
    dtype_ = DataType<DType>::kCudnnFlag;

    auto effective_layout = param_.layout.value();
    switch (effective_layout) {
      case mshadow::kNWC: effective_layout = mshadow::kNHWC; break;
      default: break;
    }

    MSHADOW_LAYOUT_SWITCH(effective_layout, Layout, {
      format_ = LayoutType<Layout>::kCudnnFlag;
    });

    // Double check to make sure this class supports the operation
    if (param.dual_scale_bias) {
      if (!Supports(param, in_shape[scale_bias_add_relu::kxDSBARData], ctx.run_ctx.ctx.dev_id)) {
        LOG(FATAL) << "Unexpected unsupported use of ScaleBiasAddRelu op.";
      }
    } else {
      if (param.fused_add) {
        if (!Supports(param, in_shape[scale_bias_add_relu::kxSBARData], ctx.run_ctx.ctx.dev_id)) {
        LOG(FATAL) << "Unexpected unsupported use of ScaleBiasAddRelu op.";
        }
      } else {
        if (!Supports(param, in_shape[scale_bias_add_relu::kxSBRData], ctx.run_ctx.ctx.dev_id)) {
        LOG(FATAL) << "Unexpected unsupported use of ScaleBiasAddRelu op.";
        }
      }
    }

    InitDescriptors(in_shape, out_shape, cudnn_fwd_compute_type, ctx);

    // Have cuDNN make a 'plan' for the fused op, returning the temp workspace size required.
    GetTempSize(ctx);

    // Create an equivalent BatchNormParam for the held instance of the NhwcBatchNormOp
    // Not needed for Backward
    bn_param_.eps = 0.0;
    bn_add_relu_param_.eps = 0.0;
    // Not needed for Backward since running mean/var are updated by forward kernel.
    bn_param_.momentum = 0.f;
    bn_add_relu_param_.momentum = 0.f;
    // Finalize kernel can respond to fix_gamma = true
    bn_param_.fix_gamma = false;
    bn_add_relu_param_.fix_gamma = false;
    // use_global_stats will only be true for inference-only graphs where backward is not needed
    bn_param_.use_global_stats = false;
    bn_add_relu_param_.use_global_stats = false;
    // Should have no effect on NHWCBatchNorm::Backward()
    bn_param_.output_mean_var = true;
    bn_add_relu_param_.output_mean_var = true;
    // ScaleBiasAddRelu only supported for NHWC layouts
    CHECK_EQ(effective_layout, mshadow::kNHWC);
    bn_param_.axis = 3;
    bn_add_relu_param_.axis = 3;
    // Only cudnn ScaleBiasAddRelu is implemented
    bn_param_.cudnn_off = false;
    bn_add_relu_param_.cudnn_off = false;
    // Copy act_type value from NormalizeConvolutionParam -> BatchNormParam
    if (param_.act_type.has_value()) {
        if (!param_.dual_scale_bias) {
            bn_param_.act_type = param_.act_type;
        }
    }
    bn_param_.bn_group = 1;
    bn_param_.xbuf_ptr = 0U;
    bn_add_relu_param_.bn_group = 1;
    bn_add_relu_param_.xbuf_ptr = 0U;
#endif  // CUDNN_VERSION >= 7600
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;

    size_t expected_inputs = scale_bias_add_relu::NumInputs(
                             param_.dual_scale_bias, param_.fused_add);
    CHECK_EQ(in_data.size(), expected_inputs);
    CHECK_EQ(out_data.size(), static_cast<size_t>(scale_bias_add_relu::kNumOutputs));

    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, fwd_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);

    DType *data_ptr = nullptr;
    DType *data_z_ptr = nullptr;
    DType *x_scale_ptr = nullptr;
    DType *x_bias_ptr = nullptr;
    DType *z_scale_ptr = nullptr;
    DType *z_bias_ptr = nullptr;

    int data_n_dims = 4;
    int in_features;

    if (param_.dual_scale_bias) {
      in_features = static_cast<int>(Features(in_data[scale_bias_add_relu::kxDSBARData].shape_));

      data_ptr = GetNdPtr(in_data[scale_bias_add_relu::kxDSBARData], data_n_dims, s);
      data_z_ptr = GetNdPtr(in_data[scale_bias_add_relu::kzDSBARData], data_n_dims, s);

      x_scale_ptr = in_data[scale_bias_add_relu::kxDSBARScale].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
      x_bias_ptr = in_data[scale_bias_add_relu::kxDSBARBias].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
      z_scale_ptr = in_data[scale_bias_add_relu::kzDSBARScale].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
      z_bias_ptr = in_data[scale_bias_add_relu::kzDSBARBias].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
    } else {
      if (param_.fused_add) {
        in_features = static_cast<int>(Features(in_data[scale_bias_add_relu::kxSBARData].shape_));
        data_ptr = GetNdPtr(in_data[scale_bias_add_relu::kxSBARData], data_n_dims, s);
        data_z_ptr = GetNdPtr(in_data[scale_bias_add_relu::kzSBARData], data_n_dims, s);

        x_scale_ptr = in_data[scale_bias_add_relu::kxSBARScale].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
        x_bias_ptr = in_data[scale_bias_add_relu::kxSBARBias].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
      } else {
        in_features = static_cast<int>(Features(in_data[scale_bias_add_relu::kxSBRData].shape_));
        data_ptr = GetNdPtr(in_data[scale_bias_add_relu::kxSBRData], data_n_dims, s);
        x_scale_ptr = in_data[scale_bias_add_relu::kxSBRScale].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
        x_bias_ptr = in_data[scale_bias_add_relu::kxSBRBias].get_with_shape<gpu, 1, DType>(
                            Shape1(in_features), s).dptr_;
      }
    }

    DType *out_ptr = GetNdPtr(out_data[scale_bias_add_relu::kOut], data_n_dims, s);
    int C = static_cast<int>(Features(out_data[scale_bias_add_relu::kOut].shape_));
    int64_t NDHW = GetNHW(out_data[scale_bias_add_relu::kOut].shape_);
    int32_t C_int32Elems = ((C + 63) & ~63) / 32;
    int32_t NDHW_int32Elems = (NDHW + 31) & ~31;
    int32_t *bitmask_ptr = out_data[scale_bias_add_relu::kBitMask].get_with_shape<gpu, 1, int32_t>(
                                   Shape1(C_int32Elems * NDHW_int32Elems), s).dptr_;

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    // This operator does not support output blending as specified by alpha or beta.
    // Set data input pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, data_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, x_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, x_bias_ptr);
    if (param_.dual_scale_bias) {
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, data_z_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQSCALE, z_scale_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQBIAS, z_bias_ptr);
    } else {
      if (param_.fused_add) {
        fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, data_z_ptr);
      }
    }

    // Set workspace input pointer in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace.dptr_);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);

    // Set data output pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, out_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ACTIVATION_BITMASK, bitmask_ptr);

    // Launch forward operation
    fwd_op_.Execute(s->dnn_handle_);

#endif  // CUDNN_VERSION < 7600
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    size_t expected_inputs = scale_bias_add_relu::NumInputs(
                             param_.dual_scale_bias, param_.fused_add);
    CHECK_EQ(in_data.size(), expected_inputs);

    // We expect to see an in_grad tensor for all inputs.
    // d_gamma and d_bias (really gradients for the like-named BNStatsFinalize inputs) we output
    // on the gradients corresponding to our saved_mean and saved_inv_stddev inputs.  They will
    // be propogated backward by Finalize() as needed (although in-place is likely in effect).
    // The equiv_scale and equiv_bias gradients are not generated (those are fp16 inputs).
    CHECK_EQ(in_grad.size(), expected_inputs);

    // In the case of dual scale bias add relu -- we want to launch two BNs on for X and one for Z
    Stream<gpu> *s = ctx.get_stream<gpu>();

    // workspace-allocation
    size_t backward_workspace_byte = 0;
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte);
    size_t workspace_size = TensorSizeBytes(workspace);

    #if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
    #else

    if (param_.dual_scale_bias) {
      // X --> [bn(a) + add + relu] --> Y
      //       [         +        ]
      //       [        bn(a)     ]
      //                  ^
      //                  |
      //                  |
      //                  Z
      // dBN for X
      TBlob not_used;
      std::vector<TBlob> bn_bwd_inputs{out_grad[scale_bias_add_relu::kxDSBARData],
            in_data[scale_bias_add_relu::kxDSBARMean],
            in_data[scale_bias_add_relu::kxDSBARInvVar],
            out_grad[scale_bias_add_relu::kBitMask],
            in_data[scale_bias_add_relu::kxDSBARData],
            in_data[scale_bias_add_relu::kxDSBARGamma],
            in_data[scale_bias_add_relu::kxDSBARBeta],
            not_used,
            not_used};
      std::vector<OpReqType> bn_bwd_req{req[scale_bias_add_relu::kxDSBARData],
            req[scale_bias_add_relu::kxDSBARGamma],
            req[scale_bias_add_relu::kxDSBARBeta],
            req[scale_bias_add_relu::kzDSBARData]
      };
      std::vector<TBlob> bn_bwd_outputs{in_grad[scale_bias_add_relu::kxDSBARData],
            in_grad[scale_bias_add_relu::kxDSBARGamma],
            in_grad[scale_bias_add_relu::kxDSBARBeta],
            in_grad[scale_bias_add_relu::kzDSBARData]};
      nhwc_bn_add_relu_op_.Init(bn_add_relu_param_);
      nhwc_bn_add_relu_op_.Backward(ctx, bn_bwd_inputs, bn_bwd_req, bn_bwd_outputs);

      // dBN for Z
      std::vector<TBlob> bnz_bwd_inputs{in_grad[scale_bias_add_relu::kzDSBARData],
          in_data[scale_bias_add_relu::kzDSBARMean],
          in_data[scale_bias_add_relu::kzDSBARInvVar],
          in_data[scale_bias_add_relu::kzDSBARData],
          in_data[scale_bias_add_relu::kzDSBARGamma],
          in_data[scale_bias_add_relu::kzDSBARBeta],
          not_used,
          not_used};
      std::vector<OpReqType> bnz_bwd_req{req[scale_bias_add_relu::kzDSBARData],
          req[scale_bias_add_relu::kzDSBARGamma],
          req[scale_bias_add_relu::kzDSBARBeta]};
      std::vector<TBlob> bnz_bwd_outputs{in_grad[scale_bias_add_relu::kzDSBARData],
          in_grad[scale_bias_add_relu::kzDSBARGamma],
          in_grad[scale_bias_add_relu::kzDSBARBeta]};
      nhwc_bn_op_.Init(bn_param_);
      nhwc_bn_op_.Backward(ctx, bnz_bwd_inputs, bnz_bwd_req, bnz_bwd_outputs, 0);
    } else {
      if (param_.fused_add) {
        TBlob not_used;
        std::vector<TBlob> bn_bwd_inputs{out_grad[scale_bias_add_relu::kxSBARData],
            in_data[scale_bias_add_relu::kxSBARMean],
            in_data[scale_bias_add_relu::kxSBARInvVar],
            out_grad[scale_bias_add_relu::kBitMask],
            in_data[scale_bias_add_relu::kxSBARData],
            in_data[scale_bias_add_relu::kxSBARGamma],
            in_data[scale_bias_add_relu::kxSBARBeta],
            not_used,
            not_used};
        std::vector<OpReqType> bn_bwd_req{req[scale_bias_add_relu::kxSBARData],
            req[scale_bias_add_relu::kxSBARGamma],
            req[scale_bias_add_relu::kxSBARBeta]};
        std::vector<TBlob> bn_bwd_outputs{in_grad[scale_bias_add_relu::kxSBARData],
            in_grad[scale_bias_add_relu::kxSBARGamma],
            in_grad[scale_bias_add_relu::kxSBARBeta],
            in_grad[scale_bias_add_relu::kzSBARData]};
        nhwc_bn_add_relu_op_.Init(bn_add_relu_param_);
        nhwc_bn_add_relu_op_.Backward(ctx, bn_bwd_inputs, bn_bwd_req, bn_bwd_outputs);
      } else {
        // Prepare inputs of NHWCBatchnorm::Backward()
        // Note that the 1st input is the same as the 1st output, i.e. the Batchnorm
        // is operating 'in place' on the gradient as output by the convolution dgrad.
        TBlob not_used;
        std::vector<TBlob> bn_bwd_inputs{out_grad[scale_bias_add_relu::kxSBRData],
            in_data[scale_bias_add_relu::kxSBRMean],
            in_data[scale_bias_add_relu::kxSBRInvVar],
            in_data[scale_bias_add_relu::kxSBRData],
            in_data[scale_bias_add_relu::kxSBRGamma],
            in_data[scale_bias_add_relu::kxSBRBeta],
            not_used,
            not_used};
        std::vector<OpReqType> bn_bwd_req{req[scale_bias_add_relu::kxSBRData],
            req[scale_bias_add_relu::kxSBRGamma],
            req[scale_bias_add_relu::kxSBRBeta]};
        std::vector<TBlob> bn_bwd_outputs{in_grad[scale_bias_add_relu::kxSBRData],
            in_grad[scale_bias_add_relu::kxSBRGamma],
            in_grad[scale_bias_add_relu::kxSBRBeta]};
        // The Init call is made prior to each Backward(), a historical result of transitioning
        // from a symbolic to a gluon (imperative) op style.
        nhwc_bn_op_.Init(bn_param_);
        // Launch batchnorm backward into the primary stream.  This will launch a kernel with
        // an offsetted workspace pointer if dual-stream is enabled.
        nhwc_bn_op_.Backward(ctx, bn_bwd_inputs, bn_bwd_req, bn_bwd_outputs, 0);
      }
    }
    #endif  // CUDNN_VERSION < 7600
  }

/*!
 * \brief Returns whether the scale_bias_add_relu operation described by `param`
 * is supported.
 */
template <typename SupportedParam>
static bool Supports(SupportedParam param,
                     const TShape& in_data_shape,
                     int dev_id) {
    using namespace mshadow;
    static_assert(std::is_same<SupportedParam, ScaleBiasAddReluParam>::value,
                  "Unsupported template specialization of ScaleBiasAddRelu::Supports()");
    // Need cuDNN version >= 7.6
    if (CUDNN_VERSION < 7600)
      return false;
    // Volta (70), Turing (75) and Ampere (80, 86) GPU architectures supported.
    static const std::set<int> supported_arches{70, 75, 80, 86};
    if (supported_arches.count(SMArch(dev_id)) == 0)
      return false;
    // Only kNHWC and kNWC format supported
    auto layout_val = param.layout.value();
    if (layout_val != kNWC && layout_val != kNHWC)
      return false;
    // Must have a multiple of 32 input features 'c' (assumes N..C layout).
    if (in_data_shape[in_data_shape.ndim()-1] % 32 != 0)
      return false;
    // Op parameters are supported, assuming datatype is float16
    return DataType<DType>::kFlag == kFloat16;
  }

 private:
  static bool NeedsOutput(size_t output_index, const std::vector<OpReqType> &req) {
    return (req.size() > output_index) && (req[output_index] != kNullOp);
  }

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

  void InitDescriptors(const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape,
                       cudnnDataType_t cudnn_fwd_compute_type,
                       const OpContext& ctx) {
    using namespace mshadow;
    size_t expected_inputs = scale_bias_add_relu::NumInputs(
                             param_.dual_scale_bias, param_.fused_add);
    CHECK_EQ(in_shape.size(), expected_inputs);
    CHECK_EQ(out_shape.size(), static_cast<size_t>(scale_bias_add_relu::kNumOutputs));

    TShape oshape;
    TShape dxshape;
    TShape dzshape;

    if ( param_.dual_scale_bias ) {
      oshape = out_shape[scale_bias_add_relu::kOut];
      dxshape = in_shape[scale_bias_add_relu::kxDSBARData];
      dzshape = in_shape[scale_bias_add_relu::kzDSBARData];
    } else {
      if ( param_.fused_add ) {
        oshape = out_shape[scale_bias_add_relu::kOut];
        dxshape = in_shape[scale_bias_add_relu::kxSBARData];
        dzshape = in_shape[scale_bias_add_relu::kzSBARData];
      } else {
        oshape = out_shape[scale_bias_add_relu::kOut];
        dxshape = in_shape[scale_bias_add_relu::kxSBRData];
      }
    }

    TShape dstride, ostride;
    dstride = ConvertLayout(Strides<4>(dxshape), param_.layout.value(), kNCHW);
    ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);

    oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
    dxshape = ConvertLayout(dxshape.get<4>(), param_.layout.value(), kNCHW);
    if (param_.dual_scale_bias || param_.fused_add) {
            dzshape = ConvertLayout(dzshape.get<4>(), param_.layout.value(), kNCHW);
    }

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    auto dual_scale_bias_ptr_type = CUDNN_PTR_16B_ALIGNED;

    // Describe i/o tensor pointer alignment for forward fused op
    if ( param_.dual_scale_bias ) {
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                  CUDNN_PARAM_ZDATA_PLACEHOLDER,
                                  CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
                                  CUDNN_PARAM_YDATA_PLACEHOLDER}, CUDNN_PTR_16B_ALIGNED);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                 CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER}, dual_scale_bias_ptr_type);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER,
                                 CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER}, dual_scale_bias_ptr_type);
    } else {
      if ( param_.fused_add ) {
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                  CUDNN_PARAM_ZDATA_PLACEHOLDER,
                                  CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
                                  CUDNN_PARAM_YDATA_PLACEHOLDER}, CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                 CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER}, dual_scale_bias_ptr_type);
      } else {
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                  CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
                                  CUDNN_PARAM_YDATA_PLACEHOLDER}, CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                 CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER}, dual_scale_bias_ptr_type);
      }
    }
    std::vector<int> dxshape_buffer(dxshape.ndim());
    nnvm::ShapeTypeCast(dxshape.begin(), dxshape.end(), dxshape_buffer.data());

    std::vector<int> dstride_buffer(dstride.ndim());
    nnvm::ShapeTypeCast(dstride.begin(), dstride.end(), dstride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_x_desc_,
                                          dtype_,
                                          static_cast<int>(dxshape.ndim()),
                                          dxshape_buffer.data(),
                                          dstride_buffer.data()));
    if (param_.dual_scale_bias || param_.fused_add) {
      std::vector<int> dzshape_buffer(dzshape.ndim());
      nnvm::ShapeTypeCast(dzshape.begin(), dzshape.end(), dzshape_buffer.data());
      CUDNN_CALL(cudnnSetTensorNdDescriptor(in_z_desc_,
                                          dtype_,
                                          static_cast<int>(dzshape.ndim()),
                                          dzshape_buffer.data(),
                                          dstride_buffer.data()));
    }

    std::vector<int> oshape_buffer(oshape.ndim());
    nnvm::ShapeTypeCast(oshape.begin(), oshape.end(), oshape_buffer.data());
    std::vector<int> ostride_buffer(ostride.ndim());
    nnvm::ShapeTypeCast(ostride.begin(), ostride.end(), ostride_buffer.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          oshape_buffer.data(),
                                          ostride_buffer.data()));

    // Always set scale/bias descriptors
    int in_features;
    if (param_.dual_scale_bias) {
      in_features = static_cast<int>(Features(in_shape[scale_bias_add_relu::kxDSBARData]));
    } else {
      if (param_.fused_add) {
        in_features = static_cast<int>(Features(in_shape[scale_bias_add_relu::kxSBARData]));
      } else {
        in_features = static_cast<int>(Features(in_shape[scale_bias_add_relu::kxSBRData]));
      }
    }

    TShape equiv_scale_bias_shape = TShape({in_features});
    std::vector<int> equiv_scale_shape = {1, static_cast<int>(in_features), 1, 1};
    std::vector<int> equiv_scale_stride = {static_cast<int>(in_features), 1,
                                          static_cast<int>(in_features),
                                          static_cast<int>(in_features)};

    CUDNN_CALL(cudnnSetTensorNdDescriptor(equiv_scale_bias_desc_,
                                        dtype_,
                                        static_cast<int>(equiv_scale_shape.size()),
                                        &equiv_scale_shape[0],
                                        &equiv_scale_stride[0]));
    if ( param_.dual_scale_bias ) {
      CUDNN_CALL(cudnnSetTensorNdDescriptor(equiv_z_scale_bias_desc_,
                                        dtype_,
                                        static_cast<int>(equiv_scale_shape.size()),
                                        &equiv_scale_shape[0],
                                        &equiv_scale_stride[0]));
    }

    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, equiv_scale_bias_desc_);
    if ( param_.dual_scale_bias ) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC, equiv_z_scale_bias_desc_);
    }

    if ( !param_.dual_scale_bias ) {
      if ( param_.fused_add ) {
        TShape equiv_scale = in_shape[scale_bias_add_relu::kxSBARScale];
        CHECK_EQ(equiv_scale, in_shape[scale_bias_add_relu::kxSBARBias]) <<
          "Expecting equal equivalent-scale and equivalent-bias input tensor shapes.";
      }
    }

    int C = in_features;
    int64_t NDHW = GetNHW(out_shape[scale_bias_add_relu::kOut]);
    int32_t C_int32Elems = ((C + 63) & ~63) / 32;
    int32_t NDHW_int32Elems = (NDHW + 31) & ~31;
    std::vector<int> bitmask_shape = {NDHW_int32Elems, C_int32Elems, 1};
    std::vector<int> bitmask_stride = {C_int32Elems, 1, C_int32Elems};
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_relu_bitmask_desc_,
                                          CUDNN_DATA_INT32,
                                          3,
                                          &bitmask_shape[0],
                                          &bitmask_stride[0]));

    // Set activation descriptor, default is no activation
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (param_.act_type.has_value()) {
      CHECK_EQ(param_.act_type.value(), activation::kReLU) <<
        "Only relu activation supported in scale bias add relu.";
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_, mode, nan_prop, dummy_clip));
    // Currently, the only way to turn off activation is to not set the descriptor
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC, activation_desc_);
    }

    // Set desc pointers
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_x_desc_);
    if ( param_.dual_scale_bias || param_.fused_add ) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ZDESC, in_z_desc_);
    }
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC, out_relu_bitmask_desc_);
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);
#endif  // CUDNN_VERSION < 7600
  }  // InitDescriptors

  void GetTempSize(const OpContext& ctx) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    // Make op plan for forward op and set forward workspace size
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(s->dnn_handle_);
  }

  int *CastTShapeToIntPtr(const TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = NULL;
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

  // Converts a TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const TShape &s) {
    uint32_t ndim = s.ndim();
    TShape strides(ndim, -1);
    for (uint32_t i = 0; i != ndim; ++i)
      strides[i] = s.ProdShape(i+1, ndim);
    return strides.get<dim>();
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words =
      std::max<size_t>(1, RoundToMultiple(size_bytes, sizeof(DType)) / sizeof(DType));
    return ctx.requested[scale_bias_add_relu::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(size_words), s);
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }

  int64_t GetNHW(const TShape &dshape) {
    uint64_t nhw = 1;
    switch (dshape.ndim()) {
      case 4:
        {
          auto sh = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
          nhw = sh[0] * sh[2] * sh[3]; break;
        }
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return nhw;
  }

  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const TShape &dshape) {
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

  // Temp workspace size in bytes needed for Forward() operation.
  size_t fwd_workspace_byte_;

  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_x_desc_;
  cudnnTensorDescriptor_t in_z_desc_;

  cudnnTensorDescriptor_t in_x_bn_eq_bias_;
  cudnnTensorDescriptor_t in_x_bn_eq_scale_;

  cudnnTensorDescriptor_t in_z_bn_eq_bias_;
  cudnnTensorDescriptor_t in_z_bn_eq_scale_;

  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t out_relu_bitmask_desc_;

  cudnnTensorFormat_t format_;

  ScaleBiasAddReluParam param_;
  cudnnTensorDescriptor_t equiv_scale_bias_desc_;
  cudnnTensorDescriptor_t equiv_z_scale_bias_desc_;

  // The assumption of the fwd_op plan as to whether sum and sum_of_squares outputs are populated.
  bool fwd_op_plan_output_stats_;

  // Batchnorm operation instance for bn(a)+relu Backward()
  NhwcBatchNormOp<DType> nhwc_bn_op_;
  BatchNormParam bn_param_;

  // Batchnorm operator instance for bn(a)+add+relu Backward()
  NhwcBatchNormAddReluOp<DType> nhwc_bn_add_relu_op_;
  BatchNormAddReluParam bn_add_relu_param_;

  // Specifies activation parameters: relu
  cudnnActivationDescriptor_t activation_desc_;

#if CUDNN_VERSION >= 7600
  // New normalized convolution forward fused-op
  CuDNNCommonOp fwd_op_;
#endif
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_SCALE_BIAS_ADD_RELU_INL_H_
