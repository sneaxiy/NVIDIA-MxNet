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
 * \file np_elemwise_unary_op_basic.cu
 * \brief GPU Implementation of numpy unary functions.
 */
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(__name$, __kernel$)       \
  NNVM_REGISTER_OP(__name$)                                               \
  .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{#__kernel$})

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npx_relu, relu);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npx_sigmoid, sigmoid);

NNVM_REGISTER_OP(_np_copy)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_negative, negation);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_reciprocal, reciprocal);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_absolute, abs);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sign, sign);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_rint, rint);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_ceil, ceil);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_floor, floor);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_bitwise_not, bitwise_not);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_trunc, trunc);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_fix, fix);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_square, square);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sqrt, sqrt);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cbrt, cbrt);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_exp, exp);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log, log);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log10, log10);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log2, log2);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log1p, log1p);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_expm1, expm1);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_logical_not, np_logical_not);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_isnan, isnan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_isinf, isinf);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_isposinf, isposinf);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_isneginf, isneginf);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_isfinite, isfinite);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sin, sin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cos, cos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_tan, tan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arcsin, arcsin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arccos, arccos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arctan, arctan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_degrees, degrees);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_radians, radians);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sinh, sinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cosh, cosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_tanh, tanh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arcsinh, arcsinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arccosh, arccosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arctanh, arctanh);

NNVM_REGISTER_OP(_npi_around)
.set_attr<FCompute>("FCompute<gpu>", AroundOpForward<gpu>);

NNVM_REGISTER_OP(_npi_nan_to_num)
.set_attr<FCompute>("FCompute<gpu>", NumpyNanToNumOpForward<gpu>);

NNVM_REGISTER_OP(_npi_backward_nan_to_num)
.set_attr<FCompute>("FCompute<gpu>", NumpyNanToNumOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
