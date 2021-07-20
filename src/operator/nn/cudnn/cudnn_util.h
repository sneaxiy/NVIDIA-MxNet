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
 * Copyright (c) 2021 by Contributors
 * \file cudnn_util.h
 * \brief
 * \author Dick Carter
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_UTIL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_UTIL_H_
#include <cudnn.h>
#include <dmlc/logging.h>

#include <algorithm>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

// Round a value 'x' up to the next multiple of 'multiple'
inline size_t RoundToMultiple(size_t x, size_t multiple) {
  size_t retVal = ((x + multiple - 1) / multiple) * multiple;
  return retVal;
}

// Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
// Always allocates at least one word.
template<typename ElemType>
inline mshadow::Tensor<gpu, 1, ElemType> AllocateTempWorkspaceTyped(const OpContext &ctx,
                                                                    size_t size_bytes,
                                                                    int temp_resource_id) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  size_t size_words =
    std::max<size_t>(1, RoundToMultiple(size_bytes, sizeof(ElemType)) / sizeof(ElemType));
  auto wksp_shape = mshadow::Shape1(size_words);
  // Check that we haven't overflowed mshadow::Shape capabilities (e.g. with index_t == int32_t)
  using mshadow_shape_size_t = decltype(wksp_shape.Size());
  auto wksp_num_elems = static_cast<std::make_unsigned_t<mshadow_shape_size_t>>(wksp_shape.Size());
  CHECK_EQ(size_words, wksp_num_elems) << "Workspace request too big: " << size_bytes << " bytes";
  return ctx.requested[temp_resource_id].get_space_typed<gpu, 1, ElemType>(wksp_shape, s);
}

// Use a double wksp elem type to help the tensor elem count from overflowing mshadow::index_t
inline mshadow::Tensor<gpu, 1, double> AllocateTempWorkspace(const OpContext &ctx,
                                                             size_t size_bytes,
                                                             int temp_resource_id) {
  return AllocateTempWorkspaceTyped<double>(ctx, size_bytes, temp_resource_id);
}

// Returns the size in bytes of the 1D Tensor of words.
template<typename ElemType>
size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, ElemType> &tensor) {
  // Interpret Size() as positive (helps expand useful range when index_t == int32_t)
  using msize_t = decltype(tensor.MSize());
  auto num_elems = static_cast<std::make_unsigned_t<msize_t>>(tensor.MSize());
  return num_elems * sizeof(ElemType);
}

#endif  // MXNET_USE_CUDNN == 1
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_UTIL_H_
