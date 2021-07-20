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
 * \file allclose_op.cu
 * \brief GPU Implementation of allclose op
 * \author Andrei Ivanov
 */
#include "./allclose_op-inl.h"
#include <cub/cub.cuh>
#include "../../common/cuda/utils.h"

namespace mxnet {
namespace op {

template<>
size_t GetAdditionalMemoryLogical<gpu>(mshadow::Stream<gpu> *s, const int num_items) {
  return GetCublasMemorySize<INTERM_DATA_TYPE, INTERM_DATA_TYPE>(num_items, s,
                                                                 cub::DeviceReduce::Min);
}

template<>
void GetResultLogical<gpu>(mshadow::Stream<gpu> *s, INTERM_DATA_TYPE *workMem,
                           size_t extraStorageBytes, int nItems, INTERM_DATA_TYPE *outPntr) {
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  cub::DeviceReduce::Min(workMem + nItems, extraStorageBytes, workMem, outPntr, nItems, stream);
}

NNVM_REGISTER_OP(_contrib_allclose)
.set_attr<FCompute>("FCompute<gpu>", AllClose<gpu>);

}  // namespace op
}  // namespace mxnet
