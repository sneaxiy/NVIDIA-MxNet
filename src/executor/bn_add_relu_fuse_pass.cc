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
 * \file bn_add_relu_fuse_pass.cc
 * \brief optimization pass which fuse add_relu into BatchNorm
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/match.h>
#include <mxnet/operator.h>

#include "./exec_pass.h"
#include "../operator/nn/batch_norm-inl.h"
#include "../operator/nn/batch_norm_add_relu-inl.h"
#include "../common/cuda/utils.h"

namespace mxnet {
namespace exec {

using namespace mxnet::op;

using mxnet::Match;

namespace {

  void FuseBNAddReluNode(const nnvm::ObjectPtr bn, const nnvm::ObjectPtr add,
                         const nnvm::ObjectPtr relu, const nnvm::NodeEntry other) {
    static const Op* bn_add_relu_op = Op::Get("BatchNormAddRelu");
    relu->attrs.op = bn_add_relu_op;
    relu->attrs.name = bn->attrs.name + "_add_relu";
    relu->attrs.dict = bn->attrs.dict;
    relu->attrs.dict.erase("act_type");  // BatchNormAddRelu does not have "act_type" parameter
    relu->inputs.resize(6);
    relu->inputs[0] = bn->inputs[0];  // data
    relu->inputs[1] = bn->inputs[1];  // gamma
    relu->inputs[2] = bn->inputs[2];  // beta
    relu->inputs[3] = bn->inputs[3];  // moving_mean
    relu->inputs[4] = bn->inputs[4];  // moving_var
    relu->inputs[5] = other;          // addend
    bn_add_relu_op->attr_parser(&(relu->attrs));
  }

}  // namespace

Graph FuseBNAddRelu(Graph&& g, bool allow_input_reorder) {
  const auto& shape_vec   = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec   = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  const auto& idx = g.indexed_graph();
  std::unordered_set<nnvm::ObjectPtr> to_delete;
  nnvm::NodeEntry relu, add, bn, other;
  Match m(g);
  m.If(&relu, IsFuseableReLU);
  m.OpName(&add, "elemwise_add");
  m.If(&bn, [&shape_vec, &dtype_vec, &context_vec, &idx, &bn, &add] {
    if (bn.index != 0) return false;
    auto nid = idx.node_id(add.node.get());
    auto eid = idx.entry_id(bn);
    return batchnormaddrelu::IsCompatibleBatchNorm(bn.node, dtype_vec[eid], shape_vec[eid],
                                                   context_vec[nid]);
  });
  m.XInput(&relu, &add);
  m.XInput(&add, &bn);
  m.Input(&add, &other);
  m.ForEach([&relu, &add, &bn, &other, &to_delete, allow_input_reorder] {
    if (!allow_input_reorder && bn.node != add.node->inputs[0].node) return;
    to_delete.insert(add.node);
    to_delete.insert(bn.node);
    FuseBNAddReluNode(bn.node, add.node, relu.node, other);
  });
  Graph ret;
  ret.outputs = g.outputs;
  return ret;
}

Graph FuseBNAmpAddRelu(Graph&& g, bool allow_input_reorder) {
  const auto& shape_vec = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  const auto& idx = g.indexed_graph();
  std::unordered_set<nnvm::ObjectPtr> to_delete;
  nnvm::NodeEntry relu, add, amp, bn, other;
  Match m(g);
  m.If(&relu, IsFuseableReLU);
  m.OpName(&add, "elemwise_add");
  m.OpName(&amp, "amp_multicast").If(&amp, [&amp] { return amp.node->inputs.size() == 2; });
  m.If(&bn, [&shape_vec, &dtype_vec, &context_vec, &idx, &add, &bn] {
    if (bn.index != 0) return false;
    auto nid = idx.node_id(add.node.get());
    auto eid = idx.entry_id(bn);
    return batchnormaddrelu::IsCompatibleBatchNorm(bn.node, dtype_vec[eid], shape_vec[eid],
                                                   context_vec[nid]);
  });
  m.XInput(&relu, &add);
  m.XInput(&add, &amp);
  m.XInput(&amp, &bn);
  m.Input(&amp, &other).If(&other, [&dtype_vec, &idx, &other] {
    return dtype_vec[idx.entry_id(other)] == mshadow::kFloat16;
  });
  m.ForEach([&relu, &add, &amp, &bn, &other, &to_delete, &dtype_vec, &idx, allow_input_reorder] {
    if (!allow_input_reorder && bn.node != amp.node->inputs[0].node) return;
    to_delete.insert(add.node);
    to_delete.insert(amp.node);
    to_delete.insert(bn.node);
    FuseBNAddReluNode(bn.node, amp.node, relu.node, other);
  });
  Graph ret;
  ret.outputs = g.outputs;
  return ret;
}

}  // namespace exec
}  // namespace mxnet
