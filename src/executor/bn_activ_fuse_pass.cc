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
 * \file bn_activation_fuse_pass.cc
 * \brief optimization pass which fuse Activation into BatchNorm
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/match.h>
#include <mxnet/operator.h>

#include "./exec_pass.h"
#include "../operator/nn/activation-inl.h"
#include "../operator/nn/batch_norm-inl.h"

namespace mxnet {
namespace exec {

using mxnet::Match;
using namespace mxnet::op;

Graph FuseBNActiv(Graph&& g) {
  const auto& shape_vec   = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec   = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  const auto& ig = g.indexed_graph();
  Match m(g);
  nnvm::NodeEntry relu, bn;
  m.If(&relu, IsFuseableReLU);
  m.OpName(&bn, "BatchNorm").If(&bn, [&bn, &shape_vec, &dtype_vec, &context_vec, &ig] {
    if (bn.index != 0) return false;
    auto nid = ig.node_id(bn.node.get());
    auto eid = ig.entry_id(nid, 0);
    return batchnorm::SupportsFusedActivation(bn.node, dtype_vec[eid], shape_vec[eid],
                                              context_vec[nid]);
  });
  m.XInput(&relu, &bn);
  static const nnvm::Op* relu_op = Op::Get("relu");
  nnvm::NodeEntryMap<nnvm::NodeEntry> entry_map;
  m.ForEach([&relu, &bn, &entry_map] {
    if (relu.node->op() == relu_op) relu.node->attrs.dict["act_type"] = "relu";
    bn.node->attrs.dict["act_type"] = relu.node->attrs.dict["act_type"];
    bn.node->attrs.name += "_activ";
    bn.node->op()->attr_parser(&(bn.node->attrs));
    entry_map.insert({nnvm::NodeEntry{relu.node, 0, 0}, nnvm::NodeEntry{bn.node, 0, 0}});
  });
  g = ReplaceNodeEntries(std::move(g), entry_map);
  return g;
}

}  // namespace exec
}  // namespace mxnet
