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
 * Copyright (c) 2020 by Contributors
 * \file match_test.cc
 * \brief Tests for NNVM graph pattern matcher
 * \author Vladimir Cherepanov
 */

#include <mxnet/match.h>

#include <nnvm/graph.h>

#include <gtest/gtest.h>

#include <set>
#include <sstream>
#include <string>
#include <tuple>

using mxnet::Match;

nnvm::NodeEntry MakeNode(const std::string& name, const std::vector<nnvm::NodeEntry>& inputs) {
  nnvm::ObjectPtr p = nnvm::Node::Create();
  p->attrs.name = name;
  p->inputs = inputs;
  return nnvm::NodeEntry(p, 0, 0);
}

nnvm::Graph CreateStarGraph(int n) {
  nnvm::Graph ret;
  std::vector<nnvm::NodeEntry> inputs;
  for (int i = 0; i < n; ++i) inputs.push_back(MakeNode("leaf" + std::to_string(i), {}));
  ret.outputs.push_back(MakeNode("root", inputs));
  return ret;
}

nnvm::Graph CreateBroomGraph(int handle_len, int width) {
  nnvm::Graph ret;
  std::vector<nnvm::NodeEntry> inputs;
  for (int i = 0; i < width; ++i) inputs.push_back(MakeNode("fiber" + std::to_string(i), {}));
  nnvm::NodeEntry node = MakeNode("center", inputs);
  for (int i = 0; i < handle_len; ++i) node = MakeNode("handle" + std::to_string(i), {node});
  ret.outputs.push_back(node);
  return ret;
}

nnvm::Graph CreateNRootGraph(int n) {
  std::vector<nnvm::NodeEntry> inputs{MakeNode("inp", {})};
  nnvm::Graph ret;
  for (int i = 0; i < n; ++i) ret.outputs.push_back(MakeNode("root" + std::to_string(i), {inputs}));
  return ret;
}

nnvm::Graph CreateWideGraph(int n_middle) {
  nnvm::Graph ret;
  auto input = MakeNode("input", {});
  std::vector<nnvm::NodeEntry> middle;
  for (int i = 0; i < n_middle; ++i) {
    auto other_input = MakeNode("other_input" + std::to_string(i), {});
    middle.push_back(MakeNode("middle" + std::to_string(i), {input, other_input}));
  }
  ret.outputs.push_back(MakeNode("output", middle));
  return ret;
}

TEST(MATCH, SingleNodeGraph) {
  auto g = CreateStarGraph(0);
  Match m(g);
  nnvm::NodeEntry e;
  m.If(&e, [] { return true; });
  int counter = 0;
  m.ForEach([&counter] { ++counter; });
  EXPECT_EQ(counter, 1);
}

TEST(MATCH, StarGraph) {
  for (auto n_leaves : {0, 1, 2, 5, 10000}) {
    nnvm::Graph g = CreateStarGraph(n_leaves);
    Match m(g);
    nnvm::NodeEntry a, b;
    m.Input(&a, &b);
    nnvm::NodeEntryMap<int> counters;
    m.ForEach([&counters, &b] { ++counters[b]; });
    EXPECT_EQ(counters.size(), n_leaves);
    for (auto& p : counters) EXPECT_EQ(p.second, 1);
  }
}

TEST(MATCH, BroomGraph) {
  for (auto n_leaves : {0, 1, 2, 5, 100}) {
    auto g = CreateBroomGraph(1, n_leaves);
    Match m(g);
    nnvm::NodeEntry a, b, c, d;
    m.Input(&a, &b);
    m.Input(&b, &c);
    m.Input(&b, &d);

    std::set<std::pair<nnvm::ObjectPtr, nnvm::ObjectPtr>> ns;
    m.ForEach([&a, &b, &c, &d, &ns] {
      EXPECT_NE(c.node, d.node);
      auto p = ns.insert({c.node, d.node});
      EXPECT_TRUE(p.second);
    });
    EXPECT_EQ(ns.size(), n_leaves * (n_leaves - 1));
  }
}

TEST(MATCH, MultiRoot) {
  for (auto n_roots : {1, 2, 3, 4, 5, 20}) {
    auto g = CreateNRootGraph(n_roots);
    Match m(g);
    nnvm::NodeEntry a, b, c, d;
    m.Input(&a, &d);
    m.Input(&b, &d);
    m.Input(&c, &d);

    std::set<std::tuple<nnvm::ObjectPtr, nnvm::ObjectPtr, nnvm::ObjectPtr>> ns;
    m.ForEach([&a, &b, &c, &d, &ns] {
      auto r = ns.insert({a.node, b.node, c.node});
      EXPECT_TRUE(r.second);
    });
    EXPECT_EQ(ns.size(), n_roots * (n_roots - 1) * (n_roots - 2));
  }
}

TEST(MATCH, WideGraph) {
  for (auto n_middle : {1, 2, 3, 5, 100}) {
    auto g = CreateWideGraph(n_middle);
    Match m(g);
    nnvm::NodeEntry a, b, c, d;
    m.Input(&a, &b, &c);
    m.Input(&b, &d);
    m.Input(&c, &d);

    std::set<std::pair<nnvm::ObjectPtr, nnvm::ObjectPtr>> ns;
    m.ForEach([&a, &b, &c, &d, &ns] {
      EXPECT_EQ(a.node->attrs.name, "output");
      EXPECT_EQ(d.node->attrs.name, "input");
      auto r = ns.insert({b.node, c.node});
      EXPECT_TRUE(r.second);
    });
    EXPECT_EQ(ns.size(), n_middle * (n_middle - 1));
  }
}

