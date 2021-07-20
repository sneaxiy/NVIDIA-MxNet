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
 * \file match.cc
 * \brief NNVM graph pattern matcher
 * \author Vladimir Cherepanov
 */

#include "mxnet/match.h"

#include <dmlc/common.h>
#include <mxnet/base.h>

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <utility>

namespace mxnet {

Match::Match(const nnvm::Graph& g) : g_(g) {
  ForEachEntry([this](const nnvm::NodeEntry& e) {
    for (const auto& ie : e.node->inputs) outputs_[ie].push_back(e);
  });
  for (const auto& o : g_.outputs) outputs_[o].push_back(o);
}

void Match::ForEach(std::function<void()> fn) {
  BuildStages();
  auto visitor = [this, fn](const nnvm::NodeEntry& e) { MatchFromRoot(e, fn); };
  ForEachEntry(visitor);
}

Match& Match::OpName(nnvm::NodeEntry* entry, const std::string& name) {
  predicates_[entry].push_back(
      [name](const nnvm::NodeEntry& e) { return e.node->op() == nnvm::Op::Get(name); });
  return *this;
}

Match& Match::If(nnvm::NodeEntry* entry, std::function<bool(const nnvm::NodeEntry&)> fn) {
  predicates_[entry].push_back(fn);
  return *this;
}

Match& Match::If(nnvm::NodeEntry* entry, std::function<bool(const nnvm::ObjectPtr&)> fn) {
  predicates_[entry].push_back([fn](const nnvm::NodeEntry& e) { return fn(e.node); });
  return *this;
}

Match& Match::If(nnvm::NodeEntry* entry, std::function<bool()> fn) {
  predicates_[entry].push_back([fn](const nnvm::NodeEntry&) { return fn(); });
  return *this;
}

Match& Match::Input(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp) {
  auto p = std::make_pair(entry, inp);
  auto it = std::find(inputs_.begin(), inputs_.end(), p);
  CHECK(it == inputs_.end()) << "Duplicate edge in a match pattern";
  inputs_.push_back(p);
  // We also use predicates_ to count all entries in a pattern, hence the following 2 lines.
  predicates_[entry];
  predicates_[inp];
  return *this;
}

Match& Match::XInput(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp) {
  Input(entry, inp);
  predicates_[inp].push_back([this](const nnvm::NodeEntry& e) { return outputs_[e].size() == 1; });
  return *this;
}

void Match::ForEachEntry(std::function<void(const nnvm::NodeEntry&)> fn) {
  std::vector<nnvm::NodeEntry> stack;
  std::unordered_set<nnvm::ObjectPtr> visited;
  for (const auto& o : g_.outputs) {
    if (visited.count(o.node)) continue;
    visited.insert(o.node);
    stack.push_back(o);
    while (!stack.empty()) {
      auto top = stack.back();
      stack.pop_back();
      fn(top);
      for (const auto& inp : top.node->inputs) {
        if (visited.count(inp.node)) continue;
        visited.insert(inp.node);
        stack.push_back(inp);
      }
    }
  }
}

void Match::MatchFromRoot(const nnvm::NodeEntry& entry, std::function<void()> fn) {
  Ctx ctx{{entry}};
  std::vector<Iterator> idx;
  idx.push_back(stages_.front()(&ctx));
  while (!idx.empty()) {
    if (!idx.back().Next()) {
      idx.pop_back();
      continue;
    }
    if (idx.size() < stages_.size()) {
      idx.push_back(stages_[idx.size()](&ctx));
    } else {
      // Full match found.
      fn();
    }
  }
}

void Match::BuildStages() {
  // First find the root.
  std::unordered_set<nnvm::NodeEntry*> roots;
  for (const auto& v : predicates_) roots.insert(v.first);
  for (const auto& e : inputs_) roots.erase(e.second);
  CHECK(!roots.empty()) << "Failed to find the pattern's root - there are no nodes without inputs";
  const auto it_root =
      std::find_if(predicates_.begin(), predicates_.end(),
                   [this, &roots](const std::pair<nnvm::NodeEntry*, std::vector<Predicate>>& p) {
                     return roots.count(p.first);
                   });
  CHECK(it_root != predicates_.end());
  const auto root = it_root->first;
  // Now do BFS and build the stages
  std::unordered_map<nnvm::NodeEntry*, std::vector<std::pair<nnvm::NodeEntry*, bool>>> neigh;
  for (auto e : inputs_) {
    neigh[e.first].push_back({e.second, false});
    neigh[e.second].push_back({e.first, true});
  }
  inputs_.clear();
  struct QE {
    nnvm::NodeEntry* ptr;
    nnvm::NodeEntry* parent_ptr;
    bool is_output;
  };
  stages_.clear();
  std::deque<QE> q{{root, nullptr, true}};
  std::unordered_set<nnvm::NodeEntry*> visited;
  using Edge = std::pair<nnvm::NodeEntry*, nnvm::NodeEntry*>;
  struct HashEdge {
    size_t operator()(const Edge& e) const {
      size_t acc = 0;
      acc = dmlc::HashCombine(acc, e.first);
      acc = dmlc::HashCombine(acc, e.second);
      return acc;
    }
  };
  std::unordered_set<Edge, HashEdge> banned_edges;
  while (!q.empty()) {
    auto e = q.front();
    q.pop_front();
    if (visited.count(e.ptr)) {
      // Add back reference
      if (e.is_output) {
        predicates_[e.ptr].push_back([e](const nnvm::NodeEntry& ne) {
          for (const auto& inp : ne.node->inputs) {
            if (nnvm::NodeEntryEqual()(inp, *e.parent_ptr)) return true;
          }
          return false;
        });
      } else {
        predicates_[e.ptr].push_back([this, e](const nnvm::NodeEntry& ne) {
          auto it_o = outputs_.find(ne);
          if (it_o == outputs_.end()) return false;
          for (const auto& o : it_o->second) {
            if (nnvm::NodeEntryEqual()(o, *e.parent_ptr)) return true;
          }
          return false;
        });
      }
      continue;
    }
    visited.insert(e.ptr);
    if (e.parent_ptr == nullptr) {
      stages_.emplace_back([this, e](Ctx* ctx) {
        return Iterator(e.ptr, &predicates_[e.ptr], ctx, ctx->root.begin(), ctx->root.end());
      });
    } else if (e.is_output) {
      stages_.emplace_back([this, e](Ctx* ctx) {
        auto it = outputs_.find(*e.parent_ptr);
        CHECK(it != outputs_.end());
        return Iterator(e.ptr, &predicates_[e.ptr], ctx, it->second.begin(), it->second.end());
      });
    } else {
      stages_.emplace_back([this, e](Ctx* ctx) {
        return Iterator(e.ptr, &predicates_[e.ptr], ctx, e.parent_ptr->node->inputs.begin(),
                        e.parent_ptr->node->inputs.end());
      });
    }
    for (auto ee : neigh[e.ptr]) {
      if (banned_edges.count({e.ptr, ee.first})) continue;
      banned_edges.insert({ee.first, e.ptr});
      q.push_back({ee.first, e.ptr, ee.second});
    }
  }
  CHECK(visited.size() == predicates_.size()) << "Pattern must be connected";
}

Match::Iterator::Iterator(nnvm::NodeEntry* ptr, std::vector<Predicate>* predicates, Ctx* ctx,
                          Match::Entries::const_iterator begin, Match::Entries::const_iterator end)
    : ptr_(ptr), predicates_(predicates), ctx_(ctx), it_(begin), end_(end) {}

Match::Iterator::~Iterator() {
  if (ptr_ == nullptr) return;
  ctx_->taken.erase(ptr_->node);
  *ptr_ = nnvm::NodeEntry();
}

Match::Iterator::Iterator(Match::Iterator&& src) noexcept
    : ptr_(src.ptr_), predicates_(src.predicates_), ctx_(src.ctx_), it_(src.it_), end_(src.end_) {
  src.ptr_ = nullptr;
}

Match::Iterator& Match::Iterator::operator=(Match::Iterator&& src) noexcept {
  ptr_ = src.ptr_;
  predicates_ = src.predicates_;
  ctx_ = src.ctx_;
  it_ = src.it_;
  end_ = src.end_;
  src.ptr_ = nullptr;
  return *this;
}

bool Match::Iterator::Next() {
  ctx_->taken.erase(ptr_->node);
  for (; it_ != end_; ++it_) {
    if (ctx_->taken.count(it_->node)) continue;
    bool matched = true;
    *ptr_ = *it_;
    for (const auto& p : *predicates_) {
      if (!p(*ptr_)) {
        *ptr_ = nnvm::NodeEntry();
        matched = false;
        break;
      }
    }
    if (!matched) continue;
    ++it_;
    ctx_->taken.insert(ptr_->node);
    return true;
  }
  return false;
}

}  // namespace mxnet
