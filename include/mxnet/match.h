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
 * \file match.h
 * \brief NNVM graph pattern matcher
 * \author Vladimir Cherepanov
 */

#ifndef MXNET_MATCH_H_
#define MXNET_MATCH_H_

#include <nnvm/graph.h>
#include <nnvm/node.h>

#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mxnet {

/*!
 * \brief Match class.
 * Can be used to search for a subgraphs in a graph, matching a specified pattern.
 */
class Match {
 public:
  explicit Match(const nnvm::Graph& g);

  /*!
   * \brief The main thing - fn() is called on every match found.
   */
  void ForEach(std::function<void()> fn);

  /*!
   * The rest of the functions below are used to incrementally build a pattern to match against.
   * They return *this in case one wants to chain related calls together.
   */

  /*! \brief Matches node by an operator name. */
  Match& OpName(nnvm::NodeEntry* entry, const std::string& name);
  /*! \brief Matches by a node entry predicate. */
  Match& If(nnvm::NodeEntry* entry, std::function<bool(const nnvm::NodeEntry&)> fn);
  /*! \brief Matches by a node predicate. */
  Match& If(nnvm::NodeEntry* entry, std::function<bool(const nnvm::ObjectPtr&)> fn);
  /*! \brief Matches using an argumentless predicate. Will use already bound placeholders. */
  Match& If(nnvm::NodeEntry* entry, std::function<bool()> fn);
  /*! \brief Establishes an output - input relashionship between 2 nodes. */
  Match& Input(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp);
  /*! \brief X stands for eXclusive - the same as above, but matches only if the input entry is used
   * once in a graph. */
  Match& XInput(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp);

  /*! \brief Syntactic sugar for multiple Input() calls for the same output. */
  template <typename... Args>
  Match& Input(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp, Args*... args) {
    Input(entry, inp);
    return Input(entry, args...);
  }

  /*! \brief Syntactic sugar for multiple XInput() calls for the same output. */
  template <typename... Args>
  Match& XInput(nnvm::NodeEntry* entry, nnvm::NodeEntry* inp, Args*... args) {
    XInput(entry, inp);
    return XInput(entry, args...);
  }

 private:
  using Predicate = std::function<bool(const nnvm::NodeEntry&)>;
  using Entries = std::vector<nnvm::NodeEntry>;

  struct Ctx {
    std::vector<nnvm::NodeEntry> root;
    std::unordered_set<nnvm::ObjectPtr> taken;
  };

  class Iterator {
   public:
    Iterator(nnvm::NodeEntry* ptr, std::vector<Predicate>* predicates, Ctx* ctx,
             Match::Entries::const_iterator begin, Match::Entries::const_iterator end);
    ~Iterator();

    Iterator(Iterator&& src) noexcept;
    Iterator& operator=(Iterator&& src) noexcept;

    bool Next();

   private:
    nnvm::NodeEntry* ptr_;
    std::vector<Predicate>* predicates_;
    Ctx* ctx_;
    Entries::const_iterator it_;
    Entries::const_iterator end_;
  };

  void ForEachEntry(std::function<void(const nnvm::NodeEntry&)> fn);
  void MatchFromRoot(const nnvm::NodeEntry& entry, std::function<void()> fn);
  void BuildStages();

  const nnvm::Graph& g_;
  nnvm::NodeEntryMap<std::vector<nnvm::NodeEntry>> outputs_;
  std::vector<std::function<Iterator(Ctx*)>> stages_;
  std::map<nnvm::NodeEntry*, std::vector<Predicate>> predicates_;

  std::vector<std::pair<nnvm::NodeEntry*, nnvm::NodeEntry*>> inputs_;
};

}  // namespace mxnet

#endif  // MXNET_MATCH_H_
