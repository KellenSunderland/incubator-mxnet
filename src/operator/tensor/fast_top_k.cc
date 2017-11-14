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
 * \file fast_top_k.cc
 * \brief Adaptation of top-k from DSSTNE
 */

// Original dsstne authors: scottlegrand, rybakov, sgkim126

#include "fast_top_k.h"

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(fast_topk)
    .describe(R"code(Returns the top *k* elements in an input array on the last axis.

Examples::

  x = [[ 0.3,  0.2,  0.4],
       [ 0.1,  0.3,  0.2]]

  // returns an index of the largest element on last axis
  topk(x) = [[ 2.],
             [ 1.]]

  // returns the value of top-2 largest elements on last axis
  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
                                   [ 0.3,  0.2]]

  // returns the value of top-2 smallest elements on last axis
  topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
                                               [ 0.1 ,  0.2]]

  // flattens and then returns list of both values and indices
  topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]

)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(TopKNumOutputs)
    .set_attr_parser(ParamParser<TopKParam>)
    .set_attr<nnvm::FInferShape>("FInferShape", TopKShape)
    .set_attr<nnvm::FInferType>("FInferType", TopKType)
    .set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", TopKNumVisibleOutputs)
    .set_attr<FCompute>("FCompute<cpu>", FastTopK)
    .set_attr<nnvm::FGradient>("FGradient",
                               [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
                                 const TopKParam& param = nnvm::get<TopKParam>(n->attrs.parsed);
                                 if (param.ret_typ == topk_enum::kReturnValue || param.ret_typ == topk_enum::kReturnBoth) {
                                   std::vector<nnvm::NodeEntry> inputs;
                                   index_t n_out = n->num_outputs();
                                   for (index_t i = 0; i < n_out; ++i) {
                                     inputs.emplace_back(nnvm::NodeEntry{ n, i, 0 });
                                   }
                                   return MakeNonlossGradNode("_backward_topk", n, {ograds[0]}, inputs, n->attrs.dict);
                                 } else {
                                   return MakeZeroGradNodes(n, ograds);
                                 }
                               })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_argument("data", "NDArray-or-Symbol", "The input array")
    .add_arguments(TopKParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet