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
 * \file storage_test.cc
 * \brief cpu/gpu storage tests
*/
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "test_util.h"
#include <operator/tensor/ordering_op-inl.h>

namespace mxnet {
namespace op {

using namespace mshadow;

TEST(Order, basic_top_k) {

  // Setup attributes.
  nnvm::NodeAttrs attrs;
  TopKParam topk_param;
  topk_param.axis = 2;
  topk_param.k = 1;
  topk_param.ret_typ = topk_enum::kReturnValue;
  attrs.parsed = topk_param;

  // Create a new context.
  OpContext ctx;

  // Request workspace
  Resource r = ResourceManager::Get()->Request(ctx.run_ctx.ctx, ResourceRequest::kTempSpace);
  ctx.requested.push_back(r);

  // Create input.
  std::vector<TBlob> inputs;
  Tensor<cpu, 3, float> input_tensor(Shape3(1, 2, 3));
  inputs.emplace_back(input_tensor);
  AllocSpace(&input_tensor);
  input_tensor[0][0][0] = 5;
  input_tensor[0][0][1] = 6;
  input_tensor[0][0][2] = 7;
  input_tensor[0][1][0] = 4;
  input_tensor[0][1][1] = 9;
  input_tensor[0][1][2] = 6;

  // Set request type.
  std::vector<OpReqType> request_types;
  request_types.emplace_back(kWriteTo);

  // Create outputs
  std::vector<TBlob> outputs;
  Tensor<cpu, 3, float> destination_values(Shape3(1, 2, 1));
  Tensor<cpu, 3, float> destination_idices(Shape3(1, 2, 1));
  AllocSpace(&destination_values);
  AllocSpace(&destination_idices);
  outputs.emplace_back(destination_values);
  outputs.emplace_back(destination_idices);


  TopK<cpu>(attrs, ctx, inputs, request_types, outputs);
  std::cout<<destination_values[0][0][0]<<std::endl;
  std::cout<<destination_values[0][1][0]<<std::endl;
  std::cout<<destination_idices[0][0][0]<<std::endl;
  std::cout<<destination_idices[0][1][0]<<std::endl;
  EXPECT_LE(std::abs(destination_values[0][0][0] - 7.0f), 1e-10);
  EXPECT_LE(std::abs(destination_values[0][1][0] - 9.0f), 1e-10);
}

} // namespace op
} // namespace mxnet