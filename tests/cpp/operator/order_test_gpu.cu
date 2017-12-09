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
#include <operator/mxnet_op.h>

namespace mxnet {
namespace op {

using namespace mshadow;


TEST(Order, top_5_from_50k_gpu) {

  // Setup attributes.
  nnvm::NodeAttrs attrs;
  TopKParam topk_param;
  topk_param.axis = 1;
  topk_param.k = 5;
  topk_param.ret_typ = topk_enum::kReturnValue;
  topk_param.is_ascend = false;
  attrs.parsed = topk_param;

  // Create a new context.
  OpContext ctx;
  ctx.run_ctx.stream = mshadow::NewStream<gpu>(true, true);

  // Request workspace
  Resource r = ResourceManager::Get()->Request(ctx.run_ctx.ctx, ResourceRequest::kTempSpace);
  ctx.requested.push_back(r);

  // Create input.
  std::vector<TBlob> inputs;
  float unsorted_values[50000];
  std::uniform_real_distribution<float> dist(0.0f, 100000.0f);
  std::random_device rd;
  std::mt19937 mt(rd());
  for (int i =0; i< 50000; i++){
    unsorted_values[i] = dist(mt);
  }
  Tensor<gpu, 2, float> input_tensor(unsorted_values, Shape2(1, 50000));
  inputs.emplace_back(input_tensor);

  // Set request type.
  std::vector<OpReqType> request_types;
  request_types.emplace_back(kWriteTo);

  // Create outputs.
  std::vector<TBlob> outputs;
  float out_values[] {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float out_indices[] {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Tensor<gpu, 3, float> destination_values(out_values, Shape3(1, 5, 1));
  Tensor<gpu, 3, float> destination_idices(out_indices, Shape3(1, 5, 1));
  outputs.emplace_back(destination_values);
  outputs.emplace_back(destination_idices);

  // Run Top-K.
  TopK<gpu>(attrs, ctx, inputs, request_types, outputs);
}


} // namespace op
} // namespace mxnet