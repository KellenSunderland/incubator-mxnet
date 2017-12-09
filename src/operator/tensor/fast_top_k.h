//
// Created by Sunderland, Kellen on 11/9/17.
//

#ifndef MXNET_FAST_TOP_K_H
#define MXNET_FAST_TOP_K_H

#include "ordering_op-inl.h"

/*!
 * \file fast_top_k.h
 * \brief Adaptation of top-k from DSSTNE
 */

// Original dsstne authors: scottlegrand, rybakov, sgkim126

namespace mxnet {
namespace op {

using namespace mshadow;

/*!
 * \brief Implementation of the TopK operation
 *
 *
 * \param ctx the running context
 * \param resource temporary resource handler
 * \param src the Source blob
 * \param ret the destination blobs
 * \param k the K elements to keep
 * \param param the topk parameters
 * \tparam xpu the device type.
 */
void FastTopKImpl(Stream<cpu>* s,
              Resource resource,
              const TBlob &src,
              const std::vector<TBlob> &ret,
              const TopKParam &param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  for (auto ret_ele : ret) {
    CHECK_EQ(ret_ele.type_flag_, src.type_flag_);
  }
  // 1. Parse and initialize information
  Tensor<cpu, 1, char> workspace;
  Tensor<cpu, 1, char> temp_workspace;
  Tensor<cpu, 1, real_t> sorted_dat;
  Tensor<cpu, 1, int> indices, batch_id, sel_indices;
  Tensor<cpu, 2, real_t> mask_val;
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  TShape target_shape;
  auto begin = std::chrono::high_resolution_clock::now();
  ParseTopKParam(src.shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  Tensor<cpu, 3, real_t> dat = src.FlatTo3D<cpu, real_t>(axis, axis, s);
  size_t temp_size = mxnet::op::SortByKeyWorkspaceSize<int, int, cpu>(src.Size());
  // std::cout<<"temp_size is: "<<temp_size<<std::endl;
  temp_size = std::max(temp_size, mxnet::op::SortByKeyWorkspaceSize<int, real_t, cpu>(src.Size()));
  // std::cout<<"temp_size is: "<<temp_size<<std::endl;
  temp_size = std::max(temp_size, mxnet::op::SortByKeyWorkspaceSize<real_t, int, cpu>(src.Size()));
  // std::cout<<"temp_size is: "<<temp_size<<std::endl;
  size_t workspace_size = temp_size + sizeof(real_t) * src.Size() + sizeof(int) * src.Size() * 2;
  if (param.ret_typ == topk_enum::kReturnMask) {
    workspace_size += sizeof(int) * batch_size * k + sizeof(real_t) * batch_size * k;
  }
  // std::cout<<"Workspace size is: "<<workspace_size<<std::endl;
  workspace = resource.get_space_typed<cpu, 1, char>(Shape1(workspace_size), s);
  char* workspace_curr_ptr = workspace.dptr_;
  sorted_dat = Tensor<cpu, 1, real_t>(reinterpret_cast<real_t*>(workspace_curr_ptr),
                                      Shape1(src.Size()), s);  // contain sorted dat
  // std::cout<<"Incrementing workspace pointer by : "<<sizeof(real_t) * src.Size()<<std::endl;
  workspace_curr_ptr += sizeof(real_t) * src.Size();
  indices = Tensor<cpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                Shape1(src.Size()), s);  // indices in the original matrix
  // std::cout<<"Incrementing workspace pointer by : "<<sizeof(int) * src.Size()<<std::endl;
  workspace_curr_ptr += sizeof(int) * src.Size();
  batch_id = Tensor<cpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                 Shape1(src.Size()), s);  // batch id in the original matrix
  // std::cout<<"Incrementing workspace pointer by : "<<sizeof(int) * src.Size()<<std::endl;
  workspace_curr_ptr += sizeof(int) * src.Size();
  // std::cout<<"dat shape: "<<dat.shape_<<std::endl;
  if (do_transpose) {
    sorted_dat = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(src.Size()));
  } else {
    sorted_dat = reshape(dat, Shape1(src.Size()));
  }

  // std::cout<<"dat shape: "<<sorted_dat.shape_<<std::endl;
  mxnet_op::Kernel<range_fwd, cpu>::Launch(s, batch_size * element_num, 1, 0, 1,
                                           kWriteTo, indices.dptr_);

  CHECK_EQ(sorted_dat.CheckContiguous(), true);
  CHECK_EQ(indices.CheckContiguous(), true);
  if (param.ret_typ == topk_enum::kReturnMask) {
    sel_indices = Tensor<cpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                      Shape1(batch_size * k), s);
    workspace_curr_ptr += sizeof(int) * batch_size * k;
    mask_val = Tensor<cpu, 2, real_t>(reinterpret_cast<real_t*>(workspace_curr_ptr),
                                      Shape2(batch_size * k, 1), s);
    workspace_curr_ptr += sizeof(real_t) * batch_size * k;
    mask_val = scalar<real_t>(1);
    CHECK_EQ(sel_indices.CheckContiguous(), true);
    CHECK_EQ(mask_val.CheckContiguous(), true);
  }
  temp_workspace = Tensor<cpu, 1, char>(workspace_curr_ptr, Shape1(temp_size), s);  // temp space
  workspace_curr_ptr += temp_size;
  // 2. Perform inplace batch sort using the `SortByKey` in MShadow
  // After sorting, each batch in `sorted_dat` will be sorted in the corresponding order
  //   and the `indices` will contain the corresponding index in `sorted_dat`
  // Sort the data and keep record of the correspondence to global indices.
  // std::cout<<"sorted_dat shape: "<<sorted_dat.shape_<<std::endl;

//  for (int i =0; i<6; i++) {
//    // std::cout<<"sorted_dat items before sort: "<<sorted_dat[i]<<std::endl;
//  }

  auto first_sort = std::chrono::high_resolution_clock::now();
  mxnet::op::SortByKey(sorted_dat, indices, is_ascend, &temp_workspace);

  // Iterate over sorted_date (shape 6)
//  for (int i =0; i<6; i++) {
//    // std::cout<<"sorted_dat items: "<<sorted_dat[i]<<std::endl;
//  }

  // Calculate the corresponding batch indices of the elements
  batch_id = indices / element_num;
  // Since the SortByKey performs stable sort, the second SortByKey will reorder
  //   the sorted_dat based on the order of the batch_id
  // std::cout<<"sorted_dat shape: "<<sorted_dat.shape_<<std::endl;
  // std::cout<<"batch_id shape: "<<batch_id.shape_<<std::endl;

  auto second_sort = std::chrono::high_resolution_clock::now();
  mxnet::op::SortByKey(batch_id, sorted_dat, true, &temp_workspace);
  // Reorder the indices
  batch_id = indices / element_num;
  // std::cout<<"batch_id shape: "<<batch_id.shape_<<std::endl;
  // std::cout<<"indices shape: "<<indices.shape_<<std::endl;
  mxnet::op::SortByKey(batch_id, indices, true, &temp_workspace);

  auto third_sort = std::chrono::high_resolution_clock::now();
  // 3. Assign results to the ret blob
  if (param.ret_typ == topk_enum::kReturnMask) {
    Tensor<cpu, 2, real_t> ret_mask =
        ret[0].get_with_shape<cpu, 2, real_t>(Shape2(ret[0].Size(), 1), s);
    ret_mask = scalar<real_t>(0);
    sel_indices = reshape(slice<1>(
        inplace_reshape(indices,
                        Shape2(batch_size,
                               element_num)), 0, k),
                          Shape1(batch_size * k));
    if (do_transpose) {
      TShape src_shape = src.shape_.FlatTo3D(axis);
      CHECK_EQ(sel_indices.CheckContiguous(), true);
      sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                      Shape3(0, 2, 1));
    }
    IndexFill(ret_mask, sel_indices, mask_val);
  } else if (param.ret_typ == topk_enum::kReturnIndices) {
    indices -= batch_id * element_num;
    if (do_transpose) {
      Tensor<cpu, 3, real_t> ret_indices = ret[0].FlatTo3D<cpu, real_t>(axis, axis, s);
      ret_indices = tcast<real_t>(transpose(
          slice<2>(inplace_reshape(indices,
                                   Shape3(ret_indices.shape_[0],
                                          ret_indices.shape_[2],
                                          element_num)),
                   0, k),
          Shape3(0, 2, 1)));
    } else {
      Tensor<cpu, 2, real_t> ret_indices =
          ret[0].get_with_shape<cpu, 2, real_t>(Shape2(batch_size, k), s);
      ret_indices = tcast<real_t>(slice<1>(
          inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k));
    }
  } else {
    indices -= batch_id * element_num;
    if (do_transpose) {
      Tensor<cpu, 3, real_t> ret_value = ret[0].FlatTo3D<cpu, real_t>(axis, axis, s);
      Tensor<cpu, 3, real_t> ret_indices = ret[1].FlatTo3D<cpu, real_t>(axis, axis, s);
      ret_value = transpose(
          slice<2>(inplace_reshape(sorted_dat,
                                   Shape3(ret_value.shape_[0], ret_value.shape_[2], element_num)),
                   0, k),
          Shape3(0, 2, 1));
      ret_indices = tcast<real_t>(transpose(
          slice<2>(inplace_reshape(indices,
                                   Shape3(ret_indices.shape_[0],
                                          ret_indices.shape_[2],
                                          element_num)),
                   0, k),
          Shape3(0, 2, 1)));
    } else {

      // std::cout<<"Source shape 0: "<<ret[0].shape_<<std::endl;
      // std::cout<<"Batch size: "<<batch_size<<std::endl;
      // std::cout<<"k: "<<k<<std::endl;
      // std::cout<<"ret size: "<<ret.size()<<std::endl;
//      for (auto tensor : ret) {
//        // std::cout<<"Return tensor shape: "<<tensor.shape_<<std::endl;
//      }
      Tensor<cpu, 2, real_t> ret_value =
          ret[0].get_with_shape<cpu, 2, real_t>(Shape2(batch_size, k), s);
      Tensor<cpu, 2, real_t> ret_indices =
          ret[1].get_with_shape<cpu, 2, real_t>(Shape2(batch_size, k), s);
      ret_value = slice<1>(inplace_reshape(sorted_dat, Shape2(batch_size, element_num)), 0, k);
      // std::cout<<"ret_value: "<<ret_value[0][0]<<std::endl;
      // std::cout<<"ret_value: "<<ret_value[1][0]<<std::endl;

      ret_indices = tcast<real_t>(slice<1>(
          inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k));
    }
  }

  auto finished = std::chrono::high_resolution_clock::now();

  std::cout<<"Tensor operations before sort:" <<
           std::chrono::duration_cast<std::chrono::nanoseconds>(first_sort-begin).count() <<
           "ns" << std::endl;

  std::cout<<"First sort:" <<
           std::chrono::duration_cast<std::chrono::nanoseconds>(second_sort-first_sort).count
               () <<
           "ns" << std::endl;

  std::cout<<"Second sort:" <<
           std::chrono::duration_cast<std::chrono::nanoseconds>(third_sort-second_sort).count
               () <<
           "ns" << std::endl;

  std::cout<<"Third sort:" <<
           std::chrono::duration_cast<std::chrono::nanoseconds>(finished-third_sort).count() <<
           "ns" << std::endl;
}

void FastTopK(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  // TODO(sxjscience) We can support inplace in the future
  CHECK_EQ(req[0], kWriteTo) << "TopK does not support inplace";
  FastTopKImpl(ctx.run_ctx.get_stream<cpu>(), ctx.requested[0], inputs[0], outputs, param);
}

}  // namespace op
}  // namespace mxnet

#endif //MXNET_FAST_TOP_K_H
