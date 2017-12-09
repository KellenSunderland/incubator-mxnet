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
 * \file fast_top_k.cu
 * \brief Adaptation of top-k from DSSTNE
 */

// Original dsstne authors: scottlegrand, rybakov, sgkim126

#include "bitonic.h"
#include "../mshadow_op.h"
#include <mshadow/tensor.h>
#include <mshadow/expression.h>
#include <mxnet/resource.h>
#include "ordering_op-inl.h"
#include "sort_op.h"
#include <mshadow/stream_gpu-inl.h>

namespace mxnet {
namespace op {

// TODO: fwd declare dsstne implementation
// Reshape the tensors as in current implementation
// Topk
// Reshape tensors back
// OR just reshape last dimension


// CUDA macros and routines

__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

// Handle arbitrary API churn from new and improved thread within thread model
#if (CUDA_VERSION >= 9000)
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
#define SHFL(x, lane) __shfl((x), (lane))
#define BALLOT(predicate) __ballot(predicate)
#define ANY(predicate) __any(predicate)
#endif // CUDA_VERSION >= 9000

#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
    }
#define LAUNCHERROR_BLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
    }

#define REDUCEERROR(error) \
    if (ANY(error != (NNFloat)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    }


#define REDUCE(a) \
    if (ANY((a) != (NNFloat)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    }

// Contains information that needs to be accessible for GPU kernels and most static hyperparameters
struct GpuData {
  unsigned int _warpSize; // Warp size
  unsigned int _warpBits; // Warp bit count
  unsigned int _warpMask; // Masks bits within a warp
  unsigned long long int* _pAccumulator; // Accumulator for error calculations
};


typedef float NNFloat;
static const float MAX_VALUE = 999999999999999.0f;
static __constant__ GpuData cData = {
    ._warpSize = 32,
    ._warpBits = 5,
    ._warpMask = 31
};

__global__ void
kCalculateTopK_32_kernel(NNFloat* pOutputBuffer, NNFloat* pKeyBuffer, uint32_t* pValueBuffer,
                         uint32_t batch, uint32_t width, uint32_t k)
{
  __shared__ volatile NNFloat sKey[64 * 4];
  __shared__ volatile uint32_t sValue[64 * 4];


//  printf("Starting ... [");
//
//  for(int i =0; i < 20; i++) {
//    printf(" %f ", pOutputBuffer[i]);
//  }
//
//  printf(" ]\n");

//  printf("cData warpmask: %d, _warpSize: %d, _warpBits: %d\n", cData._warpMask, cData._warpSize,
//         cData._warpBits);

  uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
  uint32_t tgx = threadIdx.x & cData._warpMask;

  if (pos < batch) {
    NNFloat *pOutput = pOutputBuffer + pos * width;
    uint32_t offset = threadIdx.x >> cData._warpBits;
    volatile NNFloat* psKey = &sKey[64 * offset];
    volatile uint32_t* psValue = &sValue[64 * offset];

    // Initialize values to
    NNFloat k0 = -MAX_VALUE;
    NNFloat k1 = -MAX_VALUE;
    uint32_t v0 = 0;
    uint32_t v1 = 0;

    // Read first 32 elements into registers
    uint32_t wpos = tgx;
//    printf("start - pos:  %d, tgx: %d, batch: %d, width: %d, wpos: %d\n", pos, tgx, batch, width,
//           wpos );
    if (wpos < width) {
      k0 = pOutput[wpos];
      v0 = wpos;
    }
    wpos += cData._warpSize;

    // Run through remainder of data
    NNFloat minValue = -MAX_VALUE;
    uint32_t rpos = 32;
    uint32_t bufferSize = 0;
    NNFloat key1, key2;
    uint32_t value1, value2;
    uint32_t otgx;
    bool flag;
    while (rpos < width) {
      //printf("loop ...\n");
      // Read block of data
      unsigned wpos = rpos + tgx;
      NNFloat key = -MAX_VALUE;
      uint32_t value = wpos;
      if (wpos < width) {
        key = pOutput[wpos];
      }

      // Add values > minValue to shared memory buffer
      uint32_t count = BALLOT(key > minValue);
      if (key > minValue) {
        uint32_t mask = 0xffffffff >> (32 - tgx);
        uint32_t offset = __popc(count & mask);
        offset += bufferSize;
        psKey[offset] = key;
        psValue[offset] = value;
      }
      bufferSize += __popc(count);

      // Check if buffer is full
      if (bufferSize >= 32) {
        // Sort 64 elements
        k1                  = psKey[tgx];
        v1                  = psValue[tgx];
        bool flag;
        BITONICSORT64_64();

        // Shift members in shared memory to beginning
        bufferSize         -= 32;
        if (tgx < bufferSize) {
          psKey[tgx]      = psKey[tgx + 32];
          psValue[tgx]    = psValue[tgx + 32];
        }
      }

      // Advance to next block of data
      rpos                    += cData._warpSize;

      if (rpos >= width) {
//        printf("end - pos:  %d, tgx: %d, batch: %d, width: %d, wpos: %d\n", pos, tgx, batch,
//               width,  wpos );
      }
    }

    // Do final sort if buffer has any remaining data
    if ((bufferSize > 0) || (width <= 32)) {
      // Store sentinel values in registers
      k1                       = -MAX_VALUE;
      v1                       = 0;

      // Load last block of unsorted data into registers
      if (tgx < bufferSize) {
        k1                   = psKey[tgx];
        v1                   = psValue[tgx];
      }
      BITONICSORT64_64();
    }

    // Copy results to key and value pointers
    NNFloat* pKey                = pKeyBuffer + pos * k;
    uint32_t* pValue             = pValueBuffer + pos * k;
    wpos                         = tgx;
    if (wpos < k) {
      pKey[wpos]               = k0;
      pValue[wpos]             = v0;
      printf("writing to global - pos:  %d, wpos: %d, k0: %f, v0: %d\n", pos, wpos, k0, v0);
    }

    wpos                        += cData._warpSize;
  }
}

static __always_inline int fls(int x)
{
  return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}

void kCalculateTopK(NNFloat* pOutput, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
  uint32_t blocks = (batch + 3) / 4;
  if (k <= 32) {
    std::cout<<"Launching "<<blocks<<" blocks"<<std::endl;
    kCalculateTopK_32_kernel<<<blocks, 128>>>(pOutput, pKey, pValue, batch, width, k);
    LAUNCHERROR_BLOCKING("kCalculateTopK_32_kernel");
  }
  else {
    std::cout<<"Not currently supported"<<std::endl;
  }
}

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
 */

void FastTopKImplGpu(mshadow::Stream<gpu>* s,
                  Resource resource,
                  const TBlob &src,
                  const std::vector<TBlob> &ret,
                  const TopKParam &param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  for (auto ret_ele : ret) {
    CHECK_EQ(ret_ele.type_flag_, src.type_flag_);
  }

//  std::cout<<"ret 0 shape: "<<ret[0].shape_<<std::endl;
//  std::cout<<"ret 1 shape: "<<ret[1].shape_<<std::endl;

  // 1. Parse and initialize information
  Tensor<gpu, 1, char> workspace;
  Tensor<gpu, 1, char> temp_workspace;
  Tensor<gpu, 1, real_t> sorted_dat;
  Tensor<gpu, 1, int> indices, batch_id, sel_indices;
  Tensor<gpu, 1, NNFloat> values;
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  TShape target_shape;
  ParseTopKParam(src.shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
//  std::cout<<"target_shape is: "<<target_shape<<std::endl;
//  std::cout<<"src.shape_ is: "<<src.shape_<<std::endl;
//  std::cout<<"param batch_size is: "<<batch_size<<std::endl;
//  std::cout<<"element_num is: "<<element_num<<std::endl;
  //  Tensor<gpu, 3, real_t> dat = src.FlatTo3D<gpu, real_t>(axis, axis, s);
  size_t temp_size = mxnet::op::SortByKeyWorkspaceSize<int, int, gpu>(src.Size());
  temp_size = std::max(temp_size, mxnet::op::SortByKeyWorkspaceSize<int, real_t, gpu>(src.Size()));
  temp_size = std::max(temp_size, mxnet::op::SortByKeyWorkspaceSize<real_t, int, gpu>(src.Size()));
//  std::cout<<"temp_size is: "<<temp_size<<std::endl;
  size_t workspace_size = temp_size + sizeof(real_t) * src.Size() + sizeof(int) * src.Size()
      * 2 + sizeof(NNFloat) * src.Size();  // TODO: K sized?
//  std::cout<<"workspace_size is: "<<workspace_size<<std::endl;
  workspace = resource.get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  char* workspace_curr_ptr = workspace.dptr_;
  sorted_dat = Tensor<gpu, 1, real_t>(reinterpret_cast<real_t*>(workspace_curr_ptr),
                                      Shape1(src.Size()), s);  // contain sorted dat
  workspace_curr_ptr += sizeof(real_t) * src.Size();
  indices = Tensor<gpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                Shape1(src.Size()), s);  // indices in the original matrix
  workspace_curr_ptr += sizeof(int) * src.Size();
  batch_id = Tensor<gpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                 Shape1(src.Size()), s);  // batch id in the original matrix
  workspace_curr_ptr += sizeof(int) * src.Size();
  values = Tensor<gpu, 1, NNFloat>(reinterpret_cast<NNFloat*>(workspace_curr_ptr),
                                   Shape1(src.Size()), s);
  workspace_curr_ptr += sizeof(NNFloat) * src.Size();


  sorted_dat = src.FlatTo1D<gpu, real_t>(s);
//  std::cout<<"Reshaped data size: "<<sorted_dat.shape_<<std::endl;
  mxnet_op::Kernel<range_fwd, gpu>::Launch(s, batch_size * element_num, 1, 0, 1,
                                           kWriteTo, indices.dptr_);

  CHECK_EQ(sorted_dat.CheckContiguous(), true);
  CHECK_EQ(indices.CheckContiguous(), true);
  temp_workspace = Tensor<gpu, 1, char>(workspace_curr_ptr, Shape1(temp_size), s);  // temp space
  workspace_curr_ptr += temp_size;
  // 2. Perform inplace batch sort using the `SortByKey` in MShadow
  // After sorting, each batch in `sorted_dat` will be sorted in the corresponding order
  //   and the `indices` will contain the corresponding index in `sorted_dat`
  // Sort the data and keep record of the correspondence to global indices.
  // Instead do a kCalculateTopK

  // kCalculateTopK(NNFloat* pOutput, NNFloat *pKey, uint32_t* pValue, uint32_t batch, uint32_t
  // width, uint32_t k)
  // Pass in ret[0] as first argument.
  kCalculateTopK(sorted_dat.dptr_,
                 values.dptr_,
                 reinterpret_cast<uint32_t*>(indices.dptr_), 32, 30000, 5);

//  mxnet::op::SortByKey(sorted_dat, indices, is_ascend, &temp_workspace);

  // Iterate over sorted_date (shape 6)
  // Calculate the corresponding batch indices of the elements
  //batch_id = indices / element_num;
  // Since the SortByKey performs stable sort, the second SortByKey will reorder
  //   the sorted_dat based on the order of the batch_id

  //mxnet::op::SortByKey(batch_id, sorted_dat, true, &temp_workspace);
  // Reorder the indices
  //batch_id = indices / element_num;
  //mxnet::op::SortByKey(batch_id, indices, true, &temp_workspace);
  // 3. Assign results to the ret blob

//  std::cout<<"We are quite happy"<<std::endl;
  if (param.ret_typ == topk_enum::kReturnIndices) {
    indices -= batch_id * element_num;
    Tensor<gpu, 2, real_t> ret_indices =
        ret[0].get_with_shape<gpu, 2, real_t>(Shape2(batch_size, k), s);
    ret_indices = tcast<real_t>(slice<1>(
        inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k));

  } else {
    indices -= batch_id * element_num;
    Tensor<gpu, 2, real_t> ret_value =
        ret[0].get_with_shape<gpu, 2, real_t>(Shape2(batch_size, k), s);
    Tensor<gpu, 2, real_t> ret_indices =
        ret[1].get_with_shape<gpu, 2, real_t>(Shape2(batch_size, k), s);
    ret_value = slice<1>(inplace_reshape(values, Shape2(batch_size, element_num)), 0, k);
    ret_indices = tcast<real_t>(slice<1>(
        inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k));
  }
}

void FastTopKGpu(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const std::vector<TBlob>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  // TODO(sxjscience) We can support inplace in the future
  CHECK_EQ(req[0], kWriteTo) << "TopK does not support inplace";
//  std::cout<<"inputs size: "<<inputs.size()<<std::endl;
  FastTopKImplGpu(ctx.run_ctx.get_stream<gpu>(), ctx.requested[0], inputs[0], outputs, param);
}

NNVM_REGISTER_OP(fast_topk).set_attr<FCompute>("FCompute<gpu>", FastTopKGpu);

}  // namespace op
}  // namespace mxnet