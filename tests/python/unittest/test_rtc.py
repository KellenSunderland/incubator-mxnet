# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import numpy as np
import mxnet as mx
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr


class TestRTC:
    @attr('gpu')
    def test_basic_rtc(self):
        x = mx.nd.zeros((10,), ctx=mx.gpu(0))
        x[:] = 1
        y = mx.nd.zeros((10,), ctx=mx.gpu(0))
        y[:] = 2
        rtc = mx.rtc('abc', [('x', x)], [('y', y)], """
            __shared__ float s_rec[10];
            s_rec[threadIdx.x] = x[threadIdx.x];
            y[threadIdx.x] = expf(s_rec[threadIdx.x]*5.0);""")
        rtc.push([x], [y], (1, 1, 1), (10, 1, 1))
        assert_allclose(y.asnumpy(), np.exp(x.asnumpy() * 5.0))

    @attr('gpu')
    def test_cuda_rtc(self):
        source = r'''
        extern "C" __global__ void axpy(const float *x, float *y, float alpha) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            y[i] += alpha * x[i];
        }

        extern "C" __global__ void saxpy(const float *x, float *y, float alpha) {
            extern __shared__ float smem[];
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            smem[threadIdx.x] = x[i];
            y[i] += alpha * smem[threadIdx.x];
        }
        '''
        module = mx.rtc.CudaModule(source)
        axpy = module.get_kernel("axpy", "const float *x, float *y, float alpha")
        x = mx.nd.ones((10,), ctx=mx.gpu(0))
        y = mx.nd.zeros((10,), ctx=mx.gpu(0))
        axpy.launch([x, y, 3.0], mx.gpu(0), (1, 1, 1), (10, 1, 1))
        assert (y.asnumpy() == 3).all()

        saxpy = module.get_kernel("saxpy", "const float *x, float *y, float alpha")
        saxpy.launch([x, y, 4.0], mx.gpu(0), (1, 1, 1), (10, 1, 1), 10)
        assert (y.asnumpy() == 7).all()

        saxpy.launch([x, y, 5.0], mx.gpu(0), (2, 1, 1), (5, 1, 1), 5)
        assert (y.asnumpy() == 12).all()


if __name__ == '__main__':
    import nose

    nose.runmodule()
