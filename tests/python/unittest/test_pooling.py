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

class TestPooling:
    def test_pooling_versions(self):
        def test_pooling_versions_helper(pool_op_list, data, kernel, pool_type, pad, stride,
                                         pooling_convention='valid', global_pool=False):
            ctx_list = []
            sym_list = []
            # PoolingV1 cpu
            if 'pool_v1_cpu' in pool_op_list:
                ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
                if not global_pool:
                    sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                      pooling_convention=pooling_convention, name='pool'))
                else:
                    sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
            # PoolingV1 gpu
            if 'pool_v1_gpu' in pool_op_list:
                ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
                if not global_pool:
                    sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                      pooling_convention=pooling_convention, name='pool'))
                else:
                    sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
            # Pooling cpu
            if 'pool_cpu' in pool_op_list:
                ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
                if not global_pool:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                   pooling_convention=pooling_convention, name='pool'))
                else:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
            # Pooling gpu
            if 'pool_gpu' in pool_op_list:
                ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
                if not global_pool:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                   pooling_convention=pooling_convention, cudnn_off=True, name='pool'))
                else:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, cudnn_off=True,
                                                   name='pool'))
            # CuDNNPooling
            if 'pool_cudnn' in pool_op_list:
                ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
                if not global_pool:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                   pooling_convention=pooling_convention, cudnn_off=False, name='pool'))
                else:
                    sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, cudnn_off=False,
                                                   name='pool'))
            check_consistency(sym_list, ctx_list)


        def test_1d_pooling(pool_type):
            data = (2, 3, 20)
            kernel = (4,)
            pad = (0,)
            stride = (1,)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            pad = (2,)
            stride = (2,)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            pad = (0,)
            stride = (1,)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            pad = (2,)
            stride = (2,)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         global_pool=True)


        def test_2d_pooling(pool_type):
            data = (2, 3, 20, 20)
            kernel = (4, 5)
            pad = (0, 0)
            stride = (1, 1)
            test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            # pool_v1 has bugs when pad is not 0, do not test PoolingV1 here
            pad = (2, 3)
            stride = (2, 3)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            pad = (0, 0)
            stride = (1, 1)
            test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            # pool_v1 has bugs when pad is not 0, do not test PoolingV1 here
            pad = (2, 3)
            stride = (2, 3)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         global_pool=True)

        def test_3d_pooling(pool_type):
            data = (2, 3, 20, 20, 20)
            kernel = (4, 5, 3)
            pad = (0, 0, 0)
            stride = (1, 1, 1)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            pad = (2, 3, 3)
            stride = (2, 3, 1)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='valid', global_pool=False)

            pad = (0, 0, 0)
            stride = (1, 1, 1)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            pad = (2, 3, 3)
            stride = (2, 3, 1)
            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         pooling_convention='full', global_pool=False)

            test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                         data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                         global_pool=True)

        test_1d_pooling('max')
        test_1d_pooling('avg')
        test_1d_pooling('sum')

        test_2d_pooling('max')
        test_2d_pooling('avg')
        test_2d_pooling('sum')

        test_3d_pooling('max')
        test_3d_pooling('avg')
        test_3d_pooling('sum')

if __name__ == '__main__':
    import nose
    nose.runmodule()
