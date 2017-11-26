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

from mxnet.test_utils import *
import random


def gen_rsp_random_indices(shape, density=.5, force_indices=None):
    assert 0 <= density <= 1
    indices = set()
    if force_indices is not None:
        for val in force_indices:
            indices.add(int(val))
    if not np.isclose(density, .0, rtol=1.e-3, atol=1.e-3, equal_nan=True) and len(shape) > 0:
        row_count = shape[0]
        for i in range(row_count):
            r = random.uniform(0, 1)
            if r <= density and len(indices) < shape[0]:
                indices.add(i)
    assert len(indices) <= shape[0]
    return list(indices)


# Make sure that 0's look like 0's when we do a comparison

def do_normalize(arr):
    ret = arr.copy()
    idx = np.isclose(arr, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True)
    ret[idx] = 0
    return ret


def get_result_type(call, dflt_stype):
    """Try to infer result storage type for a sparse matrix and a given unary operation"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = do_normalize(call(zero))
        if not almost_equal(result, zero, equal_nan=True):
            expected_result_type = 'default'
        else:
            if dflt_stype is not None:
                expected_result_type = dflt_stype
            else:
                expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_with_scalar(call, dflt_stype):
    """Try to infer result storage type when operating a sparse matrices and a scalar"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = call(zero, 5)

        if not almost_equal(result, zero, equal_nan=True):
            expected_result_type = 'default'
        else:
            if dflt_stype is not None:
                expected_result_type = dflt_stype
            else:
                expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_fw_bw_result_types(forward_numpy_call, fwd_res_dflt,
                           backward_numpy_call, bwd_res_dflt):
    return (get_result_type(forward_numpy_call, fwd_res_dflt),
            get_result_type(backward_numpy_call, bwd_res_dflt))


# TODO(kellens): make common.

def get_fw_bw_result_types_2(forward_numpy_call, fwd_res_dflt,
                             backward_numpy_call, bwd_res_dflt):
    return (get_result_type(forward_numpy_call, fwd_res_dflt),
            get_result_type_2(backward_numpy_call, bwd_res_dflt))


def is_scalar(var):
    return False if hasattr(var, "__len__") else True


def get_fw_bw_result_types_with_scalar(forward_numpy_call, fwd_res_dflt,
                                       backward_numpy_call, bwd_res_dflt):
    return (get_result_type_with_scalar(forward_numpy_call, fwd_res_dflt),
            get_result_type_with_scalar(backward_numpy_call, bwd_res_dflt))


def as_dense(arr):
    if arr.stype != 'default':
        return mx.nd.cast_storage(arr, stype='default')
    else:
        return arr


def get_result_type_2(call, dflt_stype):
    """Try to infer result storage type when operating on two sparse matrices"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for outer in [zero, np.ones(zero.shape)]:
            for inner in [zero, np.ones(zero.shape)]:
                result = do_normalize(call(outer, inner))
                if not almost_equal(result, zero, equal_nan=True):
                    need_default = True
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_3(call, dflt_stype):
    """Try to infer result storage type when operating on three sparse matrices"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for moon in [zero]:
            for outer in [zero]:
                for inner in [zero]:
                    res_1, res_2 = call(moon, outer, inner)
                    result = do_normalize(res_1)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                    result = do_normalize(res_2)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                if need_default is True:
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def all_zero(var):
    return 0
