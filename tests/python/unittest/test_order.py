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

from __future__ import print_function
from mxnet.test_utils import *


def test_order_simple():
    x = mx.nd.array([[(5, 6, 7), (4, 9, 6)]])
    result = mx.nd.topk(x, ret_typ='value')
    # Should be the index 2 (for val 7) and index 1 (for val 9)
    assert_almost_equal(result.asnumpy(), np.array([[(7,), (9,)]]))


def test_order():
    ctx = default_context()

    def gt_topk(dat, axis, ret_typ, k, is_ascend):
        if ret_typ == "indices":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
        elif ret_typ == "value":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(np.sort(dat, axis=axis), axis=axis, indices=indices, mode='wrap')
        else:
            assert dat.shape == (5, 5, 5, 5)
            assert axis is None or axis == 1
            ret = np.zeros(dat.shape)
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            gt_argsort = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
            if axis is None:
                ret.ravel()[gt_argsort] = 1
            else:
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            ret[i, gt_argsort[i, :, j, k], j, k] = 1
        return ret

    dshape = (5, 5, 5, 5)
    a_npy = np.arange(np.prod(dshape)).astype(np.float32)
    np.random.shuffle(a_npy)
    a_npy = a_npy.reshape(dshape)
    a = mx.sym.Variable('a')

    for axis in [1, 3, None]:
        K = [1, 3, 5, 7] if axis is None else [1, 3, 5]
        for k in K:
            for is_ascend in [True, False]:
                b = mx.sym.topk(a, axis=axis, is_ascend=is_ascend, ret_typ="value", k=k)
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=k, is_ascend=is_ascend)
                check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, ctx=ctx)
                check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

    for axis in [1, 3, None]:
        for is_ascend in [True, False]:
            b = mx.sym.sort(a, axis=axis, is_ascend=is_ascend)
            if axis is None:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=a_npy.size, is_ascend=is_ascend)
            else:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=5, is_ascend=is_ascend)
            check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, ctx=ctx)
            check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

    b = mx.sym.topk(a, axis=3, is_ascend=is_ascend, ret_typ="indices", k=3)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 3))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=3, ret_typ="indices", k=3,
                                             is_ascend=False)])

    b = mx.sym.topk(a, axis=1, is_ascend=True, ret_typ="mask", k=3)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="mask", k=3,
                                             is_ascend=True)])

    b = mx.sym.argsort(a, axis=1, is_ascend=False)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=5,
                                             is_ascend=False)])

    b = mx.sym.argmax(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=False)])

    b = mx.sym.argmin(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=True)])
