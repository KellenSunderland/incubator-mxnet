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

import numpy as np
import mxnet as mx
from nose.plugins.attrib import attr


class TestExecutor:
    @staticmethod
    def reldiff(a, b):
        diff = np.sum(np.abs(a - b))
        norm = np.sum(np.abs(a))
        reldiff = diff / norm
        return reldiff

    @staticmethod
    def check_bind_with_uniform(uf, gf, dim, sf=None, lshape=None, rshape=None):
        """check function consistency with uniform random numbers"""
        shape = tuple(np.random.randint(1, int(1000 ** (1.0 / dim)), size=dim))
        lhs = mx.symbol.Variable('lhs')
        rhs = mx.symbol.Variable('rhs')
        if sf is not None:
            ret = sf(lhs, rhs)
        else:
            ret = uf(lhs, rhs)

        assert ret.list_arguments() == ['lhs', 'rhs']
        lshape = shape if lshape is None else lshape
        rshape = shape if rshape is None else rshape

        lhs_arr = mx.nd.array(np.random.uniform(-1, 1, lshape))
        rhs_arr = mx.nd.array(np.random.uniform(-1, 1, rshape))
        lhs_grad = mx.nd.empty(lshape)
        rhs_grad = mx.nd.empty(rshape)
        executor = ret.bind(mx.Context('cpu'),
                            args=[lhs_arr, rhs_arr],
                            args_grad=[lhs_grad, rhs_grad])

        exec3 = ret.bind(mx.Context('cpu'),
                         args=[lhs_arr, rhs_arr])

        exec4 = ret.bind(mx.Context('cpu'),
                         args={'rhs': rhs_arr, 'lhs': lhs_arr},
                         args_grad={'lhs': lhs_grad, 'rhs': rhs_grad})

        executor.forward()
        exec3.forward()
        exec4.forward()
        out2 = executor.outputs[0].asnumpy()
        out1 = uf(lhs_arr.asnumpy(), rhs_arr.asnumpy())
        out3 = exec3.outputs[0].asnumpy()
        out4 = exec4.outputs[0].asnumpy()
        assert TestExecutor.reldiff(out1, out2) < 1e-6
        assert TestExecutor.reldiff(out1, out3) < 1e-6
        assert TestExecutor.reldiff(out1, out4) < 1e-6
        # test gradient
        out_grad = mx.nd.array(np.ones(out2.shape))
        lhs_grad2, rhs_grad2 = gf(out_grad.asnumpy(),
                                  lhs_arr.asnumpy(),
                                  rhs_arr.asnumpy())
        executor.backward([out_grad])

        assert TestExecutor.reldiff(lhs_grad.asnumpy(), lhs_grad2) < 1e-6
        assert TestExecutor.reldiff(rhs_grad.asnumpy(), rhs_grad2) < 1e-6

    # TODO (kellens): make paramaterized
    # Should run for both true and false
    @attr('cpu')
    def test_bind(self, disable_bulk_exec=False):
        if disable_bulk_exec:
            prev_bulk_inf_val = mx.test_utils.set_env_var("MXNET_EXEC_BULK_EXEC_INFERENCE", "0", "1")
            prev_bulk_train_val = mx.test_utils.set_env_var("MXNET_EXEC_BULK_EXEC_TRAIN", "0", "1")

        np.random.seed(0)
        nrepeat = 10
        maxdim = 4
        for repeat in range(nrepeat):
            for dim in range(1, maxdim):
                TestExecutor.check_bind_with_uniform(lambda x, y: x + y,
                                                     lambda g, x, y: (g, g),
                                                     dim)
                TestExecutor.check_bind_with_uniform(lambda x, y: x - y,
                                                     lambda g, x, y: (g, -g),
                                                     dim)
                TestExecutor.check_bind_with_uniform(lambda x, y: x * y,
                                                     lambda g, x, y: (y * g, x * g),
                                                     dim)
                TestExecutor.check_bind_with_uniform(lambda x, y: x / y,
                                                     lambda g, x, y: (g / y, -x * g / (y ** 2)),
                                                     dim)

                TestExecutor.check_bind_with_uniform(lambda x, y: np.maximum(x, y),
                                                     lambda g, x, y: (g * (x > y), g * (y > x)),
                                                     dim,
                                                     sf=mx.symbol.maximum)
                TestExecutor.check_bind_with_uniform(lambda x, y: np.minimum(x, y),
                                                     lambda g, x, y: (g * (x < y), g * (y < x)),
                                                     dim,
                                                     sf=mx.symbol.minimum)
        if disable_bulk_exec:
            mx.test_utils.set_env_var("MXNET_EXEC_BULK_EXEC_INFERENCE", prev_bulk_inf_val)
            mx.test_utils.set_env_var("MXNET_EXEC_BULK_EXEC_TRAIN", prev_bulk_train_val)

    @attr('cpu')
    def test_dot(self):
        np.random.seed(0)
        nrepeat = 10
        for repeat in range(nrepeat):
            s = tuple(np.random.randint(1, 500, size=3))
            TestExecutor.check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                                 lambda g, x, y: (np.dot(g, y.T), np.dot(x.T, g)),
                                                 2,
                                                 lshape=(s[0], s[1]),
                                                 rshape=(s[1], s[2]),
                                                 sf=mx.symbol.dot)
        for repeat in range(nrepeat):
            s = tuple(np.random.randint(1, 500, size=1))
            TestExecutor.check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                                 lambda g, x, y: (g * y, g * x),
                                                 2,
                                                 lshape=(s[0],),
                                                 rshape=(s[0],),
                                                 sf=mx.symbol.dot)

    @attr('cpu')
    def test_reshape(self):
        x = mx.sym.Variable('x')
        y = mx.sym.FullyConnected(x, num_hidden=4)

        exe = y.simple_bind(mx.cpu(), x=(5, 4), grad_req='null')
        exe.arg_arrays[0][:] = 1
        exe.arg_arrays[1][:] = mx.nd.ones((4, 4))
        exe.arg_arrays[2][:] = 0

        new_exe = exe.reshape(x=(3, 4))
        new_exe.forward(is_train=False)
        # test sub exec forward
        assert np.all(new_exe.outputs[0].asnumpy() == 4)
        # test shared memory
        assert np.all(exe.outputs[0].asnumpy()[:3] == 4)
        # test base exec forward
        exe.forward(is_train=False)
        assert np.all(exe.outputs[0].asnumpy() == 4)


if __name__ == "__main__":
    import nose

    nose.runmodule()
