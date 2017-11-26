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

import mxnet.ndarray as nd
from mxnet.contrib.autograd import *
from mxnet.test_utils import *
from nose.plugins.attrib import attr


class TestContribAutograd:
    @staticmethod
    def autograd_assert(*args, **kwargs):
        func = kwargs["func"]
        grad_f = kwargs["grad_func"]
        argnum = kwargs["argnum"] if 'argnum' in kwargs else None

        grad_func = grad_and_loss(func, argnum)
        grad_vals, output = grad_func(*args)
        res = func(*args)
        assert same(output.asnumpy(), res.asnumpy())
        grad_res = grad_f(*args)
        assert len(grad_vals) == len(grad_res)
        for a, b in zip(grad_vals, grad_res):
            assert same(a.asnumpy(), b.asnumpy())

    @attr('cpu')
    def test_unary_func(self):
        x = nd.uniform(shape=(4, 5))
        f_exp = lambda x: nd.exp(x)
        f_exp_grad = lambda x: [nd.exp(x)]
        TestContribAutograd.autograd_assert(x, func=f_exp, grad_func=f_exp_grad)
        f_half = lambda x: x / 2
        f_half_grad = lambda x: [nd.ones(x.shape) * 0.5]
        TestContribAutograd.autograd_assert(x, func=f_half, grad_func=f_half_grad)
        f_square = lambda x: x ** 2
        f_square_grad = lambda x: [2 * x]
        TestContribAutograd.autograd_assert(x, func=f_square, grad_func=f_square_grad)

    @attr('cpu')
    def test_binary_func(self):
        x = nd.uniform(shape=(4, 5))
        y = nd.uniform(shape=(4, 5))
        f_add = lambda x, y: x + y
        f_add_grad = lambda x, y: [nd.ones(x.shape), nd.ones(y.shape)]
        TestContribAutograd.autograd_assert(x, y, func=f_add, grad_func=f_add_grad)
        f_mul = lambda x, y: x * y
        f_mul_grad = lambda x, y: [y, x]
        TestContribAutograd.autograd_assert(x, y, func=f_mul, grad_func=f_mul_grad)
        f_compose = lambda x, y: x + x * y
        f_compose_grad = lambda x, y: [nd.ones(x.shape) + y, x]
        TestContribAutograd.autograd_assert(x, y, func=f_compose, grad_func=f_compose_grad)

    @attr('cpu')
    def test_operator_with_state(self):
        def f_fc(a, b, weight, bias):
            x = a * b
            fc = nd.FullyConnected(
                x, weight, bias, num_hidden=32)
            return fc

        a = nd.uniform(shape=(64, 50))
        b = nd.uniform(shape=(64, 50))
        weight = nd.uniform(shape=(32, 50))
        bias = nd.uniform(shape=(32,))

        grad_func = grad_and_loss(f_fc)
        grad_vals, outputs = grad_func(a, b, weight, bias)
        # (TODO) assert

    @attr('cpu')
    def test_argnum(self):
        def f_with_mode(a, b, mode):
            if mode:
                return a + b
            else:
                return a * b

        a = nd.uniform(shape=(3, 2))
        b = nd.uniform(shape=(3, 2))
        f_add_grad = lambda x, y, mode: [nd.ones(x.shape), nd.ones(y.shape)]
        f_mul_grad = lambda x, y, mode: [y, x]
        TestContribAutograd.autograd_assert(a, b, True,
                                            argnum=[0, 1], func=f_with_mode, grad_func=f_add_grad)
        TestContribAutograd.autograd_assert(a, b, False,
                                            argnum=[0, 1], func=f_with_mode, grad_func=f_mul_grad)

    @attr('cpu')
    def test_training(self):
        x = nd.ones((10, 10))
        with train_section():
            y = nd.Dropout(x, p=0.5)
            assert not (y.asnumpy() == x.asnumpy()).all()
            with test_section():
                y = nd.Dropout(x, p=0.5)
                assert (y.asnumpy() == x.asnumpy()).all()

    @attr('cpu')
    def test_out_grads(self):
        x = nd.ones((3, 5))
        dx = nd.zeros_like(x)
        mark_variables([x], [dx])
        da = None
        db = nd.array([1, 2, 3, 4, 5])
        dc = nd.array([5, 4, 3, 2, 1])

        with train_section():
            a, b, c = nd.split(x, axis=0, num_outputs=3, squeeze_axis=True)
            backward([a, b, c], [da, db, dc])

        assert (dx.asnumpy() == np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 3, 4, 5],
             [5, 4, 3, 2, 1]])).all()

    @attr('cpu')
    def test_detach_updated_grad(self):
        x = nd.ones((2, 2))
        dx = nd.zeros_like(x)
        y = nd.ones_like(x)
        dy = nd.zeros_like(x)
        mark_variables([x, y], [dx, dy])
        assert x._fresh_grad == False
        assert y._fresh_grad == False

        with train_section():
            x2 = x + 2
            y2 = x2 + y
            y2.backward()
        assert (dx.asnumpy() == 1).all()
        assert x._fresh_grad == True
        assert y._fresh_grad == True

        dx[:] = 0
        x._fresh_grad = False
        y._fresh_grad = False
        assert x._fresh_grad == False
        assert y._fresh_grad == False
        with train_section():
            x2 = x + 2
            x2 = x2.detach()
            y2 = x2 + y
            y2.backward()
        assert (dx.asnumpy() == 0).all()
        assert y._fresh_grad == True
        assert x._fresh_grad == False

    @attr('cpu')
    def test_retain_grad(self):
        x = mx.nd.ones((2, 2))
        dx = mx.nd.zeros((2, 2))
        mark_variables([x], [dx], grad_reqs='add')
        with train_section():
            y = x + 1
            y.backward(retain_graph=False)
        assert (dx.asnumpy() == 1).all()

        dx[:] = 0
        with train_section():
            y = x + 1
            y.backward(retain_graph=True)
            y.backward(retain_graph=False)
        assert (dx.asnumpy() == 2).all()

        # The following sequence should throw an exception. We discard the expected
        # stderr stack trace output for this operation to keep the test logs clean.
        with discard_stderr():
            try:
                with train_section():
                    y = x + 1
                    y.backward()
                    y.backward()
            except Exception:
                return

        raise AssertionError(
            "differentiating the same graph twice without retain_graph should fail")


if __name__ == "__main__":
    import nose

    nose.runmodule()
