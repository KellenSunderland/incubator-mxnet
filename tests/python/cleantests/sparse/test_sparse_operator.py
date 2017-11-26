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
import warnings
from sparse_common import *
from nose.plugins.attrib import attr


class TestSparseOperator:

    # This test takes over a minute to run, moving to nightly builds.
    @attr('nightly')
    @attr('cpu')
    @attr('gpu')
    def test_elemwise_binary_ops(self):
        def test_elemwise_binary_op(name, lhs_stype, rhs_stype, shape,
                                    forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                                    lhs_grad_stype,
                                    rhs_grad_stype,
                                    expected_result_storage_type=None,
                                    modifier_func=None,
                                    lhs_density=.5,
                                    rhs_density=.5,
                                    force_lr_overlap=False,
                                    force_grad_overlap=False,
                                    ograd_density=0.0,
                                    skip_gradient_check=False,
                                    shuffle_csr_indices=True,
                                    verbose=False):
            if lhs_grad_stype is None:
                lhs_grad_stype = lhs_stype
            if rhs_grad_stype is None:
                rhs_grad_stype = rhs_stype

            lhs_grad_stype = get_result_type_3(backward_numpy_call, lhs_grad_stype)
            rhs_grad_stype = get_result_type_3(backward_numpy_call, rhs_grad_stype)

            if verbose is True:
                print("testing: {}  lhs={}, rhs={}, lhs_grad_stype={}, rhs_grad_stype={}"
                      .format(name, lhs_stype, rhs_stype, lhs_grad_stype, rhs_grad_stype))

            # Output type should be same as lvalue type, unless otherwise specified
            if expected_result_storage_type is None:
                if lhs_stype == 'default' or rhs_stype == 'default':
                    expected_result_storage_type = 'default'
                else:
                    expected_result_storage_type = lhs_stype

            lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
            rhs = mx.symbol.Variable('rhs', stype=rhs_stype)

            grad_stypes = dict()
            grad_stypes['lhs'] = lhs_grad_stype
            grad_stypes['rhs'] = rhs_grad_stype

            if lhs_stype == 'default':
                lhs_nd = rand_ndarray(shape, 'default')
                if abs(lhs_density) < 1e-4:
                    func = all_zero
                else:
                    func = modifier_func
                lhs_nd = mx.nd.array(assign_each(lhs_nd.asnumpy(), func))
            else:
                lhs_nd = create_sparse_array_zd(
                    shape, lhs_stype, density=lhs_density,
                    modifier_func=modifier_func,
                    shuffle_csr_indices=shuffle_csr_indices,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=lhs_density,
                        force_indices=[(shape[0] / 2)] if force_lr_overlap is True else None
                    ))

            if rhs_stype == 'default':
                rhs_nd = rand_ndarray(shape, 'default')
                if abs(rhs_density) < 1e-4:
                    func = all_zero
                else:
                    func = modifier_func
                rhs_nd = mx.nd.array(assign_each(rhs_nd.asnumpy(), func))
            else:
                rhs_nd = create_sparse_array_zd(
                    shape, rhs_stype, density=rhs_density,
                    modifier_func=modifier_func,
                    shuffle_csr_indices=shuffle_csr_indices,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=rhs_density,
                        force_indices=[(shape[0] / 2)] if force_lr_overlap is True else None
                    ))

            lhs_np = lhs_nd.asnumpy()
            rhs_np = rhs_nd.asnumpy()

            if verbose is True:
                print("lhs input: {}".format(lhs_np))
                print("rhs input: {}".format(rhs_np))

            out_np = forward_numpy_call(lhs_np, rhs_np)

            if verbose is True:
                print("out_np: {}".format(out_np))

            test = forward_mxnet_call(lhs, rhs)

            location = {'lhs': lhs_nd, 'rhs': rhs_nd}

            outputs = check_symbolic_forward(test, location, [out_np], equal_nan=True)
            assert len(outputs) == 1
            assert outputs[0].stype == expected_result_storage_type

            if verbose is True:
                print("mx forward output: ", outputs[0].asnumpy())
                print("lhs_nd: ", lhs_nd.stype)
                print("rhs_nd: ", rhs_nd.stype)
                print("forward output: ", outputs[0].stype)

            if outputs[0].stype != 'default':
                out_grad = create_sparse_array_zd(
                    shape, outputs[0].stype, density=ograd_density,
                    data_init=1,
                    modifier_func=lambda x: 2,
                    shuffle_csr_indices=shuffle_csr_indices,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=ograd_density,
                        force_indices=[(shape[0] / 2)] if force_grad_overlap is True else None
                    ))
            else:
                if abs(ograd_density) < 1e-4:
                    out_grad = mx.nd.array(np.zeros(shape))
                else:
                    out_grad = mx.nd.array(np.ones(shape))

            out_grad_np = out_grad.asnumpy()

            if verbose is True:
                print("out_grad_np", out_grad_np)

            ingrad_lhs_np, ingrad_rhs_np = backward_numpy_call(out_grad_np, lhs_np, rhs_np)

            if verbose is True:
                print("out_grad", out_grad.asnumpy())
                print("ingrad_lhs_np", ingrad_lhs_np)
                print("ingrad_rhs_np", ingrad_rhs_np)

            igrads_result = check_symbolic_backward(test, location, [out_grad],
                                                    [ingrad_lhs_np, ingrad_rhs_np],
                                                    grad_stypes=grad_stypes,
                                                    equal_nan=True)

            if verbose is True:
                print("ingrad_lhs", igrads_result['lhs'].asnumpy())
                print("ingrad_rhs", igrads_result['rhs'].asnumpy())

            assert len(igrads_result) == 2

            if lhs_grad_stype is not None:
                assert igrads_result['lhs'].stype == lhs_grad_stype
            if rhs_grad_stype is not None:
                assert igrads_result['rhs'].stype == rhs_grad_stype

            if skip_gradient_check is not True:
                check_numeric_gradient(test, location,
                                       grad_stype_dict=grad_stypes)

        def check_all(l, r, check_function):
            assert l.shape == r.shape
            return check_function(l, r)

        def gt(l, r):
            return check_all(l, r, lambda a, b: a > b)

        def ge(l, r):
            return check_all(l, r, lambda a, b: a >= b)

        def lt(l, r):
            return check_all(l, r, lambda a, b: a < b)

        def le(l, r):
            return check_all(l, r, lambda a, b: a <= b)

        def least_sparse(lstype, rstype):
            if lstype == 'default' and rstype == 'default':
                return 'default'
            elif rstype != 'default':
                return rstype
            return lstype

        def most_dense(lstype, rstype):
            if lstype == rstype:
                return lstype
            return 'default'

        def check_elemwise_binary_ops(lhs_stype, rhs_stype, shape,
                                      lhs_grad_stype=None, rhs_grad_stype=None,
                                      lhs_density=.5, rhs_density=.5,
                                      force_lr_overlap=False,
                                      force_grad_overlap=False,
                                      ograd_density=0.0):
            test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_add(l, r),
                                    lambda l, r: l + r,
                                    lambda outg, l, r: (outg, outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)

            test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_sub(l, r),
                                    lambda l, r: l - r,
                                    lambda outg, l, r: (outg, -outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density,
                                    rhs_density=rhs_density,
                                    verbose=False)

            test_elemwise_binary_op("elemwise_mul", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_mul(l, r),
                                    lambda l, r: l * r,
                                    lambda outg, l, r: (outg * r, outg * l),
                                    least_sparse(lhs_stype, rhs_stype),
                                    least_sparse(lhs_stype, rhs_stype),
                                    expected_result_storage_type=most_dense(lhs_stype, rhs_stype),
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)

            test_elemwise_binary_op("elemwise_div", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_div(l, r),
                                    lambda l, r: l / r,
                                    lambda outg, l, r: (outg * (1 / r), outg * (-l / (r * r))),
                                    lhs_grad_stype, rhs_grad_stype,
                                    modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    ograd_density=ograd_density,
                                    expected_result_storage_type='default',
                                    skip_gradient_check=True,
                                    verbose=False)

            test_elemwise_binary_op("maximum", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym._internal._maximum(l, r),
                                    lambda l, r: np.maximum(l, r),
                                    lambda outg, l, r: (outg * ge(l, r), outg * lt(l, r)),
                                    lhs_grad_stype, rhs_grad_stype,
                                    modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    skip_gradient_check=True,
                                    ograd_density=ograd_density,
                                    verbose=False)

            test_elemwise_binary_op("minimum", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym._internal._minimum(l, r),
                                    lambda l, r: np.minimum(l, r),
                                    lambda outg, l, r: (outg * le(l, r), outg * gt(l, r)),
                                    lhs_grad_stype, rhs_grad_stype,
                                    modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    ograd_density=ograd_density,
                                    skip_gradient_check=True,
                                    verbose=False)

            test_elemwise_binary_op("hypot", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym._internal._hypot(l, r),
                                    lambda l, r: np.hypot(l, r),
                                    lambda outg, l, r: (
                                        outg * assign_each2(
                                            l, r, lambda a, b: a / np.sqrt(a * a + b * b)),
                                        outg * assign_each2(
                                            l, r, lambda a, b: b / np.sqrt(a * a + b * b))
                                    ),
                                    lhs_grad_stype, rhs_grad_stype,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    ograd_density=ograd_density,
                                    skip_gradient_check=True,
                                    verbose=False)

        # Run basic tests
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for ii in range(1):
                # Run defaults
                check_elemwise_binary_ops('default', 'default', rand_shape_2d())

                # Try different densities
                for lhs_density in [0.0, random.uniform(0, 1), 1.0]:
                    for rhs_density in [0.0, random.uniform(0, 1), 1.0]:
                        for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                            shape = rand_shape_2d()

                            print("lhs_density={}, rhs_density={}, ograd_density={}, shape: {}".format(
                                lhs_density, rhs_density, ograd_density, shape))

                            # Try row_sparse overlaps
                            for force_lr_overlap in [False, True]:
                                for force_grad_overlap in [False, True]:

                                    shape = rand_shape_2d()

                                    print("  force_lr_overlap={}, force_grad_overlap={}, shape={}".
                                          format(force_lr_overlap, force_grad_overlap, shape))

                                    # Left and right always overlap when one is default storage
                                    # (assuming the row_sparse one has some entries in it)
                                    if force_lr_overlap is False:
                                        check_elemwise_binary_ops('default', 'row_sparse', shape,
                                                                  lhs_density=lhs_density,
                                                                  rhs_density=rhs_density,
                                                                  force_lr_overlap=force_lr_overlap,
                                                                  force_grad_overlap=force_grad_overlap,
                                                                  ograd_density=ograd_density)
                                        check_elemwise_binary_ops('row_sparse', 'default', shape,
                                                                  lhs_density=lhs_density,
                                                                  rhs_density=rhs_density,
                                                                  force_lr_overlap=force_lr_overlap,
                                                                  force_grad_overlap=force_grad_overlap,
                                                                  ograd_density=ograd_density)

                                    # Back to left-right overlap possiblities
                                    check_elemwise_binary_ops('row_sparse', 'row_sparse', shape,
                                                              lhs_grad_stype='row_sparse',
                                                              rhs_grad_stype='row_sparse',
                                                              lhs_density=lhs_density,
                                                              rhs_density=rhs_density,
                                                              force_lr_overlap=force_lr_overlap,
                                                              force_grad_overlap=force_grad_overlap,
                                                              ograd_density=ograd_density)

                            # No overlap flags for CSR
                            check_elemwise_binary_ops('csr', 'csr', shape,
                                                      lhs_grad_stype='csr',
                                                      rhs_grad_stype='csr',
                                                      lhs_density=lhs_density,
                                                      rhs_density=rhs_density,
                                                      ograd_density=ograd_density)
                            check_elemwise_binary_ops('csr', 'csr', shape,
                                                      lhs_grad_stype='default',
                                                      rhs_grad_stype='default',
                                                      lhs_density=lhs_density,
                                                      rhs_density=rhs_density,
                                                      ograd_density=ograd_density)
                            check_elemwise_binary_ops('default', 'csr', shape,
                                                      lhs_grad_stype='csr',
                                                      rhs_grad_stype='csr',
                                                      lhs_density=lhs_density,
                                                      rhs_density=rhs_density,
                                                      ograd_density=ograd_density)
                            check_elemwise_binary_ops('csr', 'default', shape,
                                                      lhs_grad_stype='csr',
                                                      rhs_grad_stype='csr',
                                                      lhs_density=lhs_density,
                                                      rhs_density=rhs_density,
                                                      ograd_density=ograd_density)

    @attr('cpu')
    @attr('gpu')
    def test_elemwise_csr_same_zeros(self):
        # Zeroes
        a = mx.nd.sparse.zeros('csr', (1, 1))
        b = mx.nd.elemwise_add(a, a)
        res = a.asnumpy() + a.asnumpy()
        assert_almost_equal(b.asnumpy(), res)

    @attr('cpu')
    @attr('gpu')
    def test_elemwise_add_ex(self):
        def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
            lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
            rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
            lhs_nd = rand_ndarray(shape, lhs_stype)
            rhs_nd = rand_ndarray(shape, rhs_stype)
            lhs_np = lhs_nd.asnumpy()
            rhs_np = rhs_nd.asnumpy()

            out_np = lhs_np + rhs_np
            test = mx.symbol.sparse.elemwise_add(lhs, rhs)
            location = {'lhs': lhs_nd, 'rhs': rhs_nd}
            check_symbolic_forward(test, location, [out_np])
            check_numeric_gradient(test, location)
            grad_stypes = {}
            if lhs_grad_stype is not None and lhs_grad_stype != 'default':
                grad_stypes['lhs'] = lhs_grad_stype
            if rhs_grad_stype is not None and rhs_grad_stype != 'default':
                grad_stypes['rhs'] = rhs_grad_stype
            check_symbolic_backward(test, location, [out_np], [out_np, out_np],
                                    grad_stypes=grad_stypes)

        shapes = [rand_shape_2d(), rand_shape_3d()]
        for shape in shapes:
            check_elemwise_add_ex('default', 'default', shape)
            check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                                  lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')

    @attr('cpu')
    @attr('gpu')
    @attr('nightly')
    def test_cast_storage_ex(self):
        def check_cast_storage(shape, density, from_stype, to_stype, check_numeric_grad=True):
            x = mx.symbol.Variable('x', stype=from_stype)
            x_nd = rand_ndarray(shape, from_stype, density=density)
            x_np = x_nd.asnumpy()
            out_np = x_np
            test = mx.symbol.cast_storage(x, stype=to_stype)
            location = {'x': x_nd}
            check_symbolic_forward(test, location, [out_np])
            # consider disable the numeric grad check for gpu block kernel since the input is large
            if check_numeric_grad:
                check_numeric_gradient(test, location)
            grad_stypes = {'x': to_stype}
            check_symbolic_backward(test, location, [out_np], [out_np], grad_stypes=grad_stypes)

        density = [1.00, 0.50, 0.01]
        for d in density:
            shape_2d = rand_shape_2d()
            shape_3d = rand_shape_3d()
            check_cast_storage(shape_2d, d, 'csr', 'default')
            check_cast_storage(shape_2d, d, 'default', 'csr')
            check_cast_storage(shape_2d, d, 'row_sparse', 'default')
            check_cast_storage(shape_2d, d, 'default', 'row_sparse')
            check_cast_storage(shape_3d, d, 'row_sparse', 'default')
            check_cast_storage(shape_3d, d, 'default', 'row_sparse')
            for i in range(4, 6):
                shape = rand_shape_nd(i, 5)
                check_cast_storage(shape, d, 'default', 'row_sparse')
                check_cast_storage(shape, d, 'row_sparse', 'default')
            # Test specific gpu kernels
            if default_context().device_type is 'gpu':
                dim0 = rnd.randint(1, 10)
                # test gpu thread kernel
                check_cast_storage((dim0, rnd.randint(1, 32)), d, 'default', 'csr')
                # test gpu warp   kernel
                check_cast_storage((dim0, rnd.randint(32, 512)), d, 'default', 'csr')
                # test gpu block  kernel
                check_cast_storage((dim0, rnd.randint(512, 1024)), d, 'default', 'csr',
                                   check_numeric_grad=False)
                # test gpu thread kernel
                check_cast_storage((dim0, rnd.randint(1, 32)), d, 'default', 'row_sparse')
                # test gpu warp   kernel
                check_cast_storage((dim0, rnd.randint(32, 512)), d, 'default', 'row_sparse')
                # test gpu block  kernel
                check_cast_storage((dim0, rnd.randint(512, 1024)), d, 'default', 'row_sparse',
                                   check_numeric_grad=False)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_dot(self):
        def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs, lhs_density, rhs_density):
            lhs_nd = rand_ndarray(lhs_shape, 'csr', density=lhs_density, shuffle_csr_indices=False)
            lhs_dns = lhs_nd.tostype('default')
            rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density)
            rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.tostype('default')

            out = mx.nd.dot(lhs_nd, rhs_nd, transpose_a=trans_lhs)
            out_dns = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
            out_np = out_dns.asnumpy()
            assert_almost_equal(out.asnumpy(), out_np, rtol=1e-4, atol=1e-5)

            # test symbolic forward
            lhs = mx.symbol.Variable('lhs', stype='csr')
            rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
            out = mx.symbol.sparse.dot(lhs, rhs, transpose_a=trans_lhs)
            location = {'lhs': lhs_nd, 'rhs': rhs_nd}
            check_symbolic_forward(out, location, [out_np], rtol=1e-3, atol=1e-4)

            # test symbolic backward
            backward_trans = not trans_lhs
            rhs_backward_grad = mx.nd.dot(lhs_dns, out_dns, transpose_a=backward_trans).asnumpy()
            expected = {'rhs': rhs_backward_grad}
            check_symbolic_backward(out, location, [out_np], expected,
                                    grad_req={'lhs': 'null', 'rhs': 'write'},
                                    rtol=1e-3, atol=1e-4)

        def test_sparse_dot_zero_output(lhs_shape, trans_lhs, rhs_num_cols):
            """Test for nnr_out = 0. Before the fix, the test would fail."""
            lhs = mx.nd.zeros(lhs_shape)
            irow = np.random.randint(0, lhs_shape[0])
            icol = np.random.randint(0, lhs_shape[1])
            lhs[irow, icol] = 1.0
            if trans_lhs:
                rhs = rand_ndarray(shape=(lhs_shape[0], rhs_num_cols), stype='default')
                rhs[irow, :] = 0
            else:
                rhs = rand_ndarray(shape=(lhs_shape[1], rhs_num_cols), stype='default')
                rhs[icol, :] = 0
            dns_out = mx.nd.dot(lhs, rhs, transpose_a=trans_lhs)
            assert mx.nd.sum(mx.nd.abs(dns_out)).asscalar() == 0
            sps_out = mx.nd.sparse.dot(lhs.tostype('csr'), rhs.tostype('row_sparse'), transpose_a=trans_lhs)
            assert same(dns_out.asnumpy(), sps_out.asnumpy())

        density = [1.00, 0.50, 0.01]
        for lhs_d in density:
            lhs_shape = rand_shape_2d(50, 200)
            rhs_d = 1
            test_dot_csr(lhs_shape, (lhs_shape[1], 1), 'default', False, lhs_d, rhs_d)  # test gpu SpMV
            test_dot_csr(lhs_shape, (lhs_shape[0], 1), 'default', True, lhs_d, rhs_d)  # (vector kernel)
            test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(5, 10)), 'default', False, lhs_d, rhs_d)  # test gpu SpMM
            test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(5, 10)), 'default', True, lhs_d,
                         rhs_d)  # (scalar kernel)
            for rhs_d in density:
                test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False, lhs_d, rhs_d)
                test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True, lhs_d, rhs_d)

        test_sparse_dot_zero_output(rand_shape_2d(50, 200), False, 40)
        test_sparse_dot_zero_output(rand_shape_2d(50, 200), True, 40)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_slice(self):
        def check_csr_slice(shape, slice_input):
            storage_type = 'csr'
            B, _ = rand_sparse_ndarray(shape, storage_type)
            np = B.asnumpy()
            begin = rnd.randint(0, B.shape[0] - 1)
            end = rnd.randint(begin + 1, B.shape[0])
            nd_slice = mx.nd.crop(B, begin=begin, end=end)
            assert same(nd_slice.asnumpy(), np[begin:end]), (nd_slice.asnumpy(), np[begin:end])

        shape = (rnd.randint(7, 15), rnd.randint(1, 10))
        check_csr_slice(shape, True)
        check_csr_slice(shape, False)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_retain(self):
        def check_sparse_retain(shape, density, index_type=np.int64):
            num_rows = shape[0]
            rsp, _ = rand_sparse_ndarray(shape=shape, stype='row_sparse', density=density)
            length = np.random.randint(1, num_rows + 1)
            idx = random_sample(list(range(0, num_rows)), length)
            idx.sort()
            dns = rsp.asnumpy()
            tensor_retained_expected = np.zeros(shape)
            for i in idx:
                tensor_retained_expected[i][:] = dns[i]
            indices = mx.nd.array(idx, dtype=index_type)
            rsp_retained = mx.nd.sparse.retain(rsp, indices=indices)
            assert same(tensor_retained_expected, rsp_retained.asnumpy())

            # check numeric gradient
            data = mx.symbol.Variable('data')
            idx = mx.symbol.Variable('indices')
            sym = mx.sym.sparse.retain(data=data, indices=idx)
            check_numeric_gradient(sym, [rsp, indices], grad_nodes=['data'],
                                   grad_stype_dict={'data': 'row_sparse'})

        shape = rand_shape_2d()
        shape_3d = rand_shape_3d()
        densities = [0.01, 0.5, 1.0]
        index_types = [np.float32, np.int32, np.int64]
        for density in densities:
            for itype in index_types:
                check_sparse_retain(shape, density, itype)
                check_sparse_retain(shape_3d, density, itype)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_unary_with_numerics(self):
        def check_sparse_simple(name, stype, mxnet_func, forward_numpy_call,
                                backward_numpy_call, output_grad_stype=None,
                                backward_is_use_output=False):
            if output_grad_stype is None:
                output_grad_stype = stype

            expected_result_type, expected_grad_result_type = \
                get_fw_bw_result_types_2(forward_numpy_call, stype, backward_numpy_call, output_grad_stype)
            if backward_is_use_output is True:
                expected_grad_result_type = expected_result_type

            shape = (3, 4)
            data = mx.symbol.Variable("data")

            grad_stypes = {'data': expected_grad_result_type}

            y = mxnet_func(data)
            if stype == 'default':
                xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
                xa_np = xa
            else:
                xa = create_sparse_array(shape, stype, data_init=None, rsp_indices=[1],
                                         modifier_func=lambda a: a - 0.5,
                                         shuffle_csr_indices=True)
                xa_np = xa.asnumpy()

            if output_grad_stype != 'default':
                out_grad = create_sparse_array(shape, output_grad_stype, data_init=None,
                                               rsp_indices=[1, 2],
                                               modifier_func=lambda a: a - 0.5,
                                               shuffle_csr_indices=True)
                out_grad_np = out_grad.asnumpy()
            else:
                out_grad_np = np.ones(xa.shape)
                out_grad = mx.nd.array(out_grad_np)

            output_np = forward_numpy_call(xa_np)
            input_grad_np = backward_numpy_call(output_np, out_grad_np)

            outputs = check_symbolic_forward(y, [xa], [output_np])
            output = outputs[0]

            assert output.stype == expected_result_type

            input_grad_dict = check_symbolic_backward(y, location=[xa], out_grads=[out_grad],
                                                      expected=[input_grad_np],
                                                      grad_stypes=grad_stypes)
            inp_grad = input_grad_dict["data"]

            assert inp_grad.stype == expected_grad_result_type

        def check_sparse_function(name, mxnet_func, forward_numpy_call, backward_numpy_call,
                                  backward_is_use_output=False):
            check_sparse_simple(name, 'default', mxnet_func, forward_numpy_call, backward_numpy_call)
            for output_grad_stype in [None, "row_sparse", "default"]:
                check_sparse_simple(name, 'row_sparse', mxnet_func, forward_numpy_call, backward_numpy_call,
                                    output_grad_stype=output_grad_stype,
                                    backward_is_use_output=backward_is_use_output)

        check_sparse_function('relu',
                              lambda x: mx.sym.relu(x),
                              lambda x: np.maximum(x, 0.0),
                              lambda output, outg: outg * assign_each(output, lambda x: x > 0.0),
                              backward_is_use_output=True)

        check_sparse_function('sigmoid',
                              lambda x: mx.sym.sigmoid(x),
                              lambda x: np.divide(1.0, (1.0 + np.exp(-x))),
                              lambda output, outg: outg * assign_each(output, lambda x: x * (1.0 - x)),
                              backward_is_use_output=True)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_nd_zeros(self):
        def check_sparse_nd_zeros(stype, shape):
            zero = mx.nd.zeros(shape)
            sparse_zero = mx.nd.zeros(shape=shape, stype=stype)
            assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

        shape = rand_shape_2d()
        check_sparse_nd_zeros('row_sparse', shape)
        check_sparse_nd_zeros('csr', shape)
        check_sparse_nd_zeros('default', shape)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_nd_zeros_like(self):
        def check_sparse_nd_zeros_like(stype, shape):
            zero = mx.nd.zeros(shape, stype=stype)
            zero_like = mx.nd.sparse.zeros_like(zero)
            assert_almost_equal(zero.asnumpy(), zero_like.asnumpy())

        shape = rand_shape_2d()
        check_sparse_nd_zeros_like('row_sparse', shape)
        check_sparse_nd_zeros_like('csr', shape)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_axis_operations(self):
        def test_variations(func_name):
            dim0 = 30
            dim1 = 100
            axes = [0, 1]
            densities = [0, 0.5, 1]
            for density in densities:
                shape = rand_shape_2d(dim0, dim1)
                csr_array = rand_ndarray(shape=shape, stype='csr', density=density)
                dns = csr_array.tostype('default')
                for axis in axes:
                    ret = func_name(csr_array, axis=axis)
                    assert ret.stype == 'default'
                    ret_expected = func_name(dns, axis=axis)
                    assert_almost_equal(ret.asnumpy(), ret_expected.asnumpy())

        def test_fallback(func_name, axis=0, keepdims=True, exclude=True):
            dim0 = 30
            dim1 = 100
            shape = rand_shape_2d(dim0, dim1)
            csr_array = rand_ndarray(shape=shape, stype='csr', density=0.01)
            ret = func_name(csr_array, axis=axis, keepdims=keepdims,
                            exclude=exclude)

        test_variations(mx.nd.sum)
        test_fallback(mx.nd.sum, axis=0, keepdims=True, exclude=True)
        test_variations(mx.nd.mean)
        test_fallback(mx.nd.mean, axis=0, keepdims=True, exclude=True)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_square_sum(self):
        if default_context().device_type == 'cpu':
            dim0 = 30
            dim1 = 30
            axes = [0, 1]
            keepdims = [False, True]
            densities = [0, 0.01, 0.2, 0.5, 1.0]
            for density in densities:
                shape = rand_shape_2d(dim0, dim1)
                rsp = rand_ndarray(shape, 'row_sparse', density)
                dns = rsp.tostype('default')
                for axis in axes:
                    for keepdim in keepdims:
                        ret = mx.nd._internal._square_sum(rsp, axis=axis, keepdims=keepdim)
                        if axis == 1 and keepdim:
                            assert ret.stype == 'row_sparse'
                        else:
                            assert ret.stype == 'default'
                        ret_expected = mx.nd.sum(dns * dns, axis=axis, keepdims=keepdim)
                        # check forward result
                        assert_almost_equal(ret.asnumpy(), ret_expected.asnumpy())

                        rsp_data = mx.sym.Variable('data', stype='row_sparse')
                        test = mx.symbol._internal._square_sum(rsp_data, axis=axis, keepdims=keepdim)

                        # check symbolic backward since ograd can be an rsp
                        # and cannot be checked through check_numeric_gradient
                        # because it will add a loss layer as the output layer
                        # which makes ograd of the square_sum dense
                        if axis == 1 and keepdim:
                            dns_data = mx.sym.Variable('data')
                            baseline = mx.sym.sum(mx.sym.square(dns_data), axis=axis, keepdims=keepdim)
                            igrad_expected = mx.nd.empty(dns.shape)
                            baseline_exec = baseline.bind(default_context(), args=[dns],
                                                          args_grad=[igrad_expected])
                            baseline_exec.forward(is_train=True)
                            baseline_exec.backward([ret_expected])
                            # check backward when ograd is row sparse
                            check_symbolic_backward(test, [rsp], [ret_expected.tostype('row_sparse')],
                                                    [igrad_expected.asnumpy()], grad_stypes={'data': 'row_sparse'})

                            # check backward when ograd is dense
                            # the stype of output of the square_sum is deteremined in symbol binding stage.
                            # The ograd stype of the last layer is the same as the output stype of the last layer.
                            # Need to add one more layer after square_sum to trigger the kernel for ograd
                            # with default stype in square_sum op.
                            baseline1 = baseline + 1
                            baseline_exec1 = baseline1.bind(default_context(), args=[dns],
                                                            args_grad=[igrad_expected])
                            baseline_exec1.forward(is_train=True)
                            baseline_exec1.backward([ret_expected])
                            test1 = test + 1
                            check_symbolic_backward(test1, [rsp], [ret_expected], [igrad_expected.asnumpy()],
                                                    grad_stypes={'data': 'row_sparse'})

                        # check numeric gradient
                        check_numeric_gradient(test, [rsp], grad_stype_dict={'data': 'row_sparse'},
                                               atol=1e-2, rtol=0.1)

    # This test takes several minutes to run, marking as nightly.
    @attr('nightly')
    @attr('cpu')
    @attr('gpu')
    def test_sparse_storage_fallback(self):
        """ test operators which don't implement FComputeEx or FStatefulComputeEx """

        def check_broadcast_add(shape, lhs_stype, rhs_stype):
            lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
            rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
            lhs_nd = rand_ndarray(shape, lhs_stype)
            rhs_nd = rand_ndarray(shape, rhs_stype)
            lhs_dns = mx.nd.cast_storage(lhs_nd, stype='default')
            rhs_dns = mx.nd.cast_storage(rhs_nd, stype='default')

            out_dns = (lhs_dns + rhs_dns).asnumpy()
            test = mx.symbol.broadcast_add(lhs, rhs)
            location = {'lhs': lhs_nd, 'rhs': rhs_nd}
            check_symbolic_forward(test, location, [out_dns])
            check_numeric_gradient(test, location)
            check_symbolic_backward(test, location, [out_dns], [out_dns, out_dns])

        def np_softmax(x, axis=-1):
            # fix for old numpy on Travis not supporting keepdims
            x = x - np.max(x, axis=axis, keepdims=True)
            x = np.exp(x)
            x /= np.sum(x, axis=axis, keepdims=True)
            return x

        def check_softmax_with_shape(lhs_stype, rhs_stype, shape, preserve_shape=False):
            # bind with label
            ctx = default_context()
            X = mx.symbol.Variable('X', stype=lhs_stype)
            L = mx.symbol.Variable('L', stype=rhs_stype)
            Y = mx.symbol.SoftmaxOutput(data=X, label=L, preserve_shape=preserve_shape)
            x = rand_ndarray(shape, lhs_stype)
            l = rand_ndarray(shape, rhs_stype)
            l[:] = np_softmax(l.asnumpy())
            grad = mx.nd.empty(shape, ctx=ctx)
            exec1 = Y.bind(ctx, args=[x, l], args_grad={'X': grad})
            exec1.forward(is_train=True)
            out = exec1.outputs[0].asnumpy()
            assert_almost_equal(out, np_softmax(x.asnumpy()), rtol=1e-4)
            exec1.backward()
            assert_almost_equal(grad.asnumpy(), np_softmax(x.asnumpy()) - l.asnumpy(),
                                rtol=1e-3, atol=1e-4)

        def check_concat(shape, lhs_stype, rhs_stype):
            x = mx.symbol.Variable('x', stype=lhs_stype)
            w = mx.symbol.Variable('w', stype=rhs_stype)
            test = mx.sym.Concat(x, w)
            x_nd = rand_ndarray(shape, lhs_stype)
            w_nd = rand_ndarray(shape, rhs_stype)
            location = {'x': x_nd, 'w': w_nd}
            check_numeric_gradient(test, location)

        def check_operator_with_temp_resource(shape, stype):
            x = mx.symbol.Variable('x', stype=stype)
            test = mx.sym.sum(x)
            x_nd = rand_ndarray(shape, stype)
            location = {'x': x_nd}
            check_numeric_gradient(test, location)

        shape = rand_shape_2d()
        stypes = ['default', 'csr', 'row_sparse']
        for lhs in stypes:
            check_operator_with_temp_resource(shape, lhs)
            for rhs in stypes:
                check_broadcast_add(shape, lhs, rhs)
                check_concat(shape, lhs, rhs)
                check_softmax_with_shape(lhs, rhs, shape, preserve_shape=False)
                check_softmax_with_shape(rhs, rhs, shape, preserve_shape=True)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_elementwise_sum(self):
        def check_sparse_elementwise_sum_with_shape(stype, shape, n):
            # forward
            inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
            out = mx.symbol.sparse.add_n(*inputs, name='esum')
            arr = []
            arr_grad = [mx.nd.empty(shape, stype=stype) for _ in range(n)]
            densities = [0, 0.01, 0.5, 1.0]
            for i in range(n):
                arr.append(rand_ndarray(shape, stype, densities[np.random.randint(0, len(densities))]))

            exec1 = out.bind(default_context(),
                             args=arr,
                             args_grad=arr_grad)
            exec1.forward(is_train=True)
            out1 = exec1.outputs[0].asnumpy()
            out = sum(a.asnumpy() for a in arr)
            assert_almost_equal(out, out1)

            out_grad = mx.nd.empty(shape)
            out_grad[:] = np.random.uniform(-10, 10, shape)
            # backward
            exec1.backward([out_grad])
            for a in arr_grad:
                assert_almost_equal(a.asnumpy(), out_grad.asnumpy())

        for dim in range(2, 4):
            shape = tuple(np.random.randint(5, 10, size=dim))
            check_sparse_elementwise_sum_with_shape('row_sparse', shape, np.random.randint(1, 9))

    @attr('cpu')
    @attr('gpu')
    def test_sparse_embedding(self):
        ''' test sparse embedding op on cpu '''

        def check_sparse_embedding(executor, weight_ref, data_onehot, grad, density):
            # update weight based on density
            weight[:] = rand_ndarray(weight.shape, 'row_sparse', density=density)
            # check forward
            executor.forward(is_train=True)
            assert_almost_equal(executor.outputs[0].asnumpy(), np.dot(data_onehot, weight.asnumpy()))
            # check backward
            executor.backward([grad])
            assert_almost_equal(grad_map["embed_weight"].asnumpy(), np.dot(data_onehot.T, grad.asnumpy()))

        densities = [0, 0.5, 1]
        in_dim = 50
        out_dim = 3
        batch = 8
        # init executor
        data = mx.sym.Variable("data")
        weight = mx.sym.Variable("embed_weight", stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=in_dim,
                                               output_dim=out_dim, name="embed")
        grad_req = {'data': 'null', 'embed_weight': 'write'}
        exe_test = embed.simple_bind(default_context(), grad_req=grad_req, data=(batch,))
        arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
        grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
        # init data
        np_data = np.random.randint(low=0, high=in_dim, size=batch)
        np_onehot = np.zeros((batch, in_dim))
        np_onehot[np.arange(batch), np_data] = 1.0
        arg_map["data"][:] = np_data
        # init grad
        np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
        grad = mx.nd.sparse.zeros('row_sparse', np_grad.shape)
        grad[:] = np_grad
        # weight
        weight = arg_map["embed_weight"]
        for density in densities:
            check_sparse_embedding(exe_test, weight, np_onehot, grad, density)

    @attr('cpu')
    @attr('gpu')
    def test_scatter_ops(self):
        def csr_get_seen_points(name, csr_array, verbose=False):
            """Get a unique list of points int he CSR array as well as a
            corresponding parallel list of points and values"""
            seen_points = set()
            seen_point_list = list()
            values = list()
            row_count = csr_array.shape[0]
            row_pointers = csr_array.indptr.asnumpy()
            col_indexes = csr_array.indices.asnumpy()
            data = csr_array.data.asnumpy()
            for row in range(row_count):
                start_pos = row_pointers[row]
                end_pos = row_pointers[row + 1]
                for col_index in range(start_pos, end_pos):
                    col = col_indexes[col_index]
                    val = data[col_index]
                    if verbose is True:
                        print("{}: (row, col = ({}, {}) = {}".format(name, row, col, val))
                    seen_points.add((row, col))
                    seen_point_list.append((row, col))
                    values.append(val)
            return seen_points, values, seen_point_list

        def check_scatter_ops(name, shape, lhs_stype, rhs_stype, forward_mxnet_call, forward_numpy_call,
                              density=0.25, rhs_is_scalar=False, verbose=False):
            lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
            if rhs_is_scalar is False:
                rhs = mx.symbol.Variable('rhs', stype=rhs_stype)

            if verbose is True:
                print(name)

            if lhs_stype != 'default':
                lhs_nd = create_sparse_array_zd(
                    shape, lhs_stype, density=density,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=density,
                        force_indices=[(shape[0] / 2)]  # force at least one overlap
                    ))
            else:
                lhs_nd = rand_ndarray(shape, 'default')

            if rhs_is_scalar is False:
                if rhs_stype != 'default':
                    rhs_nd = create_sparse_array_zd(
                        shape, rhs_stype, density=density,
                        rsp_indices=gen_rsp_random_indices(
                            shape,
                            density=density,
                            force_indices=[(shape[0] / 2)]  # force at least one overlap
                        ))
                else:
                    rhs_nd = rand_ndarray(shape, 'default')
            else:
                rhs_nd = 9
                rhs = rhs_nd

            lhs_np = lhs_nd.asnumpy()
            rhs_np = rhs_nd if rhs_is_scalar is True else rhs_nd.asnumpy()

            if verbose is True:
                print("lhs = {}".format(lhs_np))
                print("rhs = {}".format(rhs_np))

            out_np = forward_numpy_call(lhs_np, rhs_np)

            if verbose is True:
                print("Numpy: out_np = {}".format(out_np))

            location = {'lhs': lhs_nd, 'rhs': rhs_nd}

            out = forward_mxnet_call(lhs, rhs)
            exe_test = out.bind(default_context(), args=location)
            exe_test.forward(is_train=False)
            out_nd = exe_test.outputs[0]

            if verbose is True:
                print("Sym: out_nd = {}".format(out_nd.asnumpy()))

            # For row_sparse, check that rows only exist for rows that are
            # either int lhs or rhs, and if they exist, they should equal
            # the numpy values
            if lhs_stype == 'default':
                almost_equal(out_nd.asnumpy(), out_np, equal_nan=True)
            elif lhs_stype == 'row_sparse':
                seen_rows = set()
                indices = lhs_nd.indices.asnumpy()
                for i in range(len(indices)):
                    seen_rows.add(indices[i])
                assert len(out_nd.indices.asnumpy()) == len(seen_rows)
                out_nd_np = out_nd.asnumpy()
                for row in seen_rows:
                    row_nd = out_nd_np[row]
                    row_np = out_np[row]
                    almost_equal(row_nd, row_np, equal_nan=True)
            elif lhs_stype == 'csr' and rhs_is_scalar is False:
                almost_equal(out_nd.asnumpy(), out_np, equal_nan=True)
            else:
                assert rhs_is_scalar
                lhs_seen_points, _, _ = csr_get_seen_points("lhs", lhs_nd, verbose)
                if rhs_is_scalar is False:
                    rhs_seen_points, _, _ = csr_get_seen_points("rhs", rhs_nd, verbose)
                else:
                    rhs_seen_points = set()
                input_seen_points = lhs_seen_points.union(rhs_seen_points)
                out_seen_pounts, out_values, seen_point_list = csr_get_seen_points("out_nd", out_nd, verbose)
                # Some may have been zero
                assert len(out_seen_pounts) <= len(input_seen_points)
                out_nd_np = out_nd.asnumpy()
                val_index = 0
                for row_col in seen_point_list:
                    row = row_col[0]
                    col = row_col[1]
                    val = out_values[val_index]
                    val_np = out_nd_np[row, col]
                    almost_equal(val, val_np, equal_nan=True)
                    val_index += 1

        shape = (10, 5)

        for lhs_stype in ['row_sparse', 'default', 'csr']:
            for rhs_stype in ['row_sparse', 'default', 'csr']:
                print("op: {}, lhs_stype: {}, rhs_stype: {}".format('_scatter_elemwise_div',
                                                                    lhs_stype, rhs_stype))
                check_scatter_ops('_scatter_elemwise_div', shape, lhs_stype, rhs_stype,
                                  lambda l, r: mx.sym._internal._scatter_elemwise_div(l, r),
                                  lambda l, r: l / r,
                                  verbose=False)

        for lhs_stype in ['row_sparse', 'default', 'csr']:
            print("op: {}, lhs_stype: {}".format('_scatter_plus', lhs_stype))
            check_scatter_ops('_scatter_plus', shape, lhs_stype, 'scalar',
                              lambda l, r: mx.sym._internal._scatter_plus_scalar(l, r),
                              lambda l, r: l + r,
                              rhs_is_scalar=True, verbose=False)

            print("op: {}, lhs_stype: {}".format('_scatter_minus', lhs_stype))
            check_scatter_ops('_scatter_minus', shape, lhs_stype, 'scalar',
                              lambda l, r: mx.sym._internal._scatter_minus_scalar(l, r),
                              lambda l, r: l + r,
                              rhs_is_scalar=True, verbose=False, density=0.5)


if __name__ == '__main__':
    import nose

    nose.runmodule()
