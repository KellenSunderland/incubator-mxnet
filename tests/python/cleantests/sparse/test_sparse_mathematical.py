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
from sparse_common import *
import random
import warnings
from nose.plugins.attrib import attr


class TestSparseMathematical:

    @staticmethod
    def check_sparse_mathematical_core(name, stype,
                                       forward_mxnet_call, forward_numpy_call, backward_numpy_call=None,
                                       rhs_arg=None, data_init=9., grad_init=2., output_grad_stype=None,
                                       input_grad_stype=None, force_overlap=False, density=.5,
                                       ograd_density=.5, verbose=False, shuffle_csr_indices=True):
        if verbose is True:
            print("TESTING: " + name)

        data = mx.symbol.Variable('data', stype=stype)

        temp_input_grad_stype = input_grad_stype

        if temp_input_grad_stype is None:
            temp_input_grad_stype = stype

        if rhs_arg is not None:
            if is_scalar(rhs_arg):
                expected_result_type, expected_grad_result_type = \
                    get_fw_bw_result_types_with_scalar(forward_numpy_call, stype,
                                                                              backward_numpy_call,
                                                                              temp_input_grad_stype)
            else:
                expected_result_type, expected_grad_result_type = \
                    get_fw_bw_result_types_2(forward_numpy_call, stype,
                                                  backward_numpy_call, temp_input_grad_stype)
        else:
            expected_result_type, expected_grad_result_type = \
                get_fw_bw_result_types(forward_numpy_call, stype,
                                            backward_numpy_call, temp_input_grad_stype)

        if input_grad_stype is not None and input_grad_stype != expected_grad_result_type:
            print("{}: explicit override of deduced input grad type '{}' with '{}'".format(
                name, expected_grad_result_type, input_grad_stype))
            expected_grad_result_type = input_grad_stype

        shape = rand_shape_2d()

        if verbose is True:
            print("Shape: ", shape, "density: ", density, "force_overlap", force_overlap)

        if stype == 'default':
            data_tmp = np.zeros(shape)
            if abs(density) >= 1e-4:
                data_tmp[:] = data_init
            arr_data = mx.nd.array(data_tmp)
        else:
            arr_data = create_sparse_array_zd(
                shape, stype, density=density,
                data_init=data_init,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    force_indices=[(shape[0] / 2)] if force_overlap is True else None
                )
            )
            data_tmp = arr_data.asnumpy()
            if verbose is True:
                print("arr_data indices", arr_data.indices.asnumpy())

        if verbose is True:
            print("input", data_tmp)

        if backward_numpy_call is None:
            arr_grad = None
        elif expected_grad_result_type == 'default':
            if abs(density) < 1e-4:
                arr_grad = mx.nd.zeros(shape)
            else:
                arr_grad = mx.nd.ones(shape)
        else:
            arr_grad = create_sparse_array_zd(
                shape,
                expected_grad_result_type,
                density=density,
                data_init=1,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    force_indices=[(shape[0] / 2)] if force_overlap is True else None
                )
            )

        if rhs_arg is not None:
            test = forward_mxnet_call(data, rhs_arg)
        else:
            test = forward_mxnet_call(data)

        args = list()
        args.append(arr_data)

        if arr_grad is not None:
            exe_test = test.bind(default_context(), args=args, args_grad=[arr_grad])
        else:
            exe_test = test.bind(default_context(), args=args)

        exe_test.forward(is_train=True)
        assert exe_test.outputs[0].stype == expected_result_type
        out = exe_test.outputs[0].asnumpy()

        if rhs_arg is not None:
            npout = forward_numpy_call(data_tmp, rhs_arg)
        else:
            npout = forward_numpy_call(data_tmp)

        if verbose is True:
            print("out", out)
            print("npout", npout)

        assert_almost_equal(out, npout, equal_nan=True)

        if backward_numpy_call is not None:
            if output_grad_stype == 'default' or output_grad_stype is None:
                out_grad = mx.nd.empty(shape)
                out_grad[:] = grad_init
            else:
                out_grad = create_sparse_array_zd(
                    shape, output_grad_stype,
                    density=density,
                    data_init=grad_init,
                    shuffle_csr_indices=shuffle_csr_indices,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=ograd_density,
                        force_indices=[(shape[0] / 2)] if force_overlap is True else None))

            npout_grad = out_grad.asnumpy()

            if verbose is True:
                print("npout_grad", npout_grad)

            if rhs_arg is not None:
                temp = backward_numpy_call(data_tmp, rhs_arg)
            else:
                temp = backward_numpy_call(data_tmp)
            input_grad = npout_grad * temp

            if verbose is True:
                print(arr_grad.asnumpy())
            exe_test.backward(out_grad)
            if verbose is True:
                print(arr_grad.asnumpy())

            assert arr_grad.stype == expected_grad_result_type

            arr_grad = arr_grad.asnumpy()

            if verbose is True:
                print(name)
                print("arr_grad", arr_grad)
                print("input_grad", input_grad)

            assert_almost_equal(arr_grad, input_grad, equal_nan=True)

    @attr('cpu')
    @attr('gpu')
    def test_sparse_mathematical_core(self):
        def util_sign(a):
            if np.isclose(a, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
                return 0
            elif np.isclose(a, 0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
                return 0
            elif a < 0.0:
                return -1
            else:  # a > 0.0:
                return 1

        # Check scalar binary operators
        def check_binary_op_with_scalar(stype,
                                        output_grad_stype=None,
                                        input_grad_stype=None,
                                        density=.5, ograd_density=.5,
                                        force_overlap=False, ):
            # mul_scalar
            TestSparseMathematical.check_sparse_mathematical_core("mul_scalar", stype,
                                           lambda x, y: x * y,
                                           lambda x, y: x * y,
                                           lambda input, rhs: rhs,
                                           rhs_arg=5.0,
                                           data_init=2, grad_init=3,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           density=density, ograd_density=ograd_density,
                                           force_overlap=force_overlap,
                                           verbose=False)

            # plus_scalar
            TestSparseMathematical.check_sparse_mathematical_core("plus_scalar", stype,
                                           lambda x, y: x + y,
                                           lambda x, y: x + y,
                                           lambda input, rhs: 1,
                                           rhs_arg=5.0,
                                           data_init=2, grad_init=3,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           density=density, ograd_density=ograd_density,
                                           force_overlap=force_overlap,
                                           verbose=False)

            # minus_scalar
            TestSparseMathematical.check_sparse_mathematical_core("minus_scalar", stype,
                                           lambda x, y: x - y,
                                           lambda x, y: x - y,
                                           lambda input, rhs: 1,
                                           rhs_arg=5.0,
                                           data_init=2, grad_init=3,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           density=density, ograd_density=ograd_density,
                                           force_overlap=force_overlap,
                                           verbose=False)

        # Check many basic unary operators
        def check_mathematical_core(stype, output_grad_stype=None,
                                    input_grad_stype=None, force_overlap=False,
                                    density=.5, ograd_density=.5):

            # negative
            TestSparseMathematical.check_sparse_mathematical_core("negative", stype,
                                           lambda x: mx.sym.sparse.negative(x),
                                           lambda x: np.negative(x),
                                           force_overlap=force_overlap,
                                           density=density,
                                           input_grad_stype=input_grad_stype,
                                           ograd_density=ograd_density)

            # square
            TestSparseMathematical.check_sparse_mathematical_core("square", stype,
                                           lambda x: mx.sym.sparse.square(x),
                                           lambda x: np.square(x),
                                           lambda x: 2 * x,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density,
                                           verbose=False)

            if stype != "csr":
                # sqrt
                TestSparseMathematical.check_sparse_mathematical_core("sqrt", stype,
                                               lambda x: mx.sym.sparse.sqrt(x),
                                               lambda x: np.sqrt(x),
                                               lambda x: 1.0 / (2.0 * np.sqrt(x)),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density,
                                               verbose=False)

                # rsqrt
                TestSparseMathematical.check_sparse_mathematical_core("rsqrt", stype,
                                               lambda x: mx.sym.sparse.rsqrt(x),
                                               lambda x: 1 / np.sqrt(x),
                                               lambda x: -(1.0 / (2.0 * x * np.sqrt(x))),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # tan
                TestSparseMathematical.check_sparse_mathematical_core("tan", stype,
                                               lambda x: mx.sym.sparse.tan(x),
                                               lambda x: np.tan(x),
                                               lambda x: np.tan(x) ** 2 + 1,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               density=density,
                                               ograd_density=ograd_density)

                # abs
                TestSparseMathematical.check_sparse_mathematical_core("abs", stype,
                                               lambda x: mx.sym.sparse.abs(x),
                                               lambda x: np.abs(x),
                                               lambda x: assign_each(x, function=util_sign),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # floor
                TestSparseMathematical.check_sparse_mathematical_core("floor", stype, lambda x: mx.sym.sparse.floor(x),
                                               lambda x: np.floor(x),
                                               force_overlap=force_overlap,
                                               input_grad_stype=input_grad_stype,
                                               density=density, ograd_density=ograd_density)

                # ceil
                TestSparseMathematical.check_sparse_mathematical_core("ceil", stype,
                                               lambda x: mx.sym.sparse.ceil(x),
                                               lambda x: np.ceil(x),
                                               force_overlap=force_overlap,
                                               input_grad_stype=input_grad_stype,
                                               density=density, ograd_density=ograd_density)

                # sign
                TestSparseMathematical.check_sparse_mathematical_core("sign", stype,
                                               lambda x: mx.sym.sparse.sign(x),
                                               lambda x: np.sign(x),
                                               lambda x: np.zeros(x.shape),
                                               output_grad_stype=output_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # cos
                TestSparseMathematical.check_sparse_mathematical_core("cos", stype,
                                               lambda x: mx.sym.sparse.cos(x),
                                               lambda x: np.cos(x),
                                               lambda x: -np.sin(x),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # sin
                TestSparseMathematical.check_sparse_mathematical_core("sin", stype,
                                               lambda x: mx.sym.sparse.sin(x),
                                               lambda x: np.sin(x),
                                               lambda x: np.cos(x),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # arcsin
                TestSparseMathematical.check_sparse_mathematical_core("arcsin", stype,
                                               lambda x: mx.sym.sparse.arcsin(x),
                                               lambda x: np.arcsin(x),
                                               lambda x: 1. / (1. - x ** 2) ** (1. / 2.),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # arccos
                TestSparseMathematical.check_sparse_mathematical_core("arccos", stype,
                                               lambda x: mx.sym.sparse.arccos(x),
                                               lambda x: np.arccos(x),
                                               lambda x: -1. / (1. - x ** 2.) ** (1. / 2.),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # arctan
                TestSparseMathematical.check_sparse_mathematical_core("arctan", stype,
                                               lambda x: mx.sym.sparse.arctan(x),
                                               lambda x: np.arctan(x),
                                               lambda x: 1. / (x ** 2. + 1.),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # degrees
                TestSparseMathematical.check_sparse_mathematical_core("degrees", stype,
                                               lambda x: mx.sym.sparse.degrees(x),
                                               lambda x: np.degrees(x),
                                               lambda x: assign_each(x, lambda a: 180. / np.pi),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # radians
                TestSparseMathematical.check_sparse_mathematical_core("radians", stype,
                                               lambda x: mx.sym.sparse.radians(x),
                                               lambda x: np.radians(x),
                                               lambda x: assign_each(x, lambda a: np.pi / 180.),
                                               data_init=0.6, grad_init=1,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # sinh
                TestSparseMathematical.check_sparse_mathematical_core("sinh", stype,
                                               lambda x: mx.sym.sparse.sinh(x),
                                               lambda x: np.sinh(x),
                                               lambda x: np.cosh(x),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # cosh
                TestSparseMathematical.check_sparse_mathematical_core("cosh", stype,
                                               lambda x: mx.sym.sparse.cosh(x),
                                               lambda x: np.cosh(x),
                                               lambda x: np.sinh(x),
                                               data_init=5, grad_init=5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

                # tanh
                TestSparseMathematical.check_sparse_mathematical_core("tanh", stype,
                                               lambda x: mx.sym.sparse.tanh(x),
                                               lambda x: np.tanh(x),
                                               lambda x: 1. - np.tanh(x) ** 2,
                                               data_init=0.5, grad_init=1,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # arcsinh
                TestSparseMathematical.check_sparse_mathematical_core("arcsinh", stype,
                                               lambda x: mx.sym.sparse.arcsinh(x),
                                               lambda x: np.arcsinh(x),
                                               lambda x: 1. / (x ** 2 + 1.) ** (1. / 2.),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # arccosh
                TestSparseMathematical.check_sparse_mathematical_core("arccosh", stype,
                                               lambda x: mx.sym.sparse.arccosh(x),
                                               lambda x: np.arccosh(x),
                                               lambda x: 1. / (x ** 2 - 1.) ** (1. / 2.),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # arctanh
                TestSparseMathematical.check_sparse_mathematical_core("arctanh", stype,
                                               lambda x: mx.sym.sparse.arctanh(x),
                                               lambda x: np.arctanh(x),
                                               lambda x: -1. / (x ** 2 - 1.),
                                               data_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # log1p
                TestSparseMathematical.check_sparse_mathematical_core("log1p", stype,
                                               lambda x: mx.sym.sparse.log1p(x),
                                               lambda x: np.log1p(x),
                                               lambda x: 1. / (1.0 + x),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # expm1
                TestSparseMathematical.check_sparse_mathematical_core("expm1", stype,
                                               lambda x: mx.sym.sparse.expm1(x),
                                               lambda x: np.expm1(x),
                                               lambda x: np.exp(x),
                                               data_init=0.5, grad_init=0.5,
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # log10
                TestSparseMathematical.check_sparse_mathematical_core("log10", stype,
                                               lambda x: mx.sym.sparse.log10(x),
                                               lambda x: np.log10(x),
                                               lambda x: 1. / (x * np.log(10.)),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # log2
                TestSparseMathematical.check_sparse_mathematical_core("log2", stype,
                                               lambda x: mx.sym.sparse.log2(x),
                                               lambda x: np.log2(x),
                                               lambda x: 1. / (x * np.log(2.)),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap, density=density,
                                               ograd_density=ograd_density)

                # rint
                TestSparseMathematical.check_sparse_mathematical_core("rint", stype,
                                               lambda x: mx.sym.sparse.rint(x),
                                               lambda x: np.rint(x),
                                               force_overlap=force_overlap, density=density,
                                               input_grad_stype=input_grad_stype,
                                               ograd_density=ograd_density)

                # fix
                TestSparseMathematical.check_sparse_mathematical_core("fix", stype,
                                               lambda x: mx.sym.sparse.fix(x),
                                               lambda x: np.fix(x),
                                               force_overlap=force_overlap, density=density,
                                               input_grad_stype=input_grad_stype,
                                               ograd_density=ograd_density)

                try:
                    from scipy import special as scipy_special
                    import_succeeded = True
                    # gamma
                    TestSparseMathematical.check_sparse_mathematical_core("gamma", stype,
                                                   lambda x: mx.sym.sparse.gamma(x),
                                                   lambda x: scipy_special.gamma(x),
                                                   lambda x: scipy_special.gamma(x) * scipy_special.psi(x),
                                                   output_grad_stype=output_grad_stype,
                                                   input_grad_stype=input_grad_stype,
                                                   force_overlap=force_overlap,
                                                   density=density, ograd_density=ograd_density)
                    # gammaln
                    TestSparseMathematical.check_sparse_mathematical_core("gammaln", stype,
                                                   lambda x: mx.sym.sparse.gammaln(x),
                                                   lambda x: scipy_special.gammaln(x),
                                                   lambda x: scipy_special.psi(x),
                                                   output_grad_stype=output_grad_stype,
                                                   input_grad_stype=input_grad_stype,
                                                   force_overlap=force_overlap,
                                                   density=density, ograd_density=ograd_density)

                except:
                    if import_succeeded == False:
                        print("Could not import scipy. Skipping unit tests for special functions")
                    else:
                        raise

        for i in range(1):
            print("pass", i)
            for density in [0.0, random.uniform(0, 1), 1.0]:
                for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                    for force_overlap in [False, True]:
                        print("{}, {}, {}".format(density, ograd_density, force_overlap))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # Check unary ops (unary fwd, binary bwd)
                            check_mathematical_core('default', force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)
                            check_mathematical_core('row_sparse', force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)
                            check_mathematical_core('row_sparse', output_grad_stype='default',
                                                    force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)
                            check_mathematical_core('row_sparse', output_grad_stype='row_sparse',
                                                    force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)
                            check_mathematical_core('csr', output_grad_stype='default',
                                                    force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)
                            check_mathematical_core('csr', output_grad_stype='csr',
                                                    force_overlap=force_overlap,
                                                    density=density, ograd_density=ograd_density)

                            # Check binary with scalar ops
                            check_binary_op_with_scalar('default',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('row_sparse',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('row_sparse', output_grad_stype='default',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('row_sparse',
                                                        output_grad_stype='row_sparse',
                                                        density=density, ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('csr',
                                                        output_grad_stype='csr',
                                                        input_grad_stype='default',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('csr',
                                                        output_grad_stype='csr',
                                                        input_grad_stype='csr',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)
                            check_binary_op_with_scalar('csr',
                                                        output_grad_stype='default',
                                                        density=density,
                                                        ograd_density=ograd_density,
                                                        force_overlap=force_overlap)


if __name__ == '__main__':
    import nose

    nose.runmodule()
