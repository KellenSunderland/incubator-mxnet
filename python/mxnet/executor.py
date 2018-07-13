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

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments
"""Symbolic Executor component of MXNet."""
from __future__ import absolute_import

from array import array as py_array
import ctypes
import copy
import numpy as np
import mxnet as mx
from .base import _LIB
from .base import mx_uint, NDArrayHandle, ExecutorHandle, SymbolHandle, py_str
from .base import check_call, c_handle_array, c_array_buf, c_str_array
from .ndarray import NDArray
from .ndarray import _ndarray_cls

# those functions are not used here, we just import them to keep backward compatibility
# in case the end user calls them, as they originally lives here
# pylint: disable=unused-import
from .executor_manager import _split_input_slice, _check_arguments, _load_data, _load_label

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

class Executor(object):
    """Executor is the object providing efficient symbolic graph execution and optimization.

    Examples
    --------
    >>> # typical approach to create an executor is to bind symbol
    >>> a = mx.sym.Variable('a')
    >>> b = mx.sym.Variable('b')
    >>> c = 2 * a + b
    >>> texec = c.bind(mx.cpu(), {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])})
    """
    def __init__(self, handle, symbol, ctx, grad_req, group2ctx):
        """Constructor, used Symbol.bind and Symbol.simple_bind instead.

        Parameters
        ----------
        handle: ExecutorHandle
            ExecutorHandle generated by calling `bind`.

        See Also
        --------
        Symbol.bind : to create executor.
        """
        if not isinstance(handle, ExecutorHandle):
            raise TypeError("Handle type error")
        self.handle = handle
        self.arg_arrays = []
        self.grad_arrays = []
        self.aux_arrays = []
        self.outputs = self._get_outputs()
        self._symbol = copy.deepcopy(symbol)
        self._optimized_symbol = None
        self._arg_dict = None
        self._grad_dict = None
        self._aux_dict = None
        self._output_dict = None
        self._monitor_callback = None
        self._ctx = copy.deepcopy(ctx)
        self._grad_req = copy.deepcopy(grad_req)
        self._group2ctx = copy.deepcopy(group2ctx)

    def __del__(self):
        check_call(_LIB.MXExecutorFree(self.handle))

    @staticmethod
    def _get_dict(names, ndarrays):
        """Get the dictionary given name and ndarray pairs."""
        nset = set()
        for nm in names:
            if nm in nset:
                raise ValueError('Duplicate names detected, %s' % str(names))
            nset.add(nm)
        return dict(zip(names, ndarrays))

    def _get_outputs(self):
        """List all the output NDArray.

        Returns
        -------
        A list of ndarray bound to the heads of executor.
        """
        out_size = mx_uint()
        handles = ctypes.POINTER(NDArrayHandle)()
        check_call(_LIB.MXExecutorOutputs(self.handle,
                                          ctypes.byref(out_size), ctypes.byref(handles)))
        num_output = out_size.value
        outputs = [_ndarray_cls(NDArrayHandle(handles[i])) for i in range(num_output)]
        return outputs

    def forward(self, is_train=False, **kwargs):
        """Calculate the outputs specified by the bound symbol.

        Parameters
        ----------
        is_train: bool, optional
            Whether this forward is for evaluation purpose. If True,
            a backward call is expected to follow.

        **kwargs
            Additional specification of input arguments.

        Examples
        --------
        >>> # doing forward by specifying data
        >>> texec.forward(is_train=True, data=mydata)
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> mydata.copyto(texec.arg_dict['data'])
        >>> texec.forward(is_train=True)
        >>> # doing forward by specifying data and get outputs
        >>> outputs = texec.forward(is_train=True, data=mydata)
        >>> print(outputs[0].asnumpy())
        """
        if len(kwargs) != 0:
            arg_dict = self.arg_dict
            for name, array in kwargs.items():
                if not isinstance(array, (NDArray, np.ndarray)):
                    raise ValueError('only accept keyword argument of NDArrays and numpy.ndarray')
                if name not in arg_dict:
                    raise TypeError('Unknown argument %s' % name)
                if arg_dict[name].shape != array.shape:
                    raise ValueError('Shape not match! Argument %s, need: %s, received: %s'
                                     %(name, str(arg_dict[name].shape), str(array.shape)))
                arg_dict[name][:] = array

        check_call(_LIB.MXExecutorForward(
            self.handle,
            ctypes.c_int(int(is_train))))

        return self.outputs

    def backward(self, out_grads=None, is_train=True):
        """Do backward pass to get the gradient of arguments.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray or dict of str to NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        is_train : bool, default True
            Whether this backward is for training or inference. Note that in rare
            cases you want to call backward with is_train=False to get gradient
            during inference.


        Examples
        --------
        >>> # Example for binding on loss function symbol, which gives the loss value of the model.
        >>> # Equivalently it gives the head gradient for backward pass.
        >>> # In this example the built-in SoftmaxOutput is used as loss function.
        >>> # MakeLoss can be used to define customized loss function symbol.
        >>> net = mx.sym.Variable('data')
        >>> net = mx.sym.FullyConnected(net, name='fc', num_hidden=6)
        >>> net = mx.sym.Activation(net, name='relu', act_type="relu")
        >>> net = mx.sym.SoftmaxOutput(net, name='softmax')

        >>> args =  {'data': mx.nd.ones((1, 4)), 'fc_weight': mx.nd.ones((6, 4)),
        >>>          'fc_bias': mx.nd.array((1, 4, 4, 4, 5, 6)), 'softmax_label': mx.nd.ones((1))}
        >>> args_grad = {'fc_weight': mx.nd.zeros((6, 4)), 'fc_bias': mx.nd.zeros((6))}
        >>> texec = net.bind(ctx=mx.cpu(), args=args, args_grad=args_grad)
        >>> out = texec.forward(is_train=True)[0].copy()
        >>> print out.asnumpy()
        [[ 0.00378404  0.07600445  0.07600445  0.07600445  0.20660152  0.5616011 ]]
        >>> texec.backward()
        >>> print(texec.grad_arrays[1].asnumpy())
        [[ 0.00378404  0.00378404  0.00378404  0.00378404]
         [-0.92399555 -0.92399555 -0.92399555 -0.92399555]
         [ 0.07600445  0.07600445  0.07600445  0.07600445]
         [ 0.07600445  0.07600445  0.07600445  0.07600445]
         [ 0.20660152  0.20660152  0.20660152  0.20660152]
         [ 0.5616011   0.5616011   0.5616011   0.5616011 ]]
        >>>
        >>> # Example for binding on non-loss function symbol.
        >>> # Here the binding symbol is neither built-in loss function
        >>> # nor customized loss created by MakeLoss.
        >>> # As a result the head gradient is not automatically provided.
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> # c is not a loss function symbol
        >>> c = 2 * a + b
        >>> args = {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])}
        >>> args_grad = {'a': mx.nd.zeros((2)), 'b': mx.nd.zeros((2))}
        >>> texec = c.bind(ctx=mx.cpu(), args=args, args_grad=args_grad)
        >>> out = texec.forward(is_train=True)[0].copy()
        >>> print(out.asnumpy())
        [ 4.  7.]
        >>> # out_grads is the head gradient in backward pass.
        >>> # Here we define 'c' as loss function.
        >>> # Then 'out' is passed as head gradient of backward pass.
        >>> texec.backward(out)
        >>> print(texec.grad_arrays[0].asnumpy())
        [ 8.  14.]
        >>> print(texec.grad_arrays[1].asnumpy())
        [ 4.  7.]
        """
        if out_grads is None:
            out_grads = []
        elif isinstance(out_grads, NDArray):
            out_grads = [out_grads]
        elif isinstance(out_grads, dict):
            out_grads = [out_grads[k] for k in self._symbol.list_outputs()]

        for obj in out_grads:
            if not isinstance(obj, NDArray):
                raise TypeError("inputs must be NDArray")
        ndarray = c_handle_array(out_grads)
        check_call(_LIB.MXExecutorBackwardEx(
            self.handle,
            mx_uint(len(out_grads)),
            ndarray,
            ctypes.c_int(is_train)))

    def set_monitor_callback(self, callback):
        """Install callback for monitor.

        Parameters
        ----------
        callback : function
            Takes a string and an NDArrayHandle.

        Examples
        --------
        >>> def mon_callback(*args, **kwargs):
        >>>     print("Do your stuff here.")
        >>>
        >>> texe.set_monitor_callback(mon_callback)
        """
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
        self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
        check_call(_LIB.MXExecutorSetMonitorCallback(
            self.handle,
            self._monitor_callback,
            None))

    @property
    def arg_dict(self):
        """Get dictionary representation of argument arrrays.

        Returns
        -------
        arg_dict : dict of str to NDArray
            The dictionary that maps the names of arguments to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
        """
        if self._arg_dict is None:
            self._arg_dict = Executor._get_dict(
                self._symbol.list_arguments(), self.arg_arrays)
        return self._arg_dict

    @property
    def grad_dict(self):
        """Get dictionary representation of gradient arrays.

        Returns
        -------
        grad_dict : dict of str to NDArray
            The dictionary that maps name of arguments to gradient arrays.
        """
        if self._grad_dict is None:
            self._grad_dict = Executor._get_dict(
                self._symbol.list_arguments(), self.grad_arrays)
        return self._grad_dict

    @property
    def aux_dict(self):
        """Get dictionary representation of auxiliary states arrays.

        Returns
        -------
        aux_dict : dict of str to NDArray
            The dictionary that maps name of auxiliary states to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the auxiliary states.
        """
        if self._aux_dict is None:
            self._aux_dict = Executor._get_dict(
                self._symbol.list_auxiliary_states(), self.aux_arrays)
        return self._aux_dict

    @property
    def output_dict(self):
        """Get dictionary representation of output arrays.

        Returns
        -------
        output_dict : dict of str to NDArray
            The dictionary that maps name of output names to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the outputs.
        """
        if self._output_dict is None:
            self._output_dict = Executor._get_dict(
                self._symbol.list_outputs(), self.outputs)
        return self._output_dict

    @property
    def optimized_symbol(self):
        """Get optimized symbol.

        Returns
        -------
        symbol : nnvm::Symbol
            The nnvm symbol optimized.
        """
        if self._optimized_symbol is None:
            handle = SymbolHandle()
            check_call(_LIB.MXExecutorGetOptimizedSymbol(self.handle, ctypes.byref(handle)))
        return mx.sym.Symbol(handle=handle)

    def copy_params_from(self, arg_params, aux_params=None, allow_extra_params=False):
        """Copy parameters from arg_params, aux_params into executor's internal array.

        Parameters
        ----------
        arg_params : dict of str to NDArray
            Parameters, dict of name to NDArray of arguments.

        aux_params : dict of str to NDArray, optional
            Parameters, dict of name to NDArray of auxiliary states.

        allow_extra_params : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Raises
        ------
        ValueError
            If there is additional parameters in the dict but ``allow_extra_params=False``.

        Examples
        --------
        >>> # set parameters with existing model checkpoint
        >>> model_prefix = 'mx_mlp'
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
        >>> texec.copy_params_from(arg_params, aux_params)
        """
        for name, array in arg_params.items():
            if name in self.arg_dict:
                dst = self.arg_dict[name]
                array.astype(dst.dtype).copyto(dst)
            elif not allow_extra_params:
                raise ValueError('Find name \"%s\" that is not in the arguments' % name)

        if aux_params is None:
            return

        for name, array in aux_params.items():
            if name in self.aux_dict:
                dst = self.aux_dict[name]
                array.astype(dst.dtype).copyto(dst)
            elif not allow_extra_params:
                raise ValueError('Find name %s that is not in the auxiliary states' % name)

    def reshape(self, partial_shaping=False, allow_up_sizing=False, **kwargs):
        """Return a new executor with the same symbol and shared memory,
        but different input/output shapes.
        For runtime reshaping, variable length sequences, etc.
        The returned executor shares state with the current one,
        and cannot be used in parallel with it.

        Parameters
        ----------
        partial_shaping : bool
            Whether to allow changing the shape of unspecified arguments.
        allow_up_sizing : bool
            Whether to allow allocating new ndarrays that's larger than the original.
        kwargs : dict of string to tuple of int
            New shape for arguments.

        Returns
        -------
        exec : Executor
            A new executor that shares memory with self.

        Examples
        --------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> c = 2 * a + b
        >>> texec = c.bind(mx.cpu(), {'a': mx.nd.zeros((2, 1)), 'b': mx.nd.ones((2,1))})
        >>> new_shape = {'a': (4, 2), 'b': (4, 2)}
        >>> texec.reshape(allow_up_sizing=True, **new_shape)
        """
        # pylint: disable=too-many-branches
        provided_arg_shape_data = []  # shape data
        # argument shape index in sdata,
        # e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
        provided_arg_shape_idx = [0]
        provided_arg_shape_names = []  # provided argument names
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                provided_arg_shape_names.append(k)
                provided_arg_shape_data.extend(v)
                provided_arg_shape_idx.append(len(provided_arg_shape_data))

        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []

        if self._group2ctx:
            for key, val in self._group2ctx.items():
                ctx_map_keys.append(key)
                ctx_map_dev_types.append(val.device_typeid)
                ctx_map_dev_ids.append(val.device_id)

        handle = ExecutorHandle()
        shared_handle = self.handle

        num_in_args = ctypes.c_uint()
        in_arg_handles = ctypes.POINTER(NDArrayHandle)()
        arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
        num_aux_states = ctypes.c_uint()
        aux_state_handles = ctypes.POINTER(NDArrayHandle)()

        check_call(_LIB.MXExecutorReshape(ctypes.c_int(int(partial_shaping)),
                                          ctypes.c_int(int(allow_up_sizing)),
                                          ctypes.c_int(self._ctx.device_typeid),
                                          ctypes.c_int(self._ctx.device_id),
                                          mx_uint(len(ctx_map_keys)),
                                          c_str_array(ctx_map_keys),
                                          c_array_buf(ctypes.c_int,
                                                      py_array('i', ctx_map_dev_types)),
                                          c_array_buf(ctypes.c_int,
                                                      py_array('i', ctx_map_dev_ids)),
                                          mx_uint(len(provided_arg_shape_names)),
                                          c_str_array(provided_arg_shape_names),
                                          c_array_buf(mx_uint,
                                                      py_array('I', provided_arg_shape_data)),
                                          c_array_buf(mx_uint,
                                                      py_array('I', provided_arg_shape_idx)),
                                          ctypes.byref(num_in_args),
                                          ctypes.byref(in_arg_handles),
                                          ctypes.byref(arg_grad_handles),
                                          ctypes.byref(num_aux_states),
                                          ctypes.byref(aux_state_handles),
                                          shared_handle,
                                          ctypes.byref(handle)))

        arg_arrays = [_ndarray_cls(NDArrayHandle(in_arg_handles[i]))
                      for i in range(num_in_args.value)]
        grad_arrays = [_ndarray_cls(NDArrayHandle(arg_grad_handles[i]))
                       if arg_grad_handles[i] is not None
                       else None for i in range(num_in_args.value)]
        aux_arrays = [_ndarray_cls(NDArrayHandle(aux_state_handles[i]))
                      for i in range(num_aux_states.value)]

        executor = Executor(handle, self._symbol, self._ctx, self._grad_req, self._group2ctx)
        executor.arg_arrays = arg_arrays
        executor.grad_arrays = grad_arrays
        executor.aux_arrays = aux_arrays
        return executor

    def debug_str(self):
        """Get a debug string about internal execution plan.

        Returns
        -------
        debug_str : string
            Debug string of the executor.

        Examples
        --------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.sin(a)
        >>> c = 2 * a + b
        >>> texec = c.bind(mx.cpu(), {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])})
        >>> print(texec.debug_str())
        Symbol Outputs:
	            output[0]=_plus0(0)
        Variable:a
        --------------------
        Op:_mul_scalar, Name=_mulscalar0
        Inputs:
	        arg[0]=a(0) version=0
        Attrs:
	        scalar=2
        --------------------
        Op:sin, Name=sin0
        Inputs:
	        arg[0]=a(0) version=0
        --------------------
        Op:elemwise_add, Name=_plus0
        Inputs:
	        arg[0]=_mulscalar0(0)
	        arg[1]=sin0(0)
        Total 0 MB allocated
        Total 11 TempSpace resource requested
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXExecutorPrint(
            self.handle, ctypes.byref(debug_str)))
        return py_str(debug_str.value)
