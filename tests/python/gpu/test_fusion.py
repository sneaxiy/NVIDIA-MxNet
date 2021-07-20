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

import sys
import os
import random
import itertools
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, teardown, with_seed

def check_fused_symbol(sym, **kwargs):
    inputs = sym.list_inputs()
    shapes = {inp : kwargs[inp].shape for inp in inputs}
    ctx = kwargs.get('ctx', mx.gpu(0))
    # Double identity so that there is always something to fuse
    test_sym = mx.sym.Group([mx.sym.identity(mx.sym.identity(s)) for s in sym])
    rtol = {'float16' : 1e-2,
            'float32' : 1.5e-6,
            'float64' : 1.5e-6,
            }
    atol = {'float16' : 1e-3,
            'float32' : 1e-7,
            'float64' : 1e-7,
            }
    for dtype in ['float16', 'float32', 'float64']:
        data = {inp : kwargs[inp].astype(dtype) for inp in inputs}
        for grad_req in ['write', 'add']:
            type_dict = {inp : dtype for inp in inputs}
            with environment('MXNET_USE_FUSION', '0'):
                orig_exec = test_sym.simple_bind(ctx=ctx, grad_req=grad_req, type_dict=type_dict, **shapes)
            with environment('MXNET_USE_FUSION', '1'):
                fused_exec = test_sym.simple_bind(ctx=ctx, grad_req=grad_req, type_dict=type_dict, **shapes)
            fwd_orig = orig_exec.forward(is_train=True, **data)
            out_grads = [mx.nd.ones_like(arr) for arr in fwd_orig]
            orig_exec.backward(out_grads=out_grads)
            fwd_fused = fused_exec.forward(is_train=True, **data)
            fused_exec.backward(out_grads=out_grads)
            for orig, fused in zip(fwd_orig, fwd_fused):
                assert_allclose(orig, fused, rtol=rtol[dtype], atol=atol[dtype])
            for orig, fused in zip(orig_exec.grad_arrays, fused_exec.grad_arrays):
                if orig is None and fused is None:
                    continue
                assert orig is not None
                assert fused is not None
                assert_allclose(orig, fused, rtol=rtol[dtype], atol=atol[dtype])

def announce_check(op_name):
    print("Checking fusion of " + op_name)

def check_unary_ops():
    unary_ops = [
            'relu',
            'sigmoid',
            'softsign',
            'exp',
            'expm1',
            'log',
            'log10',
            'log2',
            'log1p',
            'degrees',
            'radians',
            'sin',
            'cos',
            'tan',
            'arcsin',
            'arccos',
            'arctan',
            'sinh',
            'cosh',
            'tanh',
            'arcsinh',
            'arctanh',
            'sqrt',
            'rsqrt',
            'cbrt',
            'rcbrt',
            'square',
            'squeeze',
            'zeros_like',
            'ones_like',
            'flatten',
            'round',
            'rint',
            'fix',
            'floor',
            'ceil',
            'trunc',
            'sign',
            'reciprocal',
            'abs',
            'gamma',
            'gammaln',
            'erf',
            'negative',
            'logical_not',
            ]

    arr = mx.random.uniform(shape=rand_shape_2d())
    a = mx.sym.Variable('a')
    for op_name in unary_ops:
        announce_check(op_name)
        op = getattr(mx.sym, op_name)
        sym = op(a)
        check_fused_symbol(sym, a=arr)

    # unary ops requiring special treatment

    # arccosh needs input to be >= 1
    arr2 = arr + 1
    announce_check('arccosh')
    check_fused_symbol(mx.sym.arccosh(a), a=arr2)

    # erfinv needs -1 < input < 1, but we avoid the limits of this range where the slope nears +inf.
    arr2 = (arr - 0.5) * 1.99
    announce_check('erfinv')
    check_fused_symbol(mx.sym.erfinv(a), a=arr2)

    # Activation requires act_type attribute
    for act_type in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
        announce_check("Activation(act_type='{}')".format(act_type))
        check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=arr)
        if act_type == 'softrelu':
            # Check that softrelu implementation doesn't overflow on large inputs
            check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=1000 * arr)

    # Cast requires dtype
    for dtype in ['float16', 'float32', 'float64', 'int32']:
        announce_check("Cast(dtype='{}')".format(dtype))
        check_fused_symbol(mx.sym.Cast(a, dtype=dtype), a=arr)

    # reshape requires shape
    announce_check('reshape')
    check_fused_symbol(mx.sym.reshape(a, shape=(-1,)), a=arr)

    # expand_dims requires axis
    announce_check('expand_dims')
    check_fused_symbol(mx.sym.expand_dims(a, axis=1), a=arr)

    # clip requires a_min, a_max
    announce_check('clip')
    check_fused_symbol(mx.sym.clip(a, a_min=0.3, a_max=0.7), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=-np.inf, a_max=0.7), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=-np.inf, a_max=np.inf), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=0, a_max=np.nan), a=arr)

    # smooth_l1 requires a scalar
    announce_check('smooth_l1')
    check_fused_symbol(mx.sym.smooth_l1(a, scalar=0.3), a=arr)

def check_binary_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    announce_check('various binary ops')
    check_fused_symbol(a+b, a=arr1, b=arr2)
    check_fused_symbol(a+3, a=arr1)
    check_fused_symbol(a-b, a=arr1, b=arr2)
    check_fused_symbol(a-3, a=arr1)
    check_fused_symbol(3-a, a=arr1)
    check_fused_symbol(a*b, a=arr1, b=arr2)
    check_fused_symbol(a*3, a=arr1)
    check_fused_symbol(a/(b+1), a=arr1, b=arr2)
    check_fused_symbol(a/3, a=arr1)
    check_fused_symbol(3/a, a=arr1)
    check_fused_symbol(a**b, a=arr1, b=arr2)
    check_fused_symbol(a**3, a=arr1)
    check_fused_symbol(mx.sym.pow(3,a), a=arr1)
    check_fused_symbol(mx.sym.maximum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.minimum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,3), a=arr1)

def check_other_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.Variable('c')
    shape = rand_shape_2d()
    shape = list((5,) + shape)
    # Make sure there is at least 2 elements for the test with negative indices
    shape[1] += 1
    shape[2] += 1
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)
    arr3 = mx.random.uniform(shape=shape)

    announce_check('add_n with 3 inputs')
    check_fused_symbol(mx.sym.add_n(a,b,c), a=arr1, b=arr2, c=arr3)

    announce_check('slice_axis')
    check_fused_symbol(mx.sym.slice_axis(a, axis=0, begin=1, end=4), a=arr1)

    # Testing handling of negative axis
    check_fused_symbol(mx.sym.slice_axis(a, axis=-3, begin=1, end=4), a=arr1)

    begin = (random.randint(0, shape[0]-1),
             random.randint(0, shape[1]-1),
             random.randint(0, shape[2]-1))
    end = (random.randint(begin[0]+1, shape[0]),
           random.randint(begin[1]+1, shape[1]),
           random.randint(begin[2]+1, shape[2]))
    announce_check('slice')
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    begin = (random.randint(-shape[0], -2),
             random.randint(-shape[1], -2),
             random.randint(-shape[2], -2))
    end = (random.randint(begin[0]+1, -1),
           random.randint(begin[1]+1, -1),
           random.randint(begin[2]+1, -1))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    begin = (random.randint(-shape[0], -2),
             random.randint(-shape[1], -2),
             random.randint(-shape[2], -2))
    end = (random.randint(begin[0]+1, -1),
           random.randint(begin[1]+1, -1),
           random.randint(begin[2]+1, -1))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    arr1 = mx.random.uniform(shape=(2,3,4,5))
    arr2 = mx.random.uniform(shape=(1,2,3))
    announce_check('slice_like')
    check_fused_symbol(mx.sym.slice_like(a,b, axes=[-2, 0]), a=arr1, b=arr2)

    arr1 = mx.random.uniform(shape=(1,1,2,3))
    arr2 = mx.random.uniform(shape=(2,2,2,3))
    announce_check('broadcast_like')
    check_fused_symbol(mx.sym.broadcast_like(a, b, lhs_axes=[0], rhs_axes=[0]), a=arr1, b=arr2)


def set_back_env_var(var_name, old_env_var):
    if old_env_var is None:
        os.environ.pop(var_name)
    else:
        os.environ[var_name] = old_env_var


def check_batch_norm_activ():
    old_env_var = os.environ.get('MXNET_DISABLE_BNACTIV_FUSION', None)
    bn_name = "batchnorm"
    data = mx.sym.Variable('data')
    rtol = 1e-2
    atol = 1e-3
    for axis in [1, 3]:
        for act_type in ['relu', 'tanh']:
            for ctx in [mx.gpu(0), mx.cpu(0)]:
                for channel_size in [11, 12, 16]:
                    for dtype in ['float16', 'float32']:
                        announce_check("batchnorm+activation with axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                              axis, act_type, ctx, channel_size, dtype))
                        input_shape = [10, 5, 5, 5]
                        input_shape[axis] = channel_size
                        input_shape = tuple(input_shape)
                        bn = mx.sym.BatchNorm(data, axis=axis, name=bn_name)
                        act = mx.sym.Activation(bn, act_type=act_type)
                        os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '0'
                        executor = act.simple_bind(ctx, data=input_shape, grad_req='null', force_rebind=True,
                                                   type_dict={'data': dtype})
                        MIN_CHANNEL = 4
                        if (axis == 3 and act_type == 'relu' and ctx == mx.gpu(0) and
                            channel_size % MIN_CHANNEL == 0 and dtype == 'float16'):
                            assert executor.get_optimized_symbol().name == bn_name + "_activ"
                            arg_params = {'data': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'batchnorm_gamma': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_beta': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_moving_var': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            fused_out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            fused_out_train = executor.outputs[0].asnumpy()
                            os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '1'
                            executor = act.simple_bind(ctx, data=input_shape,
                                                       grad_req='null', force_rebind=True,
                                                       type_dict={'data': dtype})
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            out_train = executor.outputs[0].asnumpy()
                            assert_allclose(out_predict, fused_out_predict, rtol, atol)
                            assert_allclose(out_train, fused_out_train, rtol, atol)
                        else:
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
    os.environ['MXNET_USE_FUSION'] = '0'
    os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '0'
    bn = mx.sym.BatchNorm(data, axis=3, name=bn_name)
    act = mx.sym.Activation(bn, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), data=(10, 5, 5, 12),
                               grad_req='null', force_rebind=True,
                               type_dict={'data': dtype})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '1'
    bn = mx.sym.BatchNorm(data, axis=3, name=bn_name)
    act = mx.sym.Activation(bn, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), data=(10, 5, 5, 12),
                               grad_req='null', force_rebind=True,
                               type_dict={'data': dtype})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    set_back_env_var('MXNET_DISABLE_BNACTIV_FUSION', old_env_var)

def check_batch_norm_add_relu():
    old_env_var = os.environ.get('MXNET_DISABLE_BNADDRELU_FUSION', None)
    bn_name = "batchnorm"
    lhs = mx.sym.Variable('lhs')
    rhs = mx.sym.Variable('rhs')
    rtol = 1e-2
    atol = 1e-3
    for axis in [1, 3]:
        for act_type in ['relu', 'tanh']:
            for ctx in [mx.gpu(0), mx.cpu(0)]:
                for channel_size in [11, 12, 16]:
                    for dtype in ['float16', 'float32']:
                        announce_check("batchnorm+add+relu with axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                              axis, act_type, ctx, channel_size, dtype))
                        input_shape = [10, 5, 5, 5]
                        input_shape[axis] = channel_size
                        input_shape = tuple(input_shape)
                        bn = mx.sym.BatchNorm(lhs, axis=axis, name=bn_name)
                        add = bn + rhs
                        act = mx.sym.Activation(add, act_type=act_type)
                        os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
                        executor = act.simple_bind(ctx, lhs=input_shape, rhs=input_shape,
                                                   grad_req='null', force_rebind=True,
                                                   type_dict={'lhs': dtype, 'rhs': dtype})
                        MIN_CHANNEL = 4
                        if (axis == 3 and act_type == 'relu' and ctx == mx.gpu(0) and
                            channel_size % MIN_CHANNEL == 0 and dtype == 'float16'):
                            assert executor.get_optimized_symbol().name == bn_name + "_add_relu"
                            arg_params = {'lhs': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'rhs': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'batchnorm_gamma': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_beta': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_moving_var': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            fused_out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            fused_out_train = executor.outputs[0].asnumpy()
                            os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '1'
                            executor = act.simple_bind(ctx, lhs=input_shape, rhs=input_shape,
                                                       grad_req='null', force_rebind=True,
                                                       type_dict={'lhs': dtype, 'rhs': dtype})
                            print(executor.get_optimized_symbol().get_internals())
                            print(act.get_internals())
                            if ctx == mx.gpu(0):  # Still have pointwise fusion
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(act.get_internals()) - 1)
                            else:
                                assert(len(executor.get_optimized_symbol().get_internals()) ==
                                       len(act.get_internals()))
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            out_train = executor.outputs[0].asnumpy()
                            assert_allclose(out_predict, fused_out_predict, rtol, atol)
                            assert_allclose(out_train, fused_out_train, rtol, atol)
                        elif ctx == mx.gpu(0):  # Still have pointwise fusion
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()) - 1)
                        else:
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
    input_shape=(10, 5, 5, 12)
    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
    bn = mx.sym.BatchNorm(rhs, axis=3, name=bn_name)
    add = lhs + bn
    act = mx.sym.Activation(add, act_type='relu')

    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()) - 1)

    os.environ['MXNET_USE_FUSION'] = '0'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
    bn = mx.sym.BatchNorm(lhs, axis=3, name=bn_name)
    add = bn + rhs
    act = mx.sym.Activation(add, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '1'
    bn = mx.sym.BatchNorm(lhs, axis=3, name=bn_name)
    add = bn + rhs
    act = mx.sym.Activation(add, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()) - 1)

    set_back_env_var('MXNET_DISABLE_BNADDRELU_FUSION', old_env_var)

def check_norm_convolution():
    old_env_var = os.environ.get('MXNET_DISABLE_NORMCONV_FUSION', None)
    conv0_name = "conv0"
    conv1_name = "conv1"
    kernel = (3, 3)
    stride = (1, 1)
    bn_name = "batchnorm"
    data = mx.sym.Variable('data')
    rtol = 1e-2
    # We don't really check all the conditions because we trust the Supports function of the Op
    # Which are tested in unit tests of norm_convolution
    for layout, axis in [('NCHW', 1), ('NHWC', 3)]:
        for no_bias in [True, False]:
            for act_type in [None, 'relu']:
                atol = 1e-1 if act_type == 'relu' else 1.
                for ctx in [mx.gpu(0), mx.cpu(0)]:
                    for channel_size in [15, 32]:
                        for dtype in ['float16', 'float32']:
                            announce_check("norm_convolution with axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                                  axis, act_type, ctx, channel_size, dtype))
                            input_shape = [10, 5, 5, 5]
                            input_shape[axis] = channel_size
                            input_shape = tuple(input_shape)
                            # For test failure reproducibility, fix the algos of the golden copy
                            conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                                      num_filter=channel_size * 2, no_bias=no_bias,
                                                      cudnn_algo_fwd=1,
                                                      cudnn_algo_bwd_data=1,
                                                      cudnn_algo_bwd_filter=1,
                                                      name=conv0_name)
                            bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
                            conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                                      num_filter=channel_size, no_bias=no_bias,
                                                      cudnn_algo_fwd=1,
                                                      cudnn_algo_bwd_data=1,
                                                      cudnn_algo_bwd_filter=1,
                                                      name=conv1_name)
                            os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '0'
                            executor = conv.simple_bind(ctx, data=input_shape,
                                                        grad_req='write', force_rebind=True,
                                                        type_dict={'data': dtype})

                            supported_arches = [70, 75, 80, 86]
                            if (axis == 3 and no_bias and channel_size % 32 == 0 and
                                ctx.device_type == 'gpu' and mx.context.gpu_sm_arch(ctx.device_id) in supported_arches and
                                dtype == 'float16'):
                                assert executor.get_optimized_symbol().get_internals()[2].name == conv0_name + "_normalized"
                                assert executor.get_optimized_symbol().name == conv1_name + "_normalized"
                                arg_params = {'data': mx.random.uniform(shape=input_shape, dtype=dtype),
                                              'conv0_weight': mx.random.uniform(shape=(channel_size * 2,) + kernel + (channel_size,), dtype=dtype),
                                              'conv1_weight': mx.random.uniform(shape=(channel_size,) + kernel + (channel_size * 2,), dtype=dtype),
                                              'batchnorm_gamma': mx.random.uniform(shape=(channel_size * 2,), dtype='float32'),
                                              'batchnorm_beta': mx.random.uniform(shape=(channel_size * 2,), dtype='float32')}
                                aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size * 2,), dtype='float32'),
                                              'batchnorm_moving_var': mx.random.uniform(shape=(channel_size * 2,), dtype='float32')}
                                executor.copy_params_from(arg_params, aux_params)
                                executor.forward(is_train=False)
                                fused_out_predict = executor.outputs[0].asnumpy()
                                executor.forward(is_train=True)
                                fused_out_train = executor.outputs[0].asnumpy()
                                os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '1'
                                executor = conv.simple_bind(ctx, data=input_shape,
                                                           grad_req='null', force_rebind=True,
                                                           type_dict={'data': dtype})
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(conv.get_internals()))
                                executor.copy_params_from(arg_params, aux_params)
                                executor.forward(is_train=False)
                                out_predict = executor.outputs[0].asnumpy()
                                executor.forward(is_train=True)
                                out_train = executor.outputs[0].asnumpy()
                                assert_allclose(out_predict, fused_out_predict, rtol, atol)
                                assert_allclose(out_train, fused_out_train, rtol, atol)
                            else:
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(conv.get_internals()))
        os.environ['MXNET_USE_FUSION'] = '0'
        os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '0'
        input_shape = (10, 5, 5, 32)
        channel_size = 32
        conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size * 2, no_bias=True,
                                  name=conv0_name)
        bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
        conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size, no_bias=True, name=conv1_name)
        executor = conv.simple_bind(ctx, data=input_shape,
                                    grad_req='write', force_rebind=True,
                                    type_dict={'data': 'float16'})
        assert (len(executor.get_optimized_symbol().get_internals()) ==
                len(conv.get_internals()))

        os.environ['MXNET_USE_FUSION'] = '1'
        os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '1'
        input_shape = (10, 5, 5, 32)
        channel_size = 32
        conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size * 2, no_bias=True,
                                  name=conv0_name)
        bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
        conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size, no_bias=True, name=conv1_name)
        executor = conv.simple_bind(ctx, data=input_shape,
                                    grad_req='write', force_rebind=True,
                                    type_dict={'data': 'float16'})
        assert (len(executor.get_optimized_symbol().get_internals()) ==
                len(conv.get_internals()))
        set_back_env_var('MXNET_DISABLE_NORMCONV_FUSION', old_env_var)


def check_leakyrelu_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    # Testing gelu
    print("Checking fusion of LeakyReLU:gelu")
    check_fused_symbol(mx.sym.LeakyReLU(a+b, act_type='gelu'), a=arr1, b=arr2)

@with_seed()
def test_fusion():
    old_mxnet_use_fusion = os.environ.get('MXNET_USE_FUSION', None)
    os.environ['MXNET_USE_FUSION'] = '1'

    check_batch_norm_activ()
    check_batch_norm_add_relu()
    check_unary_ops()
    check_binary_ops()
    check_other_ops()
    check_norm_convolution()
    check_leakyrelu_ops()

    set_back_env_var('MXNET_USE_FUSION', old_mxnet_use_fusion)


@with_seed()
def test_fusion_compiler_cache():
    # Stresses the internal cache of CUfunctions by creating the same kernel multiple times and
    # on multiple GPUs if available.
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    # Invoke the same model twice, second time will exercise compile cache
    check_fused_symbol(a+b, ctx=mx.gpu(0), a=arr1, b=arr2)
    check_fused_symbol(a+b, ctx=mx.gpu(0), a=arr1, b=arr2)

    # On multi-GPU systems, invoke the same model on other GPUs
    num_gpus = mx.context.num_gpus()
    if num_gpus > 1:
        check_fused_symbol(a+b, ctx=mx.gpu(1), a=arr1, b=arr2)

@with_seed()
def test_fusion_reshape_executor():
    a = mx.sym.Variable("data1")
    b = mx.sym.Variable("data2")
    c = a + b + 1
    sym = mx.sym.relu(c)
    orig_shape = (10,10)
    e = sym.simple_bind(ctx=mx.gpu(), data1=orig_shape, data2=orig_shape)
    data = mx.nd.zeros(orig_shape, ctx=mx.gpu())
    out = e.forward(is_train=False)
    assert out[0].sum().asscalar() == 100
    changed_shape = (80, 2)
    new_shape = {'data1': changed_shape, 'data2': changed_shape}
    data = mx.nd.zeros(new_shape['data1'], ctx=mx.gpu())
    f = e.reshape(allow_up_sizing=True, **new_shape)
    out = f.forward(is_train=False, data1=data, data2=data)
    assert out[0].sum().asscalar() == 160
    # Reshape again
    changed_shape = (30, 5)
    new_shape = {'data1': changed_shape, 'data2': changed_shape}
    data = mx.nd.zeros(new_shape['data1'], ctx=mx.gpu())
    f = e.reshape(allow_up_sizing=True, **new_shape)
    out = f.forward(is_train=False, data1=data, data2=data)
    assert out[0].sum().asscalar() == 150

@with_seed()
@use_np
def test_fusion_boolean_inputs():
    from mxnet.gluon import HybridBlock

    class Foo(HybridBlock):
        def __init__(self, prefix=None, params=None):
            super(Foo, self).__init__(prefix=prefix, params=params)

        def hybrid_forward(self, F, valid_length):
            mask = valid_length.astype(np.float32)
            mask2 = valid_length.astype(np.float32)
            mask = mask * F.np.expand_dims(mask2, axis=-1)
            return mask

    foo = Foo()
    foo.hybridize(static_alloc=True)
    out = foo(mx.np.ones((10,), ctx=mx.gpu(), dtype=np.bool))
    mx.npx.waitall()

@with_seed()
def test_fusion_different_dimensions():
    from mxnet.gluon import HybridBlock

    class Foo(HybridBlock):
        def __init__(self, prefix=None, params=None):
            super(Foo, self).__init__(prefix=prefix, params=params)

        def hybrid_forward(self, F, x):
            mask2 = x.astype(np.float32)
            mask = F.expand_dims(mask2, axis=-1)
            return mask

    foo = Foo()
    foo.hybridize(static_alloc=True)
    # Pass 1-D data
    out = foo(mx.nd.ones((10,), ctx=mx.gpu()))
    assert np.all(out.asnumpy() == np.ones((10,1)))
    assert out.shape == (10,1)
    # Pass 2-D data
    out = foo(mx.nd.ones((10,10), ctx=mx.gpu()))
    assert np.all(out.asnumpy() == np.ones((10,10)))
    assert out.shape == (10,10,1)


@with_seed()
def test_input_reorder():
    class Block(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Block, self).__init__(**kwargs)

        def hybrid_forward(self, F, x, y, z):
            s = x * 2
            s2 = s + z
            s = F.broadcast_add(s, y * y)
            return F.dot(s, s2)

    for static_alloc in (False, True):
        arg_shapes = [(10, 10), (10, 1), (10, 10)]
        arg_data = [mx.random.uniform(shape=s) for s in arg_shapes]

        arrays = {}
        for use_fusion in ('0', '1'):
            os.environ['MXNET_USE_FUSION'] = use_fusion
            arrays[use_fusion] = {}
            n = Block()
            n.hybridize(static_alloc=static_alloc, static_shape=static_alloc)
            args = [arg.copyto(mx.gpu()) for arg in arg_data]
            for arg in args:
                arg.attach_grad()
            with autograd.record():
                r = n(*args)
            arrays[use_fusion]['result'] = r
            r.backward()
            for i, arg in enumerate(args):
                arrays[use_fusion][i] = arg.grad
        for key in ['result'] + list(range(len(arg_data))):
            assert_allclose(arrays['0'][key].asnumpy(), arrays['1'][key].asnumpy())


@with_seed()
def test_gluon_bn_add_relu():
    class BNModel(gluon.HybridBlock):

        def __init__(self, axis, relu, **kwargs):
            super(BNModel, self).__init__(**kwargs)
            self.axis = axis
            self.relu = relu

        def hybrid_forward(self, F, x, y, gamma, beta, moving_mean, moving_var):
            x = F.BatchNorm(x, gamma, beta, moving_mean, moving_var, axis=self.axis,
                            fix_gamma=False)
            return F.relu(y + x) if self.relu else y + x

    axis = 3
    channel_size = 16
    shape = [10, 5, 5, 5]
    shape[axis] = channel_size
    shape = tuple(shape)
    cshape = (channel_size,)
    dtype = 'float16'
    learnable_args = [2, 3]

    # Run the data through a BNAdd with no relu.  If the output is very
    # near 0, the compared BNAddRelu models may differ on the sign of the
    # output, and hence whether a gradient should be backpropped.
    # To avoid test failures, regenerate the input data in that case.
    min_non_relu_output = 0.0
    while min_non_relu_output < 1e-3:
        arg_data = [mx.random.uniform(shape=shape, dtype=dtype),
                    mx.random.uniform(shape=shape, dtype=dtype),
                    mx.random.uniform(shape=cshape),
                    mx.random.uniform(shape=cshape),
                    mx.random.uniform(shape=cshape),
                    mx.random.uniform(shape=cshape)]

        m = BNModel(axis, relu=False)
        static = '1'
        m.hybridize(static_alloc=static, static_shape=static)
        args = [arg.copyto(mx.gpu()) for arg in arg_data]
        with mx.autograd.record():
            r = m(*args)
        min_non_relu_output = np.abs(r.asnumpy()).min()

    arrays = {}
    for use_fusion in ('0', '1'):
        os.environ['MXNET_USE_FUSION'] = use_fusion
        arrays[use_fusion] = {}

        m = BNModel(axis, relu=True)
        static = use_fusion == '1'
        m.hybridize(static_alloc=static, static_shape=static)
        args = [arg.copyto(mx.gpu()) for arg in arg_data]
        for i in learnable_args:
            args[i].attach_grad()
        with mx.autograd.record():
            r = m(*args)
        arrays[use_fusion]['result'] = r
        r.backward()
        for i in learnable_args:
            arrays[use_fusion][i] = args[i].grad
    rtol = 5e-2
    atol = 1e-1
    for key in ['result'] + learnable_args:
        assert_allclose(arrays['0'][key].asnumpy(), arrays['1'][key].asnumpy(), rtol, atol)

@with_seed()
def test_fusion_cycle():
    class Test(gluon.nn.HybridBlock):
        def __init__(self, **kwargs):
            super(Test, self).__init__(**kwargs)

        def hybrid_forward(self, F, x, y):
            x = F.relu(x)
            y = F.relu(y)
            z1 = F.expand_dims(F.sum_axis(x, axis=1), axis=1)
            z2 = F.expand_dims(F.sum_axis(y, axis=1), axis=1)
            return x + z2, y + z1

    t = Test()
    a = mx.nd.zeros(shape=(10,1), ctx=mx.gpu())
    b = mx.nd.zeros(shape=(10,1), ctx=mx.gpu())
    t.hybridize(static_alloc=True, static_shape=True)
    out = t(a, b)
    mx.nd.waitall()


if __name__ == '__main__':
    import nose
    nose.runmodule()
