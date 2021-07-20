from __future__ import print_function

import os
import random
import mxnet as mx
import numpy as np
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

set_default_context(mx.gpu(0))

rtol = {'float16' : 1e-1,
        'float32' : 1.5e-6,
        'float64' : 1.5e-6,
        }

atol = {'float16' : 1e-2,
        'float32' : 1e-7,
        'float64' : 1e-7,
        }

def create_vars(var_names):
    return [mx.sym.Variable(name) for name in var_names]

def create_dicts(ctx, k, data_shape, weight_shape, x, z, wx, wz, mov_mean, mov_var, mov_meanz, mov_varz, beta, gamma, betaz, gammaz):
    args_dict = {'X':x, 'Wx':wx, 'Z':z, 'B':beta, 'G':gamma, 'Wz':wz, 'Bz':betaz, 'Gz':gammaz}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var, 'MovMeanz':mov_meanz, 'MovVarz':mov_varz}
    grad_dict = {'X':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Z':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'B':mx.nd.zeros((k,),dtype=np.float32,ctx=ctx),
                 'G':mx.nd.zeros((k,),dtype=np.float32,ctx=ctx),
                 'Bz':mx.nd.zeros((k,),dtype=np.float32,ctx=ctx),
                 'Gz':mx.nd.zeros((k,),dtype=np.float32,ctx=ctx),
                 'Wz':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx),
                 'Wx':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}
    return args_dict, aux_dict, grad_dict

def _has_near_zero_outputs(ctx, n, h, w, c, k,
               x, z, wx, wz, beta, gamma, betaz, gammaz,
               mov_mean, mov_var, mov_meanz, mov_varz):

    # net1 = [conv] + [bn+add+relu]
    #                      |
    #   [conv] + [bn] ------
    kernel_shape = (1,1)
    (r, s) = kernel_shape
    stride = (1,1)
    pad = (0,0)
    out_shape = (n,h,w,k)
    layout = "NHWC"
    num_filter = k
    weight_shape = (num_filter, r, s, c)
    data_shape = (n, h, w, c)

    eps = 1e-4
    momentum = 0.9
    threshold = pow(2, -13)

    X, B, G, MovMean, MovVar, Wx = create_vars(['X', 'B', 'G', 'MovMean', 'MovVar', 'Wx'])
    if type(z)!=type(None):
        Z = mx.sym.Variable('Z') # addend

    grad_dict = {'dX':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'dW':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                 'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch2a'}
    if type(wz)!=type(None):
        Wz, Bz, Gz, MovMeanz, MovVarz = create_vars(['Wz', 'Bz', 'Gz', 'MovMeanz', 'MovVarz'])
        args_dict = {'X':x, 'Wx':wx, 'Z':z, 'B':beta, 'G':gamma, 'Wz':wz, 'Bz':betaz, 'Gz':gammaz}
        aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var, 'MovMeanz':mov_meanz, 'MovVarz':mov_varz}
        conv_args1 = {'num_filter':num_filter, 'kernel':kernel_shape,
                        'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch1'}
        Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
        Zout = mx.sym.Convolution(Z,weight=Wz,no_bias=True,**conv_args1)
        Zout1, Zmean, Zvar = mx.sym.BatchNorm(Zout,gamma=Gz, beta=Bz, act_type=None,
                                eps = eps, momentum = momentum,
                                moving_mean=MovMeanz, moving_var=MovVarz,
                                fix_gamma=False, use_global_stats=False,
                                output_mean_var=True, cudnn_off=False,
                                name='bn_branch1', axis=-1)
        Zout2, Xmean, Xvar = mx.sym.BatchNorm(Xout, gamma=G, beta=B, act_type=None,
                                eps = eps, momentum = momentum,
                                moving_mean=MovMean, moving_var=MovVar,
                                fix_gamma=False, use_global_stats=False,
                                output_mean_var=True, cudnn_off=False,
                                name='bn_branch2c', axis=-1)
        group = mx.sym.Group([Zout1, Zout2])

    else:
        args_dict = {'X':x, 'Wx':wx, 'B':beta, 'G':gamma}
        aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
        Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
        group, Xmean, Xvar = mx.sym.BatchNorm(Xout, gamma=G, beta=B, act_type=None,
                                eps = eps, momentum = momentum,
                                moving_mean=MovMean, moving_var=MovVar,
                                fix_gamma=False, use_global_stats=False,
                                output_mean_var=True, cudnn_off=False,
                                name='bn_branch2c', axis=-1)

    net_exe = group.bind(ctx=ctx,args=args_dict,
                         aux_states=aux_dict,args_grad=grad_dict)

    net_fout = net_exe.forward(is_train=True)
    if type(wz)!=type(None):
        out_data1 = net_exe.output_dict['bn_branch1_output'].asnumpy()
    else:
        if type(z)!=type(None):
            out_data1 = z.asnumpy()
        else:
            out_data1 = mx.nd.zeros(out_shape,dtype=np.float16,ctx=ctx).asnumpy()
    out_data2 = net_exe.output_dict['bn_branch2c_output'].asnumpy()
    out_data = out_data1 + out_data2 # Implements the add
    out_data_abs = np.abs(out_data)
    not_feature_axes = (0, 1, 2)
    origin_dist_mins = out_data_abs.min(axis=not_feature_axes)
    bad_indices = np.nonzero(origin_dist_mins < threshold)[0]
    del net_exe
    return bad_indices

def get_beta_gamma(ctx, x, z, wx, wz,
        mov_mean, mov_var, mov_meanz, mov_varz,
        layer_info):
    n, h, w, c, k = layer_info
    bias_max = 1
    scale_max = 1.25
    equiv_scale_bias_shape = (k,)
    b_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
    g_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
    indices_to_set = np.array(range(k))
    count_max, count = 1000, 0
    while len(indices_to_set) > 0 and count < 1000:
        count += 1
        for index in indices_to_set:
            b_np[index] = np.random.uniform(-bias_max, bias_max)
            g_np[index] = np.random.uniform(1.0/scale_max, scale_max)
        beta = mx.nd.array(b_np, dtype=np.float32, ctx=ctx)
        gamma = mx.nd.array(g_np, dtype=np.float32, ctx=ctx)
        betaz = beta.copy()
        gammaz = gamma.copy()
        need_data_check = True
        if need_data_check:
            indices_to_set = _has_near_zero_outputs(ctx, n, h, w, c, k,
               x, z, wx, wz, beta, gamma, betaz, gammaz,
               mov_mean, mov_var, mov_meanz, mov_varz)
        else:
            indices_to_set = []
    return beta, gamma


@with_seed()
def test_dbar_sbar_forward():

    ctx = default_context()
    cuda_arch_list = [70, 75, 80, 86]
    cuda_arch = mx.context.gpu_sm_arch(ctx.device_id)
    if cuda_arch not in cuda_arch_list:
        print('Bypassing normalized convolution test on cuda arch {} supported versions are {}).'.format(
            cuda_arch, cuda_arch_list))
        return

    def f(data_np_x,data_np_z,x_equiv_scale,z_equiv_scale,x_equiv_bias,z_equiv_bias):
        return data_np_x+data_np_z

    def sbar(data_np_x,data_np_z,x_equiv_scale,z_equiv_scale,x_equiv_bias,z_equiv_bias):
        return data_np_x + data_np_z

    def sbr(data_np_x, x_equiv_scale, x_equiv_bias):
        return data_np_x

    ndim = 4
    shape = (208,14,14,1024)
    shape_c = (1024)

    # Not using negatives cuz RELU would filter them out
    data_np_x = mx.ndarray.random.uniform(0.0,0.20,shape,dtype=np.float16,ctx=ctx)
    data_np_z = mx.ndarray.random.uniform(0.0,0.20,shape,dtype=np.float16,ctx=ctx)

    x_mean = mx.ndarray.random.uniform(-0.20,0.20,shape_c,dtype=np.float32,ctx=ctx)
    x_invvar = mx.ndarray.random.uniform(-0.20,0.20,shape_c,dtype=np.float32,ctx=ctx)

    x_equiv_scale = mx.nd.ones(shape_c,dtype=np.float16,ctx=ctx)
    z_equiv_scale = mx.nd.ones(shape_c,dtype=np.float16,ctx=ctx)

    x_equiv_bias = mx.nd.zeros(shape_c,dtype=np.float16,ctx=ctx)
    z_equiv_bias = mx.nd.zeros(shape_c,dtype=np.float16,ctx=ctx)

    beta = mx.nd.zeros((1),dtype=np.float32,ctx=ctx)
    gamma = mx.nd.ones((1),dtype=np.float32,ctx=ctx)

    z_mean = mx.ndarray.random.uniform(-0.20,0.20,shape_c,dtype=np.float32,ctx=ctx)
    z_invvar = mx.ndarray.random.uniform(-0.20,0.20,shape_c,dtype=np.float32,ctx=ctx)

    # Testing DBAR -- L0 test
    expected = f(data_np_x,data_np_z, x_equiv_scale,
                 z_equiv_scale, x_equiv_bias, z_equiv_bias)
    output, bitmask = mx.nd.ScaleBiasAddRelu(dataX = data_np_x, dataZ = data_np_z,
                                            x_equiv_scale = x_equiv_scale , x_equiv_bias = x_equiv_bias,
                                            z_equiv_scale = z_equiv_scale, z_equiv_bias = z_equiv_bias,
                                            x_gamma = gamma, x_beta = beta, x_mean = x_mean,
                                            x_invvar = x_invvar, z_gamma = gamma, z_beta = beta,
                                            z_mean = z_mean, z_invvar = z_invvar, layout = 'NHWC',
                                            act_type='relu')
    assert_almost_equal(expected, output,
                        atol=atol['float16'], rtol=rtol['float16'])

    # Testing SBAR -- L0 Test
    expected = sbar(data_np_x, data_np_z,x_equiv_scale,
                    z_equiv_scale,x_equiv_bias,z_equiv_bias)
    output, bitmask = mx.nd.ScaleBiasAddRelu(dataX = data_np_x, dataZ = data_np_z,
                                        x_equiv_scale = x_equiv_scale,
                                        x_equiv_bias = x_equiv_bias,
                                        x_gamma = gamma, x_beta = beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        dual_scale_bias = False, fused_add = True)
    assert_almost_equal(expected, output,
                        atol=atol['float16'], rtol=rtol['float16'])

    # Testing SBR -- L0 Test
    expected = sbr(data_np_x, x_equiv_scale, x_equiv_bias)
    output, bitmask = mx.nd.ScaleBiasAddRelu(dataX = data_np_x,
                                        x_equiv_scale = x_equiv_scale,
                                        x_equiv_bias = x_equiv_bias,
                                        x_gamma = gamma, x_beta = beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        dual_scale_bias = False, fused_add = False)
    assert_almost_equal(expected, output,
                        atol=atol['float16'], rtol=rtol['float16'])

    # Test 2 -- L1 Test
    # This test is doing two things:-
    # (1) conv + [bn+add+relu] fprp and dgrad
    # (2) [conv+bn(s)] + finalizeStats() + [bn(a)+add+relu]
    # Then it compares the forward and backward for both to make sure our op is correct

    n,c,h,w,k = 208, 64, 14, 14, 64
    elements = n * h * w # fed as input to BNFinalize
    kernel_shape = (1,1)
    (r, s) = kernel_shape
    stride = (1,1)
    pad = (0,0)
    out_shape = (n,h,w,k)
    layout = "NHWC"
    num_filter = k
    weight_shape = (num_filter, r, s, c)
    data_shape = (n, h, w, c)

    eps = 1e-4
    momentum = 0.9

    x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    z = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    wx = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
    wz = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
    mov_mean = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_var = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_meanz = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_varz = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)

    beta, gamma = get_beta_gamma(ctx, x, z, wx, wz,
                                 mov_mean, mov_var,
                                 mov_meanz, mov_varz,
                                 (n, h, w, c, k))
    betaz = beta.copy()
    gammaz = gamma.copy()

    var_names = ['X', 'Z', 'B', 'G', 'MovMean', 'MovVar', 'Wx', 'Wz', 'Bz', 'Gz', 'MovMeanz', 'MovVarz']
    X, Z, B, G, MovMean, MovVar, Wx, Wz, Bz, Gz, MovMeanz, MovVarz = create_vars(var_names)

    args_dict = {'X':x, 'Wx':wx, 'Z':z, 'B':beta, 'G':gamma, 'Wz':wz, 'Bz':betaz, 'Gz':gammaz}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var, 'MovMeanz':mov_meanz, 'MovVarz':mov_varz}
    grad_dict = {'dX':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                    'dW':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}

    # net1 = [conv] + [bn+add+relu]
    #                      |
    #   [conv] + [bn] ------
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch2a'}
    conv_args1 = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch1'}

    Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
    Zout = mx.sym.Convolution(Z,weight=Wz,no_bias=True,**conv_args1)
    Zout1, Zmean, Zvar = mx.sym.BatchNorm(Zout,gamma=Gz, beta=Bz, act_type=None,
                            eps = eps, momentum = momentum,
                            moving_mean=MovMeanz, moving_var=MovVarz,
                            fix_gamma=False, use_global_stats=False,
                            output_mean_var=True, cudnn_off=False,
                            name='bn_branch1', axis=-1)
    net, Xmean, Xvar = mx.sym.BatchNormAddRelu(Xout, gamma=G, beta=B,
                                    moving_mean=MovMean, moving_var=MovVar,
                                    fix_gamma=False, use_global_stats=False,
                                    output_mean_var=True, cudnn_off=False,
                                    axis=-1,
                                    eps = eps, momentum = momentum,
                                    addend=Zout1, name="bn_add_relu")

    group1 = mx.sym.Group([net, Xmean, Xvar, Xout, Zout, Zmean, Zvar])
    net_exe = group1.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)
                        #grad_req=grad_req)
    net_fout = net_exe.forward(is_train=True)
    net_bout = net_exe.backward(is_train=True)

    # net2 = [conv+bn(s)] + [bn(a)+add+relu]
    #                               |
    #        [conv+bn(s)] -----------

    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout}
    xout, x_sum, x_sum_sq = mx.sym.NormalizedConvolution(X, weight=Wx,
                                            no_equiv_scale_bias=True, name="normalizedconvolution0",
                                            **conv_args)
    zout, z_sum, z_sum_sq = mx.sym.NormalizedConvolution(Z, weight=Wz,
                                            no_equiv_scale_bias=True, name="normalizedconvolution1",
                                            **conv_args)
    x_equiv_scale, x_equiv_bias, x_mean, x_invvar, x_gamma, x_beta = mx.sym.BNStatsFinalize(sum=x_sum,sum_squares=x_sum_sq,gamma=G,beta=B,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMean, moving_var=MovVar,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        name = "bnstatsfinalize0",
                                                                        output_mean_var=True, elem_count=int(elements))
    z_equiv_scale, z_equiv_bias, z_mean, z_invvar, z_gamma, z_beta = mx.sym.BNStatsFinalize(sum=z_sum,sum_squares=z_sum_sq,gamma=Gz,beta=Bz,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMeanz, moving_var=MovVarz,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        output_mean_var=True, elem_count=int(elements),name="bn1")

    output, bitmask = mx.sym.ScaleBiasAddRelu(dataX = xout, dataZ = zout,
                                        eps = eps,
                                        x_equiv_scale = x_equiv_scale, x_equiv_bias = x_equiv_bias,
                                        z_equiv_scale = z_equiv_scale, z_equiv_bias = z_equiv_bias,
                                        x_gamma = x_gamma, x_beta = x_beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        z_gamma = z_gamma, z_beta = z_beta,
                                        z_mean = z_mean, z_invvar = z_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        name = "scalebiasaddrelu0")
    group2 = mx.sym.Group([output,x_mean,x_invvar,xout,zout,z_invvar,z_mean,x_equiv_scale,x_equiv_bias,
                            z_equiv_scale,z_equiv_bias,z_sum,z_sum_sq,z_gamma,z_beta])
    net2_exe = group2.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)

    net2_fout = net2_exe.forward(is_train=True)
    net2_bout = net2_exe.backward(is_train=True)

    assert_almost_equal(net_exe.output_dict['res_branch2a_output'],
                         net2_exe.output_dict['normalizedconvolution0_output'],
                         atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['res_branch1_output'],
                        net2_exe.output_dict['normalizedconvolution1_output'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['bn_add_relu_mean'],
                        net2_exe.output_dict['bnstatsfinalize0_mean'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['bn_add_relu_var'],
                        net2_exe.output_dict['bnstatsfinalize0_var'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['bn_branch1_mean'],
                        net2_exe.output_dict['bn1_mean'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['bn_branch1_var'],
                        net2_exe.output_dict['bn1_var'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe.output_dict['bn_add_relu_output'],
                        net2_exe.output_dict['scalebiasaddrelu0_output'],
                        atol=atol['float16'], rtol=rtol['float16'])

def get_conv_fprop_shapes(n,c,h,w,k):
    elements = n * h * w # fed as input to BNFinalize
    kernel_shape = (1,1)
    r, s = kernel_shape
    stride = (1,1)
    pad = (0,0)
    out_shape = (n,h,w,k)
    layout = "NHWC"
    num_filter = k
    weight_shape = (num_filter, r, s, c)
    data_shape = (n,h,w,c)
    return elements, kernel_shape, r, s, stride, pad, out_shape, layout, num_filter, weight_shape, data_shape

@with_seed()
def test_bna():
    # Net 1 : conv + bn + relu
    # Net 2 : conv+bn(s) + bn(a)+relu

    ctx = default_context()
    cuda_arch_list = [70, 75, 80, 86]
    cuda_arch = mx.context.gpu_sm_arch(ctx.device_id)
    if cuda_arch not in cuda_arch_list:
        print('Bypassing normalized convolution test on cuda arch {}'.format(cuda_arch))
        return
    n, c, h, w, k = 128, 64, 14, 14, 128
    eps = 1e-4
    momentum = 0.9
    elements, kernel_shape, r, s, stride, pad, out_shape, layout, num_filter, weight_shape, data_shape = get_conv_fprop_shapes(n, c, h, w, k)

    x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    wx = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
    mov_mean = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_var = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    beta, gamma = get_beta_gamma(ctx, x, None, wx, None,
                                 mov_mean, mov_var,
                                 None, None,
                                 (n, h, w, c, k))

    X, B, G, MovMean, MovVar, Wx = create_vars(['X', 'B', 'G', 'MovMean', 'MovVar', 'Wx'])
    args_dict = {'X':x, 'Wx':wx,'B':beta, 'G':gamma}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
    grad_dict = {'X':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx)}
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch2a'}
    Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
    net = mx.sym.BatchNorm(Xout, gamma=G, beta=B,
        moving_mean=MovMean, moving_var=MovVar,
        fix_gamma=False, use_global_stats=False,
        output_mean_var=False, cudnn_off=False,
        axis=-1, eps = eps, momentum = momentum,
        act_type='relu', name="bn_relu")
    net_exe = net.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)
    net_fout = net_exe.forward(is_train=True)
    net_exe_output_dict = copy.deepcopy(net_exe.output_dict)
    dY = mx.ndarray.random.uniform(-0.2, 0.2, out_shape, dtype=np.float16, ctx=ctx )
    dY2 = copy.deepcopy(dY)
    net_bout = net_exe.backward(is_train=True, out_grads=dY)
    net_exe_grad_dict = copy.deepcopy(net_exe.grad_dict)
    del net_exe

    # net2 = [conv+bn(s)] + [bn(a)+relu]
    X, B, G, MovMean, MovVar, Wx = create_vars(['X', 'B', 'G', 'MovMean', 'MovVar', 'Wx'])
    args_dict = {'X':x, 'Wx':wx, 'B':beta, 'G':gamma}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
    grad_dict = {'X':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Wx':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout}
    xout, x_sum, x_sum_sq = mx.sym.NormalizedConvolution(X, weight=Wx,
                                            no_equiv_scale_bias=True,**conv_args)
    x_equiv_scale, x_equiv_bias, x_mean, x_invvar, x_gamma, x_beta = mx.sym.BNStatsFinalize(sum=x_sum,sum_squares=x_sum_sq,gamma=G,beta=B,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMean, moving_var=MovVar,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        output_mean_var=True, elem_count=int(elements))
    net2 = mx.sym.ScaleBiasAddRelu(dataX = xout, eps = eps,
                                        x_equiv_scale = x_equiv_scale, x_equiv_bias = x_equiv_bias,
                                        x_gamma = x_gamma, x_beta = x_beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        dual_scale_bias = False, fused_add = False,
                                        name = "bnapplyrelu"
                                        )
    net2_exe = net2.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)

    # Run netword forward
    net2_fout = net2_exe.forward(is_train=True)

    net2_bout = net2_exe.backward(is_train=True,out_grads=dY2)

    assert_almost_equal(net_exe_output_dict['bn_relu_output'],
                        net2_exe.output_dict['bnapplyrelu_output'],
                        atol=atol['float16'], rtol=rtol['float16'])
    assert_almost_equal(net_exe_grad_dict['X'],
                        net2_exe.grad_dict['X'],
                        atol = 0.3, rtol = .1)

@with_seed()
def test_sbar():

    # Net 1 : conv + bn+add+relu
    # Net 2 : conv+bn(s) + bn(a)+add+relu

    ctx = default_context()
    cuda_arch_list = [70, 75, 80, 86]
    cuda_arch = mx.context.gpu_sm_arch(ctx.device_id)
    if cuda_arch not in cuda_arch_list:
        print('Bypassing normalized convolution test on cuda arch {} supported versions are {}).'.format(
            cuda_arch, cuda_arch_list))
        return

    n, c, h, w, k = 128, 64, 14, 14, 64
    eps = 1e-4
    momentum = 0.9
    elements, kernel_shape, r, s, stride, pad, out_shape, layout, num_filter, weight_shape, data_shape = get_conv_fprop_shapes(n, c, h, w, k)

    x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    z = mx.ndarray.random.uniform(-0.5, 0.5, out_shape, dtype=np.float16, ctx=ctx)
    wx = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)

    mov_mean = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_var = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)

    beta, gamma = get_beta_gamma(ctx, x, z, wx, None,
                                 mov_mean, mov_var,
                                 None, None,
                                 (n, h, w, c, k))

    X, Z, B, G, MovMean, MovVar, Wx = create_vars(['X', 'Z', 'B', 'G', 'MovMean', 'MovVar', 'Wx'])
    args_dict = {'X':x, 'Wx':wx, 'Z':z, 'B':beta, 'G':gamma}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
    grad_dict = {'X':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Z':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Wx':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}

    # net1 = [conv] + [bn+add+relu]
    #                      |
    #   Z------------ ------
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch2a'}

    Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
    net = mx.sym.BatchNormAddRelu(Xout, gamma=G, beta=B,
                                    moving_mean=MovMean, moving_var=MovVar,
                                    fix_gamma=False, use_global_stats=False,
                                    output_mean_var=False, cudnn_off=False,
                                    axis=-1,
                                    eps = eps, momentum = momentum,
                                    addend=Z, name="bn_add_relu")
    net_exe = net.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)
    net_fout = net_exe.forward(is_train=True)
    net_exe_output_dict = copy.deepcopy(net_exe.output_dict)
    dY = mx.ndarray.random.uniform(-0.2, 0.2, out_shape, dtype=np.float16, ctx=ctx)
    dY2 = copy.deepcopy(dY)
    net_bout = net_exe.backward(is_train=True, out_grads=dY)
    net_exe_grad_dict = copy.deepcopy(net_exe.grad_dict)

    del net_exe

    # net2 = [conv+bn(s)] + [bn(a)+add+relu]
    #                               |
    #        [conv+bn(s)] -----------
    X, Z, B, G, MovMean, MovVar, Wx = create_vars(['X', 'Z', 'B', 'G', 'MovMean', 'MovVar', 'Wx'])
    args_dict = {'X':x, 'Wx':wx, 'Z':z, 'B':beta, 'G':gamma}
    aux_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
    grad_dict = {'X':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Z':mx.nd.zeros(data_shape,dtype=np.float16,ctx=ctx),
                 'Wx':mx.nd.zeros(weight_shape,dtype=np.float16,ctx=ctx)}

    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout}
    xout, x_sum, x_sum_sq = mx.sym.NormalizedConvolution(X, weight=Wx,
                                            no_equiv_scale_bias=True,**conv_args)
    x_equiv_scale, x_equiv_bias, x_mean, x_invvar, x_gamma, x_beta = mx.sym.BNStatsFinalize(sum=x_sum,sum_squares=x_sum_sq,gamma=G,beta=B,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMean, moving_var=MovVar,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        output_mean_var=True, elem_count=int(elements))
    net2 = mx.sym.ScaleBiasAddRelu(dataX = xout, dataZ = Z,
                                        eps = eps,
                                        x_equiv_scale = x_equiv_scale, x_equiv_bias = x_equiv_bias,
                                        x_gamma = x_gamma, x_beta = x_beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        dual_scale_bias = False, fused_add = True,
                                        name = "sbar"
                                        )
    net2_exe = net2.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)

    # Run netword forward
    net2_fout = net2_exe.forward(is_train=True)

    net2_bout = net2_exe.backward(is_train=True,out_grads=dY2)

    for i in ['X','Z','Wx']:
        unique, count = np.unique(np.isclose(net_exe_grad_dict[i].asnumpy(),
            net2_exe.grad_dict[i].asnumpy(),
            atol = .1, rtol = 0.1),
            return_counts=True)
        print("{}= {} {}".format(i,unique,count))

    assert_almost_equal(net_exe_output_dict['bn_add_relu_output'],
                        net2_exe.output_dict['sbar_output'],
                        atol=atol['float16'], rtol=rtol['float16'])

    assert_almost_equal(net_exe_grad_dict['Z'],
                        net2_exe.grad_dict['Z'],
                        atol = 0.3, rtol = 0.3)

    assert_almost_equal(net_exe_grad_dict['X'],
                        net2_exe.grad_dict['X'],
                        atol = 0.3, rtol = 0.3)

@with_seed()
def test_dbar_backward():

    ctx = default_context()
    cuda_arch_list = [70, 75, 80, 86]
    cuda_arch = mx.context.gpu_sm_arch(ctx.device_id)
    if cuda_arch not in cuda_arch_list:
        print('Bypassing normalized convolution test on cuda arch {} supported versions are {}).'.format(
            cuda_arch, cuda_arch_list))
        return


    # Test
    # This test is doing two things:-
    # (1) conv + [bn+add+relu] fprp and dgrad
    # (2) [conv+bn(s)] + finalizeStats() + [bn(a)+add+relu]
    # Then it compares the backward gradients for both to make sure our op is correct

    n, c , h, w, k = 128, 64, 14, 14, 64

    elements, kernel_shape, r, s, stride, pad, out_shape, layout, num_filter, weight_shape, data_shape = get_conv_fprop_shapes(n, c, h, w, k)

    eps = 1e-4
    momentum = 0.9

    x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    z = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
    wx = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
    wz = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
    mov_mean = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_var = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_meanz = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)
    mov_varz = mx.nd.zeros((k,), dtype=np.float32, ctx=ctx)

    beta, gamma = get_beta_gamma(ctx, x, z, wx, wz,
                                 mov_mean, mov_var,
                                 mov_meanz, mov_varz,
                                 (n, h, w, c, k))

    betaz = beta.copy()
    gammaz = gamma.copy()

    var_names = ['X', 'Z', 'B', 'G', 'MovMean', 'MovVar', 'Wx', 'Wz', 'Bz', 'Gz', 'MovMeanz', 'MovVarz']
    X, Z, B, G, MovMean, MovVar, Wx, Wz, Bz, Gz, MovMeanz, MovVarz = create_vars(var_names)
    args_dict, aux_dict, grad_dict = create_dicts(ctx, k, data_shape, weight_shape, x, z, wx, wz,
                                                  mov_mean, mov_var, mov_meanz, mov_varz, beta, gamma, betaz, gammaz)
    # net1 = [conv] + [bn+add+relu]
    #                      |
    #   [conv] + [bn] ------
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch2a'}
    conv_args1 = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout, 'name':'res_branch1'}

    Xout = mx.sym.Convolution(X,weight=Wx,no_bias=True,**conv_args)
    Zout = mx.sym.Convolution(Z,weight=Wz,no_bias=True,**conv_args1)
    Zout = mx.sym.BatchNorm(Zout,gamma=Gz, beta=Bz, act_type=None,
                            eps = eps, momentum = momentum,
                            moving_mean=MovMeanz, moving_var=MovVarz,
                            fix_gamma=False, use_global_stats=False,
                            output_mean_var=False, cudnn_off=False,
                            name='bn_branch1', axis=-1)
    net = mx.sym.BatchNormAddRelu(Xout, gamma=G, beta=B,
                                    moving_mean=MovMean, moving_var=MovVar,
                                    fix_gamma=False, use_global_stats=False,
                                    output_mean_var=False, cudnn_off=False,
                                    axis=-1,
                                    eps = eps, momentum = momentum,
                                    addend=Zout, name="bn_add_relu")

    net_exe = net.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)
                        #grad_req=grad_req)
    net_fout = net_exe.forward(is_train=True)
    dY = mx.ndarray.random.uniform(-0.2, 0.2, out_shape, dtype=np.float16, ctx=ctx )
    dY2 = copy.deepcopy(dY)
    net_bout = net_exe.backward(is_train=True, out_grads=dY)
    net_exe_grad_dict = copy.deepcopy(net_exe.grad_dict)
    net_exe_output_dict = copy.deepcopy(net_exe.output_dict)
    del net_exe

    # net2 = [conv+bn(s)] + [bn(a)+add+relu]
    #                               |
    #        [conv+bn(s)] -----------
    var_names = ['X', 'Z', 'B', 'G', 'MovMean', 'MovVar', 'Wx', 'Wz', 'Bz', 'Gz', 'MovMeanz', 'MovVarz']
    X, Z, B, G, MovMean, MovVar, Wx, Wz, Bz, Gz, MovMeanz, MovVarz = create_vars(var_names)
    args_dict, aux_dict, grad_dict = create_dicts(ctx, k, data_shape, weight_shape, x, z, wx, wz,
                                                  mov_mean, mov_var, mov_meanz, mov_varz, beta, gamma, betaz, gammaz)
    conv_args = {'num_filter':num_filter, 'kernel':kernel_shape,
                    'stride':stride, 'pad':pad, 'layout':layout}
    xout, x_sum, x_sum_sq = mx.sym.NormalizedConvolution(X, weight=Wx,
                                            no_equiv_scale_bias=True,**conv_args)
    zout, z_sum, z_sum_sq = mx.sym.NormalizedConvolution(Z, weight=Wz,
                                            no_equiv_scale_bias=True,**conv_args)
    x_equiv_scale, x_equiv_bias, x_mean, x_invvar, x_gamma, x_beta = mx.sym.BNStatsFinalize(sum=x_sum,sum_squares=x_sum_sq,gamma=G,beta=B,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMean, moving_var=MovVar,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        output_mean_var=True, elem_count=int(elements))
    z_equiv_scale, z_equiv_bias, z_mean, z_invvar, z_gamma, z_beta = mx.sym.BNStatsFinalize(sum=z_sum,sum_squares=z_sum_sq,gamma=Gz,beta=Bz,
                                                                        eps = eps, momentum = momentum,
                                                                        moving_mean=MovMeanz, moving_var=MovVarz,
                                                                        fix_gamma = False, use_global_stats = False,
                                                                        output_mean_var=True, elem_count=int(elements),name="bn1")
    net2 = mx.sym.ScaleBiasAddRelu(dataX = xout, dataZ = zout,
                                        eps = eps,
                                        x_equiv_scale = x_equiv_scale, x_equiv_bias = x_equiv_bias,
                                        z_equiv_scale = z_equiv_scale, z_equiv_bias = z_equiv_bias,
                                        x_gamma = x_gamma, x_beta = x_beta,
                                        x_mean = x_mean, x_invvar = x_invvar,
                                        z_gamma = z_gamma, z_beta = z_beta,
                                        z_mean = z_mean, z_invvar = z_invvar,
                                        layout = 'NHWC', act_type='relu',
                                        name = "dsbar"
                                        )
    net2_exe = net2.bind(ctx=ctx,args=args_dict,
                        aux_states=aux_dict,args_grad=grad_dict)

    # Run netword forward
    net2_fout = net2_exe.forward(is_train=True)
    net2_bout = net2_exe.backward(is_train=True,out_grads=dY2)

    for i in net_exe_grad_dict.keys():
        unique, count = np.unique(np.isclose(net_exe_grad_dict[i].asnumpy(),
                                         net2_exe.grad_dict[i].asnumpy(),
                                         atol = 0.1, rtol = 0.1),
                              return_counts=True)
        print("{} = {} {}".format(i,unique,count))

    assert_almost_equal(net_exe_output_dict['bn_add_relu_output'],
                        net2_exe.output_dict['dsbar_output'],
                        atol=atol['float16'], rtol=rtol['float16'])

    for i in ['B','G','Bz','Gz','Z','Wz','X','Wx']:
        print("checking grad tensor for {}".format(i))
        assert_almost_equal(net_exe_grad_dict[i],
                            net2_exe.grad_dict[i],
                            atol = {'G':0.8, 'Gz':0.6}.get(i, 0.5),
                            rtol = 0.3)

if __name__ == '__main__':
    import nose
    nose.runmodule()
