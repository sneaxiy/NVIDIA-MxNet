# coding: utf-8
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

"""Weight updating functions."""
import warnings
import numpy
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray, full
import os

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

__all__ = ['BERTAdam']

@register
class BERTAdam(Optimizer):
    """The Adam optimizer with weight decay regularization for BERT.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)

    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In BERTAdam, the weight decay term decoupled from gradient
    based update.

    This is also slightly different from the AdamW optimizer described in
    *Fixing Weight Decay Regularization in Adam*, where the schedule multiplier and
    learning rate is decoupled, and the bias-correction terms are removed.
    The BERTAdam optimizer uses the same learning rate to apply gradients
    w.r.t. the loss and weight decay.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`mxnet.optimizer.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 **kwargs):
        super(BERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.aggregate_num = max(1,min(50,int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE',"4"))))

    def create_state_multi_precision(self, index, weight):
        """multi-precision state creation function."""
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn('Accumulating with float16 in optimizer can lead to '
                          'poor accuracy or slow convergence. '
                          'Consider using multi_precision=True option of the '
                          'BERTAdam optimizer')
        return self.create_state(index, weight)

    def create_state(self, _, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), #mean
                zeros(weight.shape, weight.context, dtype=weight.dtype)) #variance

    def update(self, index, weight, grad, state):
        """update function"""
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        """multi-precision update function"""
        use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        """update function"""
        try:
            from mxnet.ndarray.contrib import adamw_update, multi_adamw_update
        except ImportError:
            raise ImportError('Failed to import nd.contrib.adamw_update from MXNet. '
                              'BERTAdam optimizer requires mxnet>=1.5.0b20190220. '
                              'Please upgrade your MXNet version. For example: '
                              'pip install mxnet-cu90 --pre. Otherwise, please consider '
                              'Adam optimizer with different hyper-parameters.')
        if multi_precision:
            try:
                from mxnet.ndarray.contrib import mp_adamw_update, multi_mp_adamw_update
            except ImportError:
                raise ImportError('Failed to import '
                                    'nd.contrib.mp_adamw_update from MXNet. '
                                    'BERTAdam optimizer requires mxnet>=1.5.0b20190220. '
                                    'Please upgrade your MXNet version. For example: '
                                    'pip install mxnet-cu90 --pre. Otherwise, please consider '
                                    'Adam optimizer with different hyper-parameters.')

        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')

        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        # pylint: disable=access-member-before-definition
        if not isinstance(self.rescale_grad, NDArray):
            self.rescale_grad = full(shape=(1,), val=self.rescale_grad, ctx=weight.context)
        else:
            self.rescale_grad = self.rescale_grad.as_in_context(weight.context)

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            import numpy as np
            current_index = 0
            while current_index < len(indices):
                sidx = current_index
                eidx = min(current_index + self.aggregate_num, len(indices))
                if not multi_precision:
                    mean, var = list(zip(*states[sidx:eidx]))
                    multi_adamw_update(weights[sidx:eidx],
                                        grads[sidx:eidx],
                                        mean, var,
                                        out=weights[sidx:eidx],
                                        size=len(weights[sidx:eidx]),
                                        lrs=list(np.ones(len(weights[sidx:eidx]))), #set to ones
                                        wds=wds[sidx:eidx],
                                        etas=lrs[sidx:eidx],
                                        **kwargs)
                else:
                    mean_var = list(zip(*states[sidx:eidx]))[0]
                    tmean_var = list(zip(*mean_var))
                    mean=tmean_var[0]
                    var=tmean_var[1]

                    multi_mp_adamw_update(weights[sidx:eidx],
                                            grads[sidx:eidx],
                                            mean, var,
                                            list(zip(*states[sidx:eidx]))[1],
                                            out=weights[sidx:eidx],
                                            size=len(weights[sidx:eidx]),
                                            lrs=list(np.ones(len(weights[sidx:eidx]))), #set to ones
                                            wds=wds[sidx:eidx],
                                            etas=lrs[sidx:eidx],
                                            **kwargs)
                current_index += self.aggregate_num
        else:
            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    mean, var = state
                    adamw_update(weight, grad, mean, var, out=weight,
                        lr=1, wd=wd, eta=lr, **kwargs)
                else:
                    mean, var = state[0]
                    mp_adamw_update(weight, grad, mean, var, state[1], out=weight,
                                lr=1, wd=wd, eta=lr, **kwargs)
