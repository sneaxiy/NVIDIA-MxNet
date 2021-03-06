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

# pylint: disable=line-too-long
"""Baidu ernie data, contains XNLI."""

__all__ = ['BaiduErnieXNLI', 'BaiduErnieLCQMC', 'BaiduErnieChnSentiCorp']

import os
import sys
import tarfile
from gluonnlp.data.dataset import TSVDataset
from gluonnlp.data.registry import register
from gluonnlp.base import get_home_dir
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

_baidu_ernie_data_url = 'https://ernie.bj.bcebos.com/task_data_zh.tgz'

class _BaiduErnieDataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        download_data_path = os.path.join(self._root, 'task_data.tgz')
        if not os.path.exists(download_data_path):
            urlretrieve(_baidu_ernie_data_url, download_data_path)
            tar_file = tarfile.open(download_data_path, mode='r:gz')
            tar_file.extractall(self._root)
        filename = os.path.join(self._root, 'task_data', dataset_name, '%s.tsv' % segment)
        super(_BaiduErnieDataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class BaiduErnieXNLI(_BaiduErnieDataset):
    """ The XNLI dataset redistributed by Baidu
    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.

    Original from:
    Conneau, Alexis, et al. "Xnli: Evaluating cross-lingual sentence representations."
        arXiv preprint arXiv:1809.05053 (2018).
        https://github.com/facebookresearch/XNLI

    Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
        License details: https://creativecommons.org/licenses/by-nc/4.0/

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/baidu_ernie_task_data'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> xnli_dev = BaiduErnieXNLI('dev', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(xnli_dev)
    2490
    >>> len(xnli_dev[0])
    3
    >>> xnli_dev[0]
    ['?????????????????????????????????', '????????????????????????????????????????????????????????????', 'neutral']
    >>> xnli_test = BaiduErnieXNLI('test', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(xnli_test)
    5010
    >>> len(xnli_test[0])
    2
    >>> xnli_test[0]
    ['??????????????????????????????????????????????????????????????????????????????', '?????????????????????????????????']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'baidu_ernie_data'),
                 return_all_fields=False):
        A_IDX, B_IDX, LABEL_IDX = 0, 1, 2
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(BaiduErnieXNLI, self).__init__(root, 'xnli', segment,
                                             num_discard_samples=num_discard_samples,
                                             field_indices=field_indices)

@register(segment=['train', 'dev', 'test'])
class BaiduErnieLCQMC(_BaiduErnieDataset):
    """ The LCQMC dataset original from:
    Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, Buzhou Tang,
        LCQMC: A Large-scale Chinese Question Matching Corpus,COLING2018.
    No license granted. You can request a private license via
    http://icrc.hitsz.edu.cn/LCQMC_Application_Form.pdf
    The code fits the dataset format which was redistributed by Baidu in ERNIE repo.
    (Baidu does not hold this version any more.)

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/baidu_ernie_task_data'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> lcqmc_dev = BaiduErnieLCQMC('dev', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(lcqmc_dev)
    8802
    >>> len(lcqmc_dev[0])
    3
    >>> lcqmc_dev[0]
    ['?????????????????????????????????', '????????????????????????????????????', '1']
    >>> lcqmc_test = BaiduErnieLCQMC('test', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(lcqmc_test)
    12500
    >>> len(lcqmc_test[0])
    2
    >>> lcqmc_test[0]
    ['???????????????????????????', '????????????????????????']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'baidu_ernie_data'),
                 return_all_fields=False):
        A_IDX, B_IDX, LABEL_IDX = 0, 1, 2
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(BaiduErnieLCQMC, self).__init__(root, 'lcqmc', segment,
                                              num_discard_samples=num_discard_samples,
                                              field_indices=field_indices)


@register(segment=['train', 'dev', 'test'])
class BaiduErnieChnSentiCorp(_BaiduErnieDataset):
    """ The ChnSentiCorp dataset redistributed by Baidu
    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.

    Original from Tan Songbo (Chinese Academy of Sciences, tansongbo@software.ict.ac.cn).

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/baidu_ernie_task_data'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> chnsenticorp_dev = BaiduErnieChnSentiCorp('dev', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(chnsenticorp_dev)
    1200
    >>> len(chnsenticorp_dev[0])
    2
    >>> chnsenticorp_dev[2]
    ['?????????????????????????????????????????????????????????????????????.......???????????????????????????????????????...', '0']
    >>> chnsenticorp_test = BaiduErnieChnSentiCorp('test', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(chnsenticorp_test)
    1200
    >>> len(chnsenticorp_test[0])
    1
    >>> chnsenticorp_test[0]
    ['??????????????????????????????????????????????????????????????????????????????']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'baidu_ernie_data'),
                 return_all_fields=False):
        LABEL_IDX, A_IDX = 0, 1
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(BaiduErnieChnSentiCorp, self).__init__(root, 'chnsenticorp', segment,
                                                     num_discard_samples=num_discard_samples,
                                                     field_indices=field_indices)
