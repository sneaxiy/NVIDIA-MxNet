#!/bin/bash

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


set -e

echo "BUILD make"

WITH_CAFFE_PLUGIN=0

if [ "$WITH_CAFFE_PLUGIN" == "1" ]; then
# Check out caffe
  git clone https://github.com/BVLC/caffe
  mkdir -p caffe/build
  cd caffe/build
  cmake ..
  make -j$(nproc)
  cd ../..
fi

cp make/config.mk .
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
echo "DEV=0" >> config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" >> config.mk
echo "USE_CPP_PACKAGE=1" >> config.mk

# Settings suitable to rebuild MXNet within NVIDIA's container
echo "USE_MKLDNN=0" >> config.mk
echo "USE_LAPACK_PATH=/usr/lib/x86_64-linux-gnu" >> config.mk
echo "USE_BLAS=openblas" >> config.mk
echo "USE_HOROVOD=1" >> config.mk
echo "USE_MPI_PATH=/usr/local/mpi" >> config.mk
echo "USE_NCCL=1" >> config.mk

if [ "$WITH_CAFFE_PLUGIN" == "1" ]; then
    echo "CAFFE_PATH = $(pwd)/caffe" >> config.mk
    echo "MXNET_PLUGINS += plugin/caffe/caffe.mk" >> config.mk
fi

user=`id -u -n`

make -j$(nproc)

export PYTHONPATH=${PWD}/python

if [ ! -x "$(which nosetests)" ]; then
    pip install nose scipy
fi
echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit 1
nosetests --verbose tests/python/gpu || exit 1
nosetests --verbose tests/python/train || exit 1

if [ -x "$(which mvn)" ]; then
    echo "BUILD scala_test"
    export PATH=$PATH:/opt/apache-maven/bin
    cd scala-package
    mvn integration-test || exit 1
else
    echo "No mvn, bypassing scala test."
fi
