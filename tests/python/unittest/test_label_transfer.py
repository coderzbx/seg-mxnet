# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0, (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0,
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import copy
import os
import re
import mxnet as mx
import numpy as np
from common import models
from mxnet.test_utils import rand_ndarray
import pickle as pkl

def test_symbol_basic():
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        m.list_outputs()

def test_symbol_transfer():
    ctx = mx.gpu(0)
    batch_data_shape = (4, 100, 100)
    input_data = mx.sym.Variable(name='label', shape=(100, 100))
    batch_label_shape = (batch_data_shape[0], batch_data_shape[1], batch_data_shape[2])
    l = mx.nd.array(np.random.randint(10, size=batch_label_shape))

    # input data
    l = l.as_in_context(ctx)
    print "l:", l

    sym = mx.symbol.LabelTransfer(label=input_data, ignore_label=255, label_ids=(1.0,1.0,1.0,1.0,1.0,2.0,2.0,3.0))
    exec1 = sym.bind(ctx, args = [l])
    exec1.forward(is_train=True)
    print "o:", exec1.outputs[0]

    sym = mx.symbol.LabelTransfer(label=input_data, ignore_label=255, label_ids=(8.0,4.0,6.0,0.0,0.0,9.0,4.0,2.0))
    exec1 = sym.bind(ctx, args = [l])
    exec1.forward(is_train=True)
    print "o:", exec1.outputs[0]

    #print "o/25", exec1.outputs[0] / 25.0
    # label_trans = 

def test_ndarray_transfer():
    # x = mx.nd.ones((10, 10))
    # x = mx.nd.array([   
    #     [ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  9.0,  2.0,  2.0,  2.0],
    #     [ 2.0,  0.0,  2.0,  2.0,  2.0,  2.0,  8.0,  2.0,  5.0,  2.0],
    #     [ 2.0,  0.0,  2.0,  2.0,  2.0,  4.0,  2.0,  5.0,  5.0,  2.0],
    #     [ 1.0,  2.0,  1.0,  1.0,  12.0,  2.0,  2.0,  2.0,  2.0,  0.0],
    #     [ 2.0,  2.0,  255.0,  1.0,  1.0,  2.0,  2.0,  2.0,  0.0,  2.0],
    #     [ 1.0,  2.0,  1.0,  2.0,  2.0,  13.0,  2.0,  2.0,  2.0,  2.0],
    #     [ 1.0,  2.0,  2.0,  2.0,  8.0,  1.0,  2.0,  15.0,  2.0,  2.0],
    #     [ 2.0,  2.0,  0.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0],
    #     [ 2.0,  0.0,  2.0,  255.0,  6.0,  2.0,  4.0,  2.0,  5.0,  5.0],
    #     [ 0.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]
    #     ], mx.gpu())
    # x = mx.nd.ones((24480, 20480), mx.gpu())
    x = mx.nd.ones((24480, 20480))
    label_trans1 = mx.nd.LabelTransfer(label=x, ignore_label=255, label_ids=(0.0,2.0,3.0,1.0,1.0,2.0,2.0,3.0))
    label_trans2 = mx.nd.LabelTransfer(label=x, ignore_label=255, label_ids=(8.0,4.0,6.0,0.0,0.0,9.0,4.0,2.0))
    # print(label_trans1)
    # print(label_trans2)


if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    test_symbol_transfer()
