# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init

import spconv
import spconv.functional as Fsp
from spconv import ops
from spconv.modules import SparseModule


class SparseGroup(SparseModule):
    __constants__ = [
        'stride', 'padding', 'dilation', 'subm', 'output_padding', 
    ]

    def __init__(self,
                 ndim,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False,
                 output_padding=0,
                 indice_key=None,
                 use_hash=False,
                 algo=ops.ConvAlgo.Native):
        super(SparseGroup, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.group1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.subm = subm
        self.indice_key = indice_key
        self.use_hash = use_hash
        self.algo = algo.value


    def forward(self, input):
        """
        input.features: (N_b, C)
        output.features: (N_b, m, C)
        """
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding,
                self.dilation)
        else:
            out_spatial_shape = spatial_shape
        
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        if self.group1x1:
            # (N_b, C) -> (N_b, 1, C)
            features = features.unsequeeze(dim=1)
            out_tensor = SparseConvTensor(features, input.indices,
                                          input.spatial_shape,
                                          input.batch_size)
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor


        datas = input.find_indice_pair(self.indice_key)

        if self.indice_key is not None and datas is not None:
            outids, _, indice_pairs, indice_pair_num, _ = datas
        else:
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                indices,
                batch_size,
                spatial_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
                self.subm,
                False, # transpose=False
                grid=input.grid,
                use_hash=self.use_hash)
            input.indice_dict[self.indice_key] = (outids, indices,
                                                    indice_pairs,
                                                    indice_pair_num,
                                                    spatial_shape)

        if self.subm:
            out_features = Fsp.indice_subm_group(features,
                                                indice_pairs.to(device),
                                                indice_pair_num,
                                                outids.shape[0],
                                                self.algo)


            # """
            # print for check
            # print('==> group outids.shape: ', outids.shape)
            # print('==> group outids: ', outids) 
            # 这个就是return的tensor的indices
            # print('==> group indice_pairs.shape: ', indice_pairs.shape)
            # print('==> group indice_pairs: ', indice_pairs)
            # 这个的shape是[2, 3*3*3, Length]
            # dim0: kernel_position idx 共3*3*3=27维
            # dim1: (input, output) 共2维
            # dim2: inputlen or outputlen? 好像是output的len idx            
            # print('==> group indice_pair_num: ', indice_pair_num)
            # 这个是每个kernel位置的pair长度
            # """

        else:
            out_features = Fsp.indice_group(features,
                                            indice_pairs.to(device),
                                            indice_pair_num,
                                            outids.shape[0], 
                                            self.algo)

        # 这个再转置一下
        # (N_ks, N_b, C) -> (N_b, N_ks, C)
        out_features = out_features.permute(1,0,2)

        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor




class SparseGroup3d(SparseGroup):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 use_hash=False,
                 algo=ops.ConvAlgo.Native):
        super(SparseGroup3d, self).__init__(3,
                                           in_channels,
                                           kernel_size,
                                           stride,
                                           padding,
                                           dilation,
                                           indice_key=indice_key,
                                           use_hash=use_hash,
                                           algo=algo)





class SubMGroup3d(SparseGroup):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 use_hash=False,
                 algo=ops.ConvAlgo.Native):
        super(SubMGroup3d, self).__init__(3,
                                         in_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         True,
                                         indice_key=indice_key,
                                         use_hash=use_hash,
                                         algo=algo)

