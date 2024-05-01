'''
    Copyright (C) 2024 Authors of UVS-CNNs

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from typing import Optional

class Conv_spherical(nn.modules.conv._ConvNd):
    def __init__(
        self, in_channels: int, out_channels: int, level_tag: int, kernel_size: _size_2_t, 
        stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
    ):
        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)
        super(Conv_spherical, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                             padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.convolution_list = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' + '/convolution_list_'
                                 + str(level_tag) + '.mat')['indexOfNeighborPointListForPy']))).long()

    def _conv_forward(self, input, weight):
        input = input[:,:,self.convolution_list,:] 
        input = torch.squeeze(input, dim=4)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight)


class MaxPool_spherical(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode', 'level_tag'] 
    kernel_size   : _size_2_t
    stride        : _size_2_t
    padding       : _size_2_t
    dilation      : _size_2_t
    return_indices: bool
    ceil_mode     : bool
    level_tag     : int

    def __init__(self, kernel_size: _size_2_t, level_tag: int, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool_spherical, self).__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride 
        self.padding        = padding
        self.dilation       = dilation
        self.return_indices = return_indices
        self.ceil_mode      = ceil_mode
        self.pooling_list   = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' 
                               + '/pooling_list_' + str(level_tag) + 'to' + str(level_tag-2) 
                               + '.mat')['indexOfNeighborPointList_poolingForPy']))).long()
        
    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, level_tag={level_tag}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input: Tensor) -> Tensor:
        input = input[:,:,self.pooling_list,:] 
        input = torch.squeeze(input, dim=4)
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, 
                            self.ceil_mode, self.return_indices)


class AvgPool2d_spherical(nn.modules.pooling._AvgPoolNd):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 
                     'divisor_override']
    kernel_size      : _size_2_t
    stride           : _size_2_t
    padding          : _size_2_t
    ceil_mode        : bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_2_t, level_tag: int, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> None:
        super(AvgPool2d_spherical, self).__init__()
        self.kernel_size       = kernel_size
        self.stride            = stride if (stride is not None) else kernel_size
        self.padding           = padding
        self.ceil_mode         = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override  = divisor_override
        self.pooling_list = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' 
                             + '/pooling_list_' + str(level_tag) + 'to' + str(level_tag-2) 
                             + '.mat')['indexOfNeighborPointList_poolingForPy']))).long()
        
    def forward(self, input: Tensor) -> Tensor:
        input = input[:,:,self.pooling_list,:] 
        input = torch.squeeze(input, dim=4)
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
                            self.count_include_pad, self.divisor_override)
        

class Transpose_conv_spherical(nn.modules.conv._ConvNd):
    def __init__(self, in_channels: int, out_channels: int, level_tag: int, kernel_size: _size_2_t, 
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, 
                 bias: bool = True, padding_mode: str = 'zeros'
                ):
        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)
        super(Transpose_conv_spherical, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                                       padding, dilation, False, _pair(0), groups, bias, padding_mode)
        
        self.transIndex_all     = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' + '/trans_conv_all_' 
                                   + str(level_tag) + 'to' + str(level_tag+2) + '.mat')['indexForTransConv_all']))).long()
        self.transIndex_all     = torch.squeeze(self.transIndex_all, dim=0)

        self.transIndex_zeros   = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' + '/trans_conv_zeros_'
                                   + str(level_tag) + 'to' + str(level_tag+2) + '.mat')['indexForTransConv_zeros']))).long()
        self.transIndex_zeros   = torch.squeeze(self.transIndex_zeros, dim=0)
        
        self.spherical_covIndex = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' + '/convolution_list_' 
                                   + str(level_tag+2) + '.mat')['indexOfNeighborPointListForPy']))).long()
    def _conv_forward(self, input, weight):
        input = input[:,:,self.transIndex_all,:]
        input[:,:,self.transIndex_zeros,:] = 0
        input = input[:,:,self.spherical_covIndex,:] 
        input = torch.squeeze(input, dim=4)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight)