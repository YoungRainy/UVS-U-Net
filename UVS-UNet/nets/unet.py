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
from nets.spherical_cnn_method import MaxPool_spherical,Transpose_conv_spherical,Conv_spherical

class conv_block(nn.Module):
    def __init__(self,in_c,out_c,level_tag):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            Conv_spherical(in_c, out_c, kernel_size=(1,7), level_tag=level_tag),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            Conv_spherical(out_c, out_c, kernel_size=(1,7), level_tag=level_tag, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Downsample(nn.Module):
    def __init__(self,channel, level_tag):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            MaxPool_spherical(kernel_size=(1,4), level_tag=level_tag),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channel, level_tag):
        super(Upsample, self).__init__()

        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)
        self.upsample_index = (torch.from_numpy(np.array(scipy.io.loadmat('simg_data' + '/upsampling_list_' \
                               + str(level_tag) + 'to' + str(level_tag+2))['indexOfUpsamplingForPy']))).long()

        # self.upconv = Transpose_conv_spherical(in_channels=channel, out_channels=channel, 
        #                                        level_tag=level_tag, kernel_size=(1,7))
        
    def _upsample(self,x,upsample_index):
        x_0 = x[:,:,upsample_index[:,0],:]
        x_1 = x[:,:,upsample_index[:,1],:]
        return 0.5 * (x_0 + x_1)
    
    def forward(self,x,featuremap):
        x = self._upsample(x,self.upsample_index)
        # x = self.upconv(x)
        x = self.conv1(x)
        x = torch.cat((x,featuremap),dim=1)
        return x

class UNET(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes, data_level):
        super(UNET, self).__init__()

        llist = [0,1,3,5,7,9,11,13,15,17]
        sl = (int(data_level)+1)//2
        self.layer1  = conv_block(in_channel, out_channel, llist[sl])
        self.layer2  = Downsample(out_channel, llist[sl])
        self.layer3  = conv_block(out_channel, out_channel*2, llist[sl-1])
        self.layer4  = Downsample(out_channel*2, llist[sl-1])
        self.layer5  = conv_block(out_channel*2, out_channel*4, llist[sl-2])
        self.layer6  = Downsample(out_channel*4, llist[sl-2])
        self.layer7  = conv_block(out_channel*4, out_channel*8, llist[sl-3])
        self.layer8  = Downsample(out_channel*8, llist[sl-3])
        self.layer9  = conv_block(out_channel*8, out_channel*16, llist[sl-4])
        self.layer10 = Upsample(out_channel*16, llist[sl-4])
        self.layer11 = conv_block(out_channel*16, out_channel*8, llist[sl-3])
        self.layer12 = Upsample(out_channel*8, llist[sl-3])
        self.layer13 = conv_block(out_channel*8, out_channel*4, llist[sl-2])
        self.layer14 = Upsample(out_channel*4, llist[sl-2])
        self.layer15 = conv_block(out_channel*4, out_channel*2, llist[sl-1])
        self.layer16 = Upsample(out_channel*2, llist[sl-1])
        self.layer17 = conv_block(out_channel*2, out_channel, llist[sl])
        
        self.layer18 = nn.Conv2d(out_channel, num_classes, kernel_size=(1,1), stride=1)

    def forward(self,x):
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x,f4)
        x = self.layer11(x)
        x = self.layer12(x,f3)
        x = self.layer13(x)
        x = self.layer14(x,f2)
        x = self.layer15(x)
        x = self.layer16(x,f1)
        x = self.layer17(x)
        x = self.layer18(x)
        
        return x