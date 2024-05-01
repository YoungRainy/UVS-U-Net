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
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.unet import UNET

class Unet(object):
    def __init__(self, args):
        self.args = args
        self.generate()            
    def generate(self, onnx=False):

        self.net = UNET(in_channel = self.args.in_channel,out_channel = self.args.out_channel, 
                        num_classes=self.args.num_classes, data_level=self.args.data_level)
        device   = torch.device(self.args.device)
        self.net.load_state_dict(torch.load(self.args.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.args.model_path))
        if not onnx:
            if self.args.device == 'cuda':
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
    
    def get_miou_png(self, image):
        assert(image.shape == (self.args.input_shape, 4))
        image = np.expand_dims(image, 0)
        image = np.transpose(image, [2,1,0])
        image_data = np.expand_dims(image, 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.args.device == 'cuda':
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr.argmax(axis=-1)
            image = Image.fromarray(np.uint8(pr))
        return image


