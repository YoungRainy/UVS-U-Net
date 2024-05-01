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
import os
import numpy as np
import torch
import scipy.io
from PIL import Image
from torch.utils.data.dataset import Dataset

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, data_level):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.data_level         = data_level

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        simage  = np.array(scipy.io.loadmat(os.path.join(os.path.join(self.dataset_path, 
                                            "Simage_"+self.data_level+"_RGBD"), name))['SimageForPy'])
        gt_mask = np.array(scipy.io.loadmat(os.path.join(os.path.join(self.dataset_path, 
                                            "Simage_"+self.data_level+"_mask"), name))['SimageForPy'])
        assert(simage.shape == (self.input_shape, 4))
        simage = np.expand_dims(simage, 0)
        simage = np.transpose(simage, [2,1,0])
        gt_mask[gt_mask >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[gt_mask.reshape([-1])]
        seg_labels = seg_labels.reshape((self.input_shape, 1, self.num_classes + 1))
        return simage, gt_mask, seg_labels

def dataset_collate(batch):
    simages    = []
    gt_masks   = []
    seg_labels = []
    for img, mask, labels in batch:
        simages.append(img)
        gt_masks.append(mask)
        seg_labels.append(labels)
    simages    = torch.from_numpy(np.array(simages)).type(torch.FloatTensor)
    gt_masks   = torch.from_numpy(np.array(gt_masks)).type(torch.LongTensor)
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return simages, gt_masks, seg_labels
