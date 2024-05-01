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
import argparse
import datetime
import shutil
import scipy.io
import numpy as np
from tqdm import tqdm
from arguments import get_args_parser
from unet_pre import Unet
from utils.utils_metrics import compute_mIoU, show_results

def evaluation(args, name_classes):
    image_ids = open(os.path.join('dataset_title',args.d_title,'dataset_title_val.txt'),'r').read().splitlines() 
    gt_dir = os.path.join(args.dataset_path, "Simage_"+args.data_level+"_mask/")
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    miou_out_path = "evaluation/miou_out_" + "_d" + args.d_title + "_l" + args.data_level + '_' + str(time_str)
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print("Load model.")
    deeplab = Unet(args)
    print("Load model done.")
    print("Get predict result.")
    for image_id in tqdm(image_ids,ascii=True):
        image = np.array(scipy.io.loadmat(os.path.join(os.path.join(args.dataset_path, "Simage_"+args.data_level+"_RGBD"), image_id))['SimageForPy'])
        image = deeplab.get_miou_png(image)
        image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")
    print("Get miou.")
    hist, IoUs, PAs, _= compute_mIoU(gt_dir, pred_dir, image_ids, args.num_classes, name_classes)
    print("Get miou done.")
    show_results(miou_out_path, hist, IoUs, PAs, name_classes)
    shutil.rmtree(pred_dir) 

if __name__ == '__main__':
    name_classes = ['ceiling', 'floor', 'wall', 'column','beam', 'window', 'door', 'table', 'chair', 'bookcase', 'sofa', 'board', 'clutter']
    parser = argparse.ArgumentParser('UNET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluation(args, name_classes)