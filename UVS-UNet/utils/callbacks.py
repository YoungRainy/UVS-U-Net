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
import matplotlib
import torch
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import scipy.io
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from .utils_metrics import compute_mIoU

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        self.miou       = []
        self.mPA        = []
        os.makedirs(self.log_dir)

    def append_loss(self, epoch, loss, val_loss, miou, mPA):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.miou.append(miou)
        self.mPA.append(mPA)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
            f.write(str(miou))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_mPA.txt"), 'a') as f:
            f.write(str(mPA))
            f.write("\n")
        # self.writer.add_scalar('loss', loss, epoch)
        # self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', 
                     linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', 
                     linewidth = 2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
 
       
class EvalCallback():
    def __init__(self, net, input_shape, num_classes, name_classes, image_ids, dataset_path, log_dir, cuda, data_level,
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        self.net           = net
        self.input_shape   = input_shape
        self.num_classes   = num_classes
        self.image_ids     = image_ids
        self.dataset_path  = dataset_path
        self.log_dir       = log_dir
        self.cuda          = cuda
        self.data_level    = data_level
        self.miou_out_path = miou_out_path
        self.eval_flag     = eval_flag
        self.period        = period
        self.name_classes  = name_classes
        self.image_ids     = [image_id.split()[0] for image_id in image_ids]
        self.mious         = [0]
        self.epoches       = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "eval_miou_mPA_Accuracy.txt"), 'a') as f:
                f.write(str(0) + '   '+str(0)+'   '+str(0))
                f.write("\n")

    def get_miou_png(self, image):
        assert(image.shape == (self.input_shape, 4))
        image = np.expand_dims(image, 0)
        image = np.transpose(image, [2,1,0])
        image_data = np.expand_dims(image, 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda == 'cuda':
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr.argmax(axis=-1)
            image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir   = os.path.join(self.dataset_path, "Simage_"+self.data_level+"_mask/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids,ascii=True):
                image = np.array(scipy.io.loadmat(os.path.join(os.path.join(self.dataset_path, 
                                "Simage_"+self.data_level+"_RGBD"), image_id))['SimageForPy'])
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
            print("Calculate miou.")
            _, IoUs,PAs, Accuracy = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)
            temp_miou = np.nanmean(IoUs) * 100
            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "eval_miou_mPA_Accuracy.txt"), 'a') as f:
                f.write(str(temp_miou) + '   ' + str(round(np.nanmean(PAs) * 100, 2)) + '   ' + Accuracy)
                f.write("\n")
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "eval_miou.png"))
            plt.cla()
            plt.close("all")
            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)        
        
