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
import random
import datetime
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from arguments import get_args_parser
from nets.unet import UNET
from nets.unet_training import weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, dataset_collate
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from get_miou import evaluation

def main(args):
    name_classes = ['ceiling', 'floor', 'wall', 'column','beam', 'window', 'door', 'table', 'chair', 'bookcase', 'sofa', 'board', 'clutter']
    label_ratio = [0.17384037404789865,  0.17002664771895903,  0.2625963729249342,   0.019508096683310605, 0.014504436907968913, 
                   0.016994731594287146, 0.08321331842901526,  0.020731298851232174, 0.028626771620973622, 0.048004778186652164, 
                   0.002515611224467519, 0.017173225930738712, 0.087541966989014]
    cls_weights = 1 / np.log(1.02 + np.array(label_ratio, np.float32))
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device(args.device)
        local_rank = 0
    model = UNET(in_channel = args.in_channel,out_channel = args.out_channel, num_classes=args.num_classes, 
                 data_level=args.data_level)
    weights_init(model)
    if args.model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(args.model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(args.save_dir, "logs_d" + args.d_title + "_l" + args.data_level + '_' + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)
    else:
        loss_history = None
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    if args.device == 'cuda':
        if args.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], 
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    image_ids_label = open(os.path.join('dataset_title',args.d_title,'dataset_title_train.txt'), encoding='utf-8').read().strip().split()
    random.shuffle(image_ids_label)
    Simage_ftrain  = open(os.path.join('dataset_title',args.d_title,'dataset_title_train_random.txt'), 'w')
    for i in range(len(image_ids_label)):  
        Simage_ftrain.write('%s\n'%(image_ids_label[i]))
    print("random train size",len(image_ids_label) )
    with open(os.path.join('dataset_title',args.d_title,'dataset_title_train_random.txt'),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join('dataset_title',args.d_title,'dataset_title_val.txt'),"r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    if local_rank == 0:
        show_config(
            device=args.device, num_classes=args.num_classes, model_path=args.model_path, input_shape=args.input_shape, 
            in_channel=args.in_channel, out_channel=args.out_channel,epochs=args.epochs, bs=args.bs, lr=args.lr, 
            lr_drop=args.lr_drop, lr_gamma=args.lr_gamma, optimizer_type=args.optimizer_type, 
            momentum=args.momentum, data_level=args.data_level, d_title=args.d_title,save_period=args.save_period, 
            log_dir=log_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
        )
    if local_rank == 0:
        eval_callback = EvalCallback(model, args.input_shape, args.num_classes, name_classes, val_lines, args.dataset_path, 
                                     log_dir, args.device, eval_flag=args.eval_flag, period=args.eval_period, 
                                     data_level=args.data_level)
    else:
        eval_callback = None
    train_dataset = UnetDataset(train_lines, args.input_shape, args.num_classes, True, args.dataset_path, args.data_level)
    val_dataset = UnetDataset(val_lines, args.input_shape, args.num_classes, False, args.dataset_path, args.data_level)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    batch_size = args.bs
    optimizer = {
        'adam': optim.Adam(model.parameters(), args.lr, betas = (args.momentum, 0.999), 
                             weight_decay = args.weight_decay),
        'sgd' : optim.SGD(model.parameters(), args.lr, momentum = args.momentum, nesterov=True, 
                            weight_decay = args.weight_decay)
    }[args.optimizer_type]
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_gamma)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("The dataset is too small to continue training. Please expand the dataset.")
    if args.distributed:
            batch_size = batch_size // ngpus_per_node
    gen = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, 
                     pin_memory=True, drop_last = True, collate_fn = dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, 
                         pin_memory=True, drop_last = True, collate_fn = dataset_collate, sampler=val_sampler)
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,epoch_step, epoch_step_val, 
                      gen, gen_val, args.epochs, args.device, args.dice_loss, args.focal_loss, 
                      cls_weights, args.num_classes, args.fp16, scaler, args.save_period, log_dir, local_rank)
        lr_scheduler.step()
        if args.distributed: dist.barrier()
    args.model_path = os.path.join(log_dir, 'best_epoch_weights.pth')
    evaluation(args, name_classes)
    print(args.model_path + ' evaluation ended !')
    if local_rank == 0:
        show_config(
            device=args.device, num_classes=args.num_classes, model_path=args.model_path, input_shape=args.input_shape, 
            in_channel=args.in_channel, out_channel=args.out_channel,epochs=args.epochs, bs=args.bs, lr=args.lr, 
            lr_drop=args.lr_drop, lr_gamma=args.lr_gamma, optimizer_type=args.optimizer_type, 
            momentum=args.momentum, data_level=args.data_level, d_title=args.d_title,save_period=args.save_period, 
            log_dir=log_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser('UNET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
    
