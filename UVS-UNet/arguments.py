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
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set unet detector', add_help=False)
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--sync_bn', default=False, type=bool)
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--num_classes', default=13, type=int)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bs', default=16, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--lr_gamma', default=0.85, type=float)
    parser.add_argument('--in_channel', default=4, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--optimizer_type', default="adam", type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--save_dir', default="logs", type=str)
    parser.add_argument('--eval_flag', default=True,type=bool)
    parser.add_argument('--eval_period', default=20, type=int)
    parser.add_argument('--input_shape', default=30722, type=int) #30722 7682
    parser.add_argument('--dataset_path', default="", type=str)
    parser.add_argument('--data_level', default="11", type=str) #11
    parser.add_argument('--d_title', default="1", type=str)
    parser.add_argument('--dice_loss', default=False,type=bool)
    parser.add_argument('--focal_loss', default=True,type=bool)

    return parser