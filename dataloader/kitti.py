#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import cv2
import numpy as np
import os
import sys
sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_R_PyTorch/')
import glob
import math
import multiprocessing

import pyximport
pyximport.install()

from config import cfg
from utils.data_aug import aug_data
from utils.preprocess import process_pointcloud


# %%


import torch
import torch.utils.data as Data


# %%


class Processor:
    def __init__(self, data_tag, f_rgb, f_lidar, f_label, data_dir, aug, is_testset):
        self.data_tag=data_tag
        self.f_rgb = f_rgb
        self.f_lidar = f_lidar
        self.f_label = f_label
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset
    
    def __call__(self,load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            #rgb.append( cv2.imread(f_rgb[load_index]) )
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4))
            if not self.is_testset:
                labels = [line for line in open(self.f_label[load_index], 'r').readlines()]
            else:
                labels = ['']
            tag = self.data_tag[load_index]
            voxel = process_pointcloud(raw_lidar)
            ret = [tag, rgb, raw_lidar, voxel, labels]
            
        return ret


# %%


class KITTI_Loader():
    def __init__(self, data_dir, shuffle = False, aug = False, is_testset = False):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.aug = aug
        self.is_testset = is_testset
        
        self.f_rgb = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
        self.f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))
        self.f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
        
        
        self.f_rgb.sort()
        self.f_lidar.sort()
        self.f_label.sort()
        
        self.data_tag = [name.split('/')[-1].split('.')[-2] for name in self.f_rgb]
        
        assert len(self.data_tag) != 0, "dataset folder is not correct"
        assert len(self.data_tag) == len(self.f_rgb) == len(self.f_lidar) , "dataset folder is not correct"
        
        
        nums = len(self.f_rgb)
        self.indices = list(range(nums))
        if shuffle:
            np.random.shuffle(self.indices)

#         num_batches = int(math.floor( nums / float(batch_size) ))

        self.proc=Processor(self.data_tag, self.f_rgb, self.f_lidar, self.f_label, self.data_dir, self.aug, self.is_testset)
        
    def __getitem__(self, index):
        # A list of [tag, rgb, raw_lidar, voxel, labels]
        ret = self.proc(self.indices[index])

        return ret


    def __len__(self):
        return len(self.indices)
    
    


# %%


def collate_fn(rets):
    tag = [ret[0] for ret in rets]
    rgb = [ret[1] for ret in rets]
    raw_lidar = [ret[2] for ret in rets]
    voxel = [ret[3] for ret in rets]
    labels = [ret[4] for ret in rets]
    
    # Only for voxel
    _, vox_feature, vox_number, vox_coordinate = build_input(voxel)
    
    res = (
           np.array(tag),
           np.array(labels),
           [torch.from_numpy(x) for x in vox_feature],
           np.array(vox_number),
           [torch.from_numpy(x) for x in vox_coordinate],
           np.array(rgb),
           np.array(raw_lidar)
           )

    return res

# %%


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(np.pad(coordinate, ((0, 0), (1, 0)), mode='constant', constant_values=i))

    return batch_size, feature_list, number_list, coordinate_list


# %%




