#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_PyTorch/')
from base.base_dataset import BaseDataSet 
from base.base_dataloader import BaseDataLoader

import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2 
from config import cfg
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from utils.preprocess import process_pointcloud

# In[ ]:

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


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])   # (K, T, 7); K is max number of non-empty voxels; T = 35
        number_list.append(voxel_dict['number_buffer'])     # (K,)
        coordinate = voxel_dict['coordinate_buffer']        # (K, 3)
        coordinate_list.append(np.pad(coordinate, ((0, 0), (1, 0)), mode = 'constant', constant_values = i))



    return batch_size, feature_list, number_list, coordinate_list

class KITTIDataset(BaseDataSet):
    def __init__(self, **kwargs):
        super(KITTIDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.f_rgb = glob.glob(os.path.join(self.root, 'image_2', '*.png'))
        self.f_lidar = glob.glob(os.path.join(self.root, 'velodyne', '*.bin'))
        self.f_label = glob.glob(os.path.join(self.root, 'label_2', '*.txt'))
        
        self.f_rgb.sort()
        self.f_lidar.sort()
        self.f_label.sort()
        
        self.data_tag = [name.split('/')[-1].split('.')[-2] for name in self.f_rgb]
        
        assert len(self.data_tag) != 0, 'Dataset folder is not correct!'
        assert len(self.data_tag) == len(self.f_rgb) == len(self.f_lidar), 'Dataset folder is not correct!'
        
        nums = len(self.f_rgb)
        self.indices = list(range(nums))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def _load_data(self, index, is_testset = False):
        file_id = self.indices[index]
        rgb = cv2.imread(self.f_rgb[file_id])
        raw_lidar = np.fromfile(self.f_lidar[file_id], dtype = np.float32).reshape((-1, 4))
        if not is_testset:
            labels = [line for line in open(self.f_label[file_id], 'r').readlines()]
        else:
            labels = ['']
        
        tag = self.data_tag[index]
        voxel = process_pointcloud(raw_lidar)
        return tag, rgb, raw_lidar, voxel, labels


# In[ ]:


class KITTI(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, collate_fn = collate_fn ):

        kwargs = {
            'root': data_dir
#             'split': split,
#             'val': val,
#             'shuffle': shuffle,
#             'augment': augment
        }
    
        if split == "train":
            self.dataset = KITTIDataset(**kwargs)
        elif split == "val":
            self.dataset = KITTIDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(KITTI, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)    

