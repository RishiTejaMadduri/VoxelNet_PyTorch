#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from config import cfg
from utils.data_aug import aug_data
# In[3]:



class BaseDataSet(Dataset):
    def __init__(self, root, shuffle = False, augment=False, is_testset=False, return_id = False):
        
        self.root = root
        self.aug = augment
        self.shuffle = shuffle
        self.is_testset = is_testset
        self.crop_size = (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)
        self.indices = []
        self._set_files()
        self.return_id = return_id
        
        
    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _augmentation(self, tag, rgb, raw_lidar, voxel, labels):
        return aug_data(tag, rgb, raw_lidar, voxel, labels)
    

 
    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, index, is_testset = False):
        
        if not self.is_testset:
            print("Train Set")
            tag, rgb, raw_lidar, voxel, labels = self._load_data(index)
            rgb = cv2.resize(rgb, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            
            if self.aug:
                ret = self._augmentation(tag, rgb, raw_lidar, voxel, labels)
            else:
                ret = [tag, rgb, voxel, labels]
        else:
            print("Valid set")
            tag, rgb, raw_lidar, voxel, labels = self._load_data(index)
            rgb = rgb.resize((cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), resample = Image.NEAREST)
            ret = [tag, rgb, raw_lidar, voxel, labels]
            
        return ret

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

