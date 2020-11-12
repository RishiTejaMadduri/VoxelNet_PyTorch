#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


from group_pointcloud import FeatureNet
from rpn import MiddleAndRpn


# In[5]:


sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_R_PyTorch/')


# In[ ]:


from utils import *
from config import cfg


# In[7]:


small_addon_for_BCE = 1e-6


# In[ ]:


class RPN3D(nn.Module):
    def __init__(self, cls = 'Car', alpha = 1.5, beta = 1, sigma = 3):
        super(RPN3D, self).__init__()
        
        self.cls = cls
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.feature = FeatureNet()
        self.rpn = MiddleAndRpn()
        
        self.anchors = cal_anchors()
        self.rpn_output_shape = self.rpn.output_shape
        
    def forward(self, inputs):
        self.tag = inputs[0]
        self.label = inputs[1]
        self.vox_features = inputs[2]
        self.vox_coordinate = inputs[3]
        
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(self.label, self.rpn_output_shape, self.anchors, cls = cfg.DETECT_OBJ)
        pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        
        return pos_equal_one, pos_equal_one_for_reg, pos_equal_one_sum, neg_equal_one_sum, targets
        

