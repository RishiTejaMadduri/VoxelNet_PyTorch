#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[6]:


sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_R_PyTorch/')


# In[7]:


from config import cfg


# In[9]:


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels/2)
        
        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.out_channels),
                                   nn.ReLU()
                                   )
        self.batch_norm = nn.BatchNorm2d(self.units)
        
        
    def forward(self, inputs, mask):
        temp = self.dense(inputs).transpose(1,2)
        pointwise = self.batch_norm(temp).transpose(1,2)
        
        aggregated = torch.max(pointwise, dim=1, keep_dims=True)
        repeated = aggregated.expand(-1, cfg.VOXEL_POINT_COUNT, -1)
        concatenated = torch.cat([pointwise, repeated], dim = 2)
        mask = mask.expand(-1, -1, 2*self.units)
        concatenated = concatenated*mask.float()
        
        return concatenated


# In[10]:


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        self.vfe1 = VFELayer(7,32)
        self.vfe2 = VFELayer(32,64)
        
    def forward(self, feature, number, coordinate):
        batch_size = len(feature)
        feature = torch.cat(feature, dim=0)
        coordinate = torch.cat(coordinate, dim = 0)
        vmax = torch.max(feature, dim=2, keepdim = True)
        mask = (vmax!=0)
        x = self.vfe1(feature, mask)
        x = self.vfe2(x, mask)
        
        voxelwise = torch.max(x, dim = 1)
        outputs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size([batch_size, cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))
        outputs = outputs.to_dense()
        return outputs


# In[ ]:




