#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import torch
import numpy as np
import torch.nn as nn


# In[5]:


sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_R_PyTorch/')


# In[6]:


from config import cfg


# In[7]:


small_addon_for_BCE = 1e-6


# In[8]:


class ConvMD(nn.Module):
    def __init__(M, cin, cout, kernel, stride, padding, bn=True, activation = True):
        super(ConvMD, slef).__init__()
        self.M = M
        self.cin = cin
        self.cout = cout
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.bn = bn
        self.activation = activation
        
        if self.M == 2:
            self.conv = nn.Conv2d(self.cin, self.cout, self.kernel, self.stride, self.padding)
            if bn == True:
                self.batch_norm = nn.BatchNorm2d(self.cout)
                
        if self.M == 3:
            self.conv = nn.Conv3d(self.cin, self.cout, self.kernel, self.stride, self.padding)
            if bn == True : 
                self.batch_norm = nn.BatchNorm3d(self.cout)
        else:
            raise Exception('Wrong input')
            
            

    def forward(self, inputs):
        
        out = self.conv(inputs)
        
        if self.bn:
            out = self.batch_norm(out)
            
        if self.activation:
            return F.relu(out)
        
        else:
            return out
    


# In[9]:


class Deconv2D(nn.Module):
    def __init__(self, cin, cout, kernel, stride, padding, bn=True):
        super(Deconv2D, self).__init__()
    
        self.cin = cin
        self.cout = cout
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.deconv = nn.ConvTranspose2d(self.cin, self.cout, self.kernel, self.stide, self.padding)
    
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(self.cout)
        
    def forward(self, inputs):
        out = self.deconv(out)
        
        if bn:
            out = self.batch_norm(out)
        
        return F.relu(out)
            


# In[10]:


class MiddleAndRpn(nn.Module):
    def __init__(self, alpha = 1.5, beta = 1, sigma = 3):
        super(MiddleAndRPN, self).__init__()
    
        self.middle_layer = nn.Sequential(ConvMD(3, 128, 64, 3, (2,1,1), (1,1,1)),
                                   ConvMD(3, 64, 64, 3, (1,1,1), (0,1,1)),
                                   ConvMD(3, 64, 64, 3, (2,1,1), (1,1,1))
                                  )
        if cfg.DETECT_OBJ == 'Car':
            self.conv1 = nn.Sequential(ConvMD(2, 128, 128, 3, (2,2), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1))
                                    )



        else:
            self.conv1 = nn.Sequential(ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                       ConvMD(2, 128, 128, 3, (1,1), (1,1))
                                        )

        self.deconv1 = Decon2D(128, 256, 3, (1,1), (1,1))


        self.conv2 = nn.Sequential(ConvMD(2, 128, 128, 3, (2,2), (1,1)),
                                   ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                   ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                   ConvMD(2, 128, 128, 3, (1,1), (1,1)),
                                   ConvMD(2, 128, 128, 3, (1,1), (1,1))
                                )

        self.deconv2 = Deconv2D(128, 256, 3, (2,2), (0,0))

        self.conv3 = nn.Sequential(ConvMD(2, 128, 256, 3, (2,2), (1,1)),
                                   ConvMD(2, 256, 256, 3, (1,1), (1,1)),
                                   ConvMD(2, 256, 256, 3, (1,1), (1,1)),
                                   ConvMD(2, 256, 256, 3, (1,1), (1,1)),
                                   ConvMD(2, 256, 256, 3, (1,1), (1,1))
                                )
        self.deconv3 = Deconv2D(256, 256, 4, (4,4), (0,0))
        self.prob_conv = ConvMD(2, 768, 2, 1, (1,1), (0,0), bn = False, activation = False)
        self.reg_conv = ConvMD(2, 768, 14, 1, (1,1), (0,0), bn = False, activation = False)
        self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
        
    
    def forward(self, x):
        [batch_size, depth, height, width] = x.shape
        x = x.permute(x,(0,4,1,2,3))
        
        x = self.middle_layer(x)
        x = x.view(batch_size, -1, height, width)
        x = self.conv1(x)
        x_deconv1 = self.deconv1(x)
        x = self.conv2(x)
        x_deconv2 = self.deconv2(x)
        x = self.conv3(x)
        x_deconv3 = self.deconv3(x)
        x = torch.cat([deconv3, deconv2, deconv1], dim = 1)
        
        p_map = self.prob_conv(x)
        r_map = self.reg_conv(x)
        
        return torch.sigmoid(p_map), r_map



# In[ ]:




