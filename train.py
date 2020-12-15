#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import shutil
import argparse
import cv2


# In[2]:


import numpy as np


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter


# In[4]:


import pyximport
pyximport.install()

from config import cfg
from utils.utils import box3d_to_label
from model.model import RPN3D
from dataloader.kitti import KITTI_Loader as Dataset
from dataloader.kitti import collate_fn


# In[5]:


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


# In[ ]:


def main(config, resume):
    train_logger = Logger()
    # DATA LOADERS
    train_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'training'), shuffle = True, aug = True, is_testset = False)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn,
                                  num_workers = args.workers, pin_memory = False)
    
    val_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'validation'), shuffle = False, aug = False, is_testset = False)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn,
                                num_workers = args.workers, pin_memory = False)
    val_dataloader_iter = iter(val_dataloader)

    model = RPN3D(cfg.DETECT_OBJ, args.alpha, args.beta)

    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)
    
    trainer.train()


# In[2]:


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)


# In[ ]:




