# +
from __future__ import print_function
import os
import cv2
import tqdm
import json
import math
import time
import torch
import logging
import datetime
import argparse
import numpy as np
import torch.optim

from PIL import Image
from tqdm import tqdm
from imageio import imwrite


from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader as DataLoader
from torch.utils.tensorboard import SummaryWriter

import pyximport
pyximport.install()

from config import cfg
from utils.utils import box3d_to_label
from model.model import RPN3D
from dataloader.kitti import KITTI_Loader as Dataset
from dataloader.kitti import collate_fn

# from azureml.tensorboard import Tensorboard
import warnings
warnings.filterwarnings("ignore")
from trainer import *


# -

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()
    # DATA LOADERS
    data_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/pyfuse/code/datasets/data/'
    train_loader = VOC(data_dir, 8, 'train')
    print(train_loader)
    val_loader = VOC(data_dir, 8, 'val')

    # MODEL
    model = PyFuse(50)

    # LOSS
    loss = CrossEntropyLoss2d()

    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)
    
    if test:
        test = Test(val_loader, )
    
    trainer.train()


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
