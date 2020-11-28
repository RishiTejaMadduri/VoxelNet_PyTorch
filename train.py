# +
from __future__ import print_function
import os
import cv2
import tqdm
import json
import math
import time
import torch
import Utils
import models
import logging
import datetime
import argparse
import utils_seg
# import dataloaders
import numpy as np
import torch.optim

from PIL import Image
from tqdm import tqdm
from imageio import imwrite
from utils_seg import Logger
from utils_seg import helpers
from utils_seg.losses import *
from dataloaders.voc import VOC
# from dataloaders.voc1 import VOC
from torchvision import transforms
from torchvision.utils import make_grid
from models.pyramid_fusion2 import PyFuse
from utils_seg.helpers import colorize_mask
from torch.utils.tensorboard import SummaryWriter
from utils_seg import transforms as local_transforms
from utils_seg.metrics import eval_metrics, AverageMeter
import pdb
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
    data_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/pyfuse/code/datasets/VOC/'
    train_loader = VOC(data_dir, 8, 'train')
    print(train_loader)
    val_loader = VOC(data_dir, 8, 'val')
#     train_loader = VOC("train")
#     val_loader = VOC("val")
    # MODEL
    model = PyFuse(50)
#     print(f'\n{model}\n')
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
