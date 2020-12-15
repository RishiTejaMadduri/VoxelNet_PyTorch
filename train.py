{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import time\n",
    "import shutil\n",
    "import argparse\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader as DataLoader\n",
    "from torch.nn.utils import clip_grad_norm_\n",
    "# from torch.utils.tensorboard import SummaryWriter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyximport\n",
    "pyximport.install()\n",
    "\n",
    "from config import cfg\n",
    "from utils.utils import box3d_to_label\n",
    "from model.model import RPN3D\n",
    "from dataloader.kitti import KITTI_Loader as Dataset\n",
    "from dataloader.kitti import collate_fn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_instance(module, name, config, *args):\n",
    "    # GET THE CORRESPONDING CLASS / FCT \n",
    "    return getattr(module, config[name]['type'])(*args, **config[name]['args'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(config, resume):\n",
    "    train_logger = Logger()\n",
    "    # DATA LOADERS\n",
    "    train_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'training'), shuffle = True, aug = True, is_testset = False)\n",
    "    \n",
    "    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn,\n",
    "                                  num_workers = args.workers, pin_memory = False)\n",
    "    \n",
    "    val_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'validation'), shuffle = False, aug = False, is_testset = False)\n",
    "    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn,\n",
    "                                num_workers = args.workers, pin_memory = False)\n",
    "    val_dataloader_iter = iter(val_dataloader)\n",
    "\n",
    "    model = RPN3D(cfg.DETECT_OBJ, args.alpha, args.beta)\n",
    "\n",
    "    trainer = Trainer(\n",
    "        model=model,\n",
    "        resume=resume,\n",
    "        config=config,\n",
    "        train_loader=train_loader,\n",
    "        val_loader=val_loader,\n",
    "        train_logger=train_logger)\n",
    "    \n",
    "    trainer.train()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'argparse' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-b63a91bf4eb6>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0;34m'__main__'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0;31m# PARSE THE ARGS\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m     \u001b[0mparser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0margparse\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mArgumentParser\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdescription\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'PyTorch Training'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m     parser.add_argument('-c', '--config', default='config.json',type=str,\n\u001b[1;32m      5\u001b[0m                         help='Path to the config file (default: config.json)')\n",
      "\u001b[0;31mNameError\u001b[0m: name 'argparse' is not defined"
     ]
    }
   ],
   "source": [
    "if __name__=='__main__':\n",
    "    # PARSE THE ARGS\n",
    "    parser = argparse.ArgumentParser(description='PyTorch Training')\n",
    "    parser.add_argument('-c', '--config', default='config.json',type=str,\n",
    "                        help='Path to the config file (default: config.json)')\n",
    "    parser.add_argument('-r', '--resume', default=None, type=str,\n",
    "                        help='Path to the .pth model checkpoint to resume training')\n",
    "    parser.add_argument('-d', '--device', default=None, type=str,\n",
    "                           help='indices of GPUs to enable (default: all)')\n",
    "    args = parser.parse_args()\n",
    "\n",
    "    config = json.load(open(args.config))\n",
    "    if args.resume:\n",
    "        config = torch.load(args.resume)['config']\n",
    "    if args.device:\n",
    "        os.environ[\"CUDA_VISIBLE_DEVICES\"] = args.device\n",
    "    \n",
    "    main(config, args.resume)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
