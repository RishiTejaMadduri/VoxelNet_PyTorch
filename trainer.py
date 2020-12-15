#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from base_trainer import *
from tqdm import tqdm


# In[ ]:


class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, train_loader, val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = 10
        torch.backends.cudnn.benchmark = True
        
    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        self.wrt_mode = 'train'
        
        tic = time.time()
#         self._reset_metrics()

        tbar = tqdm(self.train_loader, ncols=130)
        for (i, data) in enumerate(tbar):
            
            self.optimizer.step()
            self.lr_scheduler.step(epoch=epoch-1)
            
            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            prob_output, delta_output, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = self.model(data)
            forward_time = time.time() - start_time
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
                
            self.optimizer.step()
            loss.backward()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            
            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | cls_loss {:.2f} | reg_loss {:.2f} | cls_pos_loss {:.2f} | cls_neg_loss {:.2f} | forward_time {:.2f} | batch_time {:.2f} |'.format(
                                                epoch, loss, 
                                                cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec
                                                self.forward_time.average, self.batch_time.average))

#         # METRICS TO TENSORBOARD
        self.writer.add_scalars(str(epoch+1), {'train/loss' : loss.item(),
                                                  'train/reg_loss' : reg_loss.item(),
                                                  'train/cls_loss' : cls_loss.item(),
                                                  'train/cls_pos_loss' : cls_pos_loss_rec.item(),
                                                  'train/cls_neg_loss' : cls_neg_loss_rec.item()
                                                 }
                                  )

        # RETURN LOSS & METRICS
        log = {'loss': loss.average(),
               'cls_loss': cls_loss.average(),
               'reg_loss': reg_loss.average(),
               'cls_pos_loss': cls_pos_loss_rec.average(),
               'cls_neg_loss': cls_neg_loss_rec.average()
              }


        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        
        with torch.no_grad():
            val_visual = []
            for (i, data) in enumerate(tbar):
                
                probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data)

                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()

                # PRINT INFO
                tbar.set_description('EVAL ({}) | Loss: {:.3f} | cls_loss {:.2f} | reg_loss {:.2f} | cls_pos_loss {:.2f} | cls_neg_loss {:.2f} | forward_time {:.2f} | batch_time {:.2f} |'.format(
                                                epoch, loss, 
                                                cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec
                                                self.forward_time.average, self.batch_time.average))

            # WRTING & VISUALIZING THE MASKS
            tags, ret_box3d_scores, ret_summary = model.module.predict(val_data, probs, deltas, summary = True)
            for (tag, img) in ret_summary:
                            img = img[0].transpose(2, 0, 1)
                            self.writer.add_image(tag, img, global_counter)
                        

            # METRICS TO TENSORBOARD
            self.writer.add_scalars(str(epoch+1), {'train/loss' : loss.item(),
                                                  'train/reg_loss' : reg_loss.item(),
                                                  'train/cls_loss' : cls_loss.item(),
                                                  'train/cls_pos_loss' : cls_pos_loss_rec.item(),
                                                  'train/cls_neg_loss' : cls_neg_loss_rec.item()
                                                 }
                                  )
            log = {'loss': loss.average(),
               'cls_loss': cls_loss.average(),
               'reg_loss': reg_loss.average(),
               'cls_pos_loss': cls_pos_loss_rec.average(),
               'cls_neg_loss': cls_neg_loss_rec.average()
              }
                
        tot_val_loss += val_loss.item()
        tot_val_times += 1
        
        return log, tot_val_loss, tot_val_times

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        

