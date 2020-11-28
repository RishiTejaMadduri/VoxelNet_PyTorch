import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils_seg import transforms as local_transforms
from base_trainer import *
from utils_seg.helpers import colorize_mask
from utils_seg.metrics import eval_metrics, AverageMeter
from tqdm import tqdm


# +
class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
#         import pdb; pdb.set_trace()
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = 10
        
        self.num_classes = 21

        if self.device ==  torch.device('cpu'): prefetch = False
        torch.backends.cudnn.benchmark = True
    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        self.wrt_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            self.optimizer.step()
            self.lr_scheduler.step(epoch=epoch-1)
            
            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            target = target.permute(0, 1, 3, 2)
#             target = ((target).type(torch.LongTensor)).cuda()
#             output = output.cuda()
#             print("Loss Fn Output Size: ", output.shape)
#             print("Loss Fn Target Size: ", target.shape)
#             assert output[0].size()[2:] == target.size()[1:]
#             assert output[0].size()[1] == self.num_classes 
            loss = self.loss(output, target)
            
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
                
            self.optimizer.step()
            loss.backward()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

#             # LOGGING & TENSORBOARD
#             if batch_idx % self.log_step == 0:
#                 self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
#                 self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

#             # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            
            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))

#         # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
#         for k, v in list(seg_metrics.items())[:-1]: 
#             self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
#         for i, opt_group in enumerate(self.optimizer.param_groups):
#             self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
#             #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}
#         log = {'loss': self.total_loss.average}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
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
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                target = target.permute(0, 1, 3, 2)
#                 target = ((target).type(torch.LongTensor)).cuda()
#                 output = output.cuda()                
                
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
#             self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
#             self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
#             for k, v in list(seg_metrics.items())[:-1]: 
#                 self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
