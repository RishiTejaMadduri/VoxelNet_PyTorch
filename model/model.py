#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import sys
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%

sys.path.append('/home/rishi/Projects/VoxelNet/VoxelNet_R_PyTorch/')

from model.group_pointcloud import FeatureNet
from model.rpn import MiddleAndRpn


# %%





# %%


from utils.utils import *
from utils.colorize import colorize
from utils.nms import nms
from config import cfg


# %%


small_addon_for_BCE = 1e-6


# %%


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
        tag = inputs[0]
        label = inputs[1]
        vox_features = inputs[2]
        vox_numbers = inputs[3]
        vox_coordinates = inputs[4]
        
        features = self.feature(vox_features, vox_numbers, vox_coordinates)
        
        prob_output, delta_output = self.rpn(features)
        
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape, self.anchors, cls = cfg.DETECT_OBJ)
        pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        
        device = features.device
        pos_equal_one = torch.from_numpy(pos_equal_one).to(device).float()
        neg_equal_one = torch.from_numpy(neg_equal_one).to(device).float()
        targets = torch.from_numpy(targets).to(device).float()
        pos_equal_one_for_reg = torch.from_numpy(pos_equal_one_for_reg).to(device).float()
        pos_equal_one_sum = torch.from_numpy(pos_equal_one_sum).to(device).float()
        neg_equal_one_sum = torch.from_numpy(neg_equal_one_sum).to(device).float()
        
        pos_equal_one = pos_equal_one.permute(0, 3, 1, 2)
        neg_equal_one = neg_equal_one.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        pos_equal_one_for_reg = pos_equal_one_for_reg.permute(0, 3, 1, 2)
        
        cls_pos_loss = (-pos_equal_one * torch.log(prob_output + small_addon_for_BCE))/pos_equal_one_sum
        cls_neg_loss = (-neg_equal_one * torch.log(1-prob_output + small_addon_for_BCE))/neg_equal_one_sum
        cls_loss = torch.sum(self.alpha*cls_pos_loss + self.beta*cls_neg_loss)
        cls_pos_loss_rec = torch.sum(cls_pos_loss)
        cls_neg_loss_rec = torch.sum(cls_neg_loss)
        
        reg_loss = smooth_l1(delta_output * pos_equal_one_for_reg, targets * pos_equal_one_for_reg, self.sigma) / pos_equal_one_sum
        
        reg_loss = torch.sum(reg_loss)
        loss = cls_loss + reg_loss
        
        return prob_output, delta_output, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec
    
    
#     def predict(self, inputs, probs, deltas, summary = False, vis = False):
        
#         tag = inputs[0]
#         label = inputs[1]
#         vox_feature = inputs[2]
#         vox_number = inputs[3]
#         vox_coordinate = inputs[4]
#         img = inputs[5]
#         lidar = inputs[6]
#         batch_size = probs.shape(0)
        
#         if summary or vis:
#             batch_gt_boxes3d = label_to_gt_box3d(label, cls = self.cls, coordinate = 'lidar')
        
#         batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate = 'lidar')
#         batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
#         batch_probs = probs.reshape((batch_size, -1))
        
#         #NMS
#         ret_box3d = []
#         ret_score = []
        
#         for batch_id in range(batch_size):
#             ind = np.where(batch_probs[batch_id, :]>=cfg.RPN__SCORE_THRESH)[0]
#             tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
#             tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
#             tmp_scores = batch_probs[batch_id, ind]
            
#             boxes2d = corner_to_staandup_box2d(center_to_corner_box2d(tmp_boxes2d, cooridnate = 'lidar'))
#             ind,cnt = nms(boxes2d, tmp_scores, cfg.RPN_NMS_THRESH, RPN_NMS_POST_TOPK)
#             ind = ind[:cnt].cpu().detach.numpy()
#             tmp_boxes3d = tmp_boxes3d[ind, ...]
#             tmp_scores = tmp_scores[ind]
#             ret_box3d.append(tmp_boxes3d)
#             ret_score.append(tmp_scores)
            
#         ret_box3d_score = []
        
#         for boxes3d, scores in zip(ret_box3d, ret_score):
#             ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis], vixes3d, score[:, np.newaxis]], axis=-1))
            
#         if summary:
#             cur_tag = tag[0]
#             P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))
            
#             front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0], batch_gt_boxes3d[0], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
            
#             bird_view = lidar_to_bird_view_img(lidar[0], factor=cfg.BV_LOG_FACTOR)
            
#             bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0], batch_gt_boxes3d[0], factor = cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
            
#             heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
            
#             ret_summary = [['predict/front_view_rgb', front_image[np.newaxis, ...]],  # [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3]
#                            # [None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3]
#                            ['predict/bird_view_lidar', bird_view[np.newaxis, ...]],
#                            # [None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3]
#                            ['predict/bird_view_heatmap', heatmap[np.newaxis, ...]]]
            
#             return tag, ret_box3d_score, ret_summary
        
#         if vis:
#             front_images, bird_views, heatmaps = [], [], []
            
#             for i in range(len(img)):
#                 curr_tag = tag[i]
#                 P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
#                 front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i], batch_gt_boxes3d[i], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
#                 bird_view = lidar_to_bird_view_img(lidar[i], factor=cfg.BV_LOG_FACTOR)
#                 bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i],
#                 batch_gt_boxes3d[i], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
#                 heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)
#                 front_images.append(front_image)
#                 bird_views.append(bird_view)
#                 heatmaps.append(heatmap)
        
#             return tag, ret_box3d_score, front_images, bird_views, heatmaps
#         return tag, ret_box3d_score
        


# %%


def smooth_l1(deltas, targets, sigma = 3.0):
    
    sigma2 = sigma*sigma
    diffs = torch.sub(deltas, targets)
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0/sigma2).type('torch.FloatTensor').cuda()
    
    smooth_l1_option1 = torch.mul(diffs, diffs)*0.5*sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1-smooth_l1_signs)
    
    smooth_l1 = smooth_l1_add
    
    return smooth_l1


# %%




