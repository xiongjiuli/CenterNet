'''
load the model
window crop
and get the result
'''

import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
from model.model_v1 import CenterNet_Resnet50
from time import time
from valid_v1 import valid_slidewin, valid_crop
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'

def model_extra(model_dir):

    model = CenterNet_Resnet50()
    model = torch.nn.parallel.DataParallel(model)#, device_ids=[3, 4, 2])
    a = torch.load(model_dir)['model']

    model.load_state_dict(a)
    model = model.cuda()

    return model



def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def decode_bbox(pred_hmap, pred_whd, pred_offset, confidence, cuda):
    '''
    to get the bbox 
    '''
    pred_hmap = pool_nms(pred_hmap)

    output_w, output_h, output_d = pred_hmap.shape

    detects = []
    # * 每次只传入一张图片
    heat_map = pred_hmap.permute(1, 2, 3, 0).view([-1, 1])
    pred_whd = pred_whd.permute(1, 2, 3, 0).view([-1, 3])
    pred_offset = pred_offset.permute(1, 2, 3, 0).view([-1, 3])

    zv, yv, xv      = torch.meshgrid(torch.arange(0, output_d), torch.arange(0, output_h), torch.arange(0, output_w))
    zv, xv, yv      = xv.flatten().float(), yv.flatten().float()
    if cuda:
        xv      = xv.cuda()
        yv      = yv.cuda()

    #-------------------------------------------------------------------------#
    #   class_conf      128*128,    特征点的种类置信度
    #   class_pred      128*128,    特征点的种类
    #-------------------------------------------------------------------------#
    class_conf, class_pred  = torch.max(heat_map, dim = -1)
    mask                    = class_conf > confidence

    #-----------------------------------------#
    #   取出得分筛选后对应的结果
    #-----------------------------------------#
    pred_wh_mask        = pred_wh[mask]
    pred_offset_mask    = pred_offset[mask]
    if len(pred_wh_mask) == 0:
        detects.append([])
        # continue     

    #----------------------------------------#
    #   计算调整后预测框的中心
    #----------------------------------------#
    xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
    yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
    #----------------------------------------#
    #   计算预测框的宽高
    #----------------------------------------#
    half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
    #----------------------------------------#
    #   获得预测框的左上角和右下角
    #----------------------------------------#
    bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
    bboxes[:, [0, 2]] /= output_w
    bboxes[:, [1, 3]] /= output_h
    detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
    detects.append(detect)

    return detects










if __name__ == '__main__':

    model_dir = '/data/julia/data_lymph/save/model-centerv1-129.pt'
    model = model_extra(model_dir)

    # valid_slidewin(model, save=True)
    loss = valid_crop(model, save=True)



    