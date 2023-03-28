import numpy as np
import pandas as pd
import os
import random
import torch
import torchio as tio
import torch.nn.functional as F
import numba




def getNiiPath(path, mode):
    # 获取nii文件指定名字的路径
    if mode == 'training' or 'testing' or 'vaildation':
        pass
    else:
        print("mode erroe: mode should be 'training' or 'testing' or 'vaildation'")
    files_path = {}
    name_list = os.listdir(os.path.join(path, mode))
    for file in name_list:
        file_path = os.listdir(os.path.join(path, mode, file))
        files_path[file] = os.path.join(path, mode, file, file_path[0])

    return name_list, files_path






def focal_loss(pred, target, aerfa=2, beta=4 ):
    # * pred he target dimention both the (1, 1, 128, 128, 128)
    # pred = pred.permute(0, 2, 3, 4, 1) # ? 为什么会有这个的一个code
    '''
    pred : (b, c, 128, 128, 128)
    target : (b, c, 128, 128, 128)
    '''
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, beta)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # 把最大值和最小值控制在这一个范围之内
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, aerfa) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, aerfa) * neg_weights * neg_inds
    # embed()
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # embed()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss





def reg_l1_loss(pred, target, mask):
    '''
    pred: (1, 3, 128, 128, 128)
    target:(1, 3, 128, 128, 128)
    mask:(1, 128, 128, 128)
    '''
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0, 2, 3, 4, 1)
    target = target.permute(0, 2, 3, 4, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1, 3)
    # embed()
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')  
    # embed()
    # ？那这个岂不是对那些预测错的是没有penalty的
    loss = loss / (mask.sum() + 1e-4)
    return loss





if __name__ == '__main__':
    
    import torch
    from IPython import embed

    random.seed(1)
    pred = torch.randn(1, 3, 4, 4, 4)
    target = torch.randn( 1, 3, 4, 4, 4)
    mask = torch.randn(1, 4, 4, 4)

    # loss = focal_loss(pred, target)
    loss = reg_l1_loss(pred, target, mask)
    # embed()







