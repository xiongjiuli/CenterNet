'''
the crop valid 
'''


import torch
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm 
from data.dataloader import lymphDataset
from data.data_preprocess import getNiiPath
import torch.utils.data as data
from utils_v1 import *
from model.model_v1 import CenterNet_Resnet50






def valid_crop(model):

    path_root = '/data/julia/data_lymph'
    mode = 'validation'

    people_list, nii_path = getNiiPath(path_root, mode)

    valid_dataset = lymphDataset(path_root, people_list[0:60], mode, nii_path)

    valid_loader = data.DataLoader(valid_dataset, batch_size=5, shuffle=False)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            # embed()
            image = batch['image']['data'].cuda()
            image = image.type(torch.cuda.FloatTensor)
       
            point_gt = batch['hmap']['data'].squeeze(1)
         
            whd_gt = batch['whd']['data'].squeeze(1)
            offset_gt = batch['offset']['data'].squeeze(1)
            mask_gt = batch['mask']['data'].squeeze(1)

            pred  = model(image)#.squeeze(1)
            pred_point, pred_whd, pred_offset = pred[0], pred[1], pred[2]

            loss_1 = focal_loss(pred_point.cpu(), point_gt.unsqueeze(1))

            loss_2 = 0.1 * reg_l1_loss(pred_whd.cpu(), whd_gt, mask_gt)
            loss_3 = reg_l1_loss(pred_offset.cpu(), offset_gt, mask_gt)

            loss = loss_1 + loss_2 + loss_3

            # batch_step += 1
            # pbar.update(1)
            valid_loss += loss.item()

    valid_loss = valid_loss / len(people_list)

    return valid_loss

















if __name__ == '__main__':
    model = CenterNet_Resnet50().cuda()
    loss = valid_crop(model)


























