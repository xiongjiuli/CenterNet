import torch
from data.dataloader import lymphDataset
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import os
import argparse
import torchio as tio
from model.model_v1 import CenterNet_Resnet50
from utils_v1 import *
from valid_v1 import *
from IPython import embed
import torch.utils.data as data
import matplotlib.pyplot as plt
from time import time
from valid_v1 import save_info

# from valid_v1 import valid_slidewin
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'


def save(step, opt, model, out_path):

    data = {
        'step' : step,
        'model' : model.state_dict(),
        'opt' : opt.state_dict()
    }
    torch.save(data, os.path.join(out_path, f'model-centerv1-{step}.pt'))




def train(args):

    people_list, nii_path = getNiiPath(args.path_root, args.mode)
    # people_list = ['01190830220131']
    train_dataset = lymphDataset(args.path_root, people_list, args.mode, nii_path)
    #valid_dataset = lymphDataset(args.path_root, people_list, mode='validation', nii_path=nii_path)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = CenterNet_Resnet50()
    time_1 = time()
    model = model.cuda()
    print(time() - time_1)
    model = torch.nn.parallel.DataParallel(model)#, device_ids=[2, 3, 4])

    if args.model != 'normal':
        model_path = '/data/julia/data_lymph/save/model-centerv1-129.pt'
        model.load_state_dict(torch.load(model_path)['model'])

    # loss_mse = torch.nn.MSELoss()
    # loss_point = FocalLoss(1)
    # loss_focal = FocalLoss(2, save_choice=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

    if args.model == 'normal':
        epoch_range = range(args.epoch)
    else: 
        epoch_range = range(130, args.epoch)
    for epoch in tqdm(epoch_range):

        model.train()
        train_loss = 0
        point_loss = 0
        reg_loss = 0
        batch_step = 0
        whd_loss = 0
        L1loss = torch.nn.L1Loss()
        for batch in tqdm(train_loader):
            # embed()
            image = batch['image']['data'].cuda()
            image = image.type(torch.cuda.FloatTensor)
            # embed()
            point_gt = batch['hmap']['data'].squeeze(1)
            # bbox_gt = batch['bbox']['data'].squeeze(1).cuda()
            whd_gt = batch['whd']['data'].squeeze(1)
            offset_gt = batch['offset']['data'].squeeze(1)
            mask_gt = batch['mask']['data'].squeeze(1)

            optimizer.zero_grad() # clear optimizer parameter in x.grad

            pred  = model(image)#.squeeze(1)
            pred_point, pred_whd, pred_offset = pred[0], pred[1], pred[2]
            name = batch['name'][0]
            if batch_step == 0:
                save_info(name, output_point=pred_point[0], output_offset=pred_offset[0], output_whd=pred_whd[0], mode='training')
            # train_save_point(pred_point)
            # pred_po
            # int(b 1 w h d ) pred_bbox(b 2 w h d )
            # embed()
            # loss_1 = focal_loss(pred_point.cpu(), point_gt.unsqueeze(1))
            # # loss_1 = loss_point(pred_point, point_gt)
            # # embed()
            # loss_2 = 0.1 * reg_l1_loss(pred_whd.cpu(), whd_gt, mask_gt)
            # loss_3 = reg_l1_loss(pred_offset.cpu(), offset_gt, mask_gt)
            loss_1 = L1loss(pred_point.cpu(), point_gt.unsqueeze(1))
            # loss_1 = loss_point(pred_point, point_gt)
            # embed()
            loss_2 = reg_l1_loss(pred_whd.cpu(), whd_gt, mask_gt)
            loss_3 = reg_l1_loss(pred_offset.cpu(), offset_gt, mask_gt)
            # embed()
            loss = loss_1 + loss_2 + loss_3
            # embed()

            loss.backward() # x.grad += dloss / dx
            optimizer.step() # x.grad += -lr * x.grad
            batch_step += 1
            # pbar.update(1)
            train_loss += loss.item()
            point_loss += loss_1.item()
            reg_loss += loss_3.item()
            whd_loss += loss_2.item()
            batch_step += 1

        lr_scheduler.step()
        train_loss /= len(train_loader)


        with open('./loss/train_loss.txt', mode='a') as f:
            f.write(str(train_loss))
            f.write('\n')

        with open('./loss/whd_loss.txt', mode='a') as f:
            f.write(str(whd_loss))
            f.write('\n')

        with open('./loss/reg_loss.txt', mode='a') as f_reg:
            f_reg.write(str(reg_loss))
            f_reg.write('\n')

        with open('./loss/point_loss.txt', mode='a') as f_point:
            f_point.write(str(point_loss))
            f_point.write('\n')

        if (epoch + 1) % 10 == 0: # start validation
            # valid_loss = valid_crop(model)
            valid_loss = valid_slidewin(model, save=True)
            save_dir = '/data/julia/data_lymph/save'
            save(epoch, optimizer, model, save_dir)
            with open('./loss/valid_loss.txt', mode='a') as f:
                f.write(str(valid_loss))
                f.write('\n')



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', default='training', type=str)
    arg_parser.add_argument('--epoch', default=1000, type=int)
    arg_parser.add_argument('--batch_size', default=5, type=int)
    arg_parser.add_argument('--patch_size', default=128, type=int)
    arg_parser.add_argument('--lr', default=0.003, type=float)
    arg_parser.add_argument('--path_root', default='/data/julia/data_lymph', type=str)
    arg_parser.add_argument('--save_dir', default='/data/julia/data_lymph/save')
    arg_parser.add_argument('--model', default='normal', type=str)
    args = arg_parser.parse_args()
    if args.mode == 'training':
        train(args)




