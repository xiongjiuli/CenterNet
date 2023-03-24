import torch
from model_v5 import SegPointNet
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import os
import argparse
import torchio as tio
from data_v5 import lymphDataset
from utils_v5 import *
from valid_v5 import *
from IPython import embed
import torch.utils.data as data
import matplotlib.pyplot as plt
from time import time
# from valid_v5 import valid_slidewin
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 2'


def save(step, opt, model, out_path):

    data = {
        'step' : step,
        'model' : model.state_dict(),
        'opt' : opt.state_dict()
    }
    torch.save(data, os.path.join(out_path, f'model-v5-{step}.pt'))




def train(args):

    people_list, nii_path = getNiiPath(args.path_root, args.mode)
    people_list = ['01190830220131']
    train_dataset = lymphDataset(args.path_root, people_list, args.mode, nii_path)
    #valid_dataset = lymphDataset(args.path_root, people_list, mode='validation', nii_path=nii_path)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = SegPointNet(1, 1)
    time_1 = time()
    model = model.cuda()
    print(time() - time_1)
    model = torch.nn.parallel.DataParallel(model)#, device_ids=[2, 3, 4])

    # if args.model != 'normal':
    model_path = '/data/julia/data_lymph/save/model-v5-179.pt'
    model.load_state_dict(torch.load(model_path)['model'])

    loss_mse = torch.nn.MSELoss()
    # loss_point = FocalLoss(1)
    loss_focal = FocalLoss(2, save_choice=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

    if args.model == 'normal':
        epoch_range = range(args.epoch)
    else: 
        epoch_range = range(35, args.epoch)
    for epoch in tqdm(epoch_range):

        model.train()
        train_loss = 0
        batch_step = 0
        for batch in tqdm(train_loader):
            # embed()
            image = batch['image']['data'].cuda()
            image = image.type(torch.cuda.FloatTensor)
            # embed()
            point_gt = batch['hmap']['data'].squeeze(1).cuda()
            bbox_gt = batch['bbox']['data'].squeeze(1).cuda()

            optimizer.zero_grad() # clear optimizer parameter in x.grad

            pred_point, pred_bbox = model(image)#.squeeze(1)
            # train_save_point(pred_point)
            # pred_point(b 1 w h d ) pred_bbox(b 2 w h d )
            # embed()
            loss_1 = loss_mse(pred_point.squeeze(1), point_gt * 800)
            # loss_1 = loss_point(pred_point, point_gt)
            # embed()
            loss_2 = loss_focal(pred_bbox, bbox_gt)
            # embed()
            loss = loss_1 + 100 * loss_2
            # embed()

            loss.backward() # x.grad += dloss / dx
            optimizer.step() # x.grad += -lr * x.grad
            batch_step += 1
            # pbar.update(1)
            train_loss += loss.item()

        lr_scheduler.step()
        train_loss /= len(train_loader)


        with open('./add_segbbox/train_loss.txt', mode='a') as f:
            f.write(str(train_loss))
            f.write('\n')

        if (epoch + 1) % 10 == 0: # start validation
            valid_loss = valid_slidewin(model, args)
            save_dir = '/data/julia/data_lymph/save'
            save(epoch, optimizer, model, save_dir)
            with open('./add_segbbox/valid_loss.txt', mode='a') as f:
                f.write(str(valid_loss))
                f.write('\n')



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', default='training', type=str)
    arg_parser.add_argument('--epoch', default=1000, type=int)
    arg_parser.add_argument('--batch_size', default=20, type=int)
    arg_parser.add_argument('--patch_size', default=128, type=int)
    arg_parser.add_argument('--lr', default=0.003, type=float)
    arg_parser.add_argument('--path_root', default='/data/julia/data_lymph', type=str)
    arg_parser.add_argument('--save_dir', default='/data/julia/data_lymph/save')
    arg_parser.add_argument('--model', default='normal', type=str)
    args = arg_parser.parse_args()
    if args.mode == 'training':
        train(args)




