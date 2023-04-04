'''
the crop valid 
'''


import torch
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm 
from data.dataloader import lymphDataset
from data.data_preprocess import getNiiPath, get_Heatmap, get_WHD_offset_mask
import torch.utils.data as data
from utils_v1 import *
from model.model_v1 import CenterNet_Resnet50
from IPython import embed






def valid_crop(model, save=None):

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

            if save != None:
                print('save begin')
                # embed()
                name = batch['name'][0]
                # pred_point[0] = pred_point[0].squeeze(0)
                save_info(name, pred_point[0], pred_whd[0], pred_offset[0], mode)

            loss_1 = focal_loss(pred_point.cpu(), point_gt.unsqueeze(1))

            loss_2 = 0.1 * reg_l1_loss(pred_whd.cpu(), whd_gt, mask_gt)
            loss_3 = reg_l1_loss(pred_offset.cpu(), offset_gt, mask_gt)

            loss = loss_1 + loss_2 + loss_3

            # batch_step += 1
            # pbar.update(1)
            valid_loss += loss.item()

    valid_loss = valid_loss / len(people_list)

    return valid_loss



'''
the file is to make the silding window data crop 
the file is for testing sliding window crop

the input is the name of the testing Dataset 
the output is the p and r for each person in the txt file
and will save the 0-1 nii files

'''




def valid_slidewin(
          model,
          batch_size=3, 
          mode='validation', 
          overlap=0,
          save=None,
          ):
    
    people_list, nii_dir = getNiiPath('/data/julia/data_lymph', mode)
  
    val_loss = 0
    # people_list = ["02191126216013"] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for name in tqdm(people_list):

        image_npy, heatmap, whd_data, offset_data, mask_data =  _get_5tensorTata(name, nii_dir, mode)
        
        x, y, z = image_npy.shape   #  (512, 512, 408)
        num = torch.from_numpy(np.zeros_like(image_npy))
        num_temp = torch.from_numpy(np.zeros_like(image_npy))

        # segmentation the numpy which will be process
        step = 128 - overlap

        # get the xyz 三轴上依次要取得点数
        range_x, range_y, range_z = _get_range(image_npy.shape, step)

        
        # generate the whole input tnesor and location waited to be selectied
        info = {}
        order = 0
        for i in range_x:
            for j in range_y:
                for k in range_z:
                    patch_tensor4d = torch.from_numpy(
                        image_npy[i : i + step, \
                                  j : j + step, \
                                  k : k + step]).unsqueeze(0)
                    # a image have the all patchs
                    # patchs.append(patch_npy) 
                    start_point = torch.tensor([i, j, k, step])
                    info[order] = patch_tensor4d, start_point
                    order += 1

                    # the number of the voxel to be process
                    num_temp[i : i + step, \
                             j : j + step, \
                             k : k + step] = 1
                    num += num_temp
                    # print('the process of num')
                    # embed()
                    num_temp[:, :, :] = 0
       
        # embed()
        # setting the model

        output_point = torch.from_numpy(np.zeros((x, y, z)))
        output_point_temp = torch.from_numpy(np.zeros((x, y, z)))
        output_offset = torch.from_numpy(np.zeros((3, x, y, z)))
        output_offset_temp = torch.from_numpy(np.zeros((3, x, y, z)))
        output_whd = torch.from_numpy(np.zeros((3, x, y, z)))
        output_whd_temp = torch.from_numpy(np.zeros((3, x, y, z)))

        model.eval()
        # pick patchs from the orderd input tensor
        start = 0
        l1loss = torch.nn.L1Loss()
        while start <= len(info):
          
            # embed()
            if len(info) - start < batch_size:
                range_n = list(range(len(info) - batch_size, len(info)))
            else:
                range_n = list(range(start, start + batch_size))
            
            inputs_patch = []
            locations_patch = []
            for n in range_n:
                input = info[n][0]
                inputs_patch.append(input)
                # inputs_patch[0].shape : torch.Size([1, 128, 128, 128])
                loc = info[n][1]
                locations_patch.append(loc)
       
          
            # embed()
            # put the patchs into the model to train
            with torch.no_grad(): # 可以使gpu不叠加data，不加这个gpu会去叠加数据

                # 3. put the input to the model to train
                inputs_patch = torch.stack(inputs_patch, dim=0)
                input_tensor = inputs_patch.cuda()
                input_tensor = input_tensor.type(torch.cuda.FloatTensor)
    
                # embed()
                pred = model(input_tensor)
                pred_point, pred_whd, pred_offset = pred[0], pred[1], pred[2]


                pred_point = pred_point.cpu()#.numpy()
                pred_whd = pred_whd.cpu()#.numpy()
                pred_offset = pred_offset.cpu()

                # make the pred tensor seam to the output_point
                for l in range(len(locations_patch)):
                    # print('start sew, the patch size = 3')
                    # embed()
                    # output_point_temp = output_point_temp.astype(pred_point.dtype)

                    output_point_temp[locations_patch[l][0] : locations_patch[l][0] + locations_patch[l][3], 
                                      locations_patch[l][1] : locations_patch[l][1] + locations_patch[l][3], 
                                      locations_patch[l][2] : locations_patch[l][2] + locations_patch[l][3]] = pred_point[l, 0, :, :, :]

                    output_point += output_point_temp
                    output_point_temp[:, :, :] = 0

                    output_whd_temp[:, locations_patch[l][0] : locations_patch[l][0] + locations_patch[l][3], \
                                        locations_patch[l][1] : locations_patch[l][1] + locations_patch[l][3],\
                                        locations_patch[l][2] : locations_patch[l][2] + locations_patch[l][3]] = pred_whd[l, :, :, :, :]
                    output_whd += output_whd_temp
                    output_whd_temp[:, :, :, :] = 0

                    output_offset_temp[:, locations_patch[l][0] : locations_patch[l][0] + locations_patch[l][3], \
                                        locations_patch[l][1] : locations_patch[l][1] + locations_patch[l][3],\
                                        locations_patch[l][2] : locations_patch[l][2] + locations_patch[l][3]] = pred_offset[l, :, :, :, :]
                    output_offset += output_offset_temp
                    output_offset_temp[:, :, :, :] = 0


                    # index+=1
            start += batch_size



        # make the overlap part to be average
        output_point = output_point / num  # tensor file
        output_whd = output_whd / num
        output_offset = output_offset / num

        print('start save')
        if save != None:
            save_info(name, output_point, output_whd, output_offset, mode=mode)
        print('sew over and you should see the whole image')
        # embed()
        output_point = output_point.squeeze(1).cpu()


        # loss_1 = focal_loss(pred_point.cpu(), heatmap.unsqueeze(1))
        loss_1 = l1loss(pred_point.cpu(), heatmap.unsqueeze(1))
        loss_2 = 0.1 * reg_l1_loss(pred_whd.cpu(), whd_data, mask_data)
        loss_3 = reg_l1_loss(pred_offset.cpu(), offset_data, mask_data)

        # embed()
        loss = loss_1 + loss_2 + loss_3
        valid_loss += loss.item()

    valid_loss = valid_loss / len(people_list)

    return valid_loss




def _get_range(shape, step):

    x, y, z = shape[:]

    if x / step == x // step:
        step_x = x // step
        range_x = list(range(0, step_x * step, step))
    else: 
        step_x = x // step 
        range_x = list(range(0, step_x * step, step))
        range_x.append(x - step)
    if y / step == y // step :
        step_y = y // step
        range_y = list(range(0, step_y * step, step))
    else: 
        step_y = y // step 
        range_y = list(range(0, step_y * step, step))
        range_y.append(y - step)
    if z / step == z // step :
        step_z = z // step
        range_z = list(range(0, step_z * step, step))
    else: 
        step_z = z // step 
        range_z = list(range(0, step_z * step, step))
        range_z.append(z - step) # .append()是不会返回值的

    return range_x, range_y, range_z


def _get_5tensorTata(name, nii_dir, mode):

    npy_dir = '/data/julia/data_lymph/npy_data/resampled_clamp_norm_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(npy_dir):
        image_npy = np.load(npy_dir)
    else:
        image = tio.ScalarImage(nii_dir[name])
        # load resampled data_npy and heatmap directly
        
        prepro = tio.Compose([
            tio.Resample((0.8, 0.8, 1)),
            tio.Clamp(out_min=-160, out_max=240),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

        image_npy = prepro(image)
        image_npy = image_npy.numpy()[0, :, :, :]
        np.save(npy_dir, image_npy)

    hmap_dir = '/data/julia/data_lymph/heatmaps/resampled_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(hmap_dir):
        heatmap = np.load(hmap_dir)
    else:
        heatmap = get_Heatmap(name, mode, savenii_dir='')    


    whd_data, offset_data, mask_data =  get_WHD_offset_mask(name, mode)

    heatmap = torch.from_numpy(heatmap).type(torch.float32)
    # image_npy = torch.from_numpy(image_npy).type(torch.float32)
    whd_data = torch.from_numpy(whd_data).type(torch.float32)
    offset_data = torch.from_numpy(offset_data).type(torch.float32)
    mask_data = torch.from_numpy(mask_data).type(torch.float32)

    return image_npy, heatmap, whd_data, offset_data, mask_data




def save_info(name, output_point, output_whd, output_offset, mode):

    # output_point = torch.from_numpy(output_point)
    # output_whd = torch.from_numpy(output_whd)
    # output_offset = torch.from_numpy(output_offset)

    df = pd.read_csv('/home/julia/workfile/{}_affine_resampled.csv'.format(mode))
    affine_xyz = df[df['name'] == "'" + name]

    affine =   [[-0.8, 0, 0, 0],
                [0, -0.8, 0, 0],
                [0,    0, 1, 0],
                [0,    0, 0, 1]]

    # embed()
    affine[0][-1] = affine_xyz['affine_x'].item()
    affine[1][-1] = affine_xyz['affine_y'].item()
    affine[2][-1] = affine_xyz['affine_z'].item()
    affine = np.array(affine)

    # prediction_point = tio.ScalarImage(tensor=output_point.unsqueeze(0), affine=affine)
    prediction_point = tio.ScalarImage(tensor=output_point.detach().cpu(), affine=affine)
    prediction_whd = tio.ScalarImage(tensor=output_whd.detach().cpu(), affine=affine)
    prediction_offset = tio.ScalarImage(tensor=output_offset.detach().cpu(), affine=affine)


    prediction_point.save('/data/julia/data_lymph/temp_test/heatmap_{}.nii'.format(mode))
    prediction_whd.save('/data/julia/data_lymph/temp_test/whd_{}.nii'.format(mode))
    prediction_offset.save('/data/julia/data_lymph/temp_test/offset_{}.nii'.format(mode))
    



    return print('save done')




# if __name__ == '__main__':

#     path_root = '/data/julia/data_lymph'
#     data_type = 'validation'
#     # size = 128
#     people_list, nii_path = getNiiPath(path_root, data_type)
#     # batch_size = 30

#     # setting the model
#     model = cenert(1, 1)
#     time_1 = time()
#     model = model.cuda()
#     print(time() - time_1)
#     model = torch.nn.parallel.DataParallel(model)#, device_ids=[2, 3, 4])

#     model_path = '/data/julia/data_lymph/save/model-v5-179.pt'
#     model.load_state_dict(torch.load(model_path)['model'])


#     name = "02191126216013"
#     valid_slidewin(model)

#     # for i in tqdm(range(len(people_list))):
#     #     name = people_list[i]
#     #     valid_slidewin(name)





if __name__ == '__main__':
    model = CenterNet_Resnet50().cuda()
    time_1 = time()
    model = model.cuda()
    print(time() - time_1)
    model = torch.nn.parallel.DataParallel(model)#, device_ids=[2, 3, 4])

    model_path = '/data/julia/data_lymph/save/model-v5-179.pt'
    model.load_state_dict(torch.load(model_path)['model'])

    loss = valid_slidewin(model, save=True)


























