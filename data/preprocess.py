'''
process the data into the model

x, y, z = image.shape
heatmap[x, y, z] = (0-1)
WHD[x, y, z] = (W, H, D)
offset[x, y, z] = (.w, .h, .d)
mask[x, y, z] = 0 or 1

'''

import numpy as np
import pandas as pd
import torchio as tio
import os
import random
import torch
import scipy.ndimage as si

# * to get the W H D map very preprocess data 
def get_WHD_offset_mask(name, mode, whd=None, offset=None, mask=None):

    # * the resampled image
    csv_dir = '/data/julia/data_lymph/lymph_csv/resampled_{}.csv'.format(mode)
    df = pd.read_csv(csv_dir)
    df = df[df['name'] == "'" + name]

    coords = df[['x', 'y', 'z']].values
    whds = df[['width', 'height', 'depth']].values
    nii_dir = df[['path']][0]

    nii_image = tio.ScalarImage(nii_dir)

    whd_image = np.zeros_like(nii_image[1:])
    offset_image = np.zeros_like(nii_image[1:])
    mask_image = np.zeros_like(nii_image[1:])


    if whd == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            whd_image[coord_int] = whds[i]

        save_whd_dir = '/data/julia/data_lymph/npy_data/resampled_whd_{}/{}.npy'.format(mode, name)
        np.save(save_whd_dir, whd_image)
    if offset == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            offset_image[coord_int] = coords[i] - coord_int

        save_offset_dir = '/data/julia/data_lymph/npy_data/resampled_offset_{}/{}.npy'.format(mode, name)
        np.save(save_offset_dir, offset_image)

    if mask == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            mask_image[coord_int] = 1

        save_mask_dir = '/data/julia/data_lymph/npy_data/resampled_mask_{}/{}.npy'.format(mode, name)
        np.save(save_mask_dir, mask_image)
    
    info_dict = {}
    info_dict['whd'] = whd_image
    info_dict['offset'] = offset_image
    info_dict['mask'] = mask_image

    return info_dict
    


def crop_data(name, mode, nii_dir):

    image_dir = '/data/julia/data_lymph/npy_data/resampled_clamp_norm_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(image_dir):
        image_data = np.load(image_dir)
    else:
        image = tio.ScalarImage(nii_dir[name])

        prepro = tio.Compose([
            tio.Resample((0.8, 0.8, 1)),
            tio.Clamp(out_min=-160, out_max=240),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

        data_nii = prepro(image)
        image_data = data_nii.numpy()[0, :, :, :]
        np.save(image_dir, image_data)


    hmap_dir = '/data/julia/data_lymph/heatmaps/resampled_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(hmap_dir):
        hmap_data = np.load(hmap_dir)
    else:
        hmap_data = get_Heatmap(name, mode, '')

    whd_dir = '/data/julia/data_lymph/npy_data/resampled_whd_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(whd_dir):
        whd_data = np.load(whd_dir)
    else:
        whd_data = get_WHD_offset_mask(name, mode, whd=True)['whd'] 

    offset_dir = '/data/julia/data_lymph/npy_data/resampled_offset_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(offset_dir):
        offset_data = np.load(offset_dir)
    else:
        offset_data = get_WHD_offset_mask(name, mode, offset=True)['offset']

    mask_dir = '/data/julia/data_lymph/npy_data/resampled_mask_{}_npy/{}.npy'.format(mode, name)
    if os.path.isfile(mask_dir):
        mask_data = np.load(mask_dir)
    else:
        mask_data = get_WHD_offset_mask(name, mode, mask=True)['mask']

    voxelcoords = _getVoxelinfo(name, mode)['voxelcoords']
    # voxelwhds = getVoxelinfo(name, mode)['voxelWHDs']
    shape = list(image_data.shape)

    m, n, k = shape[0], shape[1], shape[2]
    a = 64
    if random.random() < 0.7:
        # print('============ random < 0.7 ====================')
        i = random.randint(0, len(voxelcoords)-1)
        center_crop = [random.randint(max(a, (int(voxelcoords[i][0])-a)), min((m - a), (int(voxelcoords[i][0])+a))), 
                       random.randint(max(a, (int(voxelcoords[i][1])-a)), min((n - a), (int(voxelcoords[i][1])+a))),
                       random.randint(max(a, (int(voxelcoords[i][2])-a)), min((k - a), (int(voxelcoords[i][2])+a)))]
    else:
        # print('============ random > 0.7 ====================')
        center_crop = [random.randint(a, m-a), random.randint(a, n-a), random.randint(a, k-a)]

    image_crop = image_data[center_crop[0]-a : center_crop[0]+a, 
                            center_crop[1]-a : center_crop[1]+a, 
                            center_crop[2]-a : center_crop[2]+a]
    hmap_crop = hmap_data[center_crop[0]-a : center_crop[0]+a,
                          center_crop[1]-a : center_crop[1]+a,
                          center_crop[2]-a : center_crop[2]+a]
    offset_crop = offset_data[center_crop[0]-a : center_crop[0]+a,
                              center_crop[1]-a : center_crop[1]+a,
                              center_crop[2]-a : center_crop[2]+a]
    whd_crop = whd_data[center_crop[0]-a : center_crop[0]+a,
                        center_crop[1]-a : center_crop[1]+a,
                        center_crop[2]-a : center_crop[2]+a]
    mask_crop = mask_data[center_crop[0]-a : center_crop[0]+a,
                          center_crop[1]-a : center_crop[1]+a,
                          center_crop[2]-a : center_crop[2]+a]    


    image_crop = torch.from_numpy(image_crop).type(torch.float32)
    hmap_crop = torch.from_numpy(hmap_crop).type(torch.float32)
    whd_crop = torch.from_numpy(whd_crop).type(torch.float32)
    offset_crop = torch.from_numpy(offset_crop).type(torch.float32)
    mask_crop = torch.from_numpy(mask_crop).type(torch.float32)
    # return image_crop, hmap_crop, bbox_crop

    return image_crop, hmap_crop, whd_crop, offset_crop, mask_crop




def _getVoxelinfo(name, mode):
    '''read the file and get the voxelinfo and the path
       return the dict'''
    # info = pd.read_csv(os.path.join('/data/julia/data_lymph', 'lymph_csv', mode + '_voxels.csv'))
    info = pd.read_csv('/data/julia/data_lymph/lymph_csv/resampled_{}.csv'.format(mode))
    # info = pd.read_csv('/data/julia/data_lymph/lymph_csv/transformed_{}.csv'.format(mode))
    # get the voxelcoordinate
    # print(info[info['name'] == "'" + name])
    voxels = info[info['name'] == "'" + name]
    info_dict = {}
    voxelcoords = []
    voxelWHDs = []
    for i in range(len(voxels)):
        voxel_x = voxels.iloc[i, 2]
        voxel_y = voxels.iloc[i, 3]
        voxel_z = voxels.iloc[i, 4]
        voxel_W = voxels.iloc[i, 5]
        voxel_H = voxels.iloc[i, 6]
        voxel_D = voxels.iloc[i, 7]
        voxelcoord = [np.float64(voxel_x), np.float64(voxel_y), np.float64(voxel_z)]
        voxelWHD = [np.float64(voxel_W), np.float64(voxel_H), np.float64(voxel_D)]
        voxelcoords.append(voxelcoord)
        voxelWHDs.append(voxelWHD)
    nii_path = voxels.iloc[0, 0]
    info_dict['voxelcoords'] = voxelcoords
    info_dict['voxelWHDs'] = voxelWHDs
    info_dict['path'] = nii_path

    return info_dict




def map_data(data, max, min):
    '''
    data: the data to map
    max : map to the max
    min : map to the min
    '''
    data_max = data.max()
    data_min = data.min()

    data = min + (max - min) / (data_max - data_min) * (data - data_min)
    return data



def get_Heatmap(name, mode, savenii_dir):

    dict_info = _getVoxelinfo(name=name, mode=mode)
    # the info is after resampling 
    voxelorigin = dict_info['voxelcoords'] 
    voxelWHDs = dict_info['voxelWHDs'] 
    nii_path = dict_info['path']
    image_nii = tio.ScalarImage(nii_path[2:-2])

    # to do the reampling !!!! it's a must
    resample = tio.Resample((0.8, 0.8, 1))
    resampled_image = resample(image_nii)

    _, h, w, d = resampled_image.shape

    heatmap = np.zeros((h, w, d), dtype=np.float32)
    for i in range(len(voxelorigin)):
        x, y, z = voxelorigin[i][0], voxelorigin[i][1], voxelorigin[i][2] 
        # bh, bw, bd = int(bbox_hwds[i][0]), int(bbox_hwds[i][1]), int(bbox_hwds[i][2])
        b_whd_half = int(min(voxelWHDs[i][0], voxelWHDs[i][1], voxelWHDs[i][2]) / 2)

        anchor = np.zeros((2 * b_whd_half + 1, 2 * b_whd_half + 1, 2 * b_whd_half + 1), dtype=np.float32)
        anchor[int(b_whd_half)][int(b_whd_half)][int(b_whd_half)] = 1
        anchor_hm = si.gaussian_filter(anchor, sigma=(2, 2, 2))
        anchor_hm = map_data(anchor_hm, max=1, min=1e-4)
        # if idx == 1:
        # embed()
        heatmap[int(x) - b_whd_half - 1 : int(x) + b_whd_half, int(y) - b_whd_half - 1 :int(y) + b_whd_half, int(z) - b_whd_half - 1: int(z) + b_whd_half] += anchor_hm[:, :, :]

    # save the heatmap to npy file
    np.save('/data/julia/data_lymph/heatmaps/resampled_{}_npy/{}.npy'.format(mode, name), heatmap)

    # # gerente the nii file
    # heatmap[heatmap > 0] = 1

    # # save the 0-1 heatmap as nii file
    # heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

    # nii = tio.ScalarImage(tensor=heatmap_tensor, affine=affine)
    # # print(affine)
    # # embed()
    # # Save the torchio image to nii file
    # nii.save(os.path.join(savenii_dir, '{}.nii'.format(name)))
    
    return heatmap




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


if __name__ == '__main__':
    from tqdm import tqdm
    path = '/data/julia/data_lymph'


    for mode in ['training', 'validation']:
        for name in tqdm(name_list):
            name_list, files_path = getNiiPath(path, mode)
            info_dict = get_WHD_offset_mask(name, mode, whd=True, offset=True, mask=True)