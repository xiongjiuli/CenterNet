import torch.utils.data as data
import torchio as tio
from preprocess import *
import torch


class lymphDataset(data.Dataset):
    # 创建dataset的类型,return出来的是crop之后的

    def __init__(self,
                path_root,
                people_list,
                mode,
                nii_path,
                patch_size=128,
                patch_overlap=20,

                ):
        
        super(lymphDataset, self).__init__()
        self.path_root = path_root
        self.mode = mode
        self.people_list = people_list
        self.nii_path = nii_path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap


    def __getitem__(self, idx):

        # input 
        name = self.people_list[idx]
        if self.mode == 'training':
            image_crop, hmap_crop, whd_crop, offset_crop, mask_crop = crop_data(name, self.nii_path)

            df = pd.read_csv('training_affine_resampled.csv')
            affine_xyz = df[df['name'] == "'" + name]
        
            affine = [[-0.8, 0, 0, 0],
                      [0, -0.8, 0, 0],
                      [0,    0, 1, 0],
                      [0,    0, 0, 1]]

            affine[0][-1] = affine_xyz['affine_x'].item()
            affine[1][-1] = affine_xyz['affine_y'].item()
            affine[2][-1] = affine_xyz['affine_z'].item()
            affine = np.array(affine)

            subject = tio.Subject(
                image = tio.ScalarImage(tensor=image_crop.unsqueeze(0), affine=affine),
                hmap = tio.ScalarImage(tensor=hmap_crop.unsqueeze(0), affine=affine),
                whd = tio.ScalarImage(tensor=whd_crop.unsqueeze(0), affine=affine),
                offset = tio.ScalarImage(tensor=offset_crop.unsqueeze(0), affine=affine),
                mask = tio.ScalarImage(tensor=mask_crop.unsqueeze(0), affine=affine),
            )
            # # augmentation
            # augmentation = tio.OneOf({
            #     tio.RandomFlip(axes=(0, 1, 2)) :0.19,
            #     tio.RandomAffine(scales=0.3, degrees=20, translation=30) :0.19,
            #     tio.RandomElasticDeformation() :0.05,
            #     tio.RandomBlur(std=(0.5, 1)) :0.19,
            #     tio.RandomNoise(std=0.1) :0.19,
            #     tio.RandomGamma(log_gamma=(-0.5, 0.5)) :0.19, 
            # })

            # image_subject = augmentation(subject)

            return subject

        else:   
            print('mode name have error')

    def __len__(self):

        return len(self.people_list)
       

if __name__ == '__main__':

    path_root = '/data/julia/data_lymph'
    data_type = 'training'
    size = 128
    people_list, nii_path = getNiiPath(path_root, data_type)
    













