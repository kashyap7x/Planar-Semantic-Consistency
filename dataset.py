import os
import json
import random
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from scipy.misc import imread, imresize
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline, interp2d
import pdb

trainID2Class = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}


# dataset list generation functions
def make_CityScapes(mode, root):
    # left view
    img_path = os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', mode)
    img_suffix = '_leftImg8bit.png'
    
    # segmentation label
    mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
    mask_suffix = '_gtFine_labelIds.png'
    
    # right view
    view2_path = os.path.join(root, 'rightImg8bit_trainvaltest', 'rightImg8bit', mode)
    view2_suffix = '_rightImg8bit.png'
    
    # intrinsics and extrinsics
    camera_path = os.path.join(root, 'camera_trainvaltest', 'camera', mode)
    camera_suffix = '_camera.json'
    
    # disparity map
    disp_path = os.path.join(root, 'disp_trainvaltest', 'disp', mode)
    disp_suffix = '_leftImg8bit_disp.npy'

    assert os.listdir(img_path) == os.listdir(mask_path)
    assert os.listdir(img_path) == os.listdir(view2_path)
    assert os.listdir(img_path) == os.listdir(camera_path)
    
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split(img_suffix)[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + img_suffix), 
                    os.path.join(mask_path, c, it + mask_suffix),
                    os.path.join(view2_path, c, it + view2_suffix), 
                    os.path.join(camera_path, c, it + camera_suffix),
                    os.path.join(disp_path, c, it + disp_suffix))
            items.append(item)
    return items


class CityScapes(torchdata.Dataset):
    def __init__(self, mode, root, cropSize=720, ignore_label=-1, max_sample=-1, is_train=1):
        self.list_sample = make_CityScapes(mode, root)
        self.mode = mode
        self.cropSize = cropSize
        self.is_train = is_train
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_npy(self, arr, h, w):
        y = np.arange(0, arr.shape[0]);
        x = np.arange(0, arr.shape[1]);
        spline = RectBivariateSpline(y,x, arr)
        yq = np.arange(0, h)
        xq = np.arange(0, w)
        arr_scale = spline.ev(yq[:,None], xq[None, :])
        return arr_scale
    
    def _scale_and_crop(self, img, seg, view2, disp, cropSize, is_train):
        h_s, w_s = 720, 1440
        img_scale = imresize(img, (h_s, w_s), interp='bilinear')
        seg = (seg + 1).astype(np.uint8)
        seg_scale = imresize(seg, (h_s, w_s), interp='nearest')
        seg_scale = seg_scale.astype(np.int) - 1
        view2_scale = imresize(view2, (h_s//8, w_s//8), interp='bilinear')
        disp_scale = self._scale_npy(disp, h_s//8, w_s//8)
        
        if is_train:
            # random crop
            x1_8 = random.randint(0, (w_s - cropSize)//8)
            y1_8 = random.randint(0, (h_s - cropSize)//8)
            x1 = x1_8 * 8
            y1 = y1_8 * 8
            img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
            view2_crop = view2_scale[y1_8: y1_8 + cropSize//8, x1_8: x1_8 + cropSize//8, :]
            disp_crop = disp_scale[y1_8: y1_8 + cropSize//8, x1_8: x1_8 + cropSize//8]
        else:
            # no crop
            img_crop = img_scale
            seg_crop = seg_scale
            view2_crop = view2_scale
            disp_crop = disp_scale
            
        return img_crop, seg_crop, view2_crop, disp_crop

    def _flip(self, img, seg, view2, disp):
        img_flip = img[:, ::-1, :].copy()
        seg_flip = seg[:, ::-1].copy()
        view2_flip = view2[:, ::-1, :].copy()
        disp_flip = disp[:, ::-1].copy()
        return img_flip, seg_flip, view2_flip, disp_flip

    def __getitem__(self, index):
        img_path, seg_path, view2_path, cam_path, disp_path = self.list_sample[index]
        
        
        img = imread(img_path, mode='RGB')
        seg = imread(seg_path, mode='P')
        view2 = imread(view2_path, mode='RGB')
        with open(cam_path) as f:
            cam = json.load(f)
        disp = np.load(disp_path)
        
        intrinsics = np.eye(3)
        intrinsics[0,0] = cam['intrinsic']['fx']
        intrinsics[1,1] = cam['intrinsic']['fy']
        intrinsics[0,2] = cam['intrinsic']['u0']
        intrinsics[1,2] = cam['intrinsic']['v0']
        
        baseline = cam['extrinsic']['baseline']
        
        seg_copy = seg.copy().astype(np.int)
        for k, v in self.id_to_trainid.items():
            seg_copy[seg == k] = v
        seg = seg_copy
        
        # random scale, crop, flip
        img, seg, view2, disp = self._scale_and_crop(img, seg, view2, disp,
                                        self.cropSize, self.is_train)
        if self.is_train and random.choice([-1, 1]) > 0:
            img, seg, view2, disp = self._flip(img, seg, view2, disp)

        # image to float
        img = img.astype(np.float32) / 255.
        img = img.transpose((2, 0, 1))
        
        view2 = view2.astype(np.float32) / 255.
        view2 = view2.transpose((2, 0, 1))
        
        # to torch tensor
        image = torch.from_numpy(img)
        segmentation = torch.from_numpy(seg)
        view2 = torch.from_numpy(view2)
        intrinsics = torch.from_numpy(intrinsics).type(torch.FloatTensor)
        disp = torch.from_numpy(disp).type(torch.FloatTensor)
        baseline = torch.from_numpy(np.array(baseline)).type(torch.FloatTensor)
        
        # normalize
        image = self.img_transform(image)
        view2 = self.img_transform(view2)
        
        return image, segmentation.long(), view2, intrinsics, baseline, disp, img_path

    def __len__(self):
        return len(self.list_sample)
    

def main():
   
    dataset = CityScapes('val', root='/home/selfdriving/datasets/cityscapes_full', max_sample = 32, is_train=0)
    h_s, w_s = 720, 1440
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=False)
    
    for batch_data in tqdm(loader):
        pass
        
#main()