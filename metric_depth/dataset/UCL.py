import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class UCL(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):
        # ! #jiahan 生成的txt文件中，每一行是一个样本，第一个是depth图像的路径，第二个是rgb图像的路径
        img_path = self.filelist[item].split(' ')[1]
        depth_path = self.filelist[item].split(' ')[0]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # depth = 200.0 * cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/255.0  # mm
        depth = 200.0 * cv2.imread(depth_path, -1)/255.0  # mm
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # sample['valid_mask'] = (sample['depth'] <= 80)
        sample['valid_mask'] = (sample['depth'] >= 0)  
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
    

class UCL_flip_and_swap(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __swap(self, image, depth):
        if random.random() < 0.5:
            return image, depth
        
        h,w = depth.shape
        size = int((0.5*random.random()) * h)
        x1, y1 = random.randint(0, w-size), random.randint(0, h-size)
        x2, y2 = random.randint(0, w-size), random.randint(0, h-size)
        tmp_img = image[y1:y1+size, x1:x1+size].copy()
        image[y1:y1+size, x1:x1+size] = image[y2:y2+size, x2:x2+size]
        image[y2:y2+size, x2:x2+size] = tmp_img

        tmp_depth = depth[y1:y1+size, x1:x1+size].copy()
        depth[y1:y1+size, x1:x1+size] = depth[y2:y2+size, x2:x2+size]
        depth[y2:y2+size, x2:x2+size] = tmp_depth
        
        return image, depth
    
    def __flip(self, image, depth):
        h,w = depth.shape
        size = int((0.5*random.random()) * h)
        x1, y1 = random.randint(0, w-size), random.randint(0, h-size)
        def flip_help(img, x1, y1, size, axis=0):
            tmp = img[y1:y1+size, x1:x1+size].copy()
            tmp = cv2.flip(tmp, axis)
            img[y1:y1+size, x1:x1+size] = tmp
            return img
        if random.random() < 0.5:
            image = flip_help(image, x1, y1, size, 0)
            depth = flip_help(depth, x1, y1, size, 0)
        if random.random() < 0.5:
            image = flip_help(image, x1, y1, size, 1)
            depth = flip_help(depth, x1, y1, size, 1)
        return image, depth
    
    def __getitem__(self, item):
        # ! #jiahan 生成的txt文件中，每一行是一个样本，第一个是depth图像的路径，第二个是rgb图像的路径
        img_path = self.filelist[item].split(' ')[1]
        depth_path = self.filelist[item].split(' ')[0]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # depth = 200.0 * cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/255.0  # mm
        depth = 200.0 * cv2.imread(depth_path, -1)/255.0  # mm
        
        image, depth = self.__swap(image, depth)
        image, depth = self.__flip(image, depth)
        # cv2.imwrite("/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp.jpg", image*255)
        # cv2.imwrite("/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp_depth.jpg", depth)
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # sample['valid_mask'] = (sample['depth'] <= 80)
        sample['valid_mask'] = (sample['depth'] >= 0)  
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
    

class UCL_PPS(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        self.K = torch.tensor([[227.60416, 0.0, 237.5], [0.0, 227.60416, 237.5], [0.0, 0.0, 1.0]]) * 256.0 / 475.0
        self.K[-1,-1] = 1.0
        self.h = 256
        self.w = 256
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    
    def __getitem__(self, item):
        # ! #jiahan 生成的txt文件中，每一行是一个样本，第一个是depth图像的路径，第二个是rgb图像的路径
        img_path = self.filelist[item].split(' ')[1]
        depth_path = self.filelist[item].split(' ')[0]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # depth = 200.0 * cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/255.0  # mm
        depth = 200.0 * cv2.imread(depth_path, -1)/255.0  # mm
        
        # cv2.imwrite("/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp.jpg", image*255)
        # cv2.imwrite("/home/jiahan/jiahan/codes/Depth-Anything-V2/tmp_depth.jpg", depth)
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # sample['valid_mask'] = (sample['depth'] <= 80)
        sample['valid_mask'] = (sample['depth'] >= 0)  
        sample['lightness_mask'] = (sample['image'][:, :, :] > 0.9)[0:1]
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        sample['intri'] = self.K
        sample['h_w'] = (self.h, self.w)
        
        return sample

    def __len__(self):
        return len(self.filelist)