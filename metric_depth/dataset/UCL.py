import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

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