import numpy as np
import cv2

import torch
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from depth_anything_v2.dpt import DepthAnythingV2

net_w, net_h = 518, 518
transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

def load_model(model_path, encoder='vitl', max_depth=200):
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'], strict=True)
    return model.eval()

# --------------- 生成图像 -----------------
# 生成一张中间黑，向四周逐渐变暗的图，距离中心相同距离的像素值相同
def generate_heatmap():
    heatmap = np.zeros((224, 224))
    center = (112, 112)
    for i in range(224):
        for j in range(224):
            heatmap[i, j] = 1 - np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) / 112
    return heatmap

# 给输入图片加高斯噪声
def add_gaussian_noise(image, mean=0, std=1):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# 生成一张高斯噪声的三通道图片
def generate_noise_image(h,w):
    noise = np.random.normal(0, 1, (h, w, 3))
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    return (noise*255).astype(np.uint8)


def swap_part(img, part1, part2, size):
    # h,w,_ = img.shape
    tmp = img[part1[0]:part1[0]+size[0], part1[1]:part1[1]+size[1]].copy()
    img[part1[0]:part1[0]+size[0], part1[1]:part1[1]+size[1]] = img[part2[0]:part2[0]+size[0], part2[1]:part2[1]+size[1]]
    img[part2[0]:part2[0]+size[0], part2[1]:part2[1]+size[1]] = tmp
    # tmp = img[:int(part1[1]), :int(w/2)].copy()
    # img[:int(h/2), :int(w/2)] = img[int(h/2):, int(w/2):]
    # img[int(h/2):, int(w/2):] = tmp
    return img

def flip_part(img, start, size):
    tmp = img[start[0]:start[0]+size[0], start[1]:start[1]+size[1]].copy()
    tmp = np.flip(tmp, axis=0)
    img[start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = np.flip(tmp, axis=1)
    return img

# -------------------- 展示结果 -------------------

def show_3d(pred):
    # 三维展示像素值大小
    h,w = pred.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, pred, cmap='coolwarm')
    plt.show()

def align(pred,gt):
    return pred * (np.median(gt) / np.median(pred))

def show_metric(pred, gt):
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))
    absrel = np.median(np.abs(pred - gt) / (gt+1e-6))
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"ABS REL: {absrel}")