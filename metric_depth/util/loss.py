import torch
from torch import nn
import numpy as np

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        # pred = pred.unsqueeze(3).repeat(1, 1, 1, 3) # zx
        
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class PPS_loss(nn.Module):
    def __init__(self, K, mu = 1.2):
        super().__init__()
        self.mu = mu
        self.K = K
    
    def forward(self, depth, depth_gt, mask):
        L_X_pred, A_X_pred, normal_N_pred = self.__compute_PPL(depth)
        L_X_gt, A_X_gt, normal_N_gt = self.__compute_PPL(depth_gt)

        PPS_pred = self.__compute_PPS(L_X_pred, A_X_pred, normal_N_pred)
        PPS_gt = self.__compute_PPS(L_X_gt, A_X_gt, normal_N_gt)

        loss = (mask * (PPS_pred - PPS_gt)).norm(dim=-1).mean()
        
        return loss
        

    def __compute_PPS(self, L_X, A_X, normal_N):
        B, H, W, _ = L_X.shape
        return A_X * (L_X.reshape(-1,1,3) @ normal_N.reshape(-1,1,3).transpose(1,2)).reshape(B,H,W,1)

    def __backproject_to_pointcloud_pytorch(self, depth):
        K = self.K
        B, H, W = depth.shape
        device = depth.device

        # Generate mesh grid for pixel coordinates
        x = torch.arange(W, device=device).repeat(B, H, 1)
        y = torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)

        # Normalize x and y with camera intrinsic parameters
        X = (x - K[0, 2]) * depth / K[0, 0]
        Y = (y - K[1, 2]) * depth / K[1, 1]
        Z = depth

        # Stack to get the 3D points in the camera coordinate system
        points = torch.stack([X, Y, Z], dim=-1)

        return points


    def __compute_PPL(self, depth_map):
        mu = self.mu
        
        B,H,W = depth_map.shape
        light_d_ = torch.zeros((B,H,W,3), device=depth_map.device, dtype=depth_map.dtype) # [3]
        light_d_[..., 2] = 1.0
        light_d = light_d_.view(-1,1,3).transpose(2,1) # [B, 3, 1] 光源方向
        points_X = self.__backproject_to_pointcloud_pytorch(depth_map) # [B, H, W, 3]
        points_p = points_X / points_X[..., 2:3] # [B, H, W, 3] 归一化平面
        torch.cuda.empty_cache()
        
        L_X = (points_X - points_p) / (points_X - points_p).norm(dim=-1, keepdim=True) # [B, H, W, 1] 归一化光线

        # A_X_up =  # [B, H, W, 1] 光线方向和光源方向的夹角
        # A_X_up = A_X_up ** (mu)
        # A_X_down = (points_X - points_p).norm(dim=-1, keepdim=True) # [B, H, W, 1] 光线方向和光源的距离
        A_X = ((L_X.view(-1,1,3) @ light_d).view(B,H,W,1)) ** mu / ((points_X - points_p).norm(dim=-1, keepdim=True)**2)

        normal_N = self.__compute_normals(points_X.view((B,H,W,3))) # [B, H, W, 3] 法向量
        # normal_N = self.__compute_normals2(points_X.view((B,H,W,3)).transpose(1,-1).transpose(2,3)) # [B, H, W, 3] 法向量
        del points_X, points_p, light_d
        return L_X, A_X, normal_N

    def __compute_normals(self, points):
        B, H, W, _ = points.shape
        # 初始化法向量张量，与点云相同形状
        normals = torch.zeros_like(points)
        normals[...,-1] = 1.0
        def __norm_vector(v):
            return v / (v.norm(p=2) + 1e-3)

        for b in range(B):
            for i in range(1, H-1):
                for j in range(1, W-1):
                    # 获取当前点
                    point = points[b, i, j]

                    # 定义邻域，这里简化为使用周围点
                    neighbors = points[b, max(i-1, 0):min(i+2, H), max(j-1, 0):min(j+2, W)]
                    p_N, p_NE, p_E, p_S, p_SW, p_W = neighbors[0,1], neighbors[0,2], neighbors[1,2], neighbors[2,1], neighbors[2,0], neighbors[1,0]
                    v_N, v_NE, v_E, v_S, v_SW, v_W = p_N - point, p_NE - point, p_E - point, p_S - point, p_SW - point, p_W - point

                    n_0, n_1, n_2 = torch.cross(v_E, v_NE), torch.cross(v_NE, v_N), torch.cross(v_N, v_W)
                    n_3, n_4, n_5 = torch.cross(v_W, v_SW), torch.cross(v_SW, v_S), torch.cross(v_S, v_E)
                    n_0, n_1, n_2 = __norm_vector(n_0), __norm_vector(n_1), __norm_vector(n_2)
                    n_3, n_4, n_5 = __norm_vector(n_3), __norm_vector(n_4), __norm_vector(n_5)
                    
                    normal = torch.mean(torch.stack([n_0, n_1, n_2, n_3, n_4, n_5]), dim=0)
                    normal = __norm_vector(normal)
                    
                    normals[b, i, j] = normal

        return normals

    def _image_derivatives(self, image,diff_type='center'):
        c = image.size(1)
        if diff_type=='center':
            sobel_x = 0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
            sobel_y = 0.5*torch.tensor([[0.0,1,0],[0,0,0],[0,-1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
        elif diff_type=='forward':
            sobel_x = torch.tensor([[0.0,0,0],[0,-1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
            sobel_y = torch.tensor([[0.0,1,0],[0,-1,0],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)

        dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=3)
        dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=3)
        return dp_du, dp_dv

    def __compute_normals2(self, pc, diff_type='center'):
        #pc (b,3,m,n)
        #return (b,3,m,n)
        dp_du, dp_dv = self._image_derivatives(pc,diff_type=diff_type)
        normal = torch.nn.functional.normalize( torch.cross(dp_du,dp_dv,dim=1)).transpose(1,-1).transpose(1,2)
        return normal

