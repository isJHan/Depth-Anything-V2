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

        loss = mask * (PPS_pred - PPS_gt).norm(dim=-1).mean()
        
        return loss
        

    def __compute_PPS(self, L_X, A_X, normal_N):
        B, H, W, _ = L_X.shape
        return A_X * (L_X.view(-1,3) @ normal_N.view(-1,3).T).view(B,H,W,1)

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
        light_d = torch.zeros((B,H,W,3), device=depth_map.device, dtype=depth_map.dtype) # [3]
        light_d[..., 2] = 1.0
        points_X = self.__backproject_to_pointcloud_pytorch(depth_map) # [B, H, W, 3]
        points_p = points_X / points_X[..., 2:3] # [B, H, W, 3] 归一化平面
        L_X = (points_X - points_p) / (points_X - points_p).norm(dim=-1, keepdim=True) # [B, H, W, 1] 归一化光线

        A_X_up = (L_X.view(-1,3) @ light_d.view(-1,3).T).view(B,H,W,1) # [B, H, W, 1] 光线方向和光源方向的夹角
        A_X_up = A_X_up ** (mu)
        A_X_down = (points_X - points_p).norm(dim=-1, keepdim=True) # [B, H, W, 1] 光线方向和光源的距离
        A_X = A_X_up / A_X_down

        normal_N = self.__compute_normals(points_X.view(B,-1,3)).view((B,H,W,3)) # [B, H, W, 3] 法向量
        return L_X, A_X, normal_N

    def __compute_normals(points):
        """
        计算点云的法向量。
        :param points: 点云，形状为(B, N, 3)，其中B是批次大小，N是点的数量。
        :return: 法向量，形状为(B, N, 3)。
        """
        B, N, _ = points.shape
        normals = torch.zeros_like(points)

        # 假设`neighbors`是一个形状为(B, N, K, 3)的张量，表示每个点的K个最近邻点。
        # 这里我们简化处理，直接使用`points`作为每个点的“邻居”。
        neighbors = points  # 这应该是通过k-NN得到的真实邻居

        # 计算局部协方差矩阵
        for i in range(B):
            for j in range(N):
                # 提取当前点的邻居
                P = neighbors[i, j, :, :]  # 假设所有点都是其自己的邻居
                # 计算协方差矩阵
                cov_matrix = torch.cov(P.T)
                # 计算特征值和特征向量
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                # 最小特征值对应的特征向量是法线方向
                normal = eigenvectors[:, 0]
                normals[i, j, :] = normal

        # 标准化法线向量
        normals = normals / torch.norm(normals, dim=2, keepdim=True)

        return normals

