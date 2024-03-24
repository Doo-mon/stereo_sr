import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, SimpleGate


class SKM(nn.Module):
    '''
    参考自 Stereo Image Restoration via Attention-Guided Correspondence Learning
    '''
    def __init__(self, c, r = 2, m = 4, **kwargs):
        super().__init__()
        self.channel = c
        self.r = r # 通道扩张倍数
        self.m = m # 这个表示有三个空洞卷积
        
        # ！！！ 不要简单的用python中的list来存储Module！！！
        self.dilated_conv_list = nn.ModuleList()
        for i in range(self.m):
            self.dilated_conv_list.append(nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=2*(i+1), dilation=2*(i+1)))

        self.conv1 = nn.Conv2d(self.channel, self.channel * self.r, kernel_size=1, stride=1, padding=0) # 增大通道数
        self.relu = nn.ReLU(inplace=True)

        self.proj_weight = nn.Parameter(torch.zeros((self.channel * self.r, self.m * self.channel)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, H, W = x.size()
        D_list = []
        F_s = None
        for dilated_conv in self.dilated_conv_list:
            D = dilated_conv(x)
            D_list.append(D)
            if F_s is not None:
                F_s += D
            else:
                F_s = D

        Z = torch.sum(F_s, dim=(-2, -1), keepdim=True) / (H * W) # B, C, 1, 1
        S = self.conv1(Z) # B, C*r, 1, 1
        S = self.relu(S).view(B, C * self.r) # B, C*r
        Sw = torch.matmul(S, self.proj_weight) 
        reshape_Sw = Sw.view(B, 1, self.m, C) 

        Sw = self.softmax(reshape_Sw) # B, 1, m, C
        chunks = Sw.chunk(self.m, dim=-2) # B, 1, 1, C

        out = None
        for i, chunk in enumerate(chunks):
            if out is not None:
                out += D_list[i] * torch.transpose(chunk, 1, 3) # (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
            else:
                out = D_list[i] * torch.transpose(chunk, 1, 3)
        return out


class SKSCAM(nn.Module):
    def __init__(self, c, **kwargs):
        super().__init__()
        self.scale = c ** -0.5

        self.l_proj1 = SKM(c, **kwargs)
        self.r_proj1 = SKM(c, **kwargs)

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r