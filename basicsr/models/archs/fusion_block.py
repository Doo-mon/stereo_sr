import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.NAFSSR_arch import SCAM
from basicsr.models.archs.NAFNet_arch import LayerNorm2d, SimpleGate
from einops import rearrange


# ============================ 这部分魔改 SCAM 块 ============================
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
    
class SDSCAM(nn.Module):
    def __init__(self, c, **kwargs):
        super().__init__()
        self.scale = c ** -0.5
        self.m = 3

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    
        self.dailated_conv_l= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dailated_conv_r= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=4, dilation=4)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        N_l = self.norm_l(x_l)
        N_r = self.norm_r(x_r)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        Q_l = self.l_proj1(N_l).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r = self.r_proj1(N_r).permute(0, 2, 3, 1)  # B, H, W, c

        K_l_T = self.dailated_conv_l(N_l).permute(0, 2, 1, 3) # B, H, c, W (transposed)
        K_r_T = self.dailated_conv_r(N_r).permute(0, 2, 1, 3) # B, H, c, W (transposed)


        attention_r2l = torch.matmul(Q_l, K_r_T) * self.scale
        attention_l2r = torch.matmul(Q_r, K_l_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention_r2l, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention_l2r, dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class MSDSCAM(nn.Module):
    def __init__(self, c, **kwargs):
        super().__init__()
        self.scale = c ** -0.5
        self.m = 3

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        

        self.dailated_conv_l0= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dailated_conv_r0= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=2, dilation=2)

        self.dailated_conv_l1= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dailated_conv_r1= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=4, dilation=4)

        self.dailated_conv_l2= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dailated_conv_r2= nn.Conv2d(c, c, kernel_size=3, stride=1, padding=6, dilation=6)

        self.conv1 = nn.Conv2d(3*c, c, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3*c, c, kernel_size=1, stride=1, padding=0)
        
        

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        N_l = self.norm_l(x_l)
        N_r = self.norm_r(x_r)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        Q_l = self.l_proj1(N_l).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r = self.r_proj1(N_r).permute(0, 2, 3, 1)  # B, H, W, c

        K_l_T = torch.cat((self.dailated_conv_l0(N_l),self.dailated_conv_l1(N_l),self.dailated_conv_l2(N_l)),dim=1)
        K_r_T = torch.cat((self.dailated_conv_r0(N_r),self.dailated_conv_r1(N_r),self.dailated_conv_r2(N_r)),dim=1)
        
        K_l_T = self.conv1(K_l_T).permute(0, 2, 1, 3) # B, H, c, W (transposed)
        K_r_T = self.conv2(K_r_T).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        attention_r2l = torch.matmul(Q_l, K_r_T) * self.scale
        attention_l2r = torch.matmul(Q_r, K_l_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention_r2l, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention_l2r, dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class CFM(nn.Module):
    '''
    这个主要是加了那个 mask
    '''
    def __init__(self, channel, t = 0.05, **kwargs):
        super().__init__()
        self.scale = channel ** -0.5
        self.t = t

        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1)

        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1)

        self.alpha = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

    
    def forward(self, x_l, x_r):

        F_u = self.conv1(self.norm1(x_l)) # B, C, H, W
        F_v = self.conv2(self.norm2(x_r))

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        S = torch.matmul(F_u.permute(0, 2, 3, 1), F_v.permute(0, 2, 1, 3)) * self.scale # B, C, H, H

        M_r2l = torch.softmax(S, dim=-1)
        M_l2r = torch.softmax(S.permute(0, 1, 3, 2), dim=-1)

        F_l2r = torch.matmul(M_l2r, self.conv3(x_l).permute(0, 2, 3, 1)) # B, H, W, C
        F_r2l = torch.matmul(M_r2l, self.conv4(x_r).permute(0, 2, 3, 1)) # B, H, W, C

        M_l2r = M_l2r.mean(dim=-1,keepdim=True).permute(0, 3, 1, 2) # B, 1, H, W
        M_r2l = M_r2l.mean(dim=-1,keepdim=True).permute(0, 3, 1, 2)

        V_l2r = (M_l2r > self.t).int()
        V_r2l = (M_r2l > self.t).int()
        
        F_l = F_l2r.permute(0, 3, 1, 2) * V_l2r * self.alpha
        F_r = F_r2l.permute(0, 3, 1, 2) * V_r2l * self.beta

        return x_l + F_l, x_r + F_r



# ============================ 这部分魔改 biPAM 块 ============================
class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed
class PAM(nn.Module):
    def __init__(self, channels, **kwargs):
        super(PAM, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.bq = nn.Conv2d(2*channels, channels, 1, 1, 0, groups=2, bias=True)
        self.bs = nn.Conv2d(2*channels, channels, 1, 1, 0, groups=2, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(2 * channels)
        self.bn = nn.BatchNorm2d(2 * channels)

    def __call__(self, x_left, x_right):
        b, c0, h0, w0 = x_left.shape

        catfea_left = torch.cat((self.conv1(x_left), x_left), 1)
        catfea_right = torch.cat((self.conv2(x_right), x_right), 1)


        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        
        return out_left, out_right




# ============================ 这部分魔改 Steformer 中的块 ============================


class MSMDIA(nn.Module):
    def __init__(self, c, num_heads=3,**kwargs):
        super().__init__()
        self.channel = c

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # layernorm
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # 1x1 conv
        self.qkv_l = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1)
        self.qkv_r = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1)

        # 3x3 dconv
        self.dwconv_l = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3)
        self.dwconv_r = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3)


        self.dailated_conv0= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dailated_conv1= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dailated_conv2= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=6, dilation=6)

        # conv1x1 for Q
        self.conv_Q = nn.Conv2d(self.channel*2*3, self.channel, kernel_size=1, padding=0, stride=1, groups=1)

        # output conv1x1
        self.conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1)

    
    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape

        norm_l = self.norm1(x_l)
        norm_r = self.norm2(x_r)

        Q_l, K_l, V_l = self.dwconv_l(self.qkv_l(norm_l)).chunk(3, dim=1)
        Q_r, K_r, V_r = self.dwconv_r(self.qkv_r(norm_r)).chunk(3, dim=1)
        Q = torch.cat((Q_l, Q_r), dim=1)

        Q = torch.cat((self.dailated_conv0(Q),self.dailated_conv1(Q),self.dailated_conv2(Q)),dim=1)
        Q = self.conv_Q(Q)

        # reshape
        Q = rearrange(Q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_l = rearrange(K_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_r = rearrange(K_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_l = rearrange(V_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_r = rearrange(V_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q  = torch.nn.functional.normalize(Q, dim=-1)
        K_l  = torch.nn.functional.normalize(K_l, dim=-1)
        K_r  = torch.nn.functional.normalize(K_r, dim=-1)


        A_r2l = (torch.matmul(Q, K_r.transpose(-2,-1)) * self.temperature).softmax(dim=-1)  # b head c c
        A_l2r = (torch.matmul(Q, K_l.transpose(-2,-1)) * self.temperature).softmax(dim=-1)

        F_l = rearrange(torch.matmul(A_r2l, V_l),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_r = rearrange(torch.matmul(A_l2r, V_r),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        F_l = self.conv1(F_l)
        F_r = self.conv2(F_r)

        return x_l + F_l, x_r + F_r

class AMSMDIA(nn.Module):
    def __init__(self, c, num_heads=3,**kwargs):
        super().__init__()
        self.channel = c

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # layernorm
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # 1x1 conv
        self.qkv_l = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1)
        self.qkv_r = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1)

        # 3x3 dconv
        self.dwconv_l = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3)
        self.dwconv_r = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3)


        self.dailated_conv0= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dailated_conv1= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dailated_conv2= nn.Conv2d(2*c, 2*c, kernel_size=3, stride=1, padding=6, dilation=6)

        # conv1x1 for Q
        self.conv_Q = nn.Conv2d(self.channel*2*3, self.channel, kernel_size=1, padding=0, stride=1, groups=1)

        # output conv1x1
        self.conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    
    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape

        norm_l = self.norm1(x_l)
        norm_r = self.norm2(x_r)

        Q_l, K_l, V_l = self.dwconv_l(self.qkv_l(norm_l)).chunk(3, dim=1)
        Q_r, K_r, V_r = self.dwconv_r(self.qkv_r(norm_r)).chunk(3, dim=1)
        Q = torch.cat((Q_l, Q_r), dim=1)

        Q = torch.cat((self.dailated_conv0(Q),self.dailated_conv1(Q),self.dailated_conv2(Q)),dim=1)
        Q = self.conv_Q(Q)

        # reshape
        Q = rearrange(Q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_l = rearrange(K_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_r = rearrange(K_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_l = rearrange(V_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_r = rearrange(V_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q  = torch.nn.functional.normalize(Q, dim=-1)
        K_l  = torch.nn.functional.normalize(K_l, dim=-1)
        K_r  = torch.nn.functional.normalize(K_r, dim=-1)


        A_r2l = (torch.matmul(Q, K_r.transpose(-2,-1)) * self.temperature).softmax(dim=-1)  # b head c c
        A_l2r = (torch.matmul(Q, K_l.transpose(-2,-1)) * self.temperature).softmax(dim=-1)

        F_l = rearrange(torch.matmul(A_r2l, V_l),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_r = rearrange(torch.matmul(A_l2r, V_r),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        F_l = self.conv1(F_l)*self.beta
        F_r = self.conv2(F_r)*self.gamma

        return x_l + F_l, x_r + F_r



class MDIA(nn.Module):
    '''Multi-Dconv Interactive Attention
    参考自  Restormer: Efficient Transformer for High-Resolution Image Restoration
    并由 Steformer: Efficient Stereo Image Super-Resolution with Transformer 改成双输入形式
    但是这个复现的不太准确 但是因为跑实验了,所以暂时保留 下面那个 MDIA_new 是完整的并且会用到RCSB块中
    '''
    def __init__(self, c, DW_Expand=2, **kwargs):
        super().__init__()
        self.channel = c
        self.new_channel = c * DW_Expand
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # 1x1 卷积扩大通道维
        self.conv1 = nn.Conv2d(self.channel, self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(self.channel, self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel * 2, self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True) # 这个因为要处理拼接的输入
        self.conv4 = nn.Conv2d(self.channel, self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(self.channel, self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 3x3 dconv 卷积
        self.dconv1 = nn.Conv2d(self.new_channel, self.new_channel, kernel_size=3, padding=1, groups=self.new_channel, bias=True)
        self.dconv2 = nn.Conv2d(self.new_channel, self.new_channel, kernel_size=3, padding=1, groups=self.new_channel, bias=True)
        self.dconv3 = nn.Conv2d(self.new_channel, self.new_channel, kernel_size=3, padding=1, groups=self.new_channel, bias=True)
        self.dconv4 = nn.Conv2d(self.new_channel, self.new_channel, kernel_size=3, padding=1, groups=self.new_channel, bias=True)
        self.dconv5 = nn.Conv2d(self.new_channel, self.new_channel, kernel_size=3, padding=1, groups=self.new_channel, bias=True)

        # softmax
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

        # output conv1x1
        self.conv6 = nn.Conv2d(self.new_channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv7 = nn.Conv2d(self.new_channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    
    def forward(self, x_l, x_r):

        norm_l = self.norm1(x_l)
        norm_r = self.norm2(x_r)
        cat_l_r = torch.cat((norm_l, norm_r), dim=1)

        V_l = self.dconv1(self.conv1(norm_l))
        K_l = self.dconv2(self.conv2(norm_l))
        Q = self.dconv3(self.conv3(cat_l_r))
        K_r = self.dconv4(self.conv4(norm_r)) 
        V_r = self.dconv5(self.conv5(norm_r)) 

        A_r2l = self.softmax1(torch.matmul(Q, K_r.permute(0, 1, 3, 2)) / np.sqrt(self.new_channel))
        A_l2r = self.softmax2(torch.matmul(Q, K_l.permute(0, 1, 3, 2)) / np.sqrt(self.new_channel))

        F_l = self.conv6(torch.matmul(A_r2l, V_l))
        F_r = self.conv7(torch.matmul(A_l2r, V_r))

        return x_l + F_l, x_r + F_r

class MDIA_new(nn.Module):
    ''' Multi-Dconv Interactive Attention 
    参考自  Restormer: Efficient Transformer for High-Resolution Image Restoration
    并由 Steformer: Efficient Stereo Image Super-Resolution with Transformer 改成双输入形式
    '''
    def __init__(self, c, num_heads=3, **kwargs):
        super().__init__()
        self.channel = c
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # layernorm
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # 1x1 conv
        self.qkv_l = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.qkv_r = nn.Conv2d(self.channel, self.channel*3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 3x3 dconv
        self.dwconv_l = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3, bias=True)
        self.dwconv_r = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, padding=1, groups=self.channel*3, bias=True)

        # conv1x1 for Q
        self.conv_Q = nn.Conv2d(self.channel*2, self.channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # output conv1x1
        self.conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    
    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape

        norm_l = self.norm1(x_l)
        norm_r = self.norm2(x_r)

        Q_l, K_l, V_l = self.dwconv_l(self.qkv_l(norm_l)).chunk(3, dim=1)
        Q_r, K_r, V_r = self.dwconv_r(self.qkv_r(norm_r)).chunk(3, dim=1)
        Q = torch.cat((Q_l, Q_r), dim=1)
        Q = self.conv_Q(Q)

        # reshape
        Q = rearrange(Q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_l = rearrange(K_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K_r = rearrange(K_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_l = rearrange(V_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_r = rearrange(V_r, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q  = torch.nn.functional.normalize(Q, dim=-1)
        K_l  = torch.nn.functional.normalize(K_l, dim=-1)
        K_r  = torch.nn.functional.normalize(K_r, dim=-1)


        A_r2l = (torch.matmul(Q, K_r.transpose(-2,-1)) * self.temperature).softmax(dim=-1)  # b head c c
        A_l2r = (torch.matmul(Q, K_l.transpose(-2,-1)) * self.temperature).softmax(dim=-1)

        F_l = rearrange(torch.matmul(A_r2l, V_l),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_r = rearrange(torch.matmul(A_l2r, V_r),'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        F_l = self.conv1(F_l)
        F_r = self.conv2(F_r)

        return x_l + F_l, x_r + F_r

class GDFN(nn.Module):
    '''Gated-Dconv feed-forward network
    参考自 Restormer: Efficient Transformer for High-Resolution Image Restoration
    进行了一些小修改
    '''
    def __init__(self, channel, gamma=2, **kwargs):
        super().__init__()
        self.new_channel = channel * gamma
        self.norm1 = LayerNorm2d(channel)

        self.conv1 = nn.Conv2d(channel, self.new_channel *2 , kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dconv1 = nn.Conv2d(self.new_channel*2, self.new_channel*2, kernel_size=3, padding=1, groups=self.new_channel*2, bias=True)
        # 将原来的激活函数 替换成简单门控和一个1x1卷积
        self.simple_gate = SimpleGate()
        self.conv2 = nn.Conv2d(self.new_channel //2 , self.new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.conv3 = nn.Conv2d(self.new_channel, channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, input):
        x = self.norm1(input)
        x1, x2 = self.dconv1(self.conv1(x)).chunk(2, dim=1)
        
        x2 = self.simple_gate(x2)
        x2 = self.conv2(x2)

        x = x1 * x2
        x = self.conv3(x)

        return x + input

class RCSB(nn.Module):
    '''Residual Cross Steformer Block
    参考自 Steformer: Efficient Stereo Image Super-Resolution with Transformer
    '''
    def __init__(self, channel, **kwargs):
        super().__init__()
        self.mida = MDIA_new(channel, **kwargs)
        self.gdfn1 = GDFN(channel, **kwargs)
        self.gdfn2 = GDFN(channel, **kwargs)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 =  nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x_l, x_r):
        F_l, F_r = self.mida(x_l, x_r)
        F_l = self.gdfn1(F_l)
        F_r = self.gdfn2(F_r)
        F_l = self.conv1(F_l)
        F_r = self.conv2(F_r)

        return F_l + x_l, F_r + x_r



if __name__ == '__main__':
    pass
    block = MSMDIA(48)

    x_l = torch.randn(2, 48, 64, 32)
    x_r = torch.randn(2, 48, 64, 32)

    out_l, out_r = block(x_l, x_r)
    print(out_l.size(), out_r.size())
    print(out_l, out_r)