import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock, SimpleGate
from basicsr.models.archs.NAFSSR_arch import SCAM, DropPath


class SKM(nn.Module):
    def __init__(self, c, r = 2):
        super().__init__()
        self.channel = c
        self.r = r # 通道扩张倍数
        self.m = 3 # 这个表示有三个空洞卷积
        self.dilated_x2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_x4 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilated_x6 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=6, dilation=6)

        self.conv1x1 = nn.Conv2d(self.channel, self.channel * self.r, kernel_size=1, stride=1, padding=0) # 增大通道数
        self.relu = nn.ReLU(inplace=True)

        self.proj_weight = nn.Parameter(torch.zeros((self.channel * self.r, self.m * self.channel)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, H, W = x.size()
        d_1 = self.dilated_x2(x)
        d_2 = self.dilated_x4(x)
        d_3 = self.dilated_x6(x) # B, C, H, W

        F_s = d_1 + d_2 + d_3
        Z = torch.sum(F_s, dim=(-2, -1), keepdim=True) / (H * W) # B, C, 1, 1
        S = self.conv1x1(Z) # B, C*r, 1, 1
        S = self.relu(S).view(B, C * self.r) # B, C*r
        Sw = torch.matmul(S, self.proj_weight) 
        reshape_Sw = Sw.view(B, 1, self.m, C) 

        Sw = self.softmax(reshape_Sw) # B, 1, m, C
        chunks = Sw.chunk(self.m, dim=-2) # B, 1, 1, C

        Ds = [d_1, d_2, d_3]
        out = torch.zeros((B, C, H, W))
        for i, chunk in enumerate(chunks):
            out += Ds[i] * torch.transpose(chunk,1,3) # (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)

        return out



class Fusion_Block(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x_l, x_r):
        pass


class Normal_Block(nn.Module):

    def __init__(self):
        super().__init__()
        pass


    def forward(self, x):
        pass



class Stereo_Block(nn.Module):
    def __init__(self, channel, fusion=False, drop_out_rate=0., **kwargs):
        super().__init__()
        self.blk = Normal_Block(channel, drop_out_rate = drop_out_rate, **kwargs)
        self.fusion = Fusion_Block(channel, **kwargs) if fusion else None # 用fusion来控制每个块后面是否跟着融合块

    def forward(self, *feats): # 双目的情况下应该是 （x1,x2）
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class StereoNet(nn.Module):
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False, **kwargs):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # 自定义序列类 可以处理双目的情况 # 注意 有两个不同的 dropout 值
        self.body = MySequential(
            *[
                DropPath(
                    drop_rate = drop_path_rate, 
                    module = Stereo_Block(width, fusion=(fusion_from <= i and i <= fusion_to), drop_out_rate=drop_out_rate, **kwargs),
                       ) for i in range(num_blks)
            ]
        )
        # 上采样层
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale) # 这个主要是是将通道维数据重新排列 进而扩大分辨率
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out


class newNAFSSR(Local_Base, StereoNet):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        StereoNet.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        # 这个部分是论文里面说解决训练验证不一致的情况
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            # 下面这句话 调用了一个 replace_layer 的函数
            # 作用： 寻找所有的 AdaptiveAvgPool2d 层，并用一个自定义的 AvgPool2d 层替换它们
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)





if __name__ == '__main__':

    skm = SKM(c = 16)
    x = torch.randn((2, 16, 64, 64))

    out = skm(x)


    pass



# if __name__ == '__main__':
#     pass
    # num_blks = 128
    # width = 128
    # droppath=0.1
    # train_size = (1, 6, 30, 90)

    # net = NAFSSR(up_scale=2,train_size=train_size, fast_imp=True, width=width, num_blks=num_blks, drop_path_rate=droppath)

    # inp_shape = (6, 64, 64)

    # from ptflops import get_model_complexity_info
    # FLOPS = 0
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # # params = float(params[:-4])
    # print(params)
    # macs = float(macs[:-4]) + FLOPS / 10 ** 9

    # print('mac', macs, params)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




