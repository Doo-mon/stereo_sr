import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock, SimpleGate
from basicsr.models.archs.NAFSSR_arch import SCAM, DropPath



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
































# 下面的语句是测试用
if __name__ == '__main__':
    pass
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



