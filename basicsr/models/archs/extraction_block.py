import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, SimpleGate

class MODEM(nn.Module):
    '''
    参考自 Multi-orientation depthwise extraction for stereo image super-resolution
    有略微的修改
    '''
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., **kwargs):
        super().__init__()
        new_channel = c * DW_Expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=new_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 三个深度卷积
        self.num_dconv = 3
        self.dconv1 = nn.Conv2d(in_channels=new_channel, out_channels=new_channel, kernel_size=(1,3), padding=(0,1), groups=new_channel, bias=True)
        self.dconv2 = nn.Conv2d(in_channels=new_channel, out_channels=new_channel, kernel_size=(3,3), padding=(1,1), groups=new_channel, bias=True)
        self.dconv3 = nn.Conv2d(in_channels=new_channel, out_channels=new_channel, kernel_size=(3,1), padding=(1,0), groups=new_channel, bias=True)

        self.sg = SimpleGate() # 经过这个 通道维减半
        self.conv2 = nn.Conv2d(in_channels = (new_channel * self.num_dconv) // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 其实就是nafssr中的sca
        self.cas = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c , kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        ffn_channel = FFN_Expand * c
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(inp)
        x = self.conv1(x)

        f1 = self.dconv1(x) # B 2C H W
        f2 = self.dconv2(x)
        f3 = self.dconv3(x)
        f = torch.cat((f1, f2, f3), dim = 1) # B 6C H W

        x = self.sg(f) # B 3C H W
        x = self.conv2(x) # B C H W
        x = x * self.cas(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma