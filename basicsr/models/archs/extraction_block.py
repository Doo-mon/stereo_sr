import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from basicsr.models.archs.NAFNet_arch import LayerNorm2d, SimpleGate
from basicsr.models.archs.arch_util import trunc_normal_, window_partition, window_reverse

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


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    '''
    Channel attention used in RCAN.
    有略微的修改 把非线性的部分改成了simplegate
    '''
    def __init__(self, channel, squeeze_factor = 4):
        super(ChannelAttention, self).__init__()
        new_channel = channel // squeeze_factor
        self.simplegate_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels = channel, out_channels = new_channel, kernel_size = 1, padding = 0),
            SimpleGate(),
            nn.Conv2d(in_channels = new_channel // 2, out_channels = channel, kernel_size = 1, padding = 0),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.simplegate_ca(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=2, squeeze_factor=4):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            SimpleGate(),
            nn.Conv2d(num_feat // (compress_ratio * 2), num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
            )

    def forward(self, x):
        return self.cab(x)

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HAB(nn.Module):
    '''Hybrid Attention Block.
    参考自 Activating More Pixels in Image Super-Resolution Transformer
    有略微修改
    '''
    def __init__(self, dim, num_heads, compress_ratio=3, squeeze_factor=2, conv_scale=0.01, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = 5
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale
        self.mlp_hidden_dim = int(dim * mlp_ratio) # mlp 隐层的通道维数
        
        
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size,self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cab = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, drop=drop)

    def forward(self, x):
        B, C, H, W = x.size() #  训练的时候 B 48 30 90
        input_x = x.permute(0, 2, 3, 1) # B H W C

        # LayerNorm
        x = self.norm1(x) # B C H W 因为这里用的是nafssr中的norm 所以要用的是 B C H W

        # CAB
        conv_x = self.cab(x)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # W-MSA 部分处理
     
        # partition windows
        x_windows = window_partition(x.permute(0, 2, 3, 1), self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nw*b, window_size*window_size, c
        # W-MSA  # (num_windows*b, n, c)
        attn_windows = self.attn(x_windows, mask=None)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        attn_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        attn_x = attn_x.view(B, H * W, C)

        # add
        x = input_x.view(B, H * W, C) + attn_x + conv_x * self.conv_scale # B H*W C

        # FFN
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.mlp(self.norm2(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x



if __name__=="__main__":

    # block = HAB(dim=3, num_heads=3)
    # x = torch.randn((2, 48, 30, 90))
    # out = block(x)
    # print(out)

    pass