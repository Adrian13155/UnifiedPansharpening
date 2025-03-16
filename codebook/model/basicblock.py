from collections import OrderedDict
from typing import Any
import torch.nn.functional as F
import torch.nn as nn
import torch

# from DEConv import DEConv
"""
--------------------------------------------
Kai Zhang, https://github.com/cszn/KAIR
--------------------------------------------
https://github.com/xinntao/BasicSR
--------------------------------------------
"""


def sequential(*args: Any):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64,
         out_channels=64,
         kernel_size=3,
         stride=1,
         padding=1,
         bias=True,
         mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias))
        elif t == 'T':
            L.append(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias))
        elif t == 'B':
            L.append(
                nn.BatchNorm2d(out_channels,
                               momentum=0.9,
                               eps=1e-04,
                               affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope,
                                  inplace=False))
        elif t == 'p':
            L.append(nn.PReLU())
        elif t == 'g':
            L.append(nn.GELU())
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                             padding=0))
        elif t == 'A':
            L.append(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                             padding=0))
        elif t == 'N':
            L.append(
                ConvNeXtBlock(dim=in_channels)
            )
        elif t == 'D':
            L.append(
                DEConv(dim=in_channels)
            )
            
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class SEResBlock(nn.Module):
    def __init__(self,
                 in_channels=32,
                 out_channels=32,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True):
        super(SEResBlock, self).__init__()

        # assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        # if mode[0] in ['R', 'L']:
        #     mode = mode[0].lower() + mode[1:]
        
        # self.conv_in = nn.Conv2d(in_channels=in_channels,
        #                   out_channels=out_channels,
        #                   kernel_size=kernel_size,
        #                   stride=stride,
        #                   padding=padding,
        #                   bias=bias)
        
        # self.act = nn.ReLU(inplace=True)
        
        # self.se_layer = SELayer(channel=in_channels)
        
        
        # self.conv_out =  nn.Conv2d(in_channels=in_channels,
        #                   out_channels=out_channels,
        #                   kernel_size=kernel_size,
        #                   stride=stride,
        #                   padding=padding,
        #                   bias=bias)
        
        
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.ReLU(inplace=True),
            SELayer(channel=in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=hidden_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=hidden_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=hidden_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=hidden_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=hidden_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=hidden_channels,
            #           out_channels=out_channels,
            #           kernel_size=kernel_size,
            #           stride=stride,
            #           padding=padding,
            #           bias=bias),
        )
        
    def forward(self, x):
        # x_in = self.conv_in(x)
        # x_in = self.se_layer(self.act(x_in))
        # x_in = self.conv_out(x_in)
        res = self.blocks(x)

        return x + res
    
class ResBlock(nn.Module):
    def __init__(self,
                 in_channels=96,
                 out_channels=96,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride,
                        padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


def upsample_convtranspose(in_channels=64,
                           out_channels=3,
                           kernel_size=2,
                           stride=2,
                           padding=0,
                           bias=True,
                           mode='2R',
                           negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in [
        '2', '3', '4'
    ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,
               mode, negative_slope)
    return up1


def downsample_strideconv(in_channels=64,
                          out_channels=64,
                          kernel_size=2,
                          stride=2,
                          padding=0,
                          bias=True,
                          mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in [
        '2', '3', '4'
    ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,
                 mode, negative_slope)
    return down1

# from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        # self.dwconv = DepthwiseConv2d(dim, dim, kernel_size=7, stride=1, padding=3) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 

        # x = input + x
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x