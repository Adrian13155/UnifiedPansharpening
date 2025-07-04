import torch.nn as nn
import torch.nn.functional as F

class ReduceChannelConv(nn.Module):
    def __init__(self):
        super(ReduceChannelConv, self).__init__()
        self.body4 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1, bias=False)
        )
        self.body8 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, bias=False)
        )

    def forward(self, x):
        _,C,_,_ = x.shape
        if C == 8:
            return self.body4(x)
        else:
            return self.body8(x)

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.body4 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1, bias=False)
        )
        self.body8 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        _,C,_,_ = x.shape
        if C == 4:
            return self.body4(x)
        else:
            return self.body8(x)

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

        self.body4 = nn.Sequential(
            nn.ConvTranspose2d(
                4,                # 输入通道数
                4,                # 输出通道数（保持不变）
                kernel_size=3,         # 卷积核大小
                stride=2,  # 步长=上采样因子（默认2）
                padding=1,             # 填充
                output_padding=1       # 确保尺寸正确放大
            )
        )

        self.body8 = nn.Sequential(
            nn.ConvTranspose2d(
                8,                # 输入通道数
                8,                # 输出通道数（保持不变）
                kernel_size=3,         # 卷积核大小
                stride=2,  # 步长=上采样因子（默认2）
                padding=1,             # 填充
                output_padding=1       # 确保尺寸正确放大
            )
        )

    def forward(self, x):
        _,C,_,_ = x.shape
        if C == 4:
            return self.body4(x)
        else:
            return self.body8(x)