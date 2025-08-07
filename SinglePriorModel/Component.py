import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from PromptIR.Component import TransformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        
class D_DT(nn.Module):
    def __init__(self, channel, lin_dim,nc, scale=4):
        super(D_DT, self).__init__()
        self.channel = channel
        self.nc = nc
        self.img_emb = None
        if scale == 4 or scale == 2:
            self.img_emb = nn.Conv2d(self.channel, self.nc, kernel_size=3, stride=2, padding=1)
        else:
            self.img_emb = nn.Conv2d(self.channel, self.nc, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(lin_dim,nc)
        self.gamma = nn.Parameter(torch.zeros((1, self.nc, 1, 1)), requires_grad=True)
        self.beta  = nn.Parameter(torch.zeros((1, self.nc, 1, 1)), requires_grad=True)
        self.block = TransformerBlock(dim=nc, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
        self.down = None
        if scale == 4:
            self.down = nn.Conv2d(self.nc, self.channel, kernel_size=3, stride=2, padding=1)
        else: self.down = nn.Conv2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1)
        self.DT = None
        if scale == 4 : 
            self.DT = nn.Sequential(nn.ConvTranspose2d(self.channel, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=2, padding=1,output_padding=1))
        elif scale == 2:
            self.DT = nn.Sequential(nn.ConvTranspose2d(self.channel, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1))
        else:
            self.DT = nn.Sequential(nn.ConvTranspose2d(self.channel, self.nc, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x, M, text_emb):
        B,C,H,W = x.shape
        x = self.img_emb(x)
        gating_factors = torch.sigmoid(self.fc(text_emb))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta
        f = f * gating_factors
        f = self.block(f)

        x = f + x

        x = self.down(x) - M

        x = self.DT(x)

        return x

class H_HT(nn.Module):
    def __init__(self, channel, lin_dim,nc,scale=4):
        super(H_HT, self).__init__()
        self.channel = channel
        self.nc = nc
        self.img_emb = None
        if scale == 2 or scale == 1:
            self.img_emb = nn.ConvTranspose2d(self.channel, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1) # scale 2 or 1
        else:
            self.img_emb = nn.Conv2d(self.channel, self.nc, kernel_size=3, stride=1, padding=1) # scale 4

        self.fc = nn.Linear(lin_dim,nc)
        self.gamma = nn.Parameter(torch.zeros((1, self.nc, 1, 1)), requires_grad=True)
        self.beta  = nn.Parameter(torch.zeros((1, self.nc, 1, 1)), requires_grad=True)
        self.block = TransformerBlock(dim=nc, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
        self.down = None
        if scale == 1:
            self.down = nn.ConvTranspose2d(self.nc, 1, kernel_size=3, stride=2, padding=1,output_padding=1)
        else:
            self.down = nn.Conv2d(self.nc, 1, kernel_size=3, stride=1, padding=1)

        self.HT = None
        if scale == 4:
            self.HT = nn.Sequential(nn.ConvTranspose2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1))
        elif scale == 2:
            self.HT = nn.Sequential(nn.ConvTranspose2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1))
        else:
            self.HT = nn.Sequential(nn.Conv2d(1, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, self.channel, kernel_size=3, stride=1, padding=1))
        

    def forward(self, x, P, text_emb):
        B,C,H,W = x.shape
        x = self.img_emb(x)
        gating_factors = torch.sigmoid(self.fc(text_emb))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta
        f = f * gating_factors
        f = self.block(f)

        x = f + x

        x = self.down(x) - P

        x = self.HT(x)

        return x
    

if __name__ == "__main__":

    # model = nn.Conv2d(4, 4*2, kernel_size=1, bias=False)
    
    # print(sum(p.numel() for p in model.parameters() )/1e6, "M") 

    x = torch.rand(1, 8, 128, 128)
    M = torch.rand(1, 8, 32, 32)
    P = torch.rand(1, 1, 128, 128)
    text = torch.rand(1, 384)
    # model = D_DT(8, 384, 32)
    model = H_HT(8, 384, 32)
    output = model(x, P, text)
    print(output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M")