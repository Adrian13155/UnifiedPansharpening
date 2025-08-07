import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from PromptIR.Component import Upsample,Downsample

from AdaIR.AdaIR import AdaIR
from PromptIR.SpatialChannelPrompt import SpatialChannelPrompt
from codebook.model.model3D.network3D import Network3D
from PromptIR.MoE.MoEProxnet import SpatialChannelPromptMoE,SpatialChannelPromptTextMoE
from PromptIR.Model_AMIR import *

def setup_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seeds(1)

class BaseBlock(nn.Module):
    def __init__(self, level = 1,dim=64, nc = 64, num_heads=8, num_blocks=8):
        super(BaseBlock, self).__init__()
        self.alpha_F = Parameter(0.1*torch.ones(1),requires_grad=True)
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)
        self.DT = nn.Sequential(nn.ConvTranspose3d(1, nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D = nn.Sequential(nn.Conv3d(1, nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I = nn.Sequential(nn.ConvTranspose3d(1, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(nc, 1, kernel_size=3, stride=1, padding=1))
        if level == 1:
            self.HT = nn.Sequential(nn.ConvTranspose3d(1, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(nc, 1, kernel_size=3, stride=1, padding=1))
        elif level == 2:
            self.HT = nn.Sequential(nn.ConvTranspose3d(1, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=(1, 2, 2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(nc, 1, kernel_size=3, stride=1, padding=1))
        elif level == 3:
            self.HT = nn.Sequential(nn.ConvTranspose3d(1, nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=(1, 2, 2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(nc, nc, kernel_size=3, stride=(1, 2, 2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(nc, 1, kernel_size=3, stride=1, padding=1))
        self.proxNet = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias") for i in range(num_blocks)])
    
    def forward(self, Ft, M, P):
        P = P.repeat(1,M.shape[1],1,1)
        input = Ft.unsqueeze(1)
        M = M.unsqueeze(1)
        P = P.unsqueeze(1)
        Grad_F = self.DT(self.D(input) - M) + self.I(input) - self.HT(P)
        F_middle = input - self.alpha_F[0] * Grad_F
        F_middle.squeeze(1)
        # _,C,_,_ = F_middle.shape
        F_middle = F_middle.squeeze(1)
        output = self.proxNet(F_middle)

        return output + Ft

class DURESinglePriorWithTransformerProxNet(nn.Module):
    def __init__(self, dim = 24, nc = 32):
        super(DURESinglePriorWithTransformerProxNet, self).__init__()
        self.upMode = 'bicubic'
        self.nc = nc
        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        self.patch_embed4_Ft = OverlapPatchEmbed(4, dim)
        self.patch_embed8_Ft = OverlapPatchEmbed(8, dim)

        heads = [2,4,8]
        blocks = [3,3,4]
        
        self.down1_2 = Downsample(int(dim * 2 ** 0))
        self.down1_2_Ft = Downsample(int(dim * 2 ** 0))
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.down2_3_Ft = Downsample(int(dim * 2 ** 1))
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.up3_2_Ft = Upsample(int(dim * 2 ** 2))
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.up2_1_Ft = Upsample(int(dim * 2 ** 1))

        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1)
        
        self.encoder_level_1 = BaseBlock(1, dim * 2 ** 0, nc = 32 * 2 ** 0, num_heads=heads[0], num_blocks=blocks[0])
        self.encoder_level_2 = BaseBlock(2, dim * 2 ** 1, nc = 32 * 2 ** 1, num_heads=heads[1], num_blocks=blocks[1])
        self.decoder_level_3 = BaseBlock(3, dim * 2 ** 2, nc = 32 * 2 ** 2, num_heads=heads[2], num_blocks=blocks[2])
        self.decoder_level_2 = BaseBlock(2, dim * 2 ** 1, nc = 32 * 2 ** 1, num_heads=heads[1], num_blocks=blocks[1])
        self.decoder_level_1 = BaseBlock(1, dim * 2 ** 0, nc = 32 * 2 ** 0, num_heads=heads[0], num_blocks=blocks[0])
        
        self.output4 = nn.Conv2d(int(dim*2**0), 4, kernel_size=3, stride=1, padding=1)
        self.output8 = nn.Conv2d(int(dim*2**0), 8, kernel_size=3, stride=1, padding=1)
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # 如果有偏置，将其初始化为 0
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, M, P):
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        B,C,H,W = M.shape
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        if C == 4:
            M = self.patch_embed4(M)
            inp_enc_level1 =self.patch_embed4_Ft(Ft)
        else:
            M = self.patch_embed8(M)
            inp_enc_level1 = self.patch_embed8_Ft(Ft)

        out_enc_level1 = self.encoder_level_1(inp_enc_level1, M, P)
        M = self.down1_2(M)
        out_enc_level2 = self.encoder_level_2(self.down1_2_Ft(out_enc_level1), M, P)
        M = self.down2_3(M)
        out_dec_level3 = self.decoder_level_3(self.down2_3_Ft(out_enc_level2), M, P)
        out_dec_level3 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([out_dec_level3, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        M = self.up3_2(M)

        out_dec_level2 = self.decoder_level_2(inp_dec_level2, M, P)
        out_dec_level2 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([out_dec_level2, out_enc_level1], 1)

        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        M = self.up2_1(M)
        out_dec_level1 = self.decoder_level_1(inp_dec_level1, M,P)

        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1)
        else:
            out_dec_level1 = self.output8(out_dec_level1)

        return out_dec_level1
 


if __name__ == '__main__':
    torch.cuda.set_device(2)
    model = DURESinglePriorWithTransformerProxNet().cuda()
    
    
    M = torch.rand(1, 4 ,32,32).cuda()
    Ft = F.interpolate(M , scale_factor = 4, mode = "bicubic")
    P = torch.rand(1, 1 , 128, 128).cuda()
    # text = torch.rand(1,384).cuda()

    output = model(M, P)

    print(output.shape)


    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 
    # print(sum(p.numel() for p in model.proxNet_decoder_level1.parameters() )/1e6, "M") 