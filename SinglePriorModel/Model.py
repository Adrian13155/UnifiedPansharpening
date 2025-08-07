import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from Utils import *
from ModelUtils import *
from SinglePriorModel.Component import Downsample,Upsample,ReduceChannelConv,D_DT,H_HT

from AdaIR.AdaIR import AdaIR
from PromptIR.SpatialChannelPrompt import SpatialChannelPrompt
from codebook.model.model3D.network3D import Network3D
from PromptIR.MoE.MoEProxnet import SpatialChannelPromptMoE,SpatialChannelPromptTextMoE

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


class DURE2D_3D(nn.Module): ## without alpha, with two thr
    """
    将深度展开算子和ProxNet都改成2D,保持codebook部分3D不变
    2D部分需要将通道统一为8,由于codebook返回和输入都会涉及到对应的通道
    codebook输入需要通道平均回来,输出将通道复制回去
    proxNet:
    * 3DPromptIR无任何提示 ProxNet_Prompt3D(inp_channels=1, out_channels=1, dim=8, num_blocks=[1,1,1,2])
    * 3DpromptIR+文本提示 ProxNet_Prompt3D_WithTextPrompt(inp_channels=1, out_channels=1, dim=8, num_blocks=[1,1,1,2])
    * WavBest: 
    * AdaIR:AdaIR(inp_channels=8, out_channels=8, dim = 24,num_blocks = [2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8],
        ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
    * AdaIR包含SpaFre:self.proxNet = AdaIRSpaFre(inp_channels=8, out_channels=8, dim = 20,num_blocks = [2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
    * MoreParam:self.proxNet = AdaIR(inp_channels=8, out_channels=8, dim = 32,num_blocks = [4,6,6,8], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
    * 包含Spatial prompt 和 Spectral Prompt 的proxNet:SpatialChannelPrompt(dim=16, num_blocks=[4,6,6,8], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
    * Mamba的Unet:MambaIRUNet(inp_channels=4,out_channels=4,dim=16,num_blocks=[4, 6, 6, 8],num_refinement_blocks=4,mlp_ratio=2.,bias=False,dual_pixel_task=False).cuda()
    """
    def __init__(self, Ch = 8, stages = 4, nc = 32):
        super(DURE2D_3D, self).__init__()
        self.s  = stages
        self.upMode = 'bilinear'
        self.nc = nc
        ## The modules for learning the measurement matrix D and D^T
        self.DT4 = nn.Sequential(nn.ConvTranspose2d(4, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, 4, kernel_size=3, stride=2, padding=1,output_padding=1))
        self.D4  = nn.Sequential(nn.Conv2d(4, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 4, kernel_size=3, stride=2, padding=1))
        
        self.DT8 = nn.Sequential(nn.ConvTranspose2d(8, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, 8, kernel_size=3, stride=2, padding=1,output_padding=1))
        self.D8  = nn.Sequential(nn.Conv2d(8, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 8, kernel_size=3, stride=2, padding=1))


        ## The modules for learning the measurement matrix G and G^T
        self.HT4 = nn.Sequential(nn.ConvTranspose2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, 4, kernel_size=3, stride=1, padding=1))
        self.H4  = nn.Sequential(nn.Conv2d(4, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 1, kernel_size=3, stride=1, padding=1))  
        
        self.HT8 = nn.Sequential(nn.ConvTranspose2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, 8, kernel_size=3, stride=1, padding=1))
        self.H8  = nn.Sequential(nn.Conv2d(8, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 1, kernel_size=3, stride=1, padding=1))  
        
        self.proxNet = SpatialChannelPrompt(dim=24, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)

        self.proxNetCodeBook = Network3D()

        checkpoint_path = "/data/cjj/projects/UnifiedPansharpening/experiment/03-27_23:14_3D Codebook Stage2 Shared and Task no gard/epoch=126.pth"
        
        checkpoint = torch.load(checkpoint_path)

        self.proxNetCodeBook.load_state_dict(checkpoint)


        self.alpha = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.alpha_F = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.alpha_K = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.alpha_K, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if m in self.proxNetCodeBook.modules():
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  

    def forward(self, M, P, one_hot): 
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        B,C,H,W = M.shape
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        B,_,H,W = Ft.shape
        K = Ft.clone()

        for i in range(0, self.s):
            ## F subproblem  

            if C == 4:
                Grad_F = self.DT4(self.D4(Ft) - M) + self.HT4(self.H4(Ft) - P) + self.alpha[i] * (Ft - K)
            else:
                Grad_F = self.DT8(self.D8(Ft) - M) + self.HT8(self.H8(Ft) - P) + self.alpha[i] * (Ft - K)
            F_middle = Ft - self.alpha_F[i] * Grad_F

            # F_middle.shape: [B, C, H, W]
            # print("F_middle.shape", F_middle.shape)
            Ft = self.proxNet(F_middle)

            ## K subproblem
            Grad_K = self.alpha[i] * (K - Ft)
            K_middle = K - self.alpha_K[i] * Grad_K

            K,codebook_loss,_,_ = self.proxNetCodeBook(K_middle, one_hot)

        return Ft

class DURE3Dwith2D(nn.Module): ## without alpha, with two thr
    def __init__(self, Ch = 8, stages = 4, nc = 32):
        super(DURE3Dwith2D, self).__init__()
        self.s  = stages
        self.upMode = 'bilinear'
        self.nc = nc
        ## The modules for learning the measurement matrix D and D^T
        self.DT = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))


        ## The modules for learning the measurement matrix G and G^T
        self.I = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        
        self.HT = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        
        # self.proxNet = ProxNet_Prompt2DWith3D(ce=8, dim=8, num_blocks=[1,1,1,2]).cuda()
        self.proxNet = SpatialChannelPrompt(dim=24, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)

        self.proxNetCodeBook = Network3D()

        checkpoint_path = "/data/cjj/projects/UnifiedPansharpening/experiment/03-27_23:14_3D Codebook Stage2 Shared and Task no gard/epoch=126.pth"
        
        checkpoint = torch.load(checkpoint_path)

        self.proxNetCodeBook.load_state_dict(checkpoint)


        self.alpha = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.alpha_F = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.alpha_K = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.alpha_K, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if m in self.proxNetCodeBook.modules():
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  

    def forward(self, M, P, one_hot): 
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)

        M = M.unsqueeze(1)
        P = P.unsqueeze(1).repeat(1,1,M.shape[2],1,1)
        Ft = Ft.unsqueeze(1)

        B,C,D,H,W = Ft.shape
        K = Ft.clone()

        for i in range(0, self.s):
            """
            H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
            """
            ## F subproblem  
            # print("Ft.shape", Ft.shape)
            Grad_F = self.DT(self.D(Ft) - M) + self.I(Ft) - self.HT(P) + self.alpha[i] *(Ft - K)
            # print("Grad_F.shape", Grad_F.shape)
            F_middle = Ft - self.alpha_F[i] * Grad_F

            # F_middle.shape: [B, 1, C, H, W]
            F_middle = F_middle.squeeze(1)
            # print("F_middle.shape", F_middle.shape)
            Ft = self.proxNet(F_middle)
            Ft = Ft.unsqueeze(1)
            # print("Ft.shape", Ft.shape)

           ## K subproblem
            Grad_K = self.alpha[i] * (K - Ft)
            K_middle = K - self.alpha_K[i] * Grad_K
            # K = self.proxNet(K_middle) 
            # print("K_middle.shape", K_middle.shape)
            K_middle = K_middle.squeeze(1)
            K,codebook_loss,_,_ = self.proxNetCodeBook(K_middle, one_hot)
            K = K.unsqueeze(1)
            # print("K.shape", K.shape)
        
        return Ft.squeeze(1)

class DURESinglePirorWithMoE(nn.Module): ## without alpha, with two thr
    def __init__(self, nc = 32):
        super(DURESinglePirorWithMoE, self).__init__()
        self.upMode = 'bicubic'
        self.nc = nc
        # operator encoder level 1
        self.DT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_encoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
         # operator encoder level 2
        self.DT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator encoder level 3
        self.DT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level3  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 1 ,1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 1
        self.DT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_decoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 2
        self.DT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_decoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.HT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.I_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))

        self.proxNet_encoder_level1 = SpatialChannelPromptMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        self.proxNet_encoder_level2 = SpatialChannelPromptMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_encoder_level3 = SpatialChannelPromptMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 32)
        self.proxNet_decoder_level2 = SpatialChannelPromptMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_decoder_level1 = SpatialChannelPromptMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        
        self.down1_2 = Downsample()
        self.down2_3 = Downsample()
        
        self.up3_2 = Upsample()
        self.up2_1 = Upsample()
        
        self.reduce_channel_level3_2 = ReduceChannelConv()
        self.reduce_channel_level2_1 = ReduceChannelConv()

        self.alpha_F = Parameter(0.1*torch.ones(5, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)


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
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        M = M.unsqueeze(1)
        P = P.unsqueeze(1).repeat(1,1,M.shape[2],1,1)
        Ft = Ft.unsqueeze(1)
        Loss = 0

        '''
        H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
        '''
        # 1: B,C,H,W
        Grad_F = self.DT_encoder_level1(self.D_encoder_level1(Ft) - M) + self.I_encoder_level1(Ft) - self.HT_encoder_level1(P)
        
        F_middle = Ft - self.alpha_F[0] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level1, loss_encoder_level1 = self.proxNet_encoder_level1(F_middle)
        Loss += loss_encoder_level1

        Ft = self.down1_2(Ft_encoder_level1)
        Ft = Ft.unsqueeze(1)
        #2:B,C,H/2,W/2
        Grad_F = self.DT_encoder_level2(self.D_encoder_level2(Ft) - M) + self.I_encoder_level2(Ft) - self.HT_encoder_level2(P)
        F_middle = Ft - self.alpha_F[1] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level2, loss_encoder_level2 = self.proxNet_encoder_level2(F_middle)
        Loss += loss_encoder_level2

        Ft = self.down2_3(Ft_encoder_level2)
        Ft = Ft.unsqueeze(1)
        # 3:B,C,H/4,W/4
        Grad_F = self.DT_encoder_level3(self.D_encoder_level3(Ft) - M) + self.I_encoder_level3(Ft) - self.HT_encoder_level3(P)
        F_middle = Ft - self.alpha_F[2] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level3, loss_encoder_level3 = self.proxNet_encoder_level3(F_middle)
        Loss += loss_encoder_level3

        Ft_encoder_level3 = self.up3_2(Ft_encoder_level3)
        Ft = self.reduce_channel_level3_2(torch.concat([Ft_encoder_level2,Ft_encoder_level3],1))

        # 2:B,C,H/2,W/2
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level2(self.D_decoder_level2(Ft) - M) + self.I_decoder_level2(Ft) - self.HT_decoder_level2(P)
        F_middle = Ft - self.alpha_F[3] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level2, loss_decoder_level2 = self.proxNet_decoder_level2(F_middle)
        Loss += loss_decoder_level2

        Ft_decoder_level2 = self.up3_2(Ft_decoder_level2)
        Ft = self.reduce_channel_level2_1(torch.concat([Ft_encoder_level1,Ft_decoder_level2],1))
        # 1: B,C,H,W
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level1(self.D_decoder_level1(Ft) - M) + self.I_decoder_level1(Ft) - self.HT_decoder_level1(P)
        F_middle = Ft - self.alpha_F[4] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level3, loss_decoder_level3 = self.proxNet_decoder_level1(F_middle)
        Loss += loss_decoder_level3
        

        return Ft_decoder_level3, Loss

class DURESinglePirorWithTextMoE(nn.Module): ## without alpha, with two thr
    def __init__(self, nc = 32):
        super(DURESinglePirorWithTextMoE, self).__init__()
        self.upMode = 'bicubic'
        self.nc = nc
        # operator encoder level 1
        self.DT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_encoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
         # operator encoder level 2
        self.DT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator encoder level 3
        self.DT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level3  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 1 ,1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 1
        self.DT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_decoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 2
        self.DT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_decoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.HT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.I_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))

        self.proxNet_encoder_level1 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        self.proxNet_encoder_level2 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_encoder_level3 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 32)
        self.proxNet_decoder_level2 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_decoder_level1 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        
        self.down1_2 = Downsample()
        self.down2_3 = Downsample()
        
        self.up3_2 = Upsample()
        self.up2_1 = Upsample()
        
        self.reduce_channel_level3_2 = ReduceChannelConv()
        self.reduce_channel_level2_1 = ReduceChannelConv()

        self.alpha_F = Parameter(0.1*torch.ones(5, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)


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

    def forward(self, M, P,text): 
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        M = M.unsqueeze(1)
        P = P.unsqueeze(1).repeat(1,1,M.shape[2],1,1)
        Ft = Ft.unsqueeze(1)
        Loss = 0

        '''
        H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
        '''
        # 1: B,C,H,W
        Grad_F = self.DT_encoder_level1(self.D_encoder_level1(Ft) - M) + self.I_encoder_level1(Ft) - self.HT_encoder_level1(P)
        
        F_middle = Ft - self.alpha_F[0] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level1, loss_encoder_level1 = self.proxNet_encoder_level1(F_middle,text)
        Loss += loss_encoder_level1

        Ft = self.down1_2(Ft_encoder_level1)
        Ft = Ft.unsqueeze(1)
        #2:B,C,H/2,W/2
        Grad_F = self.DT_encoder_level2(self.D_encoder_level2(Ft) - M) + self.I_encoder_level2(Ft) - self.HT_encoder_level2(P)
        F_middle = Ft - self.alpha_F[1] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level2, loss_encoder_level2 = self.proxNet_encoder_level2(F_middle,text)
        Loss += loss_encoder_level2

        Ft = self.down2_3(Ft_encoder_level2)
        Ft = Ft.unsqueeze(1)
        # 3:B,C,H/4,W/4
        Grad_F = self.DT_encoder_level3(self.D_encoder_level3(Ft) - M) + self.I_encoder_level3(Ft) - self.HT_encoder_level3(P)
        F_middle = Ft - self.alpha_F[2] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level3, loss_encoder_level3 = self.proxNet_encoder_level3(F_middle, text)
        Loss += loss_encoder_level3

        Ft_encoder_level3 = self.up3_2(Ft_encoder_level3)
        Ft = self.reduce_channel_level3_2(torch.concat([Ft_encoder_level2,Ft_encoder_level3],1))

        # 2:B,C,H/2,W/2
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level2(self.D_decoder_level2(Ft) - M) + self.I_decoder_level2(Ft) - self.HT_decoder_level2(P)
        F_middle = Ft - self.alpha_F[3] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level2, loss_decoder_level2 = self.proxNet_decoder_level2(F_middle, text)
        Loss += loss_decoder_level2

        Ft_decoder_level2 = self.up3_2(Ft_decoder_level2)
        Ft = self.reduce_channel_level2_1(torch.concat([Ft_encoder_level1,Ft_decoder_level2],1))
        # 1: B,C,H,W
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level1(self.D_decoder_level1(Ft) - M) + self.I_decoder_level1(Ft) - self.HT_decoder_level1(P)
        F_middle = Ft - self.alpha_F[4] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level3, loss_decoder_level3 = self.proxNet_decoder_level1(F_middle, text)
        Loss += loss_decoder_level3
        

        return Ft_decoder_level3, Loss

class DURESinglePirorWithAdaIRTextMoE(nn.Module): ## without alpha, with two thr
    """
    算子用3D的
    文本在ProxNet加入
    ProxNet为AdaIR(前3)+MoE(后2)
    
    """
    def __init__(self, nc = 32):
        super(DURESinglePirorWithAdaIRTextMoE, self).__init__()
        self.upMode = 'bicubic'
        self.nc = nc
        # operator encoder level 1
        self.DT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_encoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
         # operator encoder level 2
        self.DT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator encoder level 3
        self.DT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_encoder_level3  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 1 ,1), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.I_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_encoder_level3 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 1
        self.DT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))
        self.D_decoder_level1  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 2, 2), padding=1))
        self.I_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.HT_decoder_level1 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        # operator decoder level 2
        self.DT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=(1, 2, 2), padding=1,output_padding=(0, 1, 1)),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1, 1, 1), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.D_decoder_level2  = nn.Sequential(nn.Conv3d(1, self.nc, kernel_size=3, stride=(1, 2 ,2), padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, 1, kernel_size=3, stride=(1, 1, 1), padding=1))
        self.HT_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=(1,2,2), padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))
        self.I_decoder_level2 = nn.Sequential(nn.ConvTranspose3d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv3d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose3d(self.nc, 1, kernel_size=3, stride=1, padding=1))

        self.proxNet_encoder_level1 = AdaIR(inp_channels=8,out_channels=8,dim = 16,num_blocks = [1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias', decoder = True)
        self.proxNet_encoder_level2 = AdaIR(inp_channels=8,out_channels=8,dim = 16,num_blocks = [1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias', decoder = True)
        self.proxNet_encoder_level3 = AdaIR(inp_channels=8,out_channels=8,dim = 16,num_blocks = [1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias', decoder = True)
        self.proxNet_decoder_level2 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_decoder_level1 = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        
        self.down1_2 = Downsample()
        self.down2_3 = Downsample()
        
        self.up3_2 = Upsample()
        self.up2_1 = Upsample()
        
        self.reduce_channel_level3_2 = ReduceChannelConv()
        self.reduce_channel_level2_1 = ReduceChannelConv()

        self.alpha_F = Parameter(0.1*torch.ones(5, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)


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

    def forward(self, M, P,text): 
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        M = M.unsqueeze(1)
        P = P.unsqueeze(1).repeat(1,1,M.shape[2],1,1)
        Ft = Ft.unsqueeze(1)
        Loss = 0

        '''
        H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
        '''
        # 1: B,C,H,W
        Grad_F = self.DT_encoder_level1(self.D_encoder_level1(Ft) - M) + self.I_encoder_level1(Ft) - self.HT_encoder_level1(P)
        
        F_middle = Ft - self.alpha_F[0] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level1 = self.proxNet_encoder_level1(F_middle)

        Ft = self.down1_2(Ft_encoder_level1)
        Ft = Ft.unsqueeze(1)
        #2:B,C,H/2,W/2
        Grad_F = self.DT_encoder_level2(self.D_encoder_level2(Ft) - M) + self.I_encoder_level2(Ft) - self.HT_encoder_level2(P)
        F_middle = Ft - self.alpha_F[1] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level2 = self.proxNet_encoder_level2(F_middle)

        Ft = self.down2_3(Ft_encoder_level2)
        Ft = Ft.unsqueeze(1)
        # 3:B,C,H/4,W/4
        Grad_F = self.DT_encoder_level3(self.D_encoder_level3(Ft) - M) + self.I_encoder_level3(Ft) - self.HT_encoder_level3(P)
        F_middle = Ft - self.alpha_F[2] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_encoder_level3 = self.proxNet_encoder_level3(F_middle)

        Ft_encoder_level3 = self.up3_2(Ft_encoder_level3)
        Ft = self.reduce_channel_level3_2(torch.concat([Ft_encoder_level2,Ft_encoder_level3],1))

        # 2:B,C,H/2,W/2
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level2(self.D_decoder_level2(Ft) - M) + self.I_decoder_level2(Ft) - self.HT_decoder_level2(P)
        F_middle = Ft - self.alpha_F[3] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level2, loss_decoder_level2 = self.proxNet_decoder_level2(F_middle, text)
        Loss += loss_decoder_level2

        Ft_decoder_level2 = self.up3_2(Ft_decoder_level2)
        Ft = self.reduce_channel_level2_1(torch.concat([Ft_encoder_level1,Ft_decoder_level2],1))
        # 1: B,C,H,W
        Ft = Ft.unsqueeze(1)
        Grad_F = self.DT_decoder_level1(self.D_decoder_level1(Ft) - M) + self.I_decoder_level1(Ft) - self.HT_decoder_level1(P)
        F_middle = Ft - self.alpha_F[4] * Grad_F
        F_middle = F_middle.squeeze(1)
        Ft_decoder_level3, loss_decoder_level3 = self.proxNet_decoder_level1(F_middle, text)
        Loss += loss_decoder_level3
        

        return Ft_decoder_level3, Loss
    
class DURESinglePirorWithTextOPMoEProxNet(nn.Module): ## without alpha, with two thr
    """
    算子用2D的并且文本先验在算子中加入
    """
    def __init__(self, nc = 32):
        super(DURESinglePirorWithTextOPMoEProxNet, self).__init__()
        self.upMode = 'bicubic'
        self.nc = nc
        # operator encoder level 1
        self.D_DT_encoder4_1 = D_DT(channel=4,lin_dim=384,nc=self.nc,scale=4)
        self.D_DT_encoder8_1 = D_DT(channel=8,lin_dim=384,nc=self.nc,scale=4)
        self.H_HT_encoder4_1 = H_HT(channel=4,lin_dim=384,nc=self.nc,scale=4)
        self.H_HT_encoder8_1 = H_HT(channel=8,lin_dim=384,nc=self.nc,scale=4)

        # operator encoder level 2
        self.D_DT_encoder4_2 = D_DT(channel=4,lin_dim=384,nc=self.nc,scale=2)
        self.D_DT_encoder8_2 = D_DT(channel=8,lin_dim=384,nc=self.nc,scale=2)
        self.H_HT_encoder4_2 = H_HT(channel=4,lin_dim=384,nc=self.nc,scale=2)
        self.H_HT_encoder8_2 = H_HT(channel=8,lin_dim=384,nc=self.nc,scale=2)

        # operator encoder level 3
        self.D_DT_encoder4_3 = D_DT(channel=4,lin_dim=384,nc=self.nc,scale=1)
        self.D_DT_encoder8_3 = D_DT(channel=8,lin_dim=384,nc=self.nc,scale=1)
        self.H_HT_encoder4_3 = H_HT(channel=4,lin_dim=384,nc=self.nc,scale=1)
        self.H_HT_encoder8_3 = H_HT(channel=8,lin_dim=384,nc=self.nc,scale=1)

        # operator decoder level 2
        self.D_DT_decoder4_2 = D_DT(channel=4,lin_dim=384,nc=self.nc,scale=2)
        self.D_DT_decoder8_2 = D_DT(channel=8,lin_dim=384,nc=self.nc,scale=2)
        self.H_HT_decoder4_2 = H_HT(channel=4,lin_dim=384,nc=self.nc,scale=2)
        self.H_HT_decoder8_2 = H_HT(channel=8,lin_dim=384,nc=self.nc,scale=2)

        # operator decoder level 1
        self.D_DT_decoder4_1 = D_DT(channel=4,lin_dim=384,nc=self.nc,scale=4)
        self.D_DT_decoder8_1 = D_DT(channel=8,lin_dim=384,nc=self.nc,scale=4)
        self.H_HT_decoder4_1 = H_HT(channel=4,lin_dim=384,nc=self.nc,scale=4)
        self.H_HT_decoder8_1 = H_HT(channel=8,lin_dim=384,nc=self.nc,scale=4)

        self.proxNet_encoder_level1 = SpatialChannelPromptMoE(dim=12, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        self.proxNet_encoder_level2 = SpatialChannelPromptMoE(dim=12, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_encoder_level3 = SpatialChannelPromptMoE(dim=12, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 32)
        self.proxNet_decoder_level2 = SpatialChannelPromptMoE(dim=12, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 64)
        self.proxNet_decoder_level1 = SpatialChannelPromptMoE(dim=12, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True, base_size = 128)
        # dim=16, num_blocks=[1,2,2,3]
        self.down1_2 = Downsample()
        self.down2_3 = Downsample()
        
        self.up3_2 = Upsample()
        self.up2_1 = Upsample()
        
        self.reduce_channel_level3_2 = ReduceChannelConv()
        self.reduce_channel_level2_1 = ReduceChannelConv()

        self.alpha_F = Parameter(0.1*torch.ones(5, 1),requires_grad=True)
        self._initialize_weights()
        torch.nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)


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

    def forward(self, M, P,text): 
        """
        input:
        M: LRMS, [B,c,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """
        B,C,H,W = M.shape
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        Loss = 0

        '''
        H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
        '''
        # 1: B,C,H,W
        if C == 4:
            Grad_F = self.D_DT_encoder4_1(Ft,M,text) + self.H_HT_encoder4_1(Ft,P,text)
        else:
            Grad_F = self.D_DT_encoder8_1(Ft,M,text) + self.H_HT_encoder8_1(Ft,P,text)
        
        F_middle = Ft - self.alpha_F[0] * Grad_F
        Ft_encoder_level1,loss_encoder_level1 = self.proxNet_encoder_level1(F_middle)
        Loss = Loss + loss_encoder_level1
        Ft = self.down1_2(Ft_encoder_level1)
        #2:B,C,H/2,W/2
        if C == 4:
            Grad_F = self.D_DT_encoder4_2(Ft,M,text) + self.H_HT_encoder4_2(Ft,P,text)
        else:
            Grad_F = self.D_DT_encoder8_2(Ft,M,text) + self.H_HT_encoder8_2(Ft,P,text)
        F_middle = Ft - self.alpha_F[1] * Grad_F
        Ft_encoder_level2, loss_encoder_level2 = self.proxNet_encoder_level2(F_middle)
        Loss = Loss + loss_encoder_level2
        Ft = self.down2_3(Ft_encoder_level2)
        # 3:B,C,H/4,W/4
        if C == 4:
            Grad_F = self.D_DT_encoder4_3(Ft,M,text) + self.H_HT_encoder4_3(Ft,P,text)
        else:
            t = self.D_DT_encoder8_3(Ft,M,text)
            # print("t.shape",t.shape)
            Grad_F = self.D_DT_encoder8_3(Ft,M,text) + self.H_HT_encoder8_3(Ft,P,text)
        F_middle = Ft - self.alpha_F[2] * Grad_F
        Ft_encoder_level3,loss_encoder_level3 = self.proxNet_encoder_level3(F_middle)
        Loss = Loss + loss_encoder_level3
        Ft_encoder_level3 = self.up3_2(Ft_encoder_level3)
        Ft = self.reduce_channel_level3_2(torch.concat([Ft_encoder_level2,Ft_encoder_level3],1))

        # 2:B,C,H/2,W/2
        if C == 4:
            Grad_F = self.D_DT_decoder4_2(Ft,M,text) + self.H_HT_decoder4_2(Ft,P,text)
        else:
            Grad_F = self.D_DT_decoder8_2(Ft,M,text) + self.H_HT_decoder8_2(Ft,P,text)
        F_middle = Ft - self.alpha_F[3] * Grad_F
        Ft_decoder_level2, loss_decoder_level2 = self.proxNet_decoder_level2(F_middle)
        Loss += loss_decoder_level2

        Ft_decoder_level2 = self.up3_2(Ft_decoder_level2)
        Ft = self.reduce_channel_level2_1(torch.concat([Ft_encoder_level1,Ft_decoder_level2],1))
        # 1: B,C,H,W
        if C == 4:
            Grad_F = self.D_DT_decoder4_1(Ft,M,text) + self.H_HT_decoder4_1(Ft,P,text)
        else:
            Grad_F = self.D_DT_decoder8_1(Ft,M,text) + self.H_HT_decoder8_1(Ft,P,text)
        F_middle = Ft - self.alpha_F[4] * Grad_F
        Ft_decoder_level3, loss_decoder_level3 = self.proxNet_decoder_level1(F_middle)
        Loss += loss_decoder_level3
        

        return Ft_decoder_level3, Loss

if __name__ == '__main__':
    torch.cuda.set_device(4)
    # model = DURESinglePirorWithTextMoE(32).cuda()
    model = DURESinglePirorWithTextOPMoEProxNet(32).cuda()

    input = torch.rand(1, 8 ,32,32).cuda()
    P = torch.rand(1, 1 , 128, 128).cuda()
    text = torch.rand(1,384).cuda()

    output,loss = model(input, P,text)

    print(output.shape)


    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 
    print(sum(p.numel() for p in model.proxNet_decoder_level1.parameters() )/1e6, "M") 