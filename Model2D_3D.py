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
from Mamba.mambairunet_arch import MambaIRUNet

from PromptIR.SpatialChannelPrompt import SpatialChannelPrompt,SpatialChannelPromptWithJumpConnection
from codebook.model.model3D.network3D import Network3D
from channel_Adapt.DynamicChannelAdaptation import DynamicChannelAdaptation

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
        self.DT = nn.Sequential(nn.ConvTranspose2d(Ch, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, Ch, kernel_size=3, stride=2, padding=1,output_padding=1))
        self.D  = nn.Sequential(nn.Conv2d(Ch, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, Ch, kernel_size=3, stride=2, padding=1))


        ## The modules for learning the measurement matrix G and G^T
        self.HT = nn.Sequential(nn.ConvTranspose2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, Ch, kernel_size=3, stride=1, padding=1))
        self.H  = nn.Sequential(nn.Conv2d(Ch, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 1, kernel_size=3, stride=1, padding=1))  
        
        # self.proxNet = AdaIR(inp_channels=8, out_channels=8, dim = 8,num_blocks = [2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
        self.proxNet = SpatialChannelPrompt(dim=16, num_blocks=[4,6,6,8], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)

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
        if C == 4:
            M = M.repeat_interleave(2, dim=1)
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        B,_,H,W = Ft.shape
        K = Ft.clone()

        for i in range(0, self.s):
            ## F subproblem  

            Grad_F = self.DT(self.D(Ft) - M) + self.HT(self.H(Ft) - P) + self.alpha[i] * (Ft - K)
            F_middle = Ft - self.alpha_F[i] * Grad_F

            # F_middle.shape: [B, 1, C, H, W]
            Ft = self.proxNet(F_middle)

            ## K subproblem
            Grad_K = self.alpha[i] * (K - Ft)
            K_middle = K - self.alpha_K[i] * Grad_K
            # K = self.proxNet(K_middle) 
            if C == 4:
                K_middle = K_middle.float().view(B, C, 2, H , W ).mean(dim=2)
            # print("K_middle.shape: ", K_middle.shape)
            # K_middle = K_middle.unsqueeze(1)
            K,codebook_loss,_,_ = self.proxNetCodeBook(K_middle, one_hot)
            # print("K.shape: ", K.shape)
            # K = K.squeeze(1)
            if C == 4:
                K = K.repeat_interleave(2, dim=1)
        
        if C == 4:
            Ft = Ft.float().view(B, C, 2, H , W ).mean(dim=2)
        return Ft
    
class DURE2D_3DWithAdaptiveConv(nn.Module): ## without alpha, with two thr
    def __init__(self, Ch = 8, stages = 4, nc = 32):
        super(DURE2D_3DWithAdaptiveConv, self).__init__()
        self.s  = stages
        self.upMode = 'bilinear'
        self.nc = nc

        self.H4 = DynamicChannelAdaptation(in_channels=4,out_channels=1,kernel_size=7,embedding_dim=320,scale = 1,is_transpose = False)

        self.HT4 = DynamicChannelAdaptation(in_channels=1,out_channels=4,kernel_size=7,embedding_dim=320,scale = 1,is_transpose = True)

        self.D4 = DynamicChannelAdaptation(in_channels=4,out_channels=4,kernel_size=7,embedding_dim=320,scale = 4,is_transpose = False)

        self.DT4 = DynamicChannelAdaptation(in_channels=4,out_channels=4,kernel_size=7,embedding_dim=320,scale = 4,is_transpose = True)

        self.H8 = DynamicChannelAdaptation(in_channels=8,out_channels=1,kernel_size=7,embedding_dim=320,scale = 1,is_transpose = False)

        self.HT8 = DynamicChannelAdaptation(in_channels=1,out_channels=8,kernel_size=7,embedding_dim=320,scale = 1,is_transpose = True)

        self.D8 = DynamicChannelAdaptation(in_channels=8,out_channels=8,kernel_size=7,embedding_dim=320,scale = 4,is_transpose = False)

        self.DT8 = DynamicChannelAdaptation(in_channels=8,out_channels=8,kernel_size=7,embedding_dim=320,scale = 4,is_transpose = True)

        # self.proxNet = ProxNet_Prompt(inp_channels=8, out_channels=8, dim=16, num_blocks=[4,6,6,8]).cuda()
        # self.proxNet = PromptIRText(dim=16, num_blocks=[3,5,5,6]).cuda()
        self.proxNet = SpatialChannelPrompt(dim=24, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
        # self.proxNet = SpatialChannelPromptWithJumpConnection(dim=24, num_blocks=[2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)
        # self.proxNet = MambaIRUNet(inp_channels=4,out_channels=4,dim=16,num_blocks=[2, 4, 4, 6],num_refinement_blocks=4,mlp_ratio=2.,bias=False,dual_pixel_task=False).cuda()

        
        # self.proxNet = AdaIR(inp_channels=8, out_channels=8, dim = 24,num_blocks = [2,3,3,4], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)

        self.proxNetCodeBook = Network3D()
        # self.proxNetCodeBook = SpatialChannelPrompt(dim=16, num_blocks=[3,4,4,6], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)

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
        # temp = None

        for i in range(0, self.s):
            ## F subproblem  
            if C == 4:
                Grad_F = self.DT4(self.D4(Ft, one_hot) - M, one_hot) + self.HT4(self.H4(Ft,one_hot) - P, one_hot) + self.alpha[i] * (Ft - K)
            else:
                Grad_F = self.DT8(self.D8(Ft, one_hot) - M, one_hot) + self.HT8(self.H8(Ft,one_hot) - P, one_hot) + self.alpha[i] * (Ft - K)
            
            F_middle = Ft - self.alpha_F[i] * Grad_F

            # F_middle.shape: [B, C, H, W]
            Ft = self.proxNet(F_middle)

            ## K subproblem
            Grad_K = self.alpha[i] * (K - Ft)
            K_middle = K - self.alpha_K[i] * Grad_K

            K,codebook_loss,_,_ = self.proxNetCodeBook(K_middle, one_hot)
        return Ft
    
def get_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot
    
if __name__ == '__main__':
    # a = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
    # input1 = torch.randn(4, 1, 8, 128, 128)  # D=8
    # input2 = torch.randn(4, 1, 4, 128, 128)  # D=4

    # output1 = a(input1)
    # output1 = a(output1)
    # output1 = a(output1)
    # # output2 = a(input2)

    # print("output1.shape", output1.shape)
    # # print("output2.shape", output2.shape)
    # exit(0)


    # model = Network(in_ch=8, n_e=1536, out_ch=8, stage=0, depth=8, unfold_size=2, opt=None, num_block=[1,1,1]).cuda()
    # for name, module in model.named_modules():
    #     print("name:", name, "module", module)
    torch.cuda.set_device(4)
    model = DURE2D_3DWithAdaptiveConv(8, 4, 32).cuda()
    # model = DURE2D_3D(8, 4, 32).cuda()

    input = torch.rand(1, 8 ,32,32).cuda()
    P = torch.rand(1, 1 , 128, 128).cuda()
    text = torch.rand(1,384).cuda()
    one_hot = get_one_hot(2, 4)
    one_hot = one_hot.unsqueeze(0)
    # print(one_hot.shape)

    output = model(input, P, one_hot)

    print(output.shape)


    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 