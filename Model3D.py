import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from Utils import *
from ModelUtils import *
import pdb

from PromptIR.Model_AMIR import ProxNet_Prompt
from codebook.model.network import Network

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


class DURE3D(nn.Module): ## without alpha, with two thr
    def __init__(self, Ch = 8, stages = 4, nc = 32):
        super(DURE3D, self).__init__()
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
        
        self.proxNet = ProxNet_Prompt(inp_channels=8, out_channels=8, dim=16, num_blocks=[1,1,1,2])
        self.proxNetCodeBook = Network(in_ch=8, n_e=1536, out_ch=8, stage=0, depth=8, unfold_size=2, opt=None, num_block=[1,1,1])


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
        M: LRMS, [B,D,h,w]
        P: PAN, [B,1,H,W]
        H,W = h * 4, w * 4
        """  
        Ft = F.interpolate(M , scale_factor = 4, mode = self.upMode)

        M = M.unsqueeze(1)
        P = P.unsqueeze(1).repeat(1,1,M.shape[2],1,1)
        print("P.shape",P.shape)
        Ft = Ft.unsqueeze(1)

        print(Ft.shape)
        B,C,D,H,W = Ft.shape
        K = Ft.clone()
         
        for i in range(0, self.s):           
            """
            H^T(HF-P) = H^T*H*F - H^T*P = IF - H^TP
            """
            # t = self.D(Ft)
            # print("t.shape: ", t.shape)
            ## F subproblem  

            Grad_F = self.DT(self.D(Ft) - M) + self.I(Ft) - self.HT(P) + self.alpha[i] * (Ft - K)
            F_middle = Ft - self.alpha_F[i] * Grad_F

            # F_middle.shape: [B, 1, C, H, W]
            # Ft,codebook_loss, _, _, _ = self.proxNetCodeBook(F_middle, one_hot)
            print("F_middle.shape:", F_middle.shape)
            exit(0)
            # Ft = self.proxNet(F_middle) 

            ## K subproblem
            Grad_K = self.alpha[i] * (K - Ft)
            K_middle = K - self.alpha_K[i] * Grad_K
            # K = self.proxNet(K_middle) 
            K,codebook_loss, _, _, _ = self.proxNetCodeBook(K_middle, one_hot)

        if C == 4:
            Ft = Ft.float().view(B, C, 2, H , W ).mean(dim=2)

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
    # torch.cuda.set_device(6)
    model = DURE3D(8, 4, 32).cuda()

    input = torch.rand(4, 8 ,32,32).cuda()
    P = torch.rand(4, 1 , 128, 128).cuda()
    one_hot = get_one_hot(1, 4)
    one_hot = one_hot.unsqueeze(0)

    output = model(input, P, one_hot)

    # print(output.shape)


    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 