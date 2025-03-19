import torch
import torch.nn as nn
import torch.nn.functional as F
import models.unfolding_model.basicblock as B
# import spectral as spy
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from typing import Any, List, Tuple
from matplotlib.pyplot import sca
# from basic1 import *
from .blockCNL import blockCNL
from .blockCNL import AgentCNL
from .arch.network_orig import Network as Codebook



class PretrainCodebook(nn.Module):
    def __init__(self, pretrained_path='', 
                       use_codebook=True):
        super().__init__()
        
        # self.device = torch.device('cuda:%d' % device)
        
        self.model = Codebook(in_ch=4, out_ch=4, stage=0, depth=3, unfold_size=2, opt={}, num_block=1, n_e=1536) ##, n_e=1536 depth=8, num_block=3
        self.model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=True)
        self.use_codebook = use_codebook
        
    def forward(self, x):
                
        f1, f2, f3 = self.model.encode(x)
        with torch.no_grad():
            if self.use_codebook:
                fq, codebook_loss, distance_map = self.model.vq_64(f3)
            else:
                fq = f3
            f1_d, fq, f2_d, f3_d = self.model.decode(fq)
            x_rec = self.model.conv_out(f1_d)
        return x_rec
    
    ## 只训练encoder部分
    # def train_parameters(self): 
    #     return list(self.model.conv_in.parameters()) + list(self.model.encoder_conv1.parameters()) + list(self.model.encoder_256.parameters()) + \
    #             list(self.model.down1.parameters()) + list(self.model.encoder_conv2.parameters()) + list(self.model.encoder_128.parameters()) + \
    #             list(self.model.down2.parameters()) + list(self.model.encoder_conv3.parameters()) + list(self.model.encoder_64.parameters())

    #训练encoder和decoder
    def train_parameters(self, train_decoder: bool=False):
        train_paras = list(self.model.conv_in.parameters()) + list(self.model.encoder_conv1.parameters()) + list(self.model.encoder_256.parameters()) + \
                list(self.model.down1.parameters()) + list(self.model.encoder_conv2.parameters()) + list(self.model.encoder_128.parameters()) + \
                list(self.model.down2.parameters()) + list(self.model.encoder_conv3.parameters()) + list(self.model.encoder_64.parameters())+\
                list(self.model.decoder_conv1.parameters()) + list(self.model.decoder_64.parameters()) +\
                           list(self.model.decoder_conv2.parameters()) + list(self.model.decoder_128.parameters()) +\
                           list(self.model.decoder_conv3.parameters()) + list(self.model.decoder_256.parameters()) +\
                           list(self.model.conv_out.parameters()) + list(self.model.up2.parameters()) +\
                           list(self.model.up3.parameters())
        # if train_decoder:
        #     train_paras += list(self.model.decoder_conv1.parameters()) + list(self.model.decoder_64.parameters()) +\
        #                    list(self.model.decoder_conv2.parameters()) + list(self.model.decoder_128.parameters()) +\
        #                    list(self.model.decoder_conv3.parameters()) + list(self.model.decoder_256.parameters()) +\
        #                    list(self.model.conv_out.parameters()) + list(self.model.up2.parameters()) +\
        #                    list(self.model.up3.parameters())
        return train_paras


class CUNet(nn.Module):
    def __init__(self, opt=None, ms_channels=4, pan_channel=1):
        super(CUNet, self).__init__()

        self.iter = 2
        self.nc =32
        self.net_z = PretrainCodebook('/data/zjk/pannet/best_G_qb_small.pth', use_codebook=True)  ##replace with codebook  best_G_gt2_small   best_G_qb_small
        # self.net_c = blockCNL() ## NL
        self.net_c = AgentCNL(channnels = 8, num_heads = 4)
        # self.net_c1 =  nn.Sequential(  
        #     nn.Conv2d(5, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 4, 3, padding=(3 - 1) // 2, stride=1)  
        # )
        # self.net_c2 =  nn.Sequential(  
        #     nn.Conv2d(4, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 4, 3, padding=(3 - 1) // 2, stride=1)  
        # )
        # self.net_c =  nn.Sequential(  
        #     nn.Conv2d(4, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 4, 3, padding=(3 - 1) // 2, stride=1)  
        # )
        # self.net_c = nn.ModuleList([  
        #     nn.Sequential(*[
        #     nn.Conv2d(4, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #     nn.PReLU(),  
        #     nn.Conv2d(64, 4, 3, padding=(3 - 1) // 2, stride=1)]) for _ in range(self.iter)
        # ])   
        #   + [  
        #     nn.Sequential(  
        #         nn.Conv2d(4, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #         nn.PReLU(),  
        #         nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),  
        #         nn.PReLU(),  
        #         nn.Conv2d(64, 4, 3, padding=(3 - 1) // 2, stride=1)  
        #     ) for _ in range(3)  
        # ])  
        self.in_channel = 9 
        # self.net_v = NetX(in_nc=9, nc_x=[32, 64, 128], nb=4)
        self.net_v = NetX(in_nc=9, nc_x=[32, 64], nb=4)
        # or below
        # NetX(in_nc=9, nc_x=[32, 64, 128], nb=4)  ##    nb = 4     
        # self.init_weights(0)
        # # self.blue=nn.ModuleList(blue)
        self.delta_k = torch.Tensor([1.0])
        self.gamma_k = torch.Tensor([1.0]) ##stride
        self.alpha = torch.Tensor([0.1])
        self.beta = torch.Tensor([0.1])

        self.alpha_stage = self.Make_Para(self.iter, self.alpha)
        self.beta_stage = self.Make_Para(self.iter, self.beta)
        self.delta_stage = self.Make_Para(self.iter, self.delta_k)
        self.gamma_stage = self.Make_Para(self.iter, self.gamma_k)
        
        self.up_factor = 4

        self.DT = nn.Sequential(nn.ConvTranspose2d(ms_channels, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, ms_channels, kernel_size=3, stride=2, padding=1,output_padding=1))
        self.D  = nn.Sequential(nn.Conv2d(ms_channels, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, ms_channels, kernel_size=3, stride=2, padding=1))
        
    def Make_Para(self, iters, para):
        para_dimunsq = para.unsqueeze(dim=0)
        para_expand = para_dimunsq.expand(iters, -1)
        para = nn.Parameter(data=para_expand, requires_grad=True)
        return para    
        # self.blue = blue_box()
        # self.reds = red_box()

        # self.layer_out = nn.Conv2d(in_channels=self.num_filters, out_channels=1,
        #                            kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
        # nn.init.xavier_uniform_(self.layer_out.weight.data)


    def forward(self, ms, pan, ums):
        """_summary_

        Args:
            ms (_type_): color low resolution
            pan (_type_): pan
            ums (_type_): color upsampling

        Returns:
            _type_: _description_
        """
        y = pan
        # d = input['d']
        d0 = ms ## MS
        x = ums ## upsampling UMS
        f_pred = x
        v = x
        z = x
        for i in range(self.iter):

            # d_t = d0 - torch.mul(self.eta_i[k].to(d0.device), torch.mul(mat,(d0 - d0_1)) + ( d0 - f_pred))
           
            z = self.net_z(f_pred)#Here is codebook
            
            # v_1 = v - self.beta_stage[i]*(v-f_pred)

            c = self.net_c(f_pred, y) ###这个是原版的
            # c1 = self.net_c1(torch.cat([f_pred, y], dim=1))
            # c2 = self.net_c2(c1)
            # c = self.net_c(c2)

            # c = self.net_c(f_pred, y)
            # print(c.shape)
            # print(y.shape)
            v = self.net_v(torch.cat([v, c, y], dim=1))
            # d2 = self.net_x(torch.cat([d0,f_pred],dim=1))#32
            # f_pred = self.reconnect(f_pred,  v, d0, self.alpha_stage[i], self.delta_stage[i], self.gamma_stage[i], i)
            f_pred = self.reconnect(f_pred, z, v, d0, self.alpha_stage[i], self.delta_stage[i], self.gamma_stage[i], i)
        
            # for p in range(self.iter):
        return f_pred
        
    def reconnect(self, f_pred, z, v, d0, alpha, delta, gamma, i):

        # down = torch.nn.functional.interpolate(f_pred, scale_factor=1/self.up_factor, mode='bicubic', align_corners=False)
        # err1 = torch.nn.functional.interpolate(down - d0, scale_factor=self.up_factor, mode='bicubic', align_corners=False)

        down = self.D(f_pred)
        err1 = self.DT(down - d0)

        # recon = torch.mul(1 - alpha*delta - alpha*gamma, f_pred)  +  torch.mul(alpha*gamma, v) - torch.mul(alpha, err1)
        recon = torch.mul(1 - alpha*delta - alpha*gamma, f_pred)   +  torch.mul(alpha*delta, z) +  torch.mul(alpha*gamma, v) - torch.mul(alpha, err1)
        # recon = torch.mul((1 - beta - eta - delta), x) + torch.mul(eta, v) + torch.mul(beta, y) + torch.mul(delta, f) ##not modify
        # print(alpha, delta, gamma)
        return recon
    
    def train_parameters(self):
        ## train_decoder=True代表decoder也训练
        return self.net_z.train_parameters(train_decoder=True) + [self.alpha_stage, self.delta_stage, self.gamma_stage] + list(self.net_c.parameters()) + list(self.net_v.parameters()) # + list(self.D.parameters()) + list(self.DT.parameters())
        ## （）代表decoder不训练，只训练encoder
        ## return self.net_z.train_parameters() + [self.alpha_stage, self.delta_stage, self.gamma_stage] + list(self.net_c.parameters()) + list(self.net_v.parameters())

#  train_parameter = [self.alpha_stage, self.delta_stage, self.gamma_stage, self.eta_stage, self.beta_stage, self.epsilon_stage] + \
#               list(self.priorr.parameters())# if not self.opts.two_codebook else []# + self.priorr.train_parameters(not_fix_block=not_fix_block) + list(self.priorr.parameters()) + list(self.calc_T.parameters()) + list(self.calc_R.parameters()) +
#         # if self.opts.codebook:
#         #     for x in self.priort:
#         #         train_parameter += x.train_parameters()
#         # else:
#         train_parameter += list(self.priort.parameters())
            
#         return train_parameter
# class red_box(nn.Module):
#     def __init__(self):
#         super(red_box, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(8, 64, 3,padding=1), # 12
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.Conv2d(64, 64, 3,padding=1), # 12
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.Conv2d(64, 4, 3,padding=1),
#             nn.BatchNorm2d(4)
#             # nn.PReLU(),
#             # nn.Conv2d(32, 1, 3,padding=1),
#             # nn.BatchNorm2d(1)
#         )

#     def forward(self, v, y):
#        # delh = config.fwd_op_adj_mod(h[:,0:1])
#         output = self.conv(torch.cat([v,y],dim=1))
#         output = v + output
#         return output

class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256],
                 nb: int = 4):
        super(NetX, self).__init__()
        self.layer_1 = nn.Conv2d(in_channels=in_nc, out_channels=32,
                                   kernel_size=3, padding=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_1.weight.data)
        self.m_down1 = B.sequential(
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[0], nc_x[1], bias=False, mode='2'))
        # self.m_down2 = B.sequential(
        #     *[
        #         B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
        #         for _ in range(nb)
        #     ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        # self.m_down3 = B.sequential(
        #     *[
        #         B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
        #         for _ in range(nb)
        #     ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        # self.m_up3 = B.sequential(
        #     B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
        #     *[
        #         B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
        #         for _ in range(nb)
        #     ])
        # self.m_up2 = B.sequential(
        #     B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
        #     *[
        #         B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
        #         for _ in range(nb)
        #     ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], 4, bias=False, mode='C')#nc_x[0]
        # self.m_tail1 = B.conv(in_nc, 16, bias=False, mode='C')#nc_x[0]

    def forward(self, x):
        x0 = x
        x1 = self.layer_1(x0)
        x2 = self.m_down1(x1)
        # x3 = self.m_down2(x2)
        # x4 = self.m_down3(x3)
        x = self.m_body(x2)
        # x = self.m_up3(x + x4)
        # x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) #+ x1[:, :-1, :, :] 
        # x = self.m_tail1(x)
        return x


# class K_ProNet(nn.Module):
#     def __init__(self, channels):
#         super(K_ProNet, self).__init__()
#         self.channels = channels

#         self.resk1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? ?nn.BatchNorm2d(self.channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? ?nn.BatchNorm2d(self.channels),
#         )
#         self.resk2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(self.channels, self.channels, kernel_size=3, stride =1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         )
#         self.resk3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ?nn.BatchNorm2d(self.channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         )
#         self.resk4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         )
#         self.resk5 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#         # ? nn.BatchNorm2d(self.channels),
#         )
#     def forward(self, input):
#         k1 = F.relu(input + 0.1 * self.resk1(input))
#         k2 = F.relu(k1 + 0.1 * self.resk2(k1))
#         k3 = F.relu(k2 + 0.1 * self.resk3(k2))
#         k4 = F.relu(k3 + 0.1 * self.resk4(k3))
#         k5 = F.relu(input + 0.1 * self.resk5(k4))
#         return k5
