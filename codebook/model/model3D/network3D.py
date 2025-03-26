import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from torch import nn
from codebook.model.model3D.MSAB3D import MSAB3D
from codebook.model.vq import SharedAndTaskSpecificCodebookVectorQuantizer3D
import torch
from torch.nn import functional as F
from codebook.model.feature_converter import TransformerBlock
from codebook.model.vgg import Vgg19

channel_query_dict = {
    32: 128,
    64: 64,
    128: 32,
}

# channel_query_dict = {
#     16: 128,
#     32: 64,
#     64: 32,
#     128: 16,
# }



class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        self.block = nn.Sequential(#nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1),
                                   MSAB3D(dim=out_ch, num_blocks=num_blocks, dim_head=out_ch//4, heads=4))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        x = self.block(x)
        return x

from codebook.model.restormer import CrossTransformerBlock
class BasicCrossChannelAttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        ksz = 3
        self.blocks = nn.ModuleList()
        self.conv_x = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        self.conv_y = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        for i in range(num_blocks):
            self.blocks.append(CrossTransformerBlock(dim=out_ch, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, y):
        x, y = self.conv_x(x), self.conv_y(y)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, y)
        return x


class VQModule(nn.Module):
    def __init__(self,n_e_shared=512, n_e_task=128, e_dim_shared=1024, e_dim_task=4, depth=3, num_tasks=4):
        super().__init__()
        self.quantize = SharedAndTaskSpecificCodebookVectorQuantizer3D(n_e_shared=n_e_shared, n_e_task=n_e_task, e_dim_shared=e_dim_shared, e_dim_task=e_dim_task, depth=depth, num_tasks=num_tasks)

    def forward(self, x, one_hot):
        return self.quantize(x, one_hot)

    def forward_with_query(self, x, query):
        code_book = self.quantize.embedding.weight
        x_unfold = self.quantize.unfold(x).permute(0, 2, 1).reshape(-1, self.quantize.e_dim)
        z_q, alpha_list = query(x_unfold, code_book)
        z_q_fold = self.quantize.fold(z_q.contiguous(), x.shape)
        return z_q_fold, alpha_list
    
    def forward_with_local_info(self, x):
        feature_with_local_info = self.get_local_info(x)
        z_q, codebook_loss, indices = self.quantize(feature_with_local_info)
        return z_q, codebook_loss, indices, feature_with_local_info

    def lookup(self, indices, shape):
        z_q = 0
        for index in indices:
            z_q += self.quantize.embedding(index)
        z_q_fold = self.quantize.fold(z_q, shape)
        return z_q_fold


class QueryModule(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.codebook_transform = None
        self.depth = depth

    def forward(self, z, codebook):
        N, d = codebook.shape
        z_t = z
        codebook_t = self.codebook_transform
        z_q, residual = 0, z_t.detach()
        maps = []
        for i in range(self.depth):
            dist_map = self.dist(residual, codebook_t)
            maps.append(dist_map)
            pred = torch.argmin(dist_map, keepdim=False, dim=1)
            pred_one_hot = F.one_hot(pred, N).float()
            delta = torch.einsum("bm,md->bd", pred_one_hot, codebook)
            z_q = z_q + delta
            residual = residual - delta

        return z_q, maps

    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum(y ** 2, dim=1) - 2 * \
                   torch.matmul(x, y.t())

from codebook.model.network_swinir import RSTB
class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x


class PermuteBeforeLinear(nn.Module):
    def __init__(self):
        super(PermuteBeforeLinear, self).__init__()
        
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class PermuteAfterLinear(nn.Module):
    def __init__(self):
        super(PermuteAfterLinear, self).__init__()
        
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


from codebook.model.basicblock import *
# import basicblock as B
class UNet(nn.Module):
    def __init__(self,
                 in_nc=12,
                 out_nc=4,
                 nc_x=[32, 64, 128, 256],   ##[64, 128, 256, 512]
                #  vgg_nc = [64, 128, 256, 1024],
                 depths=[1, 1, 1, 1]):
        super(UNet, self).__init__()

        self.ms_channels = in_nc
        self.out_channel = out_nc

        self.layer_1 = nn.Conv2d(in_channels=self.ms_channels, out_channels=nc_x[0],
                                   kernel_size=3, padding=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_1.weight.data)
        self.m_down1 = sequential(
            *[
                ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(depths[0])
            ], downsample_strideconv(nc_x[0], nc_x[1], bias=False, mode='2'))
        self.m_down2 = sequential(
            *[
                ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(depths[1])
            ], downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = sequential(
            *[
                ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(depths[2])
            ], downsample_strideconv(nc_x[2], nc_x[-1], bias=False, mode='2'))

        self.m_body = sequential(*[
            ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(depths[-1])
        ])

        self.m_up3 = sequential(
            upsample_convtranspose(nc_x[-1], nc_x[2], bias=False, mode='2'),
            *[
                ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(depths[2])
            ])
        self.m_up2 = sequential(
            upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(depths[1])
            ])
        self.m_up1 = sequential(
            upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(depths[0])
            ])

        self.m_tail = conv(nc_x[0], self.out_channel, bias=False, mode='C')#nc_x[0]
        # self.m_tail1 = conv(in_nc, 16, bias=False, mode='C')#nc_x[0]
        
        # 64 128 128      32 128 128
        # 128 64 64       64 64 64
        # 256 32 32       128 32 32
        # 512 16 16       256 16 16
        # 512 8 8

            
        self.fuse = nn.ModuleList([])
        for c in nc_x:
            self.fuse.append(nn.Conv2d(c * 2, c, 1, 1, 0))

    def forward(self, x):
        b, c, h, w = x.shape
        
        x0 = x
        x1 = self.layer_1(x0)

 
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4) #x4
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) #+ x1[:, :-1, :, :] 
        # x = self.m_up3(self.fuse[3](torch.cat([x, x4], dim=1)))
        # x = self.m_up2(self.fuse[2](torch.cat([x, x3], dim=1)))
        # x = self.m_up1(self.fuse[1](torch.cat([x, x2], dim=1)))
        # x = self.m_tail(self.fuse[0](torch.cat([x, x1], dim=1))) #+ x1[:, :-1, :, :] 
        # x = self.m_tail1(x)
        return x #+ x0[:, 8:]
    



class Network3D(nn.Module):
    def __init__(self, n_e=1024, depth=6, num_block=[1,1,1]):
        super().__init__()
        self.in_ch = 1
        self.out_ch = 1

        curr_res = max(channel_query_dict.keys())
        feature_channel = min(channel_query_dict.keys())
        feature_depths = len(channel_query_dict.keys())
        self.feature_channel = feature_channel
        self.feature_depths = feature_depths


        self.conv_in = nn.Conv3d(in_channels=self.in_ch, out_channels=self.feature_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # 1, 16

        self.encoder_conv1 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1) # 16, 16
        self.encoder_256 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[0]) # 16. 16

        self.down1 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=(1, 2, 2), padding=1) # 16, 32
        curr_res = curr_res // 2  # 128
        self.encoder_conv2 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1) # 32, 32
        self.encoder_128 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[1]) # 32,32

        self.down2 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=(1, 2, 2), padding=1) # 32, 64
        curr_res = curr_res // 2  # 64
        self.encoder_conv3 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1) # 64, 64
        self.encoder_64 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[2]) # 64, 64

        # self.down3 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=(1, 2, 2), padding=1) # 64 , 128
        # curr_res = curr_res // 2  # 32
        # self.encoder_conv4 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        # self.encoder_32 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[3])

        self.vq_64 = VQModule(n_e_shared=512, n_e_task=128, e_dim_shared=feature_channel**2, e_dim_task=4, depth=3, num_tasks=4)

        self.decoder_conv1 = nn.Conv3d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_64 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[2])

        self.up2 = nn.Upsample(scale_factor=(1,2,2))
        curr_res *= 2

        self.decoder_conv2 = nn.Conv3d(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_128 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[1])

        self.up3 = nn.Upsample(scale_factor=(1,2,2))
        curr_res *= 2

        self.decoder_conv3 = nn.Conv3d(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_256 = BasicBlock3D(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[0])

        self.conv_out = nn.Conv3d(channel_query_dict[curr_res], self.out_ch, 3, 1, 1)

    def decode(self, fq):
        fq_mid = self.decoder_conv1(fq)
        f3_d = self.decoder_64(fq_mid)
        f3_d_mid = self.decoder_conv2(self.up2(f3_d))
        f2_d = self.decoder_128(f3_d_mid)
        f2_d_mid = self.decoder_conv3(self.up3(f2_d))
        f1_d = self.decoder_256(f2_d_mid)

        # print("f3_d.shape",f3_d.shape)
        # print("f2_d.shape",f2_d.shape)
        # print("f1_d.shape",f1_d.shape)
        return f1_d, f2_d_mid, f3_d_mid, fq_mid
    
    def encode(self, x):
        x = self.conv_in(x)
        f1 = self.encoder_256(self.encoder_conv1(x))
        f2 = self.encoder_128(self.encoder_conv2(self.down1(f1)))
        f3 = self.encoder_64(self.encoder_conv3(self.down2(f2)))
        # print("f1.shape",f1.shape)
        # print("f2.shape",f2.shape)
        # print("f3.shape",f3.shape)
        return f1, f2, f3
    
    def forward(self, x, one_hot):
        x = x.unsqueeze(1)
        f1, f2, f3 = self.encode(x)
        vq_output = self.vq_64(f3, one_hot)

        fq, codebook_loss = vq_output

        
        f1_d, fq, f2_d, f3_d = self.decode(fq)

        x_rec = self.conv_out(f1_d)
        x_rec = x_rec.squeeze(1)
        return x_rec, codebook_loss, [f1, f2, f3], [fq, f2_d, f3_d]
        
def get_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot
    

        
if __name__ == "__main__":

    # q = torch.randn(1, 4, 4, 65536).cuda()
    # k = torch.randn(1, 4, 4, 65536).cuda()
    # q = torch.randn(1, 4, 65536, 4).cuda()
    # k = torch.randn(1, 4, 65536, 4).cuda()

    # v = k @ q.transpose(-2, -1)
    model = Network3D().cuda()
    one_hot = torch.tensor([
        [0, 0, 1, 0],  # 第一个样本属于任务 0
        [0, 0, 1, 0]   # 第二个样本属于任务 1
    ], dtype=torch.float32)  # [batch, num_tasks]
    input_4_channels = torch.randn(2, 8, 128, 128).cuda()
    # input_8_channels = torch.randn(1, 1, 8, 128, 128).cuda()
    output_4_channels,_,_,_ = model(input_4_channels, one_hot)
    # output_8_channels = model(input_8_channels)

    print(output_4_channels.shape)
    # print(output_8_channels.shape)
    # model = Network3D(in_ch=8, n_e=1536, out_ch=8, stage=0, depth=8, unfold_size=2, opt=None, num_block=[1,1,1]).cuda()

    # t = torch.rand(1,8,64,64).cuda()
    # one_hot = get_one_hot(1, 4)
    # one_hot = one_hot.unsqueeze(0)
    # output = model(t, one_hot)

    # print(sum(p.numel() for p in model.parameters() )/1e6, "M")