import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from torch import nn
from codebook.model.MSAB import MSAB, MCAB, FeedForward, Trash
from codebook.model.vq import BlockBasedResidualVectorQuantizer, DualCodebookVectorQuantizer, SharedAndTaskSpecificCodebookVectorQuantizer
import torch
from torch.nn import functional as F
from codebook.model.feature_converter import TransformerBlock
from codebook.model.vgg import Vgg19

channel_query_dict = {
    64: 256,
    128: 128,
    256: 64,
}



class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        ksz = 3
        self.block = nn.Sequential(#nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1),
                                   MSAB(dim=out_ch, num_blocks=num_blocks, dim_head=out_ch//4, heads=4))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        x = self.block(x)
        return x

class BasicMCABlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        ksz = 3
        self.blocks = nn.ModuleList()
        # self.conv_x = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        # self.conv_y = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        self.mapx = nn.Sequential(*[
            nn.Linear(in_ch, 2 * in_ch),
            nn.LayerNorm(2 * in_ch),
            nn.LeakyReLU(),
            nn.Linear(2 * in_ch, 2 * in_ch),
            nn.LayerNorm(2 * in_ch),
            nn.LeakyReLU(),
            nn.Linear(2 * in_ch, in_ch),
        ])
        
        for i in range(num_blocks):
            self.blocks.append(MCAB(dim=out_ch, num_blocks=num_blocks, dim_head=out_ch//4, heads=4))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, y):
        # x, y = self.conv_x(x), self.conv_y(y)
        x = self.mapx(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        # y = self.mapy(y)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, y)
        return x
    
class fuseR(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=1):
        super().__init__()
        ksz = 3
        self.blocks = nn.ModuleList()
        self.conv_x = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        self.conv_y = nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1)
        self.mapy = nn.Sequential(*[
            nn.Linear(in_ch, 2 * in_ch),
            nn.LayerNorm(2 * in_ch),
            nn.LeakyReLU(),
            nn.Linear(2 * in_ch, in_ch),
            nn.LayerNorm(2 * in_ch),
            nn.LeakyReLU(), 
            nn.Linear(2 * in_ch, in_ch),
        ])
        for i in range(num_blocks):
            self.blocks.append(MCAB(dim=out_ch, num_blocks=num_blocks, dim_head=out_ch//4, heads=4))
        self.conv11 = nn.Conv2d(out_ch * 2, out_ch, 1, 1, 0)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, y):
        x, y = self.conv_x(x), self.conv_y(y)
        y = y.permute(0, 2, 3, 1)
        y = self.mapy(y)
        y = y.permute(0, 3, 1, 2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, y)
        return self.conv11(torch.cat([x, y], dim=1))

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
    def __init__(self, in_ch, e_dim=512, n_e=512, depth=6, unfold_size=2, mlp_codebook=False):
        super().__init__()
        self.opt = (in_ch, e_dim, n_e, depth)
        self.quantize = SharedAndTaskSpecificCodebookVectorQuantizer(n_e=n_e, e_dim=e_dim, depth=depth, unfold_size=unfold_size, mlp_codebook=mlp_codebook, num_tasks=4)
        # self.quantize = BlockBasedResidualVectorQuantizer(n_e=n_e, e_dim=e_dim, depth=depth, unfold_size=unfold_size, mlp_codebook=mlp_codebook)
        # self.quantize = DualCodebookVectorQuantizer(n_e_shared=256, n_e_task=256, e_dim=e_dim, depth=depth, unfold_size=unfold_size, mlp_codebook=mlp_codebook)

    def forward(self, x, one_hot):
        # z_q, codebook_loss, cosine_loss, indices = self.quantize(x)
        # z_q, codebook_loss, indices = self.quantize(x, one_hot)
        # z_q, codebook_loss, kl_loss, indices = self.quantize(x, one_hot)
        # return z_q, codebook_loss, cosine_loss, indices
        # return z_q, codebook_loss, indices
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


class semanticFuseModule(nn.Module):
    def __init__(self, in_channel, num_class=21):
        super(semanticFuseModule, self).__init__()
        
        self.num_class = num_class
        self.mlp_channel = [in_channel // num_class for i in range(num_class)]
        self.add_channel = [1 if i < in_channel % num_class else 0 for i in range(num_class)]
        self.mlps_channel = [i + j for i, j in zip(self.mlp_channel, self.add_channel)]
        self.mlps = nn.ModuleList([nn.Linear(in_channel, i) for i in self.mlps_channel])
    
        # self.conv_in = nn.Conv2d(in_channel * 2, in_channel * 2, 3, 1, 1)
        # self.channelAttention = BasicBlock(in_ch=in_channel * 2, out_ch=in_channel * 2, num_blocks=1)
        self.in_mlp = nn.Linear(in_channel, in_channel)
        self.channelAttention = BasicMCABlock(in_ch=in_channel, out_ch=in_channel, num_blocks=1)

        # self.out = nn.Linear(in_channel, in_channel)
        
    def forward(self, x, semantic):
        b, c, h, w = semantic.shape
        bx, cx, hx, wx = x.shape
        
        
        semantic = F.relu(semantic)
        if h != hx:
            semantic = F.interpolate(semantic, size=(hx, wx), mode='bilinear', align_corners=False)
            b, c, h, w = semantic.shape
        
        semantic_feature = [x * semantic[:, i].unsqueeze(1).repeat(1, cx, 1, 1) for i in range(self.num_class)]
        semantic_feature = [b(xx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for b, xx in zip(self.mlps, semantic_feature)]
        semantic_feature = torch.cat(semantic_feature, dim=1)
        # all_features = torch.cat([x, semantic_feature], dim=1)
        all_features = semantic_feature
        
        # out_features = self.conv_in(all_features)
        # out_features = self.channelAttention(out_features)
        out_features = self.in_mlp(all_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        out_features = self.channelAttention(x, out_features)
        
        # res = x + self.out(out_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # x + 
        res = out_features   # x + 
        return res
        

class Merge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.zero_conv_in = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        
        self.feature_convert = BasicMCABlock(in_channels, out_channels, num_blocks=1)
        
        self.zero_conv_out = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # torch.nn.init.zeros_(self.zero_conv.weight.data)
        # torch.nn.init.zeros_(self.zero_conv.bias.data)
    def init_params(self):
        torch.nn.init.zeros_(self.zero_conv_in.weight.data)
        torch.nn.init.zeros_(self.zero_conv_in.bias.data)
        torch.nn.init.zeros_(self.zero_conv_out.weight.data)
        torch.nn.init.zeros_(self.zero_conv_out.bias.data)

    def forward(self, x, y):
        # features = torch.cat([x,y], dim=1)
        x = self.zero_conv_in(x)
        output = self.zero_conv_out(self.feature_convert(x, y))
        return output



class trashLayer(nn.Module):
    def __init__(self, dim, number_block=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Trash(dim) for i in range(number_block)]
        )
        
    def forward(self, x):
        return self.blocks(x)

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
    



class Network(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_e=1024, stage=0, depth=6, num_block=3, 
                 unfold_size=2, opt=None): # , illumination_loss=False
        super().__init__()
        
        global channel_query_dict
        assert stage in [0, 1, 2, 4, 5, 6, 7, 8]
        self.stage = stage
        curr_res = max(channel_query_dict.keys())
        
        feature_channel = min(channel_query_dict.keys())
        feature_depths = len(channel_query_dict.keys())
        self.feature_channel = feature_channel
        self.feature_depths = feature_depths
        self.in_ch = in_ch
        self.opt = opt
        
        self.conv_in = nn.Conv2d(self.in_ch, feature_channel, 3, 1, 1)
        
        
        self.encoder_conv1 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.encoder_256 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[0])

        self.down1 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=2, padding=1)
        curr_res = curr_res // 2  # 128
        self.encoder_conv2 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.encoder_128 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[1])

        self.down2 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=2, padding=1)
        curr_res = curr_res // 2  # 64
        self.encoder_conv3 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.encoder_64 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[2])
        
        # 如果改变了patch_size,这里也要�???????????????? ###################################################################################
        self.vq_64 = VQModule(feature_channel * (2**(feature_depths - 1)), feature_channel * (2**(feature_depths - 1)) * unfold_size * unfold_size, n_e, depth=depth, unfold_size=unfold_size, mlp_codebook=False)
        
        
        #####################################################################################################################
        self.decoder_conv1 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_64 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[2])
        

        self.up2 = nn.Upsample(scale_factor=2)
        curr_res *= 2
        self.decoder_conv2 = nn.Conv2d(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_128 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[1])

        self.up3 = nn.Upsample(scale_factor=2)
        curr_res *= 2
        self.decoder_conv3 = nn.Conv2d(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], 3, stride=1, padding=1)
        self.decoder_256 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block[0])

        self.conv_out = nn.Conv2d(channel_query_dict[curr_res], out_ch, 3, 1, 1)
        


    def forward(self, x, y=None, semantic=None, lms=None, ms=None, pan=None):
        return self.forward_s1(x,y)
        
    def encode(self, x):
        x = self.conv_in(x)
        f1 = self.encoder_256(self.encoder_conv1(x))
        f2 = self.encoder_128(self.encoder_conv2(self.down1(f1)))
        f3 = self.encoder_64(self.encoder_conv3(self.down2(f2)))
        return f1, f2, f3
    
    def decode(self, fq):
        fq_mid = self.decoder_conv1(fq)
        f3_d = self.decoder_64(fq_mid)
        f3_d_mid = self.decoder_conv2(self.up2(f3_d))
        f2_d = self.decoder_128(f3_d_mid)
        f2_d_mid = self.decoder_conv3(self.up3(f2_d))
        f1_d = self.decoder_256(f2_d_mid)
        return f1_d, f2_d_mid, f3_d_mid, fq_mid
    
    def forward_s1(self, x, one_hot):
        B,C,H,W = x.shape
        if C == 4:
            x = x.repeat_interleave(2, dim=1)
        f1, f2, f3 = self.encode(x)
        vq_output = self.vq_64(f3, one_hot)
        # 根据返回值的数量来决定如何解包
        
        if len(vq_output) == 3:
            fq, codebook_loss, distance_map = vq_output
            kl_loss = None
        elif len(vq_output) == 4:
            fq, codebook_loss, kl_loss, distance_map = vq_output

        f1_d, fq, f2_d, f3_d = self.decode(fq)
        x_rec = self.conv_out(f1_d)

        if C == 4:
            x_rec = x_rec.float().view(B, C, 2, H , W ).mean(dim=2)

        if kl_loss is None:
            return x_rec, codebook_loss, distance_map, [f1, f2, f3], [fq, f2_d, f3_d]
        else:
            return x_rec, codebook_loss, kl_loss, distance_map, [f1, f2, f3], [fq, f2_d, f3_d]
        
def get_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot
        
if __name__ == "__main__":
    checkpoint_path = "/data/cjj/projects/codebookCode/Checkpoint/test_Stage2_Iter12:6_Length1024:256/models/epoch_632_step_27840_2s_G.pth"
    checkpoint = torch.load(checkpoint_path)

    model = Network(in_ch=8, n_e=1536, out_ch=8, stage=0, depth=8, unfold_size=2, opt=None, num_block=[1,1,1]).cuda()
    model.load_state_dict(checkpoint, strict=False)

    t = torch.rand(1,8,64,64).cuda()
    one_hot = get_one_hot(1, 4)
    one_hot = one_hot.unsqueeze(0)
    output = model(t, one_hot)

    print(sum(p.numel() for p in model.parameters() )/1e6, "M")