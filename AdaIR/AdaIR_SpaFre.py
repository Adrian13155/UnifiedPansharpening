import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from AdaIR.SIFTNet import SpaFre


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=8):
        super(FreModule, self).__init__()


        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        high_feature, low_feature = self.fft(x) 

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        B,C,H,W = x.shape
        if C == 4:
            x = x.repeat_interleave(2, dim=1)
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low


##########################################################################
##---------- AdaIR -----------------------

class AdaIRSpaFre(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=4, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias', 
        decoder = True,
    ):

        super(AdaIRSpaFre, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)   
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        self.conv_p1 = nn.Conv2d(1, dim, 3, 1, 1)

        # nn.Conv2d(1, channels, 3, 1, 1)           
        self.decoder = decoder
        
        if self.decoder:
            self.fre1 = SpaFre(dim*2**3)
            self.fre2 = SpaFre(dim*2**2)
            self.fre3 = SpaFre(dim*2**1)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output4 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output8 = nn.Conv2d(int(dim*2**1), 8, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img,pan):
        B,C,H,W = inp_img.shape
        if C == 4:
            inp_enc_level1 = self.patch_embed4(inp_img)
        else:
            inp_enc_level1 = self.patch_embed8(inp_img)

        

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        if self.decoder:
            latent,_ = self.fre1(latent, pan)
      
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        if self.decoder:
            out_dec_level3,_ = self.fre2(out_dec_level3, pan)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2,_ = self.fre3(out_dec_level2, pan)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        

        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img

        return out_dec_level1
    
if __name__ == '__main__':
    torch.cuda.set_device(0)
    lr = torch.randn(1,8,128,128).cuda()
    pan = torch.rand(1,1,128,128).cuda()
    model = AdaIRSpaFre(inp_channels=8, 
        out_channels=8, 
        dim = 20,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias', 
        decoder = True,
    ).cuda()
    output = model(lr, pan)
    print("output.shape: ",output.shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))