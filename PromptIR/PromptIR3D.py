import torch
import torch.nn as nn
from Model_AMIR import *

def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')

def to_5d(x,d,h,w):
    return rearrange(x, 'b (d h w) c -> b c d h w',d=d,h=h,w=w)

class LayerNorm3D(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3D, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_5d(self.body(to_3d(x)),d, h, w)
    
class Attention3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        # b,c,h,w = x.shape
        b, c, d, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1) # b, dim * 3, d, h, w 
        
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out
    
class FeedForward3D(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward3D, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock3D, self).__init__()

        self.norm1 = LayerNorm3D(dim, LayerNorm_type)
        self.attn = Attention3D(dim, num_heads, bias)
        self.norm2 = LayerNorm3D(dim, LayerNorm_type)
        self.ffn = FeedForward3D(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed3D(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed3D, self).__init__()

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    
class Downsample3D(nn.Module):
    def __init__(self, n_feat):
        super(Downsample3D, self).__init__()

        self.conv = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        Returns: [B, C*2, D, H//2, W//2]
        """
        x = self.conv(x)  # [B, C//2, D, H, W]
        b, c, d, h, w = x.shape

        # make sure H and W are even
        assert h % 2 == 0 and w % 2 == 0, "H and W must be divisible by 2"

        x = rearrange(x, 'b c d (h2 h) (w2 w) -> b (c h2 w2) d h w', h2=2, w2=2)
        
        return x
    
class Upsample3D(nn.Module):
    def __init__(self, n_feat):
        super(Upsample3D, self).__init__()
        self.conv = nn.Conv3d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        Returns: [B, C//2, D, H*2, W*2]
        """
        x = self.conv(x)  # [B, C//2, D, H, W]
        # b, c, d, h, w = x.shape

        x = rearrange(x, 'b (c h2 w2) d  h  w -> b c d (h2 h) (w2 w)', h2 = 2, w2 = 2)

        return x

class ProxNet_Prompt3D(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 16,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(ProxNet_Prompt3D, self).__init__()

        self.patch_embed = OverlapPatchEmbed3D(inp_channels, dim)
        
        
        self.decoder = decoder
        

        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1) ,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock3D(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample3D(dim) ## From Level 1 to Level 2

        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**2) ,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample3D(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**3) ,int(dim*2**3),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample3D(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample3D(int(dim*2**3)) ## From Level 4 to Level 3
        
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample3D(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample3D(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock3D(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, D, H, W = inp_img.shape
        
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim
        print("out_enc_level1:", out_enc_level1.shape)
       


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        print("inp_enc_level2:", inp_enc_level2.shape) 

        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2)  # dim * 2 * 2
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3) #  dim * 2 * 2 * 2
        
        latent = self.latent(inp_enc_level4) 

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 

        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_img

 
        return out_dec_level1
    
if __name__== "__main__":
    torch.cuda.set_device(0)
    # model = ProxNet_AMIR_Prompt(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3])
    model = ProxNet_Prompt3D(inp_channels=1, out_channels=1, dim=8, num_blocks=[1,1,1,2]).cuda()

    F_middle = torch.rand(4, 1, 4, 128, 128).cuda()

    output = model(F_middle)
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M")

    # trans = TransformerBlock3D(dim=16, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias").cuda()

    # t_output  = trans(output)