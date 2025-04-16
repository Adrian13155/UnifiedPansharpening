import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import torch
import torch.nn as nn
from PromptIR.Model_AMIR import *

class ProxNet_Prompt2DWith3D(nn.Module):
    """
    ce表示通道扩展的个数
    """
    def __init__(self, 
        ce=4, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(ProxNet_Prompt2DWith3D, self).__init__()

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        """
        这里用卷积或者复制可能都要试试
        """
        self.channel_expansion = nn.Conv3d(1, ce, kernel_size=(1,3,3), stride=1,padding=(0, 1, 1))
        
        
        self.decoder = decoder
        prompt_dim1 = 16
        prompt_dim2 = 32
        prompt_dim3 = 64
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=prompt_dim1,prompt_len=5,prompt_size = 64,lin_dim = dim * 2)
            self.prompt2 = PromptGenBlock(prompt_dim=prompt_dim2,prompt_len=5,prompt_size = 32,lin_dim = dim * 2 ** 2)
            self.prompt3 = PromptGenBlock(prompt_dim=prompt_dim3,prompt_len=5,prompt_size = 16,lin_dim = dim * 2 ** 3)

        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1 + prompt_dim1) ,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**2 + prompt_dim2) ,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**3 + prompt_dim3) ,int(dim*2**3),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output1 = nn.Conv2d(int(dim*2**1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output2 = nn.Conv2d(int(dim*2**1), 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        """
        input:[B,1,4/8,H,W]
        """
        B,C,D,H,W = inp_img.shape
        
        inp_enc_level1 = self.channel_expansion(inp_img)
        inp_enc_level1 = rearrange(inp_enc_level1, 'b c d h w -> (b d) c h w')
        # print("inp_enc_level1.shape:", inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        dec3_param = self.prompt3(latent)
        latent = torch.cat([latent, dec3_param], 1)
        latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        
        dec2_param = self.prompt2(out_dec_level3)
        out_dec_level3 = torch.cat([inp_dec_level3, dec2_param], 1)
        out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        
        dec1_param = self.prompt1(out_dec_level2)
        out_dec_level2 = torch.cat([inp_dec_level2, dec1_param], 1)
        out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 
        t1 = self.output2(out_dec_level1)
        t2 = self.output2(out_dec_level1)
        t2 = rearrange(t2,'(b d) c h w -> b c d h w', b=B)
        # print("t2.shape:", t2.shape)

        out_dec_level1 = t2 + inp_img


        return out_dec_level1
    

if __name__== "__main__":
    torch.cuda.set_device(7)
    # model = ProxNet_AMIR_Prompt(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3])
    model = ProxNet_Prompt2DWith3D(ce=8, dim=8, num_blocks=[1,1,1,2]).cuda()

    inp_ms = torch.rand(4, 1, 8, 128, 128).cuda()

    output = model(inp_ms)
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 