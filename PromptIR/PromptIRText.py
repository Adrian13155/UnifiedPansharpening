import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from PromptIR.Model_AMIR import *
from PromptIR.D_GDM import D_GDM

class PromptIRText(nn.Module):
    def __init__(self, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(PromptIRText, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        
        
        self.decoder = decoder
        prompt_dim1 = 16
        prompt_dim2 = 32
        prompt_dim3 = 64
        
        if self.decoder:
            self.prompt1 = D_GDM(prompt_len=5, prompt_dim=prompt_dim1, prompt_size=64, lin_dim=384, dim=dim * 2, num_heads=2, bias=False)
            self.prompt2 = D_GDM(prompt_len=5, prompt_dim=prompt_dim2, prompt_size=32, lin_dim=384, dim=dim * 2 ** 2, num_heads=2, bias=False)
            self.prompt3 = D_GDM(prompt_len=5, prompt_dim=prompt_dim3, prompt_size=16, lin_dim=384, dim=dim * 2 ** 3, num_heads=2, bias=False)

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
                    
        self.output4 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output8 = nn.Conv2d(int(dim*2**1), 8, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, text_emb):
        B,C,H,W = inp_img.shape

        if C == 4:
            inp_enc_level1 = self.patch_embed4(inp_img)
        else:
            inp_enc_level1 = self.patch_embed8(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        # print("latent.shape", latent.shape)
        dec3_param = self.prompt3(latent, text_emb)
        # print("dec3_param.shape", dec3_param.shape)
        latent = torch.cat([latent, dec3_param], 1)
        latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        
        dec2_param = self.prompt2(out_dec_level3, text_emb)
        out_dec_level3 = torch.cat([inp_dec_level3, dec2_param], 1)
        out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        
        dec1_param = self.prompt1(out_dec_level2, text_emb)
        out_dec_level2 = torch.cat([inp_dec_level2, dec1_param], 1)
        out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img
 
        return out_dec_level1
if __name__ == "__main__":
    torch.cuda.set_device(1)
    # model = ProxNet_AMIR_Prompt(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3])
    model = PromptIRText( dim=16, num_blocks=[4,6,6,8]).cuda()
    text_emb = torch.rand(1,384).cuda()
    inp_ms = torch.rand(1, 4, 128, 128).cuda()

    output = model(inp_ms, text_emb)
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M")     
