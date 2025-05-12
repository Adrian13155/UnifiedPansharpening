import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from PromptIR.Model_AMIR import *

class SpaChaPromptGenBlock(nn.Module):
    def __init__(self,spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size = 32, spectral_prompt_dim = 64):
        super(SpaChaPromptGenBlock,self).__init__()
        self.spatial_prompt = nn.Parameter(torch.rand(1,spatial_prompt_num,spatial_prompt_size,spatial_prompt_size), requires_grad=True)
        self.spectral_prompt = nn.Parameter(torch.rand(spectral_prompt_num,spectral_prompt_dim), requires_grad=True)
        self.linear_layer_spatial = nn.Linear(spectral_prompt_dim,spatial_prompt_num)
        self.linear_layer_spectral = nn.Linear(spectral_prompt_dim,spectral_prompt_num)
        self.conv = nn.Conv2d(spectral_prompt_dim * 2,spectral_prompt_dim,kernel_size=1,stride=1,bias=False)

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        # print("emb.shape: ",emb.shape)
        prompt_weights_spatial = F.softmax(self.linear_layer_spatial(emb),dim=1)
        prompt_weights_spectral = F.softmax(self.linear_layer_spectral(emb),dim=1)
        prompt_spatial = prompt_weights_spatial.unsqueeze(-1).unsqueeze(-1) * self.spatial_prompt.repeat(B,1,1,1)
        prompt_spectral = prompt_weights_spectral.unsqueeze(-1) * self.spectral_prompt.unsqueeze(0).repeat(B,1,1)
        spatial_prompt = torch.sum(prompt_spatial,dim=1)
        spectral_prompt = torch.sum(prompt_spectral,dim=1)
        
        spatial_feature = spatial_prompt.unsqueeze(1) * x
        spectral_feature = spectral_prompt.unsqueeze(-1).unsqueeze(-1) * x

        feature = torch.concat((spatial_feature, spectral_feature),dim = 1)
        out = self.conv(feature)

        return out + x
    
class SpatialChannelPrompt(nn.Module):
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

        super(SpatialChannelPrompt, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        
        self.decoder = decoder

        self.encoder_prompt_dim0 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=128,spectral_prompt_dim=dim * 2 ** 0)
        self.encoder_prompt_dim1 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=64,spectral_prompt_dim=dim * 2 ** 1)
        self.encoder_prompt_dim2 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=dim * 2 ** 2)
        self.encoder_prompt_dim3 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=16,spectral_prompt_dim=dim * 2 ** 3)
        self.decoder_prompt_dim2 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=dim * 2 ** 2)
        self.decoder_prompt_dim1 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=64,spectral_prompt_dim=dim * 2 ** 1)
        self.decoder_prompt_dim0 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=128,spectral_prompt_dim=dim * 2 ** 1)
        
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


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output4 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output8 = nn.Conv2d(int(dim*2**1), 8, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B,C,H,W = inp_img.shape

        # if C == 4:
        #     inp_img = inp_img.repeat_interleave(2, dim=1)

        if C == 4:
            inp_enc_level1 = self.patch_embed4(inp_img)
        else:
            inp_enc_level1 = self.patch_embed8(inp_img)
        
        inp_enc_level1 = self.encoder_prompt_dim0(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim,128,128


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        inp_enc_level2 = self.encoder_prompt_dim1(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # dim * 2,64, 64


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.encoder_prompt_dim2(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # dim * 4, 32,32


        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4 = self.encoder_prompt_dim3(inp_enc_level4)
        latent = self.latent(inp_enc_level4) # dim * 8, 16, 16


        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.decoder_prompt_dim2(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.decoder_prompt_dim1(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.decoder_prompt_dim0(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 
        # out_dec_level1 = self.output8(out_dec_level1) + inp_img

        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img
 
        return out_dec_level1
    
class SpatialChannelPromptWithJumpConnection(nn.Module):
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

        super(SpatialChannelPromptWithJumpConnection, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        
        self.decoder = decoder

        self.encoder_prompt_dim0 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=128,spectral_prompt_dim=dim * 2 ** 0)
        self.encoder_prompt_dim1 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=64,spectral_prompt_dim=dim * 2 ** 1)
        self.encoder_prompt_dim2 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=dim * 2 ** 2)
        self.encoder_prompt_dim3 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=16,spectral_prompt_dim=dim * 2 ** 3)
        self.decoder_prompt_dim2 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=dim * 2 ** 2)
        self.decoder_prompt_dim1 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=64,spectral_prompt_dim=dim * 2 ** 1)
        self.decoder_prompt_dim0 = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=128,spectral_prompt_dim=dim * 2 ** 1)

        self.conv1 = nn.Conv2d((dim * 2 ** 0) * 3, dim * 2 ** 0, 1, 1)
        self.conv2 = nn.Conv2d((dim * 2 ** 1) * 2, dim * 2 ** 1, 1, 1)
        self.conv3 = nn.Conv2d((dim * 2 ** 2) * 2, dim * 2 ** 2, 1, 1)
        
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


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output4 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output8 = nn.Conv2d(int(dim*2**1), 8, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, temp):
        B,C,H,W = inp_img.shape

        if C == 4:
            inp_enc_level1 = self.patch_embed4(inp_img)
        else:
            inp_enc_level1 = self.patch_embed8(inp_img)
        
        if temp is not None:
            inp_enc_level1 = self.conv1(torch.concat([inp_enc_level1, temp[2]], dim=1))
        inp_enc_level1 = self.encoder_prompt_dim0(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim,128,128

        
        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        if temp is not None:
            inp_enc_level2 = self.conv2(torch.concat([inp_enc_level2, temp[1]], dim=1))
        inp_enc_level2 = self.encoder_prompt_dim1(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # dim * 2,64, 64

        inp_enc_level3 = self.down2_3(out_enc_level2) 
        if temp is not None:
            inp_enc_level3 = self.conv3(torch.concat([inp_enc_level3, temp[0]], dim=1))
        inp_enc_level3 = self.encoder_prompt_dim2(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # dim * 4, 32,32


        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4 = self.encoder_prompt_dim3(inp_enc_level4)
        latent = self.latent(inp_enc_level4) # dim * 8, 16, 16

        tempD = []

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.decoder_prompt_dim2(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        tempD.append(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.decoder_prompt_dim1(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        tempD.append(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.decoder_prompt_dim0(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        tempD.append(out_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img
 
        return out_dec_level1, tempD

if __name__== "__main__":
    # torch.cuda.set_device(1)
    # model = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=64).cuda()
    # feature = torch.rand(1,64,32,32).cuda()
    # output = model(feature)
    model = SpatialChannelPrompt(dim=32, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)#.cuda()
    inp_ms = torch.rand(1, 4, 128, 128)#.cuda()
    

    output= model(inp_ms)
    # for i in temp:
    #     print(i.shape)

    output= model(inp_ms)
    
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 