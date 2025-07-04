import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from PromptIR.MoE.MoE import MoE
import torch
import torch.nn as nn
import torch.nn.functional as F
from PromptIR.Model_AMIR import OverlapPatchEmbed, TransformerBlock, Downsample, Upsample


class SpaPromptGenBlock(nn.Module):
    def __init__(self,spatial_prompt_num=5,spatial_prompt_size = 32, dim = 64):
        super(SpaPromptGenBlock,self).__init__()
        self.spatial_prompt = nn.Parameter(torch.rand(1,spatial_prompt_num,spatial_prompt_size,spatial_prompt_size), requires_grad=True)
        self.linear_layer_spatial = nn.Linear(dim,spatial_prompt_num)
        self.conv = nn.Conv2d(dim,dim,kernel_size=1,stride=1,bias=False)

        self.moe = MoE(dim, dim, mlp_ratio=2.66, num_experts=4, noisy_gating=True, use_experts=2) 
        nn.init.xavier_uniform_(self.spatial_prompt)

    def forward(self,x,text):
        B,C,H,W = x.shape
        
        emb = x.mean(dim=(-2,-1))
        print("emb.shape: ",emb.shape)
        prompt_weights_spatial = F.softmax(self.linear_layer_spatial(emb),dim=1)
        prompt_spatial = prompt_weights_spatial.unsqueeze(-1).unsqueeze(-1) * self.spatial_prompt.repeat(B,1,1,1)
        spatial_prompt = torch.sum(prompt_spatial,dim=1)
        
        spatial_feature = spatial_prompt.unsqueeze(1) * x

        out = self.conv(spatial_feature)

        out, loss = self.moe(x, out) 
        return out + x, loss
    
class ChaPromptGenBlock(nn.Module):
    def __init__(self,spectral_prompt_num=5, dim = 64):
        super(ChaPromptGenBlock,self).__init__()
        self.spectral_prompt = nn.Parameter(torch.rand(spectral_prompt_num,dim), requires_grad=True)
        self.linear_layer_spectral = nn.Linear(dim,spectral_prompt_num)
        self.conv = nn.Conv2d(dim,dim,kernel_size=1,stride=1,bias=False)

        self.moe = MoE(dim, dim, mlp_ratio=2.66, num_experts=4, noisy_gating=True, use_experts=2)
        nn.init.xavier_uniform_(self.spectral_prompt)

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        # print("emb.shape: ",emb.shape)
        prompt_weights_spectral = F.softmax(self.linear_layer_spectral(emb),dim=1)
        prompt_spectral = prompt_weights_spectral.unsqueeze(-1) * self.spectral_prompt.unsqueeze(0).repeat(B,1,1)
        spectral_prompt = torch.sum(prompt_spectral,dim=1)

        spectral_feature = spectral_prompt.unsqueeze(-1).unsqueeze(-1) * x
        out = self.conv(spectral_feature)

        out, loss = self.moe(x, out) 
        return out + x, loss
    
class SpaTextPromptGenBlock(nn.Module):
    def __init__(self,spatial_prompt_num=5,spatial_prompt_size = 32, dim = 64):
        super(SpaTextPromptGenBlock,self).__init__()
        self.spatial_prompt = nn.Parameter(torch.rand(1,spatial_prompt_num,spatial_prompt_size,spatial_prompt_size), requires_grad=True)
        self.linear_text = nn.Linear(384,spatial_prompt_num)
        self.conv = nn.Conv2d(dim,dim,kernel_size=1,stride=1,bias=False)

        self.moe = MoE(dim, dim, mlp_ratio=2.66, num_experts=4, noisy_gating=True, use_experts=2) 
        nn.init.xavier_uniform_(self.spatial_prompt)

    def forward(self,x,text):
        B,C,H,W = x.shape
        text_emb = self.linear_text(text)
        # print("emb.shape: ",emb.shape)
        prompt_weights_text = F.softmax(text_emb,dim=1)
        prompt_text = prompt_weights_text.unsqueeze(-1).unsqueeze(-1) * self.spatial_prompt.repeat(B,1,1,1)
        text_prompt = torch.sum(prompt_text,dim=1)
        
        text_feature = text_prompt.unsqueeze(1) * x
        # print(text_feature.shape)
        out = self.conv(text_feature)

        out, loss = self.moe(x, out) 
        return out + x, loss

class SpatialChannelPromptMoE(nn.Module):
    def __init__(self, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
        base_size = 128
    ):

        super(SpatialChannelPromptMoE, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        
        self.decoder = decoder

        self.encoder_prompt_dim0 = SpaPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size,dim=dim * 2 ** 0)
        self.encoder_prompt_dim1 = SpaPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 2, dim=dim * 2 ** 1)
        self.encoder_prompt_dim2 = SpaPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 4, dim=dim * 2 ** 2)
        self.encoder_prompt_dim3 = ChaPromptGenBlock(spectral_prompt_num=5,dim=dim * 2 ** 3)
        self.decoder_prompt_dim2 = ChaPromptGenBlock(spectral_prompt_num=5,dim=dim * 2 ** 2)
        self.decoder_prompt_dim1 = ChaPromptGenBlock(spectral_prompt_num=5,dim=dim * 2 ** 1)
        self.decoder_prompt_dim0 = ChaPromptGenBlock(spectral_prompt_num=5,dim=dim * 2 ** 1)
        
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
        
        inp_enc_level1,loss_tmp = self.encoder_prompt_dim0(inp_enc_level1)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim,128,128


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        inp_enc_level2, loss_tmp = self.encoder_prompt_dim1(inp_enc_level2)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # dim * 2,64, 64


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3, loss_tmp = self.encoder_prompt_dim2(inp_enc_level3)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # dim * 4, 32,32


        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4, loss_tmp = self.encoder_prompt_dim3(inp_enc_level4)
        loss_importance = loss_importance + loss_tmp
        latent = self.latent(inp_enc_level4) # dim * 8, 16, 16


        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3, loss_tmp = self.decoder_prompt_dim2(inp_dec_level3)
        loss_importance = loss_importance + loss_tmp
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2, loss_tmp = self.decoder_prompt_dim1(inp_dec_level2)
        loss_importance = loss_importance + loss_tmp
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1, loss_tmp = self.decoder_prompt_dim0(inp_dec_level1)
        loss_importance = loss_importance + loss_tmp
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 
        # out_dec_level1 = self.output8(out_dec_level1) + inp_img

        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img
 
        return out_dec_level1, loss_importance
    
class SpatialChannelPromptTextMoE(nn.Module):
    def __init__(self, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
        base_size = 128
    ):

        super(SpatialChannelPromptTextMoE, self).__init__()

        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        
        self.decoder = decoder

        self.encoder_prompt_dim0 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size,dim=dim * 2 ** 0)
        self.encoder_prompt_dim1 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 2, dim=dim * 2 ** 1)
        self.encoder_prompt_dim2 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 4, dim=dim * 2 ** 2)
        self.encoder_prompt_dim3 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 8,dim=dim * 2 ** 3)
        self.decoder_prompt_dim2 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 4, dim=dim * 2 ** 2)
        self.decoder_prompt_dim1 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size // 2, dim=dim * 2 ** 1)
        self.decoder_prompt_dim0 = SpaTextPromptGenBlock(spatial_prompt_num=5,spatial_prompt_size=base_size,dim=dim * 2 ** 1)
        
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

    def forward(self, inp_img, text):
        B,C,H,W = inp_img.shape

        # if C == 4:
        #     inp_img = inp_img.repeat_interleave(2, dim=1)

        if C == 4:
            inp_enc_level1 = self.patch_embed4(inp_img)
        else:
            inp_enc_level1 = self.patch_embed8(inp_img)
        
        inp_enc_level1,loss_tmp = self.encoder_prompt_dim0(inp_enc_level1,text)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # dim,128,128


        inp_enc_level2 = self.down1_2(out_enc_level1) # dim * 2
        inp_enc_level2, loss_tmp = self.encoder_prompt_dim1(inp_enc_level2,text)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # dim * 2,64, 64


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3, loss_tmp = self.encoder_prompt_dim2(inp_enc_level3,text)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # dim * 4, 32,32


        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4, loss_tmp = self.encoder_prompt_dim3(inp_enc_level4,text)
        loss_importance = loss_importance + loss_tmp
        latent = self.latent(inp_enc_level4) # dim * 8, 16, 16


        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3, loss_tmp = self.decoder_prompt_dim2(inp_dec_level3,text)
        loss_importance = loss_importance + loss_tmp
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2, loss_tmp = self.decoder_prompt_dim1(inp_dec_level2,text)
        loss_importance = loss_importance + loss_tmp
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1, loss_tmp = self.decoder_prompt_dim0(inp_dec_level1,text)
        loss_importance = loss_importance + loss_tmp
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 
        # out_dec_level1 = self.output8(out_dec_level1) + inp_img

        if C == 4:
            out_dec_level1 = self.output4(out_dec_level1) + inp_img
        else:
            out_dec_level1 = self.output8(out_dec_level1) + inp_img
 
        return out_dec_level1, loss_importance
    
if __name__== "__main__":
    # torch.cuda.set_device(1)
    # model = SpaChaPromptGenBlock(spatial_prompt_num=5,spectral_prompt_num=5,spatial_prompt_size=32,spectral_prompt_dim=64).cuda()
    # feature = torch.rand(1,64,32,32).cuda()
    # output = model(feature)
    model = SpatialChannelPromptTextMoE(dim=16, num_blocks=[1,2,2,3], num_refinement_blocks = 2,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', decoder = True)#.cuda()
    inp_ms = torch.rand(1, 4, 128, 128)#.cuda()
    text = torch.rand(1, 384)#.cuda()

    output, loss= model(inp_ms, text)
    
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 