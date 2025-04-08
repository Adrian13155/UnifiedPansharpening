import torch
import torch.nn as nn
# from .Component import *
from Component import *
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)

        return out

class Spatial_channel_gate(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        promptsize = 128,
        promptlen = 16,
        atom_dim = 256,
        usetransformer = True,
        useres = False
    ):

        super(Spatial_channel_gate, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        atom_num = 32 
        ## 生成ms_prompt 
        self.ms_prompt_gen = RIN(in_dim=inp_channels - 1, atom_num=atom_num, atom_dim=atom_dim) 
        ## 生成pan_prompt
        self.pan_prompt_gen = PAN_PromptBlock_multiscale(prompt_size = promptsize)
        self.prompt_encoder_level1 = SCgate(dim=16, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_encoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_encoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_decoder_latent = SCgate(dim=128, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_decoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_decoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim, useres=useres)
        self.prompt_decoder_level1 = SCgate(dim=32, channel_prompt_dim=atom_dim, useres=useres)
        if (usetransformer == True):
            self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
            self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
            self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
            self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
            self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
            self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
            self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
            self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        else:
            self.encoder_level1 = nn.Sequential(*[ResBlock(in_channels=dim, out_channels=dim) for i in range(num_blocks[0])])
            self.encoder_level2 = nn.Sequential(*[ResBlock(in_channels=int(dim*2**1), out_channels=int(dim*2**1)) for i in range(num_blocks[1])])
            self.encoder_level3 = nn.Sequential(*[ResBlock(in_channels=int(dim*2**2), out_channels=int(dim*2**2)) for i in range(num_blocks[2])])
            self.latent = nn.Sequential(*[ResBlock(in_channels=int(dim*2**3), out_channels=int(dim*2**3)) for i in range(num_blocks[3])])
            self.decoder_level3 = nn.Sequential(*[ResBlock(in_channels=int(dim*2**2), out_channels=int(dim*2**2)) for i in range(num_blocks[2])])
            self.decoder_level2 = nn.Sequential(*[ResBlock(in_channels=int(dim*2**1), out_channels=int(dim*2**1)) for i in range(num_blocks[1])])
            self.decoder_level1 = nn.Sequential(*[ResBlock(in_channels=int(dim*2**1), out_channels=int(dim*2**1)) for i in range(num_blocks[0])])
            self.refinement = nn.Sequential(*[ResBlock(in_channels=int(dim*2**1), out_channels=int(dim*2**1)) for i in range(num_refinement_blocks)])
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
                
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
                
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
       
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
   

    def forward(self, inp_ms, inp_pan): 
        B, C, H, W = inp_ms.shape
        if C == 4 :
            inp_ms = inp_ms.repeat_interleave(2, dim=1)

        # ms_prompt = self.ms_prompt_gen(inp_ms)
        inp_ms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
        inp_img = torch.concat((inp_ms,inp_pan), dim=1)
       
        pan_prompt_list = self.pan_prompt_gen(inp_pan)
        ms_prompt = self.ms_prompt_gen(inp_ms)

        inp_enc_level1 = self.patch_embed(inp_img)
          
        inp_enc_level1 = self.prompt_encoder_level1(inp_enc_level1, pan_prompt_list[0], ms_prompt) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 


        inp_enc_level2 = self.down1_2(out_enc_level1) 
        inp_enc_level2 = self.prompt_encoder_level2(inp_enc_level2, pan_prompt_list[1], ms_prompt)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.prompt_encoder_level3(inp_enc_level3, pan_prompt_list[2], ms_prompt)
        #out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        #inp_enc_level4 = self.down3_4(out_enc_level3)        
        #inp_enc_level4 = self.prompt_decoder_latent(inp_enc_level4, pan_prompt_list[3], ms_prompt)
        #latent = self.latent(inp_enc_level4) 
        
                        
        #inp_dec_level3 = self.up4_3(latent)
        #inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        #inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        #inp_dec_level3 = self.prompt_decoder_level3(inp_dec_level3, pan_prompt_list[2], ms_prompt)
        #out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        
        inp_dec_level2 = self.up3_2(inp_enc_level3)
        #inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.prompt_decoder_level2(inp_dec_level2, pan_prompt_list[1], ms_prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # pan_last_prompt = torch.cat([pan_prompt_list[0],pan_prompt_list[0]],dim=1) 
        inp_dec_level1 = self.prompt_decoder_level1(inp_dec_level1, pan_prompt_list[0], ms_prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_ms

        if C == 4 :
            out_dec_level1 = out_dec_level1.float().view(B, C, 2, H*4, W*4).mean(dim=2)
 
        return out_dec_level1 

class Spatial_channel_gate_concat(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        promptsize = 128,
        promptlen = 16,
        atom_dim = 256
    ):

        super(Spatial_channel_gate_concat, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        atom_num = 32 
        ## 生成ms_prompt 
        self.ms_prompt_gen = RIN(in_dim=inp_channels - 1, atom_num=atom_num, atom_dim=atom_dim) 
        ## 生成pan_prompt
        self.pan_prompt_gen = PAN_PromptBlock_multiscale(prompt_size = promptsize)
        self.prompt_encoder_level1 = SCgate_concat_no_cross(dim=16, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level2 = SCgate_concat_no_cross(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level3 = SCgate_concat_no_cross(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_latent = SCgate_concat_no_cross(dim=128, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level3 = SCgate_concat_no_cross(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level2 = SCgate_concat_no_cross(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level1 = SCgate_concat_no_cross(dim=32, channel_prompt_dim=atom_dim)

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
          
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        

    def forward(self, inp_ms, inp_pan): 
        B, C, H, W = inp_ms.shape
        if C == 4 :
            inp_ms = inp_ms.repeat_interleave(2, dim=1)

        # ms_prompt = self.ms_prompt_gen(inp_ms)
        inp_ms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
        inp_img = torch.concat((inp_ms,inp_pan), dim=1)
       
        pan_prompt_list = self.pan_prompt_gen(inp_pan)
        ms_prompt = self.ms_prompt_gen(inp_ms)

        inp_enc_level1 = self.patch_embed(inp_img)
          
        inp_enc_level1 = self.prompt_encoder_level1(inp_enc_level1, pan_prompt_list[0], ms_prompt) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 


        inp_enc_level2 = self.down1_2(out_enc_level1) 
        inp_enc_level2 = self.prompt_encoder_level2(inp_enc_level2, pan_prompt_list[1], ms_prompt)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.prompt_encoder_level3(inp_enc_level3, pan_prompt_list[2], ms_prompt)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        inp_enc_level4 = self.prompt_decoder_latent(inp_enc_level4, pan_prompt_list[3], ms_prompt)
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.prompt_decoder_level3(inp_dec_level3, pan_prompt_list[2], ms_prompt)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.prompt_decoder_level2(inp_dec_level2, pan_prompt_list[1], ms_prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # pan_last_prompt = torch.cat([pan_prompt_list[0],pan_prompt_list[0]],dim=1) 
        inp_dec_level1 = self.prompt_decoder_level1(inp_dec_level1, pan_prompt_list[0], ms_prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_ms

        if C == 4 :
            out_dec_level1 = out_dec_level1.float().view(B, C, 2, H*4, W*4).mean(dim=2)
 
        return out_dec_level1 

class Spatial_channel_gate_wo_level1(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        promptsize = 128,
        promptlen = 16,
        atom_dim = 256
    ):

        super(Spatial_channel_gate_wo_level1, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        atom_num = 32 
        ## 生成ms_prompt 
        self.ms_prompt_gen = RIN(in_dim=inp_channels - 1, atom_num=atom_num, atom_dim=atom_dim) 
        ## 生成pan_prompt
        self.pan_prompt_gen = PAN_PromptBlock_multiscale(prompt_size = promptsize)
        ## 嵌入prompt
        self.prompt_encoder_level1 = SCgate(dim=16, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_latent = SCgate(dim=128, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level1 = SCgate(dim=32, channel_prompt_dim=atom_dim)

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
          
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
   

    def forward(self, inp_ms, inp_pan): 
        B, C, H, W = inp_ms.shape
        if C == 4 :
            inp_ms = inp_ms.repeat_interleave(2, dim=1)

        # ms_prompt = self.ms_prompt_gen(inp_ms)
        inp_ms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
        inp_img = torch.concat((inp_ms,inp_pan), dim=1)
       
        pan_prompt_list = self.pan_prompt_gen(inp_pan)
        ms_prompt = self.ms_prompt_gen(inp_ms)

        inp_enc_level1 = self.patch_embed(inp_img)
          
        # inp_enc_level1 = self.prompt_encoder_level1(inp_enc_level1, pan_prompt_list[0], ms_prompt) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 


        inp_enc_level2 = self.down1_2(out_enc_level1) 
        inp_enc_level2 = self.prompt_encoder_level2(inp_enc_level2, pan_prompt_list[1], ms_prompt)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.prompt_encoder_level3(inp_enc_level3, pan_prompt_list[2], ms_prompt)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        inp_enc_level4 = self.prompt_decoder_latent(inp_enc_level4, pan_prompt_list[3], ms_prompt)
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.prompt_decoder_level3(inp_dec_level3, pan_prompt_list[2], ms_prompt)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.prompt_decoder_level2(inp_dec_level2, pan_prompt_list[1], ms_prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # pan_last_prompt = torch.cat([pan_prompt_list[0],pan_prompt_list[0]],dim=1) 
        # inp_dec_level1 = self.prompt_decoder_level1(inp_dec_level1, pan_prompt_list[0], ms_prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_ms

        if C == 4 :
            out_dec_level1 = out_dec_level1.float().view(B, C, 2, H*4, W*4).mean(dim=2)
 
        return out_dec_level1 

class Spatial_channel_gate_Affine(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        promptsize = 128,
        promptlen = 16,
        atom_dim = 256
    ):

        super(Spatial_channel_gate_Affine, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        atom_num = 32 
        ## 生成ms_prompt 
        self.ms_prompt_gen = RIN(in_dim=inp_channels - 1, atom_num=atom_num, atom_dim=atom_dim) 
        ## 生成pan_prompt
        self.pan_prompt_gen = PAN_PromptBlock_multiscale(prompt_size = promptsize)
        self.prompt_encoder_level1 = Affine_prompt(dim=16, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level2 = Affine_prompt(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level3 = Affine_prompt(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_latent = Affine_prompt(dim=128, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level3 = Affine_prompt(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level2 = Affine_prompt(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level1 = Affine_prompt(dim=32, channel_prompt_dim=atom_dim)

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
          
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
   

    def forward(self, inp_ms, inp_pan): 
        B, C, H, W = inp_ms.shape
        if C == 4 :
            inp_ms = inp_ms.repeat_interleave(2, dim=1)

        # ms_prompt = self.ms_prompt_gen(inp_ms)
        inp_ms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
        inp_img = torch.concat((inp_ms,inp_pan), dim=1)
       
        pan_prompt_list = self.pan_prompt_gen(inp_pan)
        ms_prompt = self.ms_prompt_gen(inp_ms)

        inp_enc_level1 = self.patch_embed(inp_img)
          
        inp_enc_level1 = self.prompt_encoder_level1(inp_enc_level1, pan_prompt_list[0], ms_prompt) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 


        inp_enc_level2 = self.down1_2(out_enc_level1) 
        inp_enc_level2 = self.prompt_encoder_level2(inp_enc_level2, pan_prompt_list[1], ms_prompt)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.prompt_encoder_level3(inp_enc_level3, pan_prompt_list[2], ms_prompt)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        inp_enc_level4 = self.prompt_decoder_latent(inp_enc_level4, pan_prompt_list[3], ms_prompt)
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.prompt_decoder_level3(inp_dec_level3, pan_prompt_list[2], ms_prompt)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.prompt_decoder_level2(inp_dec_level2, pan_prompt_list[1], ms_prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # pan_last_prompt = torch.cat([pan_prompt_list[0],pan_prompt_list[0]],dim=1) 
        inp_dec_level1 = self.prompt_decoder_level1(inp_dec_level1, pan_prompt_list[0], ms_prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_ms

        if C == 4 :
            out_dec_level1 = out_dec_level1.float().view(B, C, 2, H*4, W*4).mean(dim=2)
 
        return out_dec_level1 

class Restormer_pan(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 42,

        num_blocks = [4,6,6,8],  
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        ):

        super(Restormer_pan, self).__init__()
        self.upMode = 'bilinear'  


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

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
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)



    # def forward(self, ms, pan):
    def forward(self, inp_ms, inp_pan):
        B, C, H, W = inp_ms.shape
        ## 4通道复制成8通道
        if C == 4 :
            inp_ms = inp_ms.repeat_interleave(2, dim=1)
        inp_ms = F.interpolate(inp_ms, scale_factor = 4, mode = self.upMode)
        inp_img = torch.concat([inp_ms, inp_pan],dim=1)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)       
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

        out_dec_level1 = self.output(out_dec_level1) + inp_ms
        
        ## 还原成4通道
        if C == 4 :
            out_dec_level1 = out_dec_level1.float().view(B, C, 2, H*4, W*4).mean(dim=2)

        return out_dec_level1
    
class ProxNet_AMIR_Prompt(nn.Module):
    def __init__(self, 
        inp_channels=9, 
        out_channels=8, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        promptsize = 128,
        promptlen = 16,
        atom_dim = 256
    ):

        super(ProxNet_AMIR_Prompt, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        atom_num = 32 
        ## 生成ms_prompt 
        self.ms_prompt_gen = RIN(in_dim=inp_channels - 1, atom_num=atom_num, atom_dim=atom_dim) 
        ## 生成pan_prompt
        self.pan_prompt_gen = PAN_PromptBlock_multiscale(prompt_size = promptsize)
        ## 嵌入prompt
        self.prompt_encoder_level1 = SCgate(dim=16, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_encoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_latent = SCgate(dim=128, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level3 = SCgate(dim=64, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level2 = SCgate(dim=32, channel_prompt_dim=atom_dim)
        self.prompt_decoder_level1 = SCgate(dim=32, channel_prompt_dim=atom_dim)

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
          
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
   

    def forward(self, inp_ms, inp_pan): 
        B, C, H, W = inp_ms.shape

        ms_prompt = self.ms_prompt_gen(inp_ms)
        pan_prompt_list = self.pan_prompt_gen(inp_pan)
        inp_img = torch.concat((inp_ms,inp_pan), dim=1)
        

        inp_enc_level1 = self.patch_embed(inp_img)
          
        inp_enc_level1 = self.prompt_encoder_level1(inp_enc_level1, pan_prompt_list[0], ms_prompt) 
        # print(f"inp_enc_level1.shape: {inp_enc_level1.shape}, pan_prompt_list[0].shape: {pan_prompt_list[0].shape}, ms_prompt.shape:{ms_prompt.shape}")
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 


        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2 = self.prompt_encoder_level2(inp_enc_level2, pan_prompt_list[1], ms_prompt)
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        inp_enc_level3 = self.prompt_encoder_level3(inp_enc_level3, pan_prompt_list[2], ms_prompt)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)        
        inp_enc_level4 = self.prompt_decoder_latent(inp_enc_level4, pan_prompt_list[3], ms_prompt)
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.prompt_decoder_level3(inp_dec_level3, pan_prompt_list[2], ms_prompt)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.prompt_decoder_level2(inp_dec_level2, pan_prompt_list[1], ms_prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # pan_last_prompt = torch.cat([pan_prompt_list[0],pan_prompt_list[0]],dim=1) 
        inp_dec_level1 = self.prompt_decoder_level1(inp_dec_level1, pan_prompt_list[0], ms_prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1) 


        out_dec_level1 = self.output(out_dec_level1) + inp_ms

 
        return out_dec_level1
    
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt
    
class ProxNet_Prompt(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(ProxNet_Prompt, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        
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
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        
        inp_enc_level1 = self.patch_embed(inp_img)
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


        out_dec_level1 = self.output(out_dec_level1) + inp_img

 
        return out_dec_level1
    

if __name__== "__main__":
    torch.cuda.set_device(1)
    # model = ProxNet_AMIR_Prompt(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 2, 2, 3])
    model = ProxNet_Prompt(inp_channels=8, out_channels=8, dim=16, num_blocks=[1,1,1,2]).cuda()

    inp_ms = torch.rand(4, 8, 128, 128).cuda()

    output = model(inp_ms)
    print("output: ",output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6, "M") 