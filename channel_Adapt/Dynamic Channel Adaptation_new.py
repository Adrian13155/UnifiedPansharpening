import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WavelengthsAndMTF():
    gf1_wtf = torch.tensor([0.15,0.15,0.15,0.15])
    qb_wtf = torch.tensor([0.22,0.25,0.27,0.18])
    wv2_wtf = torch.tensor([0.35,0.32,0.30,0.28,0.25,0.22,0.20,0.18])
    wv4_wtf = torch.tensor([0.30,0.28,0.25,0.20])

    wtfs = [gf1_wtf, qb_wtf, wv2_wtf, wv4_wtf]

    gf1_wavelength = torch.tensor([485,555,660,830])
    qb_wavelength = torch.tensor([485,560,660,830])
    wv2_wavelength = torch.tensor([425,480,545,605,660,725,832,950])
    wv4_wavelength = torch.tensor([480,545,672,850])

    wavelengths = [gf1_wavelength, qb_wavelength, wv2_wavelength, wv4_wavelength]


class CrossAttentionFusion(nn.Module):
    """使用Cross Attention融合两个卷积核"""
    def __init__(self, kernel_size, embed_dim, num_heads=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        
        # Query来自k_wave, Key/Value来自k_mtf
        self.q_proj = nn.Linear(kernel_size**2, embed_dim)
        self.k_proj = nn.Linear(kernel_size**2, embed_dim)
        self.v_proj = nn.Linear(kernel_size**2, embed_dim)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, kernel_size**2)
        
    def forward(self, k_wave, k_mtf):
        """
        Args:
            k_wave: [out, C, K*K] 波长相关核
            k_mtf: [out, C, K*K] MTF相关核
        Returns:
            fused_kernel: [out, C, K*K]
        """
        # 投影到embedding空间
        q = self.q_proj(k_wave)  # [out, C, D]
        k = self.k_proj(k_mtf)
        v = self.v_proj(k_mtf)
        
        # 调整维度为 [C, out, D] (因为nn.MultiheadAttention期望seq_len在前)
        q = q.permute(1, 0, 2)  # [C, out, D]
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        
        # Cross Attention
        attn_output, _ = self.multihead_attn(q, k, v)  # [C, out, D]
        attn_output = attn_output.permute(1, 0, 2)  # [out, C, D]
        
        # 投影回原始空间
        fused = self.out_proj(attn_output)  # [out, C, K*K]
        
        # 残差连接
        return fused + k_wave

class MTFLearner(nn.Module):
    """改进的MTF学习器，每个通道独立学习"""
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels * kernel_size * kernel_size)
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, mtf):
        # mtf: [C] -> [C, out*K*K]
        kernel = self.mlp(mtf.view(-1, 1))  # [C, out*K*K]
        return kernel.view(-1, self.out_channels, self.kernel_size, self.kernel_size)  # [C, out, K, K]

class DynamicSpectralConv(nn.Module):
    """动态卷积"""
    def __init__(self, 
                 all_channels=65,  # 400-1050nm @10nm间隔 (65 channels)
                 out_channels=1,
                 kernel_size=7,
                 embedding_dim=64,
                 scale_factor=4,
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.all_channels = all_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        # 波长配置 (400, 410, ..., 1050nm)
        # self.register_buffer('wavelength_bank', 
        #                    torch.arange(400, 1051, 10).float()[:in_channels])
        # assert len(self.wavelength_bank) == 65, "波长通道数应为65"
        
        
        # 1. 可学习的大卷积核 [out, 65, K, K]
        self.large_kernel = nn.Parameter(
            torch.randn(out_channels, all_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
    
        # 2. 波长相关的小卷积核生成器
        # self.pos_encoder = SinusoidalTimeEmbedding(embedding_dim)
        # self.wavelength_mlp = nn.Sequential(
        #     nn.Linear(embedding_dim, 128),
        #     nn.GELU(),
        #     nn.Linear(128, out_channels * kernel_size * kernel_size)
        # )
        
        # 3. MTF学习的卷积核 (每个输入通道独立)
        self.mtf_learner = MTFLearner(out_channels, kernel_size)
        
        # 4. 核融合Transformer
        # self.kernel_transformer = nn.TransformerEncoderLayer(
        #     d_model=kernel_size*kernel_size,
        #     nhead=num_heads,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.kernel_fusion = nn.Linear(kernel_size**2 * 3, kernel_size**2)
        # 4. Cross Attention融合模块
        self.cross_attn = CrossAttentionFusion(
            kernel_size=kernel_size,
            embed_dim=embedding_dim
        )    
        # 偏置生成
        # self.bias_mlp = nn.Sequential(
        #     nn.Linear(out_channels * 3, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, out_channels)
        # )

    def get_channel_index(self, wavelengths):
        """精确计算波长对应的通道索引：(λ-400)//10"""
        # wavelengths: [C] -> [C]
        indices = ((wavelengths - 400) / 10).long()
        return indices.clamp(0, self.all_channels-1)

    def forward(self, x, one_hot):
        """
        Args:
            x: [B, C, H, W] 输入多光谱图像
            input_wavelengths: [C] 每个通道的中心波长
            mtf_values: [C] 每个通道的MTF值
        """
        B, C, H, W = x.shape
        K = self.kernel_size

        task_id = one_hot[0].argmax().item()
        input_wavelengths, mtf_values =  WavelengthsAndMTF.wavelengths[task_id], WavelengthsAndMTF.wtfs[task_id]

        
        # 1. 从大核中选择对应通道 [out, C, K, K]
        indices = self.get_channel_index(input_wavelengths)  # [C]
        selected_kernel = self.large_kernel[:, indices, :, :]  # [out, C, K, K]
        
        # 2. 生成波长相关的小核 [out, C, K, K]
        # pos_emb = self.pos_encoder(input_wavelengths)  # [C, D]
        # small_kernel = self.wavelength_mlp(pos_emb)  # [C, out*K*K]
        # small_kernel = small_kernel.view(C, self.out_channels, K, K)  # [C, out, K, K]
        # small_kernel = small_kernel.permute(1, 0, 2, 3)  # [out, C, K, K]
        
        # 3. 生成MTF相关的核 [out, C, K, K]
        mtf_kernel = self.mtf_learner(mtf_values)  # [C, out, K, K]
        mtf_kernel = mtf_kernel.permute(1, 0, 2, 3)  # [out, C, K, K]
        
        # 4. 核融合
        # 展平所有核 [out, C, K*K]
        k_wave = selected_kernel.flatten(2)  # [out, C, K*K]
        # k_wave = small_kernel.flatten(2)
        k_mtf = mtf_kernel.flatten(2)
        
        # 拼接特征 [out, C, 3*K*K]
        # combined = torch.cat([k_wave, k_mtf], dim=-1)
        
        # Transformer融合 [out, C, K*K]
        fused_kernel = self.cross_attn(k_wave,k_mtf).view(self.out_channels,C,K,K)

        
        # 5. 偏置融合
        bias = self.bias.view(1, -1)  # [1, out]
        # b_wave = small_kernel.mean(dim=(1,2,3)).view(1, -1)  # [1, out]
        # b_mtf = mtf_kernel.mean(dim=(1,2,3)).view(1, -1)  # [1, out]
        # fused_bias = self.bias_mlp(
        #     torch.cat([b_large, b_wave, b_mtf], dim=-1))  # [1, out]
        
        # 6. 执行卷积
        out = F.conv2d(x, 
                      weight=fused_kernel, 
                      bias=bias.view(-1),
                      stride=self.scale_factor,
                      padding=K//2)
        return out



class DynamicSpectralTransposeConv(nn.Module):
    """动态转置卷积"""
    def __init__(self, 
                 all_channels=65,
                 out_channels=8,  # MS波段数
                 kernel_size=7,
                 embedding_dim=64,
                 scale_factor=4):  # 上采样倍数
        super().__init__()
        self.all_channels = all_channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        # 1. 可学习的大转置卷积核 [out, 65, K, K]
        self.large_kernel = nn.Parameter(torch.randn(out_channels, all_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # 3. MTF学习器（使用目标MS的MTF值）
        self.mtf_learner = MTFLearner(out_channels, kernel_size)
        
        # 4. 转置卷积专用融合模块
        self.cross_attn = CrossAttentionFusion(
            kernel_size=kernel_size,
            embed_dim=embedding_dim
        )

    def get_channel_index(self, wavelengths):
        """精确计算波长对应的通道索引：(λ-400)//10"""
        # wavelengths: [C] -> [C]
        indices = ((wavelengths - 400) / 10).long()
        return indices.clamp(0, self.all_channels-1)

    def forward(self, pan, one_hot):
        """
        Args:
            pan: [B, 1, H, W] 输入PAN图像
            mtf_values: [out_channels] 目标MS波段的MTF值
        """
        B, _, H, W = pan.shape
        K = self.kernel_size

        task_id = one_hot[0].argmax().item()
        input_wavelengths, mtf_values =  WavelengthsAndMTF.wavelengths[task_id], WavelengthsAndMTF.wtfs[task_id]

        # 1. 基础大核 [out, 1, K, K]
        indices = self.get_channel_index(input_wavelengths)  # [C]
        selected_kernel = self.large_kernel[:, indices, :, :]  # [out, 1, K, K]
        if _ ==1 : #若输入为PAN图像则卷积核通道取平均处理
            selected_kernel = torch.mean(selected_kernel,dim=1,keepdim=True)

        # 3. 生成MTF相关核 [out, 1, K, K]
        mtf_kernel = self.mtf_learner(mtf_values.view(-1,1))
        if _ ==1 :
            mtf_kernel = torch.mean(mtf_kernel,dim=1,keepdim=True)

        k_wave = selected_kernel.flatten(2)
        k_mtf = mtf_kernel.flatten(2)
        
        # 4. 核融合（输出[out, 1, 2K, 2K]）
        fused_kernel = self.cross_attn(k_wave,k_mtf).view(_,self.out_channels,K,K) 
        
        # 5. 执行转置卷积
        out = F.conv_transpose2d(
            pan,
            weight=fused_kernel,
            bias=self.bias,
            stride=self.scale_factor,
            padding=K//2,
            output_padding=self.scale_factor-1
        )
        
        return out 


if __name__ == "__main__":
    # 测试用例
    H = DynamicSpectralConv(all_channels=65,  # 400-1050nm @10nm
                              out_channels=1,
                              kernel_size=7,
                              scale_factor=1
                              )
    x4 = torch.randn(4, 4, 64, 64)
    one_hot4 = torch.tensor([[0, 1, 0, 0]]
         ,dtype=torch.float32)
    task_id4 = one_hot4.argmax().item()
    out = H(x4, one_hot4)
    print("Output H 4 shape:", out.shape)  # [4, 1, 64, 64]

    x8 = torch.randn(4, 8, 64, 64)
    one_hot8 = torch.tensor([[0, 0, 1, 0]]
         ,dtype=torch.float32)
    task_id8 = one_hot8.argmax().item()
    out = H(x8, one_hot8)
    print("Output H 8 shape:", out.shape)  # [4, 1, 64, 64]

    D4 = DynamicSpectralConv(all_channels=65,  # 400-1050nm @10nm
                              out_channels=4,
                              kernel_size=7,
                              scale_factor=4
                              )

    D8 = DynamicSpectralConv(all_channels=65,  # 400-1050nm @10nm
                              out_channels=8,
                              kernel_size=7,
                              scale_factor=4
                              )
    
    out = D8(x8, one_hot8)
    print("Output D 8 shape:", out.shape)  # [4, 8, 16, 16]

    out = D4(x4, one_hot4)
    print("Output D 4 shape:", out.shape)  # [4, 4, 16, 16]

    HT4 = DynamicSpectralTransposeConv(
                all_channels=65,
                out_channels=4,
                kernel_size=7,
                scale_factor=1,
                )
    pan4 = torch.randn(4, 1, 64, 64)
    out = HT4(pan4, one_hot4)
    print("Output HT 4 shape:", out.shape)  # [4, 4, 64, 64]

    DT4 = DynamicSpectralTransposeConv(
                all_channels=65,
                out_channels=4,
                kernel_size=7,
                scale_factor=4,
                )
        # 模拟输入 (4张8通道的多光谱图像)
    x = torch.randn(4, 4, 16, 16)

    out = DT4(x,one_hot4)
    print("Output DT 4 shape:", out.shape)  # [4, 4, 64, 64]


    HT8 = DynamicSpectralTransposeConv(
                all_channels=65,
                out_channels=8,
                kernel_size=7,
                scale_factor=1,
                )
    # 模拟输入 (4张8通道的多光谱图像)
    pan8 = torch.randn(4, 1, 64, 64)
    out = HT8(pan8, one_hot8)
    print("Output HT 8 shape:", out.shape)  # [4, 8, 64, 64]

    DT8 = DynamicSpectralTransposeConv(
                all_channels=65,
                out_channels=8,
                kernel_size=7,
                scale_factor=4,
                )
        # 模拟输入 (4张8通道的多光谱图像)
    x = torch.randn(4, 8, 16, 16)

    out = DT8(x,one_hot8)
    print("Output DT 8 shape:", out.shape)  # [4, 8, 64, 64]