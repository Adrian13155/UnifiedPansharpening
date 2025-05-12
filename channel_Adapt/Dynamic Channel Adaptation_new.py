import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module): #正余弦位置编码
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downscale_freq_shift = 1
        self.max_period = 10000

    def forward(self, wavelength):
        position = wavelength.float().unsqueeze(1)
        d_model = self.embedding_dim
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros_like(wavelength).unsqueeze(1).repeat(1, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class DynamicChannelAdaptation(nn.Module):  #通过光谱波长学习动态的卷积
    def __init__(self, in_channels, out_channels, batch_size, L, scale, kernel_size=3, embedding_dim=64, num_heads=1, dropout=0.1,is_transpose=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.L = L
        self.scale = scale
        self.K = kernel_size
        self.is_transpose = is_transpose
        
        # 波长位置编码
        self.pos_encoder = SinusoidalTimeEmbedding(embedding_dim)
        # 波长嵌入mlp
        tem = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.wavelength_embedding = nn.ModuleList([tem for _ in range(in_channels)])
        
        # 动态权重生成的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 查询token
        self.weight_query = nn.Parameter(torch.randn(L, out_channels))
        self.bias_query = nn.Parameter(torch.randn(1, out_channels))
        
        # 权重和偏置生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(out_channels, out_channels * kernel_size * kernel_size),
            nn.GELU(),
            nn.Linear(out_channels * kernel_size * kernel_size, out_channels * kernel_size * kernel_size)
        )
        
        self.bias_generator = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, wavelengths):
        """
        Args:
            x: input tensor of shape [B, Cin, H, W]
            wavelengths: tensor of shape [Cin] containing central wavelengths for each channel
        Returns:
            output tensor of shape [B, Cout, H, W]
        """
        B, Cin, H, W = x.shape
        
        # 1. 波长位置编码+嵌入 
        e_lambda = self.pos_encoder(wavelengths.view(-1)).view(self.in_channels,-1).to(torch.float32)
        emb = torch.zeros(self.in_channels,self.out_channels)
        for i, md_embed in enumerate(self.wavelength_embedding):
                md_emb = md_embed(e_lambda[i, :])  # (N, Dout)
                emb[i,:] = md_emb # (N, Dout)
        e_lambda=emb

        # 2. Transformer处理
        # 拼接序列
        transformer_input = torch.cat([self.weight_query, e_lambda, self.bias_query], dim=0)  # [L+Cin+1, Dout]
        # Transformer处理
        transformer_output = self.transformer(transformer_input)  # [L+Cin+1, Dout]
        # 分离输出
        zw = transformer_output[:self.L, :]  # [L, Dout]
        zb = transformer_output[-1:, :]  # [1, Dout]
        
        # 3. 动态生成卷积权重和偏置
        # 生成权重
        weights = self.weight_generator(zw + e_lambda)  # [Cin, Dout*K*K]
        if self.is_transpose:
            weights = weights.view(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size).contiguous()
        else:
            weights = weights.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).contiguous()  # # [Dout, Cin, K, K] 
        # 生成偏置
        bias = self.bias_generator(zb)  # [1 , Dout]
        
        # 4. 执行动态卷积
        if self.is_transpose:
            print("x.shape", x.shape)
            print("weights.shape", weights.shape)
            padding,output_padding=self.auto_transpose_params(input_size=H,output_size=H*self.scale,kernel_size=self.K,stride=self.scale)
            out = F.conv_transpose2d(x, weight=weights, bias=bias.view(-1), stride=self.scale, padding=padding, output_padding=output_padding).contiguous()  
        else:
            padding = self.auto_conv_params(input_size=H,output_size=H//self.scale,kernel_size=self.K,stride=self.scale)
            out = F.conv2d(x, weight=weights, bias=bias.view(-1), stride=self.scale, padding=padding).contiguous()  # padding与K相适应
        return out

    def auto_transpose_params( #根据输入Kernel_Size和Stride等自适应计算padding 与 output_padding
        self,
        input_size: int,       # 输入特征图大小，例如 16
        output_size: int,      # 目标输出图大小，例如 64
        kernel_size: int,      # 卷积核大小，例如 9
        stride: int            # 步长，例如 4
    ):
        for padding in range(kernel_size):
            for output_padding in [0, 1]:
                calc_out = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
                if calc_out == output_size:
                    return padding,output_padding

        raise ValueError("No valid (padding, output_padding) combination found.")
    
    def auto_conv_params( #根据输入Kernel_Size和Stride等自适应计算padding
        self,
        input_size: int,       # 输入特征图大小，例如 16
        output_size: int,      # 目标输出图大小，例如 64
        kernel_size: int,      # 卷积核大小，例如 9
        stride: int            # 步长，例如 4
    ):
        for padding in range(kernel_size):
            calc_out = (input_size + 2 * padding - kernel_size) // stride + 1
            if calc_out == output_size:
                return padding
        raise ValueError("No valid padding found to reach desired output size.")
    
    

if __name__=="__main__":
    # 初始化模块
    # in_channels = 8  # 输入通道数/卷积核通道数
    # # out_channels =1  # 输出通道数/卷积核个数
    # kernel_size = 7  #卷积核尺寸
    # embedding_dim = 320  #位置编码的维度
    # batch_size = 4 
    # height, width = 64, 64 #输入图片分辨率
    # L = in_channels #初始化可学习权重的形状L*out_channels；可学习偏置1*out_channels;后续需要zw + e_lambda，因此L = in_channels
    # scale = 1 #下采样率/卷积的stride参数
    

    H = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=1,
        batch_size = 4,
        kernel_size=7,
        embedding_dim=320,
        L=8,
        scale = 1,
        is_transpose = False
    )

    HT = DynamicChannelAdaptation(
        in_channels=1,
        out_channels=8,
        batch_size = 4,
        kernel_size=7,
        embedding_dim=320,
        L=1,
        scale = 1,
        is_transpose = True
    )

    D = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=8,
        batch_size = 4,
        kernel_size=7,
        embedding_dim=320,
        L=8,
        scale = 4,
        is_transpose = False
    )

    DT = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=8,
        batch_size = 4,
        kernel_size=7,
        embedding_dim=320,
        L=8,
        scale = 4,
        is_transpose = True
    )

    # 示例输入
    MS = torch.randn(4, 8, 64, 64)  #bs,C,H,H
    PAN = torch.randn(4, 1, 64, 64) #bs,c,H,W
    MS_downsample = torch.randn(4,8,16,16) #bs,C,h,w
    # 假设每个通道的中心波长 (可以实际替换为遥感数据的真实波长)
    wavelengths_MS = torch.randn(8)
    wavelengths_PAN = torch.randn(1)

    # 前向变换
    H_MS = H(MS, wavelengths_MS)   #MS->PAN
    HT_PAN = HT(PAN, wavelengths_PAN)  #PAN->MS
    exit(0)
    D_MS = D(MS, wavelengths_MS)   #MS->MS_downsample
    DT_MS_downsample = DT(MS_downsample, wavelengths_MS) #MS_downsample->MS

    print('H_MS:',H_MS.shape)
    print('HT_PAN:',HT_PAN.shape)
    print('D_MS:',D_MS.shape)
    print('DT_MS_downsample:',DT_MS_downsample.shape)