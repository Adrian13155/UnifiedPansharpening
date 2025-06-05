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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).cuda()
        pe = torch.zeros_like(wavelength).unsqueeze(1).repeat(1, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class DynamicChannelAdaptation(nn.Module):  #通过光谱波长学习动态的卷积
    gf1_wavelength = torch.tensor([485,555,660,830], dtype=torch.float64)
    qb_wavelength = torch.tensor([485,560,660,830], dtype=torch.float64)
    wv2_wavelength = torch.tensor([425,480,545,605,660,725,832,950], dtype=torch.float64)
    wv4_wavelength = torch.tensor([480,545,672,850], dtype=torch.float64)
    # gf1_wavelength = torch.tensor([0.15,0.15,0.15,0.15], dtype=torch.float64)
    # qb_wavelength = torch.tensor([0.22,0.25,0.27,0.18], dtype=torch.float64)
    # wv2_wavelength = torch.tensor([0.35,0.32,0.30,0.28,0.25,0.22,0.20,0.18], dtype=torch.float64)
    # wv4_wavelength = torch.tensor([0.30,0.28,0.25,0.20], dtype=torch.float64)
    # 四数据集用wavelengths = [gf1_wavelength, qb_wavelength, wv2_wavelength, wv4_wavelength]
    wavelengths = [gf1_wavelength, qb_wavelength, wv2_wavelength, wv4_wavelength]
    # 三数据集用wavelengths = [gf1_wavelength, qb_wavelength, wv4_wavelength]
    def __init__(self, in_channels, out_channels, scale, kernel_size=3, embedding_dim=64, num_heads=1, dropout=0.1,is_transpose=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.num_channels = max(self.out_channels,self.in_channels)
        self.L = self.num_channels  ##初始化可学习权重的形状L*out_channels；可学习偏置1*out_channels;后续需要zw + e_lambda，因此L要确定
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
        self.wavelength_embedding = nn.ModuleList([tem for _ in range(self.num_channels)])  #为了PAN->MS也用上wavelength_MS的信息，因此加上max,min
        
        # 动态权重生成的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 查询token
        self.weight_query = nn.Parameter(torch.randn(self.L, out_channels))
        self.bias_query = nn.Parameter(torch.randn(1, out_channels))
        
        # 权重和偏置生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(out_channels, min(in_channels,out_channels) * kernel_size * kernel_size),
            nn.GELU(),
            nn.Linear(min(in_channels,out_channels) * kernel_size * kernel_size, min(in_channels,out_channels) * kernel_size * kernel_size)
        )
        
        self.bias_generator = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, one_hot):
        """
        Args:
            x: input tensor of shape [B, Cin, H, W]
            wavelengths: tensor of shape [Cin] containing central wavelengths for each channel
        Returns:
            output tensor of shape [B, Cout, H, W]
        """
        B, Cin, H, W = x.shape
        out = []
        for batch_idx in range(B):
            x_batch = x[batch_idx].unsqueeze(0)

            # 1. 波长位置编码+嵌入 
            task_id = one_hot[batch_idx].argmax().item()
            wavelengths = DynamicChannelAdaptation.wavelengths[task_id].cuda()
            e_lambda = self.pos_encoder(wavelengths.view(-1)).view(self.num_channels,-1).to(torch.float32)
            emb = torch.zeros(self.num_channels,self.out_channels).cuda()
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
                padding,output_padding=self.auto_transpose_params(input_size=H,output_size=H*self.scale,kernel_size=self.K,stride=self.scale)
                outT = F.conv_transpose2d(x_batch, weight=weights, bias=bias.view(-1), stride=self.scale, padding=padding, output_padding=output_padding).contiguous()  
            else:
                padding = self.auto_conv_params(input_size=H,output_size=H//self.scale,kernel_size=self.K,stride=self.scale)
                outT = F.conv2d(x_batch, weight=weights, bias=bias.view(-1), stride=self.scale, padding=padding).contiguous()  # padding与K相适应

            out.append(outT.squeeze(0))
        
        out = torch.stack(out, dim=0)
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
    
    
    H4 = DynamicChannelAdaptation(
        in_channels=4,
        out_channels=1,
        kernel_size=7,
        embedding_dim=320,
        scale = 1,
        is_transpose = False
    )

    HT4 = DynamicChannelAdaptation(
        in_channels=1,
        out_channels=4,
        kernel_size=7,
        embedding_dim=320,
        scale = 1,
        is_transpose = True
    )

    D4 = DynamicChannelAdaptation(
        in_channels=4,
        out_channels=4,
        kernel_size=7,
        embedding_dim=320,
        scale = 4,
        is_transpose = False
    )

    DT4 = DynamicChannelAdaptation(
        in_channels=4,
        out_channels=4,
        kernel_size=7,
        embedding_dim=320,
        scale = 4,
        is_transpose = True
    )

    H8 = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=1,
        kernel_size=7,
        embedding_dim=320,
        scale = 1,
        is_transpose = False
    )

    HT8 = DynamicChannelAdaptation(
        in_channels=1,
        out_channels=8,
        kernel_size=7,
        embedding_dim=320,
        scale = 1,
        is_transpose = True
    )

    D8 = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=8,
        kernel_size=7,
        embedding_dim=320,
        scale = 4,
        is_transpose = False
    )

    DT8 = DynamicChannelAdaptation(
        in_channels=8,
        out_channels=8,
        kernel_size=7,
        embedding_dim=320,
        scale = 4,
        is_transpose = True
    )


    one_hot = torch.tensor([
        [0, 1, 0, 0],  # 第一个样本属于任务 0
        [1, 0, 0, 0]   # 第二个样本属于任务 1
    ], dtype=torch.float32)  # [batch, num_tasks]

    # 示例输入
    MS = torch.randn(2, 4, 64, 64)  #bs,C,H,H
    PAN = torch.randn(2, 1, 64, 64) #bs,c,H,W
    MS_downsample = torch.randn(2,4,16,16) #bs,C,h,w
    # 假设每个通道的中心波长 (可以实际替换为遥感数据的真实波长)


    # 前向变换
    H_MS = H4(MS, one_hot)   #MS->PAN
    HT_PAN = HT4(PAN, one_hot)  #PAN->MS
    D_MS = D4(MS, one_hot)   #MS->MS_downsample
    DT_MS_downsample = DT4(MS_downsample, one_hot) #MS_downsample->MS

    print('H_MS:',H_MS.shape)
    print('HT_PAN:',HT_PAN.shape)
    print('D_MS:',D_MS.shape)
    print('DT_MS_downsample:',DT_MS_downsample.shape)

    one_hot = torch.tensor([
        [0, 0, 1, 0],  # 第一个样本属于任务 0
        [0, 0, 1, 0]   # 第二个样本属于任务 1
    ], dtype=torch.float32)  # [batch, num_tasks]

    MS8 = torch.randn(2, 8, 64, 64)  #bs,C,H,H
    PAN = torch.randn(2, 1, 64, 64) #bs,c,H,W
    MS_downsample8 = torch.randn(2,8,16,16) #bs,C,h,w

    H_MS = H8(MS8, one_hot)   #MS->PAN
    HT_PAN = HT8(PAN, one_hot)  #PAN->MS
    D_MS = D8(MS8, one_hot)   #MS->MS_downsample
    DT_MS_downsample = DT8(MS_downsample8, one_hot) #MS_downsample->MS

    print('H_MS:',H_MS.shape)
    print('HT_PAN:',HT_PAN.shape)
    print('D_MS:',D_MS.shape)
    print('DT_MS_downsample:',DT_MS_downsample.shape)