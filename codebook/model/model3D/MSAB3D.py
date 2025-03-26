import torch
import torch.nn as nn
import torch.nn.functional as F
from ..MSAB import GELU,PreNorm
from einops import rearrange

class FeedForward3D(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=dim * mult),
            GELU(),
            nn.Conv3d(dim * mult, dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
        )
        self.dim = dim
        self.mult = mult

    def forward(self, x):
        """
        x: [b, d, h, w, c]
        return out: [b, d, h, w, c]
        """
        # 将输入数据的形状从 [b, d, h, w, c] 转换为 [b, c, d, h, w]
        out = self.net(x.permute(0, 4, 1, 2, 3).contiguous())
        # 将输出数据的形状从 [b, c, d, h, w] 转换回 [b, d, h, w, c]
        return out.permute(0, 2, 3, 4, 1).contiguous()
    
class MS_MSA3D(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 合并 QKV 计算，减少计算量
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        # 使用3D卷积代替2D卷积进行位置编码
        self.pos_emb = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=dim),
            GELU(),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=dim),
        )

    def forward(self, x_in):
        """
        x_in: [b, d, h, w, c]
        return out: [b, d, h, w, c]
        """
        # 输入验证
        assert x_in.dim() == 5, "Input tensor must be 5D [b, d, h, w, c]"
        b, d, h, w, c = x_in.shape
        # 将 d, h, w 展平
        x = x_in.view(b, d * h * w, c)  
        # 计算 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 一次性计算 Q, K, V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1) # b x heads x dim_head x (hwd)
        
        # 归一化
        q, k = map(lambda t: F.normalize(t, dim=-1, p=2), (q, k))
        # 计算注意力
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # 加权 V
        x = attn @ v
        # 还原形状
        x = rearrange(x, 'b h dd (d hh ww)-> b (d hh ww) (h dd)', d=d, hh=h, ww=w)
        out_c = self.proj(x).view(b, d, h, w, c)

        # 位置编码
        out_p = self.pos_emb(x_in.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        return out_c + out_p
    
class MSAB3D(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA3D(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward3D(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b, c, d, h, w]
        return out: [b, c, d, h, w]
        """
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [b, d, h, w, c]
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 4, 1, 2, 3).contiguous()  # [b, c, d, h, w]
        return out
    
