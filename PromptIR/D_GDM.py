import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import torch
import torch.nn as nn
import torch.nn.functional as F
from PromptIR.Component import Attention
from einops import rearrange

class D_GDM(nn.Module):
    def __init__(self, prompt_len, prompt_dim, prompt_size, lin_dim,
                 dim, num_heads, bias):
        super(D_GDM,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, prompt_dim, kernel_size=1, bias=bias)

    def forward(self,x,text_emb):
        B,C,H,W = x.shape
        # print("x.shape: ",x.shape)
        weights = F.softmax(self.linear_layer(text_emb),dim=1)
        prompt = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        key = torch.sum(prompt,dim=1)
        # print("key.shape: ", key.shape)

        qv = self.qkv_dwconv(self.qkv(x))
        q,v = qv.chunk(2, dim=1)   

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out)

        # print(out.shape)
        return out



if __name__ == "__main__":
    torch.cuda.set_device(2)
    dim = 8
    model = D_GDM(prompt_len=5, prompt_dim=32, prompt_size=128, lin_dim=384,
                  dim=dim, num_heads=2, bias=False).cuda()

    text_emb = torch.rand(2,384).cuda()
    x = torch.rand(2, dim, 128, 128).cuda()

    output = model(x, text_emb)
    print(output.shape)

