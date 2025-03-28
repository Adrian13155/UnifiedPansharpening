import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
from torch import nn
import torch
from torch.nn import functional as F
from codebook.model.vq import ResidualVectorQuantizer

class BlockBasedResidualVectorQuantizer3D_forone(ResidualVectorQuantizer):
    """
    只有一种码本的codebook, 第一阶段的codebook
    """
    def __init__(self, n_shared=1024, n_task = 256, e_dim=256, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False):
        super().__init__(1, 1, 0.25, False, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(kernel_size=(self.unfold_size, self.unfold_size))
        self.beta = beta
        self.e_dim = e_dim
        self.depth = depth  # 匹配次数  
        
        self.n_shared = n_shared
        self.n_task = n_task

        # 共享的/公有的codebook
        self.shared_codebook = nn.Embedding(self.n_shared , e_dim)
        self.shared_codebook.weight.data.uniform_(-1.0 / self.n_shared , 1.0 / self.n_shared)

    def forward(self,z,one_hot):
        b,c,d,h,w = z.shape # 1,64,4,32,32
        z_flattened = z.view(-1, c, h ,w) # 64,4,32,32
        z_flattened = self.unfold(z_flattened).permute(0, 2, 1) # 64, 961, 16
        # print("z_flattened1.shape", z_flattened.shape)
        z_flattened = z_flattened.reshape(-1, self.e_dim)
        # print("z_flattened2.shape", z_flattened.shape)
        z_q = z_flattened

        codebook = self.shared_codebook.weight
        z_q, residual, indices = 0, z_flattened, []

        for i in range(self.depth):
            d = self.dist(residual, codebook)  # b x N
            min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
            delta = self.shared_codebook(min_encoding_indices)

            z_q = z_q + delta
            residual = residual - delta

        z_q = self.fold(z_q, z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        codebook_loss = q_latent_loss + e_latent_loss * self.beta

        z_q = z + (z_q - z).detach()

        return z_q, codebook_loss
    
    def fold(self, z_t, shape_z):
        b, c, d, h, w = shape_z
        z_t = z_t.view(b * d, -1, self.e_dim).permute(0, 2, 1)
        # print("z_t.shape", z_t.shape)
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
        count_t = torch.ones(1, c, h, w).to(z_t.device)#.cuda()
        # print("count_t.shape", count_t.shape)
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        # print("z_q.shape",z_q.shape)
        z_q = z_q / count_t
        return z_q.view(shape_z)



class BlockBasedResidualVectorQuantizer3D(ResidualVectorQuantizer):
    """
    既有公有码本又有私有码本的codebook
    """
    def __init__(self, n_shared=1024, n_task = 256, e_dim=256, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False):
        super().__init__(1, 1, 0.25, False, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(kernel_size=(self.unfold_size, self.unfold_size))
        self.beta = beta
        self.e_dim = e_dim
        self.depth = depth  # 匹配次数  
        
        self.n_shared = n_shared
        self.n_task = n_task

        # 共享的/公有的codebook
        self.shared_codebook = nn.Embedding(self.n_shared , e_dim)
        self.shared_codebook.weight.data.uniform_(-1.0 / self.n_shared , 1.0 / self.n_shared)

        self.task_codebooks = nn.ModuleList([nn.Embedding(self.n_task, e_dim) for _ in range(4)])
        for codebook in self.task_codebooks:
            codebook.weight.data.uniform_(-1.0 / self.n_shared, 1.0 / self.n_shared)

    def forward(self,z,one_hot):
        b,c,d,h,w = z.shape # 1,64,4,32,32
        z_q = torch.zeros_like(z)
        total_codebook_loss = 0.0
        for batch_idx in range(b):
            z_batch = z[batch_idx].unsqueeze(0)  # 取出单个 batch, 形状 [1, c, d, h, w]
            

            z_flattened = z_batch.view(-1, c, h, w) 
            z_flattened = self.unfold(z_flattened).permute(0, 2, 1) # 64, 961, 16
            z_flattened = z_flattened.reshape(-1, self.e_dim)

            z_q_batch = torch.zeros_like(z_flattened)
            residual = z_flattened


            shared_codebook = self.shared_codebook.weight
            for i in range(self.depth):
                d = self.dist(residual, shared_codebook)  # b x N
                min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
                delta = self.shared_codebook(min_encoding_indices)

                z_q_batch = z_q_batch + delta
                residual = residual - delta

            task_id = one_hot[batch_idx].argmax().item()  # 获取当前 batch 的任务索引
            task_codebook = self.task_codebooks[task_id].weight  # 选择任务特定 codebook

            for i in range(self.depth // 2):
                d = self.dist(residual, task_codebook)  # b x N
                min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
                delta = self.shared_codebook(min_encoding_indices)

                z_q_batch = z_q_batch + delta
                residual = residual - delta


            z_q_batch = self.fold(z_q_batch, z_batch.shape)

            # **计算 codebook loss**
            e_latent_loss = torch.mean((z_q_batch.detach() - z_batch) ** 2)
            q_latent_loss = torch.mean((z_q_batch - z_batch.detach()) ** 2)
            codebook_loss = q_latent_loss + e_latent_loss * self.beta
            total_codebook_loss += codebook_loss

            z_q[batch_idx] = z_batch + (z_q_batch - z_batch).detach()

        z_q = z + (z_q - z).detach()

        return z_q, total_codebook_loss / b
    
    def fold(self, z_t, shape_z):
        b, c, d, h, w = shape_z
        z_t = z_t.view(b * d, -1, self.e_dim).permute(0, 2, 1)
        # print("z_t.shape", z_t.shape)
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
        count_t = torch.ones(1, c, h, w).to(z_t.device)#.cuda()
        # print("count_t.shape", count_t.shape)
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        # print("z_q.shape",z_q.shape)
        z_q = z_q / count_t
        return z_q.view(shape_z)

if __name__ == "__main__":
    torch.cuda.set_device(5)
    model = BlockBasedResidualVectorQuantizer3D(n_shared=1024, n_task = 256, e_dim=256, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False).cuda()

    x = torch.rand(2, 64, 8, 32, 32).cuda()

    one_hot = torch.tensor([
        [1, 0, 0, 0],  # 第一个样本属于任务 0
        [0, 1, 0, 0]   # 第二个样本属于任务 1
    ], dtype=torch.float32)  # [batch, num_tasks]
    model(x, one_hot)

    # input_tensor = torch.randn(1, 3, 5, 5)
    # unfold = nn.Unfold(kernel_size=3, stride=1, padding=0)
    # output = unfold(input_tensor)
    # print(output.shape) # [1,27,9]