from torch import nn
import torch
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            # (A-B)^2 = A^2 + B^2 - 2AB
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum(y ** 2, dim=1) - 2 * \
                   torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)  # b x N

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
            # + self.beta * self.dist(z_q_gt.detach(), z)
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

    def get_k_nearest_neighbors(self, z, k=8):
        # reshape z -> (batch, height, width, channel) and flatten
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # b h w c
        z_flattened = z.view(-1, self.e_dim)  # bhw c

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)  # b x N
        _, idx = torch.topk(-d, k, dim=-1)
        centers = self.embedding(idx)
        centers = centers.view(b, h*w, k, c)
        return centers


class ResidualVectorQuantizer(VectorQuantizer):
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, depth=6):
        super().__init__(n_e, e_dim, beta, LQ_stage)
        self.depth = depth

    def forward(self, z, gt_indices=None, current_iter=None):
        b, c, h, w = z.shape
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if gt_indices is not None:
            # gt_indices.shape = b x d x n
            z_q_gt = 0
            for i in range(len(gt_indices)):
                z_q_gt = z_q_gt + self.embedding(gt_indices[i])
            z_q_gt = z_q_gt.view(z.shape)

        codebook = self.embedding.weight
        z_q, residual, indices = 0, z_flattened, []
        for i in range(self.depth):
            d = self.dist(residual, codebook)  # b x N
            min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
            delta = self.embedding(min_encoding_indices)
            z_q = z_q + delta
            residual = residual - delta
            indices.append(min_encoding_indices.clone())

        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # d = self.dist(z_flattened, codebook)
            # d_gt = self.dist(z_q_gt, codebook)
            # codebook_loss = F.kl_div(F.log_softmax(-d, dim=-1), F.softmax(-d_gt, dim=-1))
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = torch.stack(indices, dim=1).reshape(b, h, w, -1)
        return z_q, codebook_loss, indices

    def get_codebook_entry(self, indices):
        b, d, h, w = indices.shape
        gt_indices = indices.reshape(b, d, h * w)
        z_q = torch.sum(self.embedding(gt_indices), dim=1, keepdim=False)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

class SharedAndTaskSpecificCodebookVectorQuantizer(ResidualVectorQuantizer):
    """
    公有码本和私有码本的想法:
    原特征和公有码本匹配的向量相减得到一个残差，这个残差再去跟私有码本匹配，匹配之后的向量（私有特征）再与匹配的公有特征相加。
    私有码本是根据数据集分的，每一个数据集对应一个私有码本。
    """
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False, num_tasks=4):
        super().__init__(1, e_dim, beta, LQ_stage, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(kernel_size=(self.unfold_size, self.unfold_size))
        self.z_length = e_dim // (self.unfold_size * self.unfold_size)
        self.z_stride = self.z_length // 2
        self.mlp_codebook = mlp_codebook
        if self.mlp_codebook:
            self.codebook = nn.ModuleList([nn.Linear(e_dim, e_dim) for i in range(3)])

        self.depth4shared = 12
        self.depth4taskd = 6

        self.n_shared = 1024
        self.n_tasks = 256

        # 共享的/公有的codebook
        self.shared_codebook = nn.Embedding(self.n_shared , e_dim)
        self.shared_codebook.weight.data.uniform_(-1.0 / self.n_shared , 1.0 / self.n_shared)# 这个初始化很重要,没有就是10个psnr的区别


        # 私有的/Task-specific的codebook
        self.task_codebookss = nn.ModuleList([nn.Embedding(self.n_tasks, e_dim) for _ in range(num_tasks)])
        for codebook in self.task_codebookss:
            codebook.weight.data.uniform_(-1.0 / self.n_tasks, 1.0 / self.n_tasks)

        # 第二阶段：训练共享码本
        for param in self.shared_codebook.parameters():
            param.requires_grad = False  # 冻结共享码本的权重更新

        for task_codebook in self.task_codebookss:
            for param in task_codebook.parameters():
                param.requires_grad = False  # 允许任务特定码本的权重
        
    def fold(self, z_t, shape_z):
        b, c, h, w = shape_z
        z_t = z_t.view(b, -1, self.e_dim).permute(0, 2, 1)
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
        count_t = torch.ones(1, self.z_length, h, w).to(z_t.device)#.cuda()
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        z_q = z_q / count_t
        return z_q.view(shape_z)

    def forward(self, z, one_hot, gt_indices=None, current_iter=None):
        assert self.z_length == z.size(1)
        z_flattened = self.unfold(z).permute(0, 2, 1)
        z_flattened = z_flattened.reshape(-1, self.e_dim)

        # Shared codebook matching
        z_q_shared = 0
        residual = z_flattened
        for _ in range(self.depth4shared):
            d_shared = self.dist(residual, self.shared_codebook.weight)
            min_encoding_indices_shared = torch.argmin(d_shared, dim=1)
            delta_shared = self.shared_codebook(min_encoding_indices_shared)
            z_q_shared += delta_shared
            residual -= delta_shared


        # Task-specific codebook matching
        task_index = torch.argmax(one_hot, dim=1)
        z_q_task = torch.zeros_like(z_q_shared)

        for i in range(one_hot.size(0)):  # Iterate over batch
            task_codebook = self.task_codebookss[task_index[i]]
            residual_task = residual[i:i+1]  # Initialize residual_task for each sample

            for _ in range(self.depth4taskd):
                d_task = self.dist(residual_task, task_codebook.weight)
                min_encoding_indices_task = torch.argmin(d_task, dim=1)
                delta_task = task_codebook(min_encoding_indices_task)
                residual_task -= delta_task  # 更新残差
                z_q_task[i:i+1] += delta_task  # 累加 delta_task



        # 计算每个任务特定码本的完整分布
        # task_distributions = [F.softmax(codebook.weight, dim=0) for codebook in self.task_codebooks]

        # 确保分布是归一化的并避免 log(0)
        # epsilon = 1e-6
        # task_distributions = [dist + epsilon for dist in task_distributions]

        # Calculate KL divergence between each pair of task-specific codebook means
        # kl_loss = 0
        # for i in range(len(task_distributions)):
        #     for j in range(i + 1, len(task_distributions)):
        #         kl_ij = F.kl_div((task_distributions[i] + epsilon).log(), task_distributions[j] + epsilon, reduction='batchmean')
        #         kl_ji = F.kl_div((task_distributions[j] + epsilon).log(), task_distributions[i] + epsilon, reduction='batchmean')
        #         kl_loss += kl_ij + kl_ji

        # Combine shared and task-specific quantized outputs
        z_q = z_q_shared  + z_q_task

        z_q = self.fold(z_q, z.shape)

        # Calculate losses
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # 动态调整 kl_loss 的权重
        # if codebook_loss.item() > 0 and kl_loss.item() > 0:
        #     codebook_loss_magnitude = torch.floor(torch.log10(codebook_loss)).item()
        #     kl_loss_magnitude = torch.floor(torch.log10(kl_loss)).item()
        #     target_kl_loss_magnitude = codebook_loss_magnitude - 1
        #     kl_loss_weight = 10 ** (target_kl_loss_magnitude - kl_loss_magnitude)
        # else:
        # kl_loss_weight = 0.0001  # 默认值，防止出现 log(0) 的情况

        z_q = z + (z_q - z).detach()

        # kl_loss = kl_loss * kl_loss_weight
        return z_q, codebook_loss, None
        # return z_q, codebook_loss, kl_loss, None
class DualCodebookVectorQuantizer(ResidualVectorQuantizer):
    """
    公有码本和私有码本的想法，在余弦相似度上做损失来约束公有码本和私有码本
    """
    def __init__(self, n_e_shared, n_e_task, e_dim, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False):
        super().__init__(1, e_dim, beta, LQ_stage, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(kernel_size=(self.unfold_size, self.unfold_size))
        self.z_length = e_dim // (self.unfold_size * self.unfold_size)
        self.z_stride = self.z_length // 2
        self.mlp_codebook = mlp_codebook

        # 定义共享码本和特定码本
        self.shared_embedding = nn.Embedding(n_e_shared, e_dim)
        self.task_specific_embedding = nn.Embedding(n_e_task, e_dim)

        # 初始化码本
        self.shared_embedding.weight.data.uniform_(-1.0 / n_e_shared, 1.0 / n_e_shared)
        self.task_specific_embedding.weight.data.uniform_(-1.0 / n_e_task, 1.0 / n_e_task)

        if self.mlp_codebook:
            self.codebook = nn.ModuleList([nn.Linear(e_dim, e_dim) for i in range(3)])

    def forward(self, z, gt_indices=None, current_iter=None):
        assert self.z_length == z.size(1)
        z_flattened = self.unfold(z).permute(0, 2, 1)
        z_flattened = z_flattened.reshape(-1, self.e_dim)

        # 使用共享码本进行量化
        shared_codebook = self.shared_embedding.weight
        z_q_shared, residual_shared, indices_shared = self.quantize(z_flattened, shared_codebook)
        z_q_shared = self.fold(z_q_shared, z.shape)

        # 使用私有码本进行量化
        task_specific_codebook = self.task_specific_embedding.weight
        z_q_task_specific, residual_task_specific, indices_task_specific = self.quantize(z_flattened, task_specific_codebook)
        z_q_task_specific = self.fold(z_q_task_specific, z.shape)

        # 合并两个量化结果
        z_q_combined = self.fuse(z_q_shared, z_q_task_specific)

        # 计算量化损失
        e_latent_loss_shared = torch.mean((z_q_shared.detach() - z) ** 2)
        q_latent_loss_shared = torch.mean((z_q_shared - z.detach()) ** 2)
        codebook_loss_shared = q_latent_loss_shared + e_latent_loss_shared * self.beta

        e_latent_loss_task_specific = torch.mean((z_q_task_specific.detach() - z) ** 2)
        q_latent_loss_task_specific = torch.mean((z_q_task_specific - z.detach()) ** 2)
        codebook_loss_task_specific = q_latent_loss_task_specific + e_latent_loss_task_specific * self.beta

        # 计算余弦相似度相关的损失
        cosine_loss_shared = self.compute_shared_cosine_loss(z_flattened, shared_codebook)
        cosine_loss_task_specific = self.compute_task_specific_cosine_loss(z_flattened, task_specific_codebook)

        # 总的量化损失
        codebook_loss = codebook_loss_shared + codebook_loss_task_specific
        cosine_loss = cosine_loss_shared + cosine_loss_task_specific

        # preserve gradients
        z_q_combined = z + (z_q_combined - z).detach()

        return z_q_combined, codebook_loss, cosine_loss, (indices_shared, indices_task_specific)
    
    def quantize(self, z_flattened, codebook):
        z_q, residual, indices = 0, z_flattened, []
        for i in range(self.depth):
            d = self.dist(residual, codebook)
            min_encoding_indices = torch.argmin(d, dim=1)
            delta = codebook[min_encoding_indices]
            z_q = z_q + delta
            residual = residual - delta
            indices.append(min_encoding_indices.clone())
        return z_q, residual, indices

    def fuse(self, z_q_shared, z_q_task_specific):
        # 这里可以选择不同的合并方式，比如加权平均、拼接等
        return z_q_shared * 0.7 + z_q_task_specific * 0.3
    
    def compute_shared_cosine_loss(self, z_flattened, shared_codebook):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(z_flattened.unsqueeze(1), shared_codebook.unsqueeze(0), dim=-1)
        # 确保每个特征与共享码本中每个向量的匹配度都大于一个阈值
        threshold = 0.5
        loss = F.relu(threshold - cos_sim).mean()
        return loss

    def compute_task_specific_cosine_loss(self, z_flattened, task_specific_codebook):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(z_flattened.unsqueeze(1), task_specific_codebook.unsqueeze(0), dim=-1)
        # 确保最匹配的向量与其他向量的匹配度差异较大
        max_sim, _ = cos_sim.max(dim=1, keepdim=True)
        margin = 0.55
        loss = F.relu(max_sim - cos_sim - margin).mean()
        return loss
    
    def fold(self, z_t, shape_z):
        b, c, h, w = shape_z
        z_t = z_t.view(b, -1, self.e_dim).permute(0, 2, 1)
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
        count_t = torch.ones(1, self.z_length, h, w).to(z_t.device)#.cuda()
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        z_q = z_q / count_t
        return z_q.view(shape_z)

class BlockBasedResidualVectorQuantizer(ResidualVectorQuantizer):
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, depth=6, unfold_size=2, mlp_codebook=False):
        super().__init__(n_e, e_dim, beta, LQ_stage, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(kernel_size=(self.unfold_size, self.unfold_size))
        self.z_length = e_dim // (self.unfold_size * self.unfold_size)
        self.z_stride = self.z_length // 2
        print(f"RQ Depth = {self.depth} ...")
        self.mlp_codebook = mlp_codebook
        if self.mlp_codebook:
            self.codebook = nn.ModuleList([nn.Linear(e_dim, e_dim) for i in range(3)])
        

    def forward(self, z, gt_indices=None, current_iter=None):
        assert self.z_length == z.size(1)
        z_flattened = self.unfold(z).permute(0, 2, 1)
        z_flattened = z_flattened.reshape(-1, self.e_dim)
        z_q = z_flattened
        if self.mlp_codebook:
            for block in self.codebook:
                z_q = block(z_q)
                z_q = F.relu(z_q)
            z_q = self.fold(z_q, z.shape)
            
            e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
            q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
            codebook_loss = q_latent_loss + e_latent_loss * self.beta
            
            z_q = z + (z_q - z).detach()
            return z_q, codebook_loss, None
            
        else:
            codebook = self.embedding.weight
            z_q, residual, indices = 0, z_flattened, []
            for i in range(self.depth):
                d = self.dist(residual, codebook)  # b x N
                min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
                delta = self.embedding(min_encoding_indices)

                """
                pred_one_hot = F.one_hot(min_encoding_indices, self.n_e).float()
                delta = torch.einsum("bm,md->bd", pred_one_hot, codebook)
                """

                z_q = z_q + delta
                residual = residual - delta
                # indices.append(min_encoding_indices.clone())
                indices.append(d)

            z_q = self.fold(z_q, z.shape)

            e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
            q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

            z_q = z + (z_q - z).detach()

            # indices = torch.stack(indices, dim=1).view(z.size(0), -1, self.depth)  # b x n x d
            return z_q, codebook_loss, indices

    def fold(self, z_t, shape_z):
        b, c, h, w = shape_z
        z_t = z_t.view(b, -1, self.e_dim).permute(0, 2, 1)
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
        count_t = torch.ones(1, self.z_length, h, w).to(z_t.device)#.cuda()
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        z_q = z_q / count_t
        return z_q.view(shape_z)
    
    # def fold_mlp(self, z_t, shape_z):
    #     b, c, h, w = shape_z
    #     z_t = z_t.permute(0, 2, 1)
    #     fold = nn.Fold(output_size=(h, w), kernel_size=(self.unfold_size, self.unfold_size))
    #     count_t = torch.ones(1, self.z_length, h, w).to(z_t.device)#.cuda()
    #     count_t = self.unfold(count_t)
    #     count_t = fold(count_t)
    #     z_q = fold(z_t)
    #     z_q = z_q / count_t
    #     return z_q.view(shape_z)

