import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, mlp_ratio, num_experts, noisy_gating=True, use_experts=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = use_experts
        # instantiate experts
        self.experts = nn.ModuleList([Mlp(input_size, h_feat=int(input_size*mlp_ratio), out_feat=output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.randn(2*input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(2*input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts) 

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, prompt):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """ 
        
        # import pdb 
        # pdb.set_trace() 
        
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x) 
        
        
        x = rearrange(x, 'b c h w -> (b h w) c') 
        prompt = rearrange(prompt, 'b c h w -> (b h w) c') 
        
        x_gating = torch.cat((x, prompt), dim=1) #[B, 2C, H, W]
        
        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs) 
        
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        return y, loss
    
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Spatial_Routing(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(Spatial_Routing, self).__init__() 
        
        self.fc = nn.Linear(atom_dim, dim) 
        self.moe = MoE(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=4, noisy_gating=True, use_experts=2) 

    def forward(self, x, prompt): 
        d = self.fc(prompt) 
        out, loss = self.moe(x, d) 
        return out + x, loss




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    




##########################################################################
#########################  生成prompt   ##################################
##########################################################################
##生成PAN_prompt
class PAN_PromptBlock(nn.Module): ## prompt_size = 64 or 512
    def __init__(self, prompt_dim = 1, prompt_len=16, prompt_size=64, lin_dim=1, learnable_input_prompt = True):
        super(PAN_PromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.randn(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.dim = prompt_dim
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1)) ## B,C
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1) ## B, prompt_len

        # prompt_param = nn.Parameter(torch.rand(1, 16, self.dim, H, W), requires_grad=False).to(x.device)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1).to(x.device) ## B, prompt_len, prompt_dim, size, size 
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param  ## B, prompt_len, prompt_dim, size, size
        prompt = torch.sum(prompt, dim=1) ## B, prompt_dim, size, size
        prompt = self.dec_conv3x3(prompt)
        return prompt

class PAN_PromptBlock_multiscale(nn.Module): ## prompt_size = 64 or 512
    def __init__(self, prompt_dim = 1, prompt_len=16, prompt_size=64, lin_dim=1, learnable_input_prompt = True):
        super(PAN_PromptBlock_multiscale, self).__init__()
        self.prompt_param = nn.Parameter(torch.randn(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.dim = prompt_dim
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.down_conv1 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_conv2 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_conv3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_list = [self.down_conv1, self.down_conv2, self.down_conv3]
    def forward(self, x):
        B, C, H, W = x.shape
        prompt_list = []
        emb = x.mean(dim=(-2, -1)) ## B,C
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1) ## B, prompt_len

        # prompt_param = nn.Parameter(torch.rand(1, 16, self.dim, H, W), requires_grad=False).to(x.device)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1).to(x.device) ## B, prompt_len, prompt_dim, size, size 
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param  ## B, prompt_len, prompt_dim, size, size
        prompt = torch.sum(prompt, dim=1) ## B, prompt_dim, size, size
        prompt = self.dec_conv3x3(prompt)
        prompt_list.append(prompt)
        for i in range(3):
            prompt = self.conv_list[i](prompt)
            prompt_list.append(prompt)
        return prompt_list

class PAN_PromptBlock_multiscale_interpolate(nn.Module): ## prompt_size = 64 or 512
    def __init__(self, prompt_dim = 1, prompt_len=16, prompt_size=64, lin_dim=1, learnable_input_prompt = True):
        super(PAN_PromptBlock_multiscale_interpolate, self).__init__()
        self.prompt_param = nn.Parameter(torch.randn(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.dim = prompt_dim
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        prompt_list = []
        emb = x.mean(dim=(-2, -1)) ## B,C
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1) ## B, prompt_len

        # prompt_param = nn.Parameter(torch.rand(1, 16, self.dim, H, W), requires_grad=False).to(x.device)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1).to(x.device) ## B, prompt_len, prompt_dim, size, size 
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param  ## B, prompt_len, prompt_dim, size, size
        prompt = torch.sum(prompt, dim=1) ## B, prompt_dim, size, size
        prompt = self.dec_conv3x3(prompt)
        prompt_list.append(prompt)
        for i in range(3):
            prompt = F.interpolate(prompt, scale_factor=0.5, mode='bilinear')
            prompt_list.append(prompt)
        return prompt_list

## 生成MS_prompt
class RIN(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(RIN, self).__init__()
        
        # Condtion network 
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1)) 
        
        self.lastOut = nn.Linear(32, atom_num) 
        self.act = nn.GELU()
        
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1) ##展平，（B，C，H，W）->（B，C*H*W）B, 32
        out = self.lastOut(out) # B, 16
        logits = F.softmax(out, -1) # B, 16
        out = logits @ self.dictionary # B, 256
        out = self.act(out)
        return out 
    
class RIN(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(RIN, self).__init__()
        
        # Condtion network 
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1)) 
        
        self.lastOut = nn.Linear(32, atom_num) 
        self.act = nn.GELU()
        
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1) ##展平，（B，C，H，W）->（B，C*H*W）B, 32
        out = self.lastOut(out) # B, 16
        logits = F.softmax(out, -1) # B, 16
        out = logits @ self.dictionary # B, 256
        out = self.act(out)
        return out 
    



##########################################################################
#########################  加入prompt   ##################################
##########################################################################
class Channel_Routing(nn.Module):

    def __init__(self, atom_dim, dim):
        super(Channel_Routing, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True) 

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        out = x * self.gamma + self.beta  
        out = out * gating_factors 
             
        return x + out
        # return out
    

class Channel_Routing_notres(nn.Module):
    def __init__(self, atom_dim, dim):
        super(Channel_Routing_notres, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True) 

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        out = x * self.gamma + self.beta  
        out = out * gating_factors    
        return out



class Spatial_prompt_in(nn.Module):
    def __init__(self):
        super(Spatial_prompt_in, self).__init__()
        # self.down_conv = nn.ModuleList()
        # for _ in range(down_num):
        #     self.down_conv.append(nn.Sequential(   
        #         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(dim, dim*2, kernel_size=1)
        #     ))
        #     dim = dim*2
        # self.change_channel = nn.Conv2d()
    def forward(self, x, prompt):
        B,C,H,W = x.shape
        prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
        prompt = torch.sigmoid(prompt)  
        return x + x*prompt

class Spatial_prompt_in_notres(nn.Module):
    def __init__(self):
        super(Spatial_prompt_in_notres, self).__init__()
    def forward(self, x, prompt):
        B,C,H,W = x.shape
        prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
        prompt = torch.sigmoid(prompt)  
        return x*prompt
    
class Spatial_prompt_in_notres_add(nn.Module):
    def __init__(self):
        super(Spatial_prompt_in_notres_add, self).__init__()
    def forward(self, x, prompt):
        B,C,H,W = x.shape
        prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
        prompt = torch.sigmoid(prompt)  
        return x+prompt

class SCgate(nn.Module):
    """
    通道prompt与X_in作积，再与光谱prompt相加(仿射变换),再做残差
    """
    def __init__(self, dim, channel_prompt_dim, ratio=4):
        super(SCgate, self).__init__()
        self.fc_channel_in = nn.Linear(channel_prompt_dim, int(dim/ratio))
        self.fc_channel_out = nn.Linear(int(dim/ratio), dim)
        self.fc_spatial_in = nn.Conv2d(in_channels=1, out_channels=int(dim/ratio), kernel_size=3, stride=1, padding=1)
        self.fc_spatial_out = nn.Conv2d(in_channels=int(dim/ratio), out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward(self, x, spatial_prompt, channel_prompt):
        
        channel_prompt = F.gelu(self.fc_channel_in(channel_prompt)) 
        channel_prompt = torch.sigmoid(self.fc_channel_out(channel_prompt))
        channel_prompt = channel_prompt.unsqueeze(2).unsqueeze(3)   # B，dim，1，1


        spatial_prompt = self.fc_spatial_in(spatial_prompt)
        spatial_prompt = self.fc_spatial_out(spatial_prompt)
        # print("x.shape:", x.shape)
        x = x * channel_prompt + spatial_prompt + x
        return x

## 去掉中间的channel concat到spatial的操作，最后结果concat起来
class SCgate_concat_no_cross(nn.Module):
    def __init__(self, dim, channel_prompt_dim, ratio=4):
        super(SCgate_concat_no_cross, self).__init__()
        self.fc_channel_in = nn.Linear(channel_prompt_dim, int(dim/ratio))
        self.fc_channel_out = nn.Linear(int(dim/ratio), dim)
        self.fc_spatial_in = nn.Linear(1, int(dim/ratio))
        self.fc_spatial_out = nn.Linear(int(dim/ratio), 1)
        self.reduce_chan = nn.Conv2d(int(dim*2**1), dim, kernel_size=1)
    def forward(self, x, spatial_prompt, channel_prompt):
        
        channel_prompt = F.gelu(self.fc_channel_in(channel_prompt)) 
        channel_prompt = channel_prompt.unsqueeze(1).unsqueeze(1)   # B，1，1，dim/ratio
        spatial_prompt = torch.permute(spatial_prompt, (0,2,3,1))   # B，H，W，1
        spatial_prompt = F.gelu(self.fc_spatial_in(spatial_prompt)) # B，H，W，dim/ratio
        B, H, W, C = spatial_prompt.shape
        channel_prompt_expand = channel_prompt.expand(B, H, W, C)   # B, H, W,dim/ratio
        spatial_prompt = torch.sigmoid(self.fc_spatial_out(spatial_prompt)) # B, H, W, 1
        channel_prompt = torch.sigmoid(self.fc_channel_out(channel_prompt)) # B, 1, 1, dim
        gate = channel_prompt * spatial_prompt # B ,H ,W ,C
        gate = torch.permute(gate, (0,3,1,2)) # B, C, H, W
        # return x * gate
        return self.reduce_chan(torch.cat((x,gate),dim=1))
    
class Affine_prompt(nn.Module):
    def __init__(self, dim, channel_prompt_dim):
        super(Affine_prompt,self).__init__()
        self.fc_channel = nn.Linear(channel_prompt_dim, dim)
        self.fc_spatial = nn.Linear(1, dim)
    def forward(self, x, spatial_prompt, channel_prompt):
        B, C, H, W = x.shape
        alpha = F.gelu(self.fc_channel(channel_prompt)).unsqueeze(1).unsqueeze(1) # B, 1, 1, C
        spatial_prompt = torch.permute(spatial_prompt, (0,2,3,1))                 # B, H, W, 1
        beta = F.gelu(self.fc_spatial(spatial_prompt)).permute(0,3,1,2)           # B, C, H, W
        # alpha = alpha.expand(B, H, W, C).permute(0,3,1,2)
        alpha = alpha.permute(0,3,1,2)

        return x + (x * alpha + beta)
    
if __name__ == "__main__":
    x = torch.randn(4, 16, 32, 32)
    # spatial_prompt = torch.randn(4, 1, 32, 32)
    channel_prompt = torch.randn(4, 16)
    channel_prompt = channel_prompt.unsqueeze(2).unsqueeze(3)
    print(channel_prompt.shape)
    output = x * channel_prompt
    print(output.shape)
    # aff = Affine_prompt(16, 256)
    # x = aff(x,spatial_prompt, channel_prompt)
    # print(x.shape)

        