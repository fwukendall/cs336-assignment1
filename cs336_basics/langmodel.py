import torch
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
import numpy as np


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: Int,
        num_heads: Int,
        max_seq_len: Int = 0,
        theta: Float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype

        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.d_k = d_k
        self.proj_qkv = Linear(d_model, 3 * d_model, device, dtype)
        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        self.proj_out = Linear(d_model, d_model, device, dtype)
    
    def forward(
        self,
        x: Float[torch.Tensor, " ... seq_len d_in"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ):
        assert x.shape[-1] == self.d_model
        qkv = self.proj_qkv(x)
        num_heads = self.num_heads
        Q_all, K_all, V_all = qkv.split(self.d_model, dim=-1)
        Q_mh = torch.stack(Q_all.split(self.d_k, dim=-1))
        K_mh = torch.stack(K_all.split(self.d_k, dim=-1))
        V_mh = torch.stack(V_all.split(self.d_k, dim=-1))
        if self.rope:
            if token_positions is None:
                xpos = torch.arange(0, x.shape[-2]).expand(x.shape[:-1])
            else:
                xpos = token_positions
            Q_mh = self.rope(Q_mh, xpos)
            K_mh = self.rope(K_mh, xpos)
        mask_T = torch.triu(torch.ones(*x.shape[:-1], x.shape[-2])) == 1
        mask_mh = rearrange(mask_T.broadcast_to((num_heads, *mask_T.shape)), '... a b -> ... b a')
        raw_attention_out = scaled_dot_product_attention(Q_mh, K_mh, V_mh, mask_mh)
        attention_out = rearrange(
            raw_attention_out,
            'heads ... d_in d_k -> ... d_in (heads d_k)',
        )
        return self.proj_out(attention_out)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    assert K.shape[-2] == V.shape[-2]
    QK = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    denom = np.sqrt(Q.shape[-1])
    if mask is not None:
        QK_masked = (QK / denom).masked_fill(~mask, -torch.inf)
    sm = softmax(QK_masked, -1)
    return einsum(sm, V, '... queries values, ... values d_v -> ... queries d_v')


def softmax(
    x: Float,
    i: Int,
) -> torch.Tensor:
    '''
    Apply softmax() to the i-th dimension of x
    '''
    x_max, _ = x.max(dim=i, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=i, keepdim=True)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: Float,
        d_k: Int,
        max_seq_len: Int,
        device: torch.device | None = None,
    ):
        '''
        Pre-compute a rotation matrix, register buffer
        '''
        super().__init__()
        self.d_k = d_k
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.theta = theta
        self._update_cache(max_seq_len)
            
    def forward(
        self,
        x: Float[torch.Tensor, '... seq_len d_k'],
        token_positions: Int[torch.Tensor, '... seq_len'],
    ) -> torch.Tensor:
        odd_x = x[..., ::2]
        even_x = x[..., 1::2]
        if int(token_positions.max()) >= self.max_seq_len:
            rounded_seq_len = 2 ** int(np.ceil(np.log2(float(token_positions.max()+1))))
            self._update_cache(rounded_seq_len)
        cosines = self.cosines[token_positions]
        sines = self.sines[token_positions]
        odd_out = cosines * odd_x - sines * even_x
        even_out = sines * odd_x + cosines * even_x
        return torch.stack([odd_out, even_out], -1).flatten(-2)
    
    def _update_cache(
        self,
        max_seq_len: Int,
    ):
        d_k = self.d_k
        theta = self.theta
        self.max_seq_len = max_seq_len
        denom_exponents = torch.arange(0, d_k, 2) / d_k
        denom_invs = theta ** (-denom_exponents)
        positions = torch.arange(0, max_seq_len)
        angles = einsum(positions, denom_invs, 'd_seq, d_k -> d_seq d_k')
        cosines = torch.cos(angles)
        sines = torch.sin(angles)
        cosines = cosines.to(self.device)
        sines = sines.to(self.device)
        self.register_buffer('cosines', cosines, persistent=True)
        self.register_buffer('sines', sines, persistent=True)
        return


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: Int,
        d_ff: Int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        '''
        W2.dot(SiLU(W1.dot(x)) * W3.dot(x))
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    
    def forward(self, x: Float[torch.Tensor, '... d_model']) -> torch.Tensor:
        w1_out = self.w1(x)
        return self.w2(sigmoid(w1_out) * w1_out * self.w3(x))



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: Int,
        eps: Float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(
            size=(d_model,),
            dtype=dtype,
            device=device,
        ), requires_grad=True)
    
    def forward(self, x: Float[torch.Tensor, '... d_model']) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        if self.dtype != torch.float32:
            g = self.g.to(torch.float32)
        else:
            g = self.g
        denom = torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + self.eps)
        out = x * denom * g
        return out.to(in_dtype)



class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: Int,
        embedding_dim: Int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_vocab = num_embeddings
        self.d_model = embedding_dim
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    self.d_vocab,
                    self.d_model,
                    dtype=dtype,
                    device=device,
                ),
                mean=0,
                std=1,
                a=-3,
                b=3,
            ),
            requires_grad=True,
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embeddings[token_ids]
    

class Linear(nn.Module):
    '''
    Apply the linear transformation to the input.
    Make sure to:
    subclass nn.Module
    call the superclass constructor
    construct and store your parameter as W (not W⊤) for memory ordering reasons, putting it in
    an nn.Parameter
For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to
    initialize the weights.
    To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter
    should load the given weights into your Linear module. You can use Module.load_state_dict for
    this purpose. Then, run uv run pytest -k test_linear.
    '''
    
    def __init__(
        self,
        in_features: Int,
        out_features: Int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        '''
        linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype
        sigma = np.sqrt(2 / (self.d_in + self.d_out))
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_out, self.d_in, dtype=dtype, device=device),
                mean=0,
                std=sigma,
                a=-3*sigma,
                b=3*sigma,
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return einsum(self.W, x, 'd_out d_in, ... d_in -> ... d_out')