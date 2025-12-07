import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """RoPE - Rotary Position Embedding"""
    def __init__(self, dim: int, max_position: int, base: float = 10000.0):
        super().__init__()
        pos = torch.arange(max_position, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        cos = self.cos[pos][:, None, None, : x.size(-1)]
        sin = self.sin[pos][:, None, None, : x.size(-1)]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot_x = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + rot_x * sin


class Attention(nn.Module):
    """Multi-head Self-Attention with optional RoPE"""
    def __init__(self, hidden_size: int, num_heads: int, max_position: int, use_rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, max_position, rope_base) if use_rope else None

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        bsz, seq, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        out = attn.transpose(1, 2).reshape(bsz, seq, -1)
        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward network with configurable activation"""
    def __init__(self, hidden_size: int, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=True)
        
        if activation == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
