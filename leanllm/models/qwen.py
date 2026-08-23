import torch
import torch.nn as nn

from .base import RMSNorm, RotaryEmbedding


class QwenAttention(nn.Module):
    """Qwen3 attention with separate q/k/v and MQA (num_kv_heads)"""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int, rope_theta: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_kv_heads  # Use kv_heads to compute head_dim (important for Qwen3!)
        self.rope_base = rope_theta

        # Qwen3: q gets all heads, k/v are grouped for MQA
        q_proj_out = num_heads * self.head_dim  # 16 * 128 = 2048
        kv_proj_out = num_kv_heads * self.head_dim  # 8 * 128 = 1024

        self.q_proj = nn.Linear(hidden_size, q_proj_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_proj_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=False)
        self.o_proj = nn.Linear(q_proj_out, hidden_size, bias=False)
        # rope is applied on the projected head_dim derived from q/k shapes
        self.rope = RotaryEmbedding(self.head_dim, max_position, base=rope_theta)

    def _reshape(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # Infer head_dim from current tensor shape to avoid mismatch when head_dim != hidden_size // num_heads
        bsz, seq, dim = x.shape
        head_dim = dim // num_heads
        return x.view(bsz, seq, num_heads, head_dim), head_dim

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        bsz, seq, _ = x.shape
        q, head_dim_q = self._reshape(self.q_proj(x), self.num_heads)
        k, head_dim_k = self._reshape(self.k_proj(x), self.num_kv_heads)
        v, _ = self._reshape(self.v_proj(x), self.num_kv_heads)

        # Update rope dim to match actual head_dim if it differs
        if self.rope.cos.size(-1) != head_dim_q:
            # re-init rope to correct head_dim
            self.rope = RotaryEmbedding(head_dim_q, self.rope.cos.size(0), base=self.rope_base).to(q.device)

        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        out = attn.transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    """SwiGLU activation: gate * gate_proj + up_proj -> down_proj"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        inner_dim = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, inner_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    """Qwen3 Transformer Block: RMSNorm + RoPE Attention + SwiGLU MLP"""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int, rope_theta: float, intermediate_size: int):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attn = QwenAttention(hidden_size, num_heads, num_kv_heads, max_position, rope_theta)
        self.ln2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, pos):
        x = x + self.attn(self.ln1(x), pos)
        x = x + self.mlp(self.ln2(x))
        return x


class Qwen3(nn.Module):
    """Qwen3 model: Embedding -> L×Block -> RMSNorm -> LM Head"""
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, num_kv_heads: int,
                 max_position_embeddings: int, rope_theta: float, intermediate_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            QwenBlock(hidden_size, num_heads, num_kv_heads, max_position_embeddings, rope_theta, intermediate_size)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, pos: torch.Tensor):
        """Forward pass: (batch, seq) -> (batch, seq, vocab)"""
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, pos)
        x = self.ln_f(x)
        return self.lm_head(x)


def load_qwen_from_hf(model: Qwen3, state_dict: dict):
    """Load HF Qwen weights into our Qwen3 model"""
    sd = state_dict
    with torch.no_grad():
        model.embed.weight.copy_(sd["model.embed_tokens.weight"])
        
        for i, blk in enumerate(model.blocks):
            prefix = f"model.layers.{i}."
            # HF weight shapes: (out_features, in_features)
            # q_proj: (2048, 1024), k_proj: (1024, 1024), v_proj: (1024, 1024), o_proj: (1024, 2048)
            blk.attn.q_proj.weight.copy_(sd[prefix + "self_attn.q_proj.weight"])  # (2048, 1024)
            blk.attn.k_proj.weight.copy_(sd[prefix + "self_attn.k_proj.weight"])  # (1024, 1024)
            blk.attn.v_proj.weight.copy_(sd[prefix + "self_attn.v_proj.weight"])  # (1024, 1024)
            blk.attn.o_proj.weight.copy_(sd[prefix + "self_attn.o_proj.weight"])  # (1024, 2048) -> weight is (1024, 2048)
            
            blk.ln1.weight.copy_(sd[prefix + "input_layernorm.weight"])
            blk.ln2.weight.copy_(sd[prefix + "post_attention_layernorm.weight"])
            
            blk.mlp.gate_proj.weight.copy_(sd[prefix + "mlp.gate_proj.weight"])
            blk.mlp.up_proj.weight.copy_(sd[prefix + "mlp.up_proj.weight"])
            blk.mlp.down_proj.weight.copy_(sd[prefix + "mlp.down_proj.weight"])
        
        model.ln_f.weight.copy_(sd["model.norm.weight"])
    
    return model
