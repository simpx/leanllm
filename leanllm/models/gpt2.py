import torch
import torch.nn as nn

from .base import Attention, MLP


class GPT2Block(nn.Module):
    """Transformer block for GPT-2: PreNorm + Attn/MLP + Residual"""
    def __init__(self, hidden_size: int, num_heads: int, max_position: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, max_position, use_rope=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x, pos):
        x = x + self.attn(self.ln1(x), pos)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 model: Embedding -> L×Block -> Norm -> LM Head"""
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, max_position_embeddings: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        self.blocks = nn.ModuleList([GPT2Block(hidden_size, num_heads, max_position_embeddings) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, pos: torch.Tensor):
        """Forward pass: (batch, seq) -> (batch, seq, vocab)"""
        x = self.embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x, pos)
        x = self.ln_f(x)
        return self.lm_head(x)


def load_gpt2_from_hf(model: GPT2, state_dict: dict):
    """Load HF GPT-2 weights into our GPT2 model
    
    HF GPT-2 uses Conv1D with shape (in, out), PyTorch Linear uses (out, in)
    Need to transpose all linear weights
    """
    sd = state_dict
    with torch.no_grad():
        model.embed.weight.copy_(sd["transformer.wte.weight"])
        model.pos_embed.weight.copy_(sd["transformer.wpe.weight"])
        for i, blk in enumerate(model.blocks):
            prefix = f"transformer.h.{i}."
            # c_attn.weight: (768, 2304) -> need (2304, 768) for Linear
            blk.attn.qkv.weight.copy_(sd[prefix + "attn.c_attn.weight"].T)
            blk.attn.qkv.bias.copy_(sd[prefix + "attn.c_attn.bias"])
            
            blk.attn.proj.weight.copy_(sd[prefix + "attn.c_proj.weight"].T)
            blk.attn.proj.bias.copy_(sd[prefix + "attn.c_proj.bias"])
            
            blk.ln1.weight.copy_(sd[prefix + "ln_1.weight"])
            blk.ln1.bias.copy_(sd[prefix + "ln_1.bias"])
            
            blk.mlp.fc1.weight.copy_(sd[prefix + "mlp.c_fc.weight"].T)
            blk.mlp.fc1.bias.copy_(sd[prefix + "mlp.c_fc.bias"])
            
            blk.mlp.fc2.weight.copy_(sd[prefix + "mlp.c_proj.weight"].T)
            blk.mlp.fc2.bias.copy_(sd[prefix + "mlp.c_proj.bias"])
            
            blk.ln2.weight.copy_(sd[prefix + "ln_2.weight"])
            blk.ln2.bias.copy_(sd[prefix + "ln_2.bias"])
            
        model.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(sd["transformer.ln_f.bias"])
    return model
