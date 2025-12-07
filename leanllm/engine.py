import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .models import GPT2, load_gpt2_from_hf


class LLM:
    """Simple vLLM-like API for inference"""

    def __init__(self, model_id: str, device: str = None, dtype: str = "auto"):
        self.model_id = model_id
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = self._parse_dtype(dtype)

        # Load config and tokenizer from HF
        hf_cfg = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build and load model
        if "gpt2" in model_id.lower():
            self.model = self._load_gpt2(hf_cfg)
        else:
            raise ValueError(f"Model {model_id} not supported yet")

        self.model.eval()

    def _parse_dtype(self, dtype_str):
        if dtype_str == "auto":
            return torch.float16 if self.device.type == "cuda" else torch.float32
        elif dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")

    def _load_gpt2(self, hf_cfg):
        """Load GPT-2 model from HF"""
        model = GPT2(
            vocab_size=hf_cfg.vocab_size,
            hidden_size=hf_cfg.hidden_size,
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            max_position_embeddings=hf_cfg.max_position_embeddings,
        )

        # Load weights from HF
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        load_gpt2_from_hf(model, hf_model.state_dict())
        del hf_model  # Free memory
        
        # Move to target device/dtype after loading weights
        model = model.to(device=self.device, dtype=self.dtype)
        return model

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Generate text from prompt (greedy decoding for v0)"""
        with torch.no_grad():
            ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            for _ in range(max_new_tokens):
                pos = torch.arange(ids.size(1), device=self.device)
                logits = self.model(ids, pos)[0, -1]
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
                if next_id.item() == self.tokenizer.eos_token_id:
                    break
            return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
