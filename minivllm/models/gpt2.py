"""The GPT-2 124M trained in the nanoGPT project: pre-LN, LayerNorm, GELU, biases, RoPE, tied embeddings.

Numerics follow nanoGPT's generation under bf16 autocast: the token embedding, residual stream and
LayerNorms stay in fp32; linear layers, GELU and attention run in bf16; RoPE rotates in fp32.
"""

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.attention import AttnMetadata, store_kv


@dataclass
class GPT2Config:
    vocab_size: int  # padded to 50304; ids from 50257 up are not real tokens
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    rope_base: float = 10000.0
    real_vocab_size: int = 50257


def rope_cache(head_dim, base, max_pos):
    # computed on the CPU like nanoGPT, so the tables match it exactly
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x [T, H, D]; cos, sin [T, 1, D/2] fp32; dimension i is paired with i + D/2
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class GPT2Attention(nn.Module):
    def __init__(self, cfg: GPT2Config, attn_backend):
        super().__init__()
        self.num_heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn = attn_backend
        self.k_cache = None  # [num_blocks, block_size, H, D], assigned by the model runner
        self.v_cache = None

    def forward(self, x, cos, sin, meta: AttnMetadata):
        T = x.shape[0]
        q, k, v = self.c_attn(x).view(T, 3, self.num_heads, self.head_dim).unbind(1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        store_kv(k, v, self.k_cache, self.v_cache, meta.slot_mapping)
        o = self.attn(q, k, v, self.k_cache, self.v_cache, meta)
        return self.c_proj(o.reshape(T, -1))


class GPT2MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2Config, attn_backend):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = GPT2Attention(cfg, attn_backend)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = GPT2MLP(cfg)

    def forward(self, x, cos, sin, meta):
        # x is fp32; the bf16 branch outputs are promoted back to fp32 when added
        dtype = self.mlp.c_fc.weight.dtype
        x = x + self.attn(self.ln_1(x).to(dtype), cos, sin, meta)
        return x + self.mlp(self.ln_2(x).to(dtype))


class GPT2ForCausalLM(nn.Module):
    def __init__(self, cfg: GPT2Config, attn_backend):
        super().__init__()
        self.cfg = cfg
        self.max_positions = cfg.block_size
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList(GPT2Block(cfg, attn_backend) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head_weight = None  # bf16 copy of wte.weight
        self.cos = self.sin = None
        self.num_kv_heads, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.attn_layers = [block.attn for block in self.layers]

    def forward(self, input_ids, positions, meta: AttnMetadata):
        """input_ids, positions: [T]. Returns final hidden states [T, n_embd] in fp32."""
        x = self.wte(input_ids)
        cos, sin = self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)
        for block in self.layers:
            x = block(x, cos, sin, meta)
        return self.ln_f(x)

    def compute_logits(self, hidden):
        return F.linear(hidden.to(self.lm_head_weight.dtype), self.lm_head_weight)


def load_gpt2(ckpt_path, attn_backend, device="cuda", dtype=torch.bfloat16):
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    args = ckpt["model_args"]
    assert args["pos_emb"] == "rope" and args["norm"] == "layernorm" and args["mlp"] == "gelu" and args["bias"]
    cfg = GPT2Config(args["vocab_size"], args["n_layer"], args["n_head"], args["n_embd"], args["block_size"])
    model = GPT2ForCausalLM(cfg, attn_backend)
    sd = {re.sub(r"^h\.", "layers.", k.removeprefix("transformer.")): v for k, v in ckpt["model"].items()}
    sd.pop("lm_head.weight")
    model.load_state_dict(sd, strict=True)
    # linear layers in bf16; embedding and LayerNorms stay fp32
    for block in model.layers:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.to(device)
    model.lm_head_weight = model.wte.weight.detach().to(dtype)
    cos, sin = rope_cache(model.head_dim, cfg.rope_base, cfg.block_size)
    model.cos, model.sin = cos.to(device), sin.to(device)
    return model.eval().requires_grad_(False)
