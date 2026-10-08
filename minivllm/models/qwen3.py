"""Qwen3 dense model on the flat token layout. Numerics follow transformers' Qwen3 implementation.

With fused = True (set by the model runner), forward runs the elementwise parts in the fused kernels of
minivllm/fused.py; the plain PyTorch path stays for comparisons.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from minivllm import fused
from minivllm.attention import AttnMetadata, store_kv


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cache(head_dim, theta, max_pos, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.arange(max_pos, dtype=torch.float32, device=device)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x [T, H, D]; cos, sin [T, 1, D]; dimension i is paired with i + D/2
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, attn_backend):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.attn = attn_backend
        self.k_cache = None  # [num_blocks, block_size, Hkv, D], assigned by the model runner
        self.v_cache = None

    def forward(self, x, cos, sin, meta: AttnMetadata):
        T = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = apply_rope(self.q_norm(q.view(T, self.num_heads, self.head_dim)), cos, sin)
        k = apply_rope(self.k_norm(k.view(T, self.num_kv_heads, self.head_dim)), cos, sin)
        v = v.view(T, self.num_kv_heads, self.head_dim)
        store_kv(k, v, self.k_cache, self.v_cache, meta.slot_mapping)
        o = self.attn(q, k, v, self.k_cache, self.v_cache, meta)
        return self.o_proj(o.reshape(T, self.q_size))

    def forward_fused(self, x, positions, cos, sin, meta: AttnMetadata):
        """forward with the q/k norms, RoPE and the K/V write in one kernel; cos and sin are the whole bf16 tables."""
        need_kv = meta.is_prefill and getattr(self.attn, "reads_new_kv", True)
        q, k, v = fused.qk_norm_rope_store(self.qkv_proj(x), self.num_heads, self.num_kv_heads, self.q_norm, self.k_norm,
                                           positions, cos, sin, self.k_cache, self.v_cache, meta.slot_mapping, need_kv)
        o = self.attn(q, k, v, self.k_cache, self.v_cache, meta)
        return self.o_proj(o.reshape(x.shape[0], self.q_size))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def forward_fused(self, x):
        return self.down_proj(fused.silu_mul(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, attn_backend):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg, attn_backend)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x, cos, sin, meta):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, attn_backend, max_positions=4096):
        super().__init__()
        assert cfg.tie_word_embeddings
        self.cfg = cfg
        self.max_positions = max_positions
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg, attn_backend) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.cos = self.sin = None  # [max_positions, head_dim] fp32, built by init_rope()
        self.rope_bf16 = None  # the same tables in bf16, as the reference forward rounds them
        self.num_kv_heads, self.head_dim = cfg.num_key_value_heads, cfg.head_dim
        self.attn_layers = [layer.self_attn for layer in self.layers]
        self.fused = False

    def init_rope(self, device):
        self.cos, self.sin = rope_cache(self.cfg.head_dim, self.cfg.rope_theta, self.max_positions, device)
        self.rope_bf16 = (self.cos.to(torch.bfloat16), self.sin.to(torch.bfloat16))

    def forward(self, input_ids, positions, meta: AttnMetadata):
        """input_ids, positions: [T]. Returns final hidden states [T, hidden]."""
        x = self.embed_tokens(input_ids)
        if self.fused:
            return self.forward_fused(x, positions, meta)
        cos = self.cos[positions].to(x.dtype).unsqueeze(1)
        sin = self.sin[positions].to(x.dtype).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, cos, sin, meta)
        return self.norm(x)

    def forward_fused(self, x, positions, meta):
        # each residual add is done by the kernel of the RMSNorm after it, on x in place
        h = fused.add_rms_norm(x, None, self.layers[0].input_layernorm)
        for i, layer in enumerate(self.layers):
            a = layer.self_attn.forward_fused(h, positions, *self.rope_bf16, meta)
            m = layer.mlp.forward_fused(fused.add_rms_norm(x, a, layer.post_attention_layernorm))
            h = fused.add_rms_norm(x, m, self.layers[i + 1].input_layernorm if i + 1 < len(self.layers) else self.norm)
        return h

    def compute_logits(self, hidden):
        return F.linear(hidden, self.embed_tokens.weight)


def load_qwen3(path, attn_backend, device="cuda", dtype=torch.bfloat16, max_positions=4096):
    cfg = Qwen3Config.from_json(os.path.join(path, "config.json"))
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg, attn_backend, max_positions)
    with safe_open(os.path.join(path, "model.safetensors"), "pt", device=str(device)) as f:
        def get(name):
            return f.get_tensor(name).to(dtype)

        sd = {"embed_tokens.weight": get("model.embed_tokens.weight"), "norm.weight": get("model.norm.weight")}
        for i in range(cfg.num_hidden_layers):
            src, dst = f"model.layers.{i}.", f"layers.{i}."
            a, m = src + "self_attn.", src + "mlp."
            sd[dst + "self_attn.qkv_proj.weight"] = torch.cat([get(a + p + "_proj.weight") for p in "qkv"])
            sd[dst + "self_attn.o_proj.weight"] = get(a + "o_proj.weight")
            sd[dst + "self_attn.q_norm.weight"] = get(a + "q_norm.weight")
            sd[dst + "self_attn.k_norm.weight"] = get(a + "k_norm.weight")
            sd[dst + "mlp.gate_up_proj.weight"] = torch.cat([get(m + "gate_proj.weight"), get(m + "up_proj.weight")])
            sd[dst + "mlp.down_proj.weight"] = get(m + "down_proj.weight")
            sd[dst + "input_layernorm.weight"] = get(src + "input_layernorm.weight")
            sd[dst + "post_attention_layernorm.weight"] = get(src + "post_attention_layernorm.weight")
    model.load_state_dict(sd, assign=True, strict=True)
    model.init_rope(device)
    return model.eval().requires_grad_(False)
