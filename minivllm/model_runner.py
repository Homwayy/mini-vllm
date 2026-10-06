"""Loads the model, owns the KV pool, and turns a scheduled batch into one forward pass plus sampling."""

import os

import numpy as np
import torch

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.kv_manager import cdiv
from minivllm.models.gpt2 import load_gpt2
from minivllm.models.qwen3 import load_qwen3
from minivllm.sampler import Sampler
from minivllm.sequence import Sequence

MODELS = {
    "qwen3": dict(path=os.path.join("checkpoints", "Qwen3-0.6B"), eos=(151643, 151645), vocab_limit=None),
    "gpt2": dict(path=r"A:\GitHub project\nanoGPT\out\final\ckpt.pt", eos=(50256,), vocab_limit=50257),
}


class ModelRunner:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        info = MODELS[cfg.model]
        if cfg.attention == "flex":
            attn = FlexAttention(cfg.block_size, cfg.max_num_batched_tokens, cfg.max_num_seqs, device)
        elif cfg.attention == "gather":
            attn = GatherAttention()
        else:
            raise ValueError(f"unknown attention backend {cfg.attention}")
        path = cfg.model_path or info["path"]
        if cfg.model == "qwen3":
            self.model = load_qwen3(path, attn, device, max_positions=cfg.max_model_len)
        else:
            self.model = load_gpt2(path, attn, device)
            assert cfg.max_model_len <= self.model.max_positions
        self.sampler = Sampler(info["vocab_limit"])
        self.eos_token_ids = set(info["eos"])
        self.max_blocks_per_seq = cdiv(cfg.max_model_len, cfg.block_size)
        shape = (cfg.num_kv_blocks, cfg.block_size, self.model.num_kv_heads, self.model.head_dim)
        for a in self.model.attn_layers:
            a.k_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
            a.v_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.kv_bytes = 2 * len(self.model.attn_layers) * a.k_cache.numel() * a.k_cache.element_size()

    def to_device(self, x, dtype):
        return torch.from_numpy(np.asarray(x, dtype=dtype)).to(self.device)

    def prepare_prefill(self, seqs: list[Sequence]):
        bs = self.cfg.block_size
        ids, pos, slots, cu = [], [], [], [0]
        for seq in seqs:
            p = np.arange(len(seq))
            ids.append(seq.token_ids)
            pos.append(p)
            slots.append(np.asarray(seq.block_table)[p // bs] * bs + p % bs)
            cu.append(cu[-1] + len(seq))
        meta = AttnMetadata(is_prefill=True, slot_mapping=self.to_device(np.concatenate(slots), np.int64), cu_seqlens=cu)
        input_ids = self.to_device(np.concatenate(ids), np.int64)
        last = self.to_device(cu[1:], np.int64) - 1  # logits only for each sequence's last token
        return input_ids, self.to_device(np.concatenate(pos), np.int64), meta, last

    def prepare_decode(self, seqs: list[Sequence]):
        bs = self.cfg.block_size
        pos = [len(seq) - 1 for seq in seqs]
        slots = [seq.block_table[p // bs] * bs + p % bs for seq, p in zip(seqs, pos)]
        block_tables = np.zeros((len(seqs), self.max_blocks_per_seq), dtype=np.int32)
        for i, seq in enumerate(seqs):
            block_tables[i, : len(seq.block_table)] = seq.block_table
        meta = AttnMetadata(
            is_prefill=False, slot_mapping=self.to_device(slots, np.int64),
            block_tables=self.to_device(block_tables, np.int32),
            context_lens=self.to_device([p + 1 for p in pos], np.int32), max_context_len=max(pos) + 1,
        )
        input_ids = self.to_device([seq.token_ids[-1] for seq in seqs], np.int64)
        return input_ids, self.to_device(pos, np.int64), meta, None

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions, meta, last = (self.prepare_prefill if is_prefill else self.prepare_decode)(seqs)
        hidden = self.model(input_ids, positions, meta)
        if last is not None:
            hidden = hidden[last]
        logits = self.model.compute_logits(hidden)
        return self.sampler(logits, [seq.params for seq in seqs]).tolist()
