"""Loads the model, owns the KV pool, and turns a scheduled batch into one forward pass plus sampling.

Decode batches are padded to a bucket size. A padding row is a dummy one-token sequence at position 0 whose
K/V goes to a scratch block outside the managed pool, so padding needs no special case anywhere. The decode
inputs live in static buffers that are refilled in place every step. With CUDA graphs, each bucket's decode
forward is captured once and replayed; logits are computed outside the graph for the real rows only.
"""

import os
import time

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


def decode_buckets(max_num_seqs):
    sizes = [b for b in (1, 2, 4, 8) if b < max_num_seqs] + list(range(16, max_num_seqs, 16))
    return sizes + [max_num_seqs]


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

        # the pool has one extra block at the end that padding rows write into
        self.scratch_block = cfg.num_kv_blocks
        shape = (cfg.num_kv_blocks + 1, cfg.block_size, self.model.num_kv_heads, self.model.head_dim)
        for a in self.model.attn_layers:
            a.k_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
            a.v_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.kv_bytes = 2 * len(self.model.attn_layers) * a.k_cache.numel() * a.k_cache.element_size()

        self.buckets = decode_buckets(cfg.max_num_seqs)
        self.max_blocks_per_seq = cdiv(cfg.max_model_len, cfg.block_size)
        n, w = self.buckets[-1], self.max_blocks_per_seq
        self.dec_ids = torch.zeros(n, dtype=torch.int64, device=device)
        self.dec_pos = torch.zeros(n, dtype=torch.int64, device=device)
        self.dec_slots = torch.zeros(n, dtype=torch.int64, device=device)
        self.dec_block_tables = torch.zeros(n, w, dtype=torch.int32, device=device)
        self.dec_context_lens = torch.zeros(n, dtype=torch.int32, device=device)
        self.graphs = {}  # bucket -> (CUDAGraph, hidden-state output)
        self.use_cuda_graph = False
        self.graph_capture_s = 0.0
        self.graph_memory_bytes = 0
        if cfg.cuda_graph:
            self.capture_decode_graphs()

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

    def fill_decode_buffers(self, seqs: list[Sequence], padded: int):
        """Writes the decode inputs of seqs, plus padding rows up to `padded`, into the static buffers."""
        bs = self.cfg.block_size
        ids = np.zeros(padded, np.int64)
        pos = np.zeros(padded, np.int64)
        slots = np.full(padded, self.scratch_block * bs, np.int64)
        context_lens = np.ones(padded, np.int32)
        block_tables = np.zeros((padded, self.max_blocks_per_seq), np.int32)
        block_tables[:, 0] = self.scratch_block
        for i, seq in enumerate(seqs):
            p = len(seq) - 1
            ids[i], pos[i], context_lens[i] = seq.token_ids[-1], p, p + 1
            slots[i] = seq.block_table[p // bs] * bs + p % bs
            block_tables[i, : len(seq.block_table)] = seq.block_table
        for buf, x in ((self.dec_ids, ids), (self.dec_pos, pos), (self.dec_slots, slots),
                       (self.dec_context_lens, context_lens), (self.dec_block_tables, block_tables)):
            buf[:padded].copy_(torch.from_numpy(x))
        return int(context_lens.max())

    def decode_hidden(self, padded: int, max_context_len: int):
        meta = AttnMetadata(
            is_prefill=False, slot_mapping=self.dec_slots[:padded], block_tables=self.dec_block_tables[:padded],
            context_lens=self.dec_context_lens[:padded], max_context_len=max_context_len,
        )
        return self.model(self.dec_ids[:padded], self.dec_pos[:padded], meta)

    @torch.inference_mode()
    def capture_decode_graphs(self):
        # empty_cache before and after: compiling flex can leave large cached blocks that nothing uses
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]
        max_len = self.cfg.max_model_len  # graphs need a fixed shape for the gather backend's padding
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        # largest bucket first, so the smaller graphs reuse its memory
        for padded in reversed(self.buckets):
            self.fill_decode_buffers([], padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # compiles flex for this batch size before capturing
                    self.decode_hidden(padded, max_len)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                hidden = self.decode_hidden(padded, max_len)
            self.graphs[padded] = (graph, hidden)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.graph_capture_s = time.perf_counter() - start
        self.graph_memory_bytes = free_before - torch.cuda.mem_get_info()[0]
        self.use_cuda_graph = True

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions, meta, last = self.prepare_prefill(seqs)
            hidden = self.model(input_ids, positions, meta)[last]
        else:
            padded = next(b for b in self.buckets if b >= len(seqs))
            max_context_len = self.fill_decode_buffers(seqs, padded)
            if self.use_cuda_graph:
                graph, hidden = self.graphs[padded]
                graph.replay()
            else:
                hidden = self.decode_hidden(padded, max_context_len)
            hidden = hidden[: len(seqs)]
        logits = self.model.compute_logits(hidden)
        return self.sampler(logits, [seq.params for seq in seqs]).tolist()
