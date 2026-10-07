"""Loads the model, owns the KV pool, and turns a scheduled batch into one forward pass plus sampling.

Decode batches are padded to a bucket size. A padding row is a dummy one-token sequence at position 0 whose
K/V goes to a scratch block outside the managed pool, so padding needs no special case anywhere. The decode
inputs live in static buffers that are refilled in place every step. With CUDA graphs, each bucket's decode
forward is captured once and replayed; logits are computed outside the graph for the real rows only.

With chunked prefill, a step that computes more than one token of some sequence (a mixed step) is padded to
a token-count bucket instead. Its sequences sit in a fixed number of slots: the padding tokens form one
more sequence that reads and writes only the scratch block, and unused slots have no tokens. The prefill
kernel then runs with a grid that depends only on the bucket, so mixed steps are captured as well.

With speculative decoding, a decode step that checks draft tokens (a verify step) is a mixed step too: each
sequence computes its last token and its drafts, so a sequence without drafts adds a single row.

With EAGLE drafting, after every step the draft model first catches up on the positions the target just
verified (again a mixed step, over the draft's own KV pool), then drafts num_spec_tokens tokens per sequence
in a chain of single-row steps; both are captured as CUDA graphs. After a prefill, the catch-up is the
draft's prefill over the prompt, with the target's prefill layout.
"""

import itertools
import os
import time

import numpy as np
import torch
import triton

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.attention.triton_paged import TritonAttention, prefill_block_m
from minivllm.kv_manager import cdiv
from minivllm.models.gpt2 import load_gpt2
from minivllm.models.qwen3 import load_qwen3
from minivllm.sampler import Sampler
from minivllm.sequence import Sequence
from minivllm.spec.eagle import load_eagle
from minivllm.spec.rejection import verify_drafts

MODELS = {
    "qwen3": dict(path=os.path.join("checkpoints", "Qwen3-0.6B"), eos=(151643, 151645), vocab_limit=None),
    "gpt2": dict(path=r"A:\GitHub project\nanoGPT\out\final\ckpt.pt", eos=(50256,), vocab_limit=50257),
}


# rows per tile of the prefill kernel in mixed steps: few enough that the decode rows stay bound by reading K/V
MIXED_BLOCK_M = 32


def decode_buckets(max_num_seqs):
    sizes = [b for b in (1, 2, 4, 8) if b < max_num_seqs] + list(range(16, max_num_seqs, 16))
    return sizes + [max_num_seqs]


def mixed_buckets(max_tokens, smallest=64):
    sizes, b = [], smallest
    while b < max_tokens:
        sizes.append(b)
        b *= 2
    return sizes + [max_tokens]


class ModelRunner:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        info = MODELS[cfg.model]
        if cfg.attention == "flex":
            attn = FlexAttention(cfg.block_size, cfg.max_num_batched_tokens, cfg.max_num_seqs, device)
        elif cfg.attention in ("triton", "triton-decode"):
            attn = TritonAttention(cfg.block_size, cfg.max_num_batched_tokens, cfg.max_num_seqs, device,
                                   flex_prefill=cfg.attention == "triton-decode")
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
        # mixed steps: prompt chunks with decode rows, or (speculative decoding) decode rows with their drafts
        self.mixed_buckets, self.mixed_block_m = [], MIXED_BLOCK_M
        if cfg.chunked_prefill:
            self.mixed_buckets = mixed_buckets(cfg.max_num_batched_tokens)
        elif cfg.spec:
            self.mixed_buckets = mixed_buckets(cfg.max_num_seqs * (cfg.num_spec_tokens + 1), smallest=8)
            group = self.model.attn_layers[0].num_heads // self.model.num_kv_heads
            self.mixed_block_m = prefill_block_m(cfg.num_spec_tokens + 1, triton.next_power_of_2(group))
        if self.mixed_buckets:
            t, n = self.mixed_buckets[-1], cfg.max_num_seqs + 1  # one more slot for the padding tokens
            self.mix_ids = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_pos = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_slots = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
            self.mix_seq_lens = torch.ones(n, dtype=torch.int32, device=device)
            # wide enough for the padding sequence, which can be longer than any real one
            w = max(w, cdiv(t, cfg.block_size))
            self.mix_block_tables = torch.zeros(n, w, dtype=torch.int32, device=device)
        self.mixed_graphs = {}  # token bucket -> (CUDAGraph, hidden-state output)
        # EAGLE: the draft model, with one more KV pool in the target's layout (draft position t in the slot of
        # target position t), and the static buffers of its catch-up and drafting steps
        self.draft = None
        self.draft_graphs = {}  # token bucket -> (CUDAGraph, draft output) for catching up
        self.chain_graphs = {}  # batch bucket -> CUDAGraph that drafts num_spec_tokens tokens
        self.last = None  # the last forward, for drafting: (sequences, hidden states, first row of each, prefill meta and positions)
        if cfg.spec == "eagle":
            self.draft = load_eagle(cfg.draft_path, self.model, attn, device)
            a = self.draft.block.attn
            a.k_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
            a.v_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.kv_bytes += 2 * a.k_cache.numel() * a.k_cache.element_size()
            d, s, g = self.model.cfg.n_embd, self.buckets[-1], cfg.num_spec_tokens
            self.mix_feat = torch.zeros(self.mixed_buckets[-1], d, dtype=torch.float32, device=device)
            self.ch_g = torch.zeros(s, d, dtype=torch.float32, device=device)
            self.ch_pos = torch.zeros(s, dtype=torch.int64, device=device)
            self.ch_temp = torch.zeros(s, dtype=torch.float32, device=device)
            # the chain writes up to num_spec_tokens - 2 positions past a sequence; past its blocks, into scratch
            self.ch_block_tables = torch.zeros(s, cdiv(cfg.max_model_len + g, cfg.block_size), dtype=torch.int32, device=device)
            self.ch_tokens = torch.zeros(s, g, dtype=torch.int64, device=device)
            self.ch_probs = torch.zeros(s, g, len(self.draft.vocab_ids), dtype=torch.float32, device=device)
        self.use_cuda_graph = False
        self.graph_capture_s = 0.0
        self.graph_memory_bytes = 0
        if cfg.cuda_graph:
            self.capture_graphs()

    def to_device(self, x, dtype):
        return torch.from_numpy(np.asarray(x, dtype=dtype)).to(self.device)

    def prepare_prefill(self, seqs: list[Sequence]):
        bs = self.cfg.block_size
        ids, pos, slots, cu = [], [], [], [0]
        block_tables = np.zeros((len(seqs), max(len(seq.block_table) for seq in seqs)), np.int32)
        for i, seq in enumerate(seqs):
            # tokens before num_cached_tokens are in shared blocks from the prefix cache
            p = np.arange(seq.num_cached_tokens, len(seq))
            ids.append(seq.token_ids[seq.num_cached_tokens :])
            pos.append(p)
            slots.append(np.asarray(seq.block_table)[p // bs] * bs + p % bs)
            cu.append(cu[-1] + len(p))
            block_tables[i, : len(seq.block_table)] = seq.block_table
        meta = AttnMetadata(
            is_prefill=True, slot_mapping=self.to_device(np.concatenate(slots), np.int64), cu_seqlens=cu,
            block_tables=self.to_device(block_tables, np.int32),
            context_lens=self.to_device([len(seq) for seq in seqs], np.int32),
        )
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

    def fill_mixed_buffers(self, entries, padded: int):
        """Writes a mixed step into the static buffers; returns the first row of each entry, then the end.

        entries is [(sequence, start, tokens)]: each sequence computes `tokens`, at positions start, start + 1, ...
        """
        bs, num_slots, k = self.cfg.block_size, self.mix_seq_lens.shape[0], len(entries)
        n = np.array([len(tokens) for _, _, tokens in entries], np.int64)
        start = np.array([first for _, first, _ in entries], np.int64)
        cu = np.full(num_slots + 1, padded, np.int32)  # unused slots have no tokens
        cu[0] = 0
        cu[1 : k + 1] = np.cumsum(n)
        t = int(cu[k])
        block_tables = np.full(self.mix_block_tables.shape, self.scratch_block, np.int32)
        for i, (seq, _, _) in enumerate(entries):
            block_tables[i, : len(seq.block_table)] = seq.block_table
        seq_lens = np.ones(num_slots, np.int32)
        seq_lens[:k] = start + n
        # the padding tokens: one more sequence, at positions 0, 1, ..., that only touches the scratch block
        seq_lens[k] = max(padded - t, 1)
        pos = np.concatenate([np.arange(t) - np.repeat(cu[:k], n) + np.repeat(start, n), np.arange(padded - t)])
        owner = np.repeat(np.arange(k + 1), np.append(n, padded - t))
        slots = block_tables[owner, pos // bs].astype(np.int64) * bs + pos % bs
        ids = np.zeros(padded, np.int64)
        ids[:t] = list(itertools.chain.from_iterable(tokens for _, _, tokens in entries))
        for buf, x in ((self.mix_ids, ids), (self.mix_pos, pos), (self.mix_slots, slots)):
            buf[:padded].copy_(torch.from_numpy(x))
        for buf, x in ((self.mix_cu, cu), (self.mix_seq_lens, seq_lens), (self.mix_block_tables, block_tables)):
            buf.copy_(torch.from_numpy(x))
        return cu[: k + 1]

    def mixed_hidden(self, padded: int):
        meta = AttnMetadata(
            is_prefill=True, slot_mapping=self.mix_slots[:padded], block_tables=self.mix_block_tables,
            context_lens=self.mix_seq_lens, backend_state=(self.mix_cu, self.mixed_block_m),
        )
        return self.model(self.mix_ids[:padded], self.mix_pos[:padded], meta)

    def draft_hidden(self, padded: int):
        """The draft model over a mixed step in the static buffers, with features from mix_feat."""
        meta = AttnMetadata(
            is_prefill=True, slot_mapping=self.mix_slots[:padded], block_tables=self.mix_block_tables,
            context_lens=self.mix_seq_lens, backend_state=(self.mix_cu, self.mixed_block_m),
        )
        return self.draft(self.mix_feat[:padded], self.mix_ids[:padded], self.mix_pos[:padded], meta)

    def fill_chain_buffers(self, seqs: list[Sequence], padded: int):
        """Writes where each sequence's draft chain starts: its last position, block table and temperature."""
        pos = np.zeros(padded, np.int64)
        temp = np.zeros(padded, np.float32)
        block_tables = np.full((padded, self.ch_block_tables.shape[1]), self.scratch_block, np.int32)
        for i, seq in enumerate(seqs):
            pos[i], temp[i] = len(seq) - 1, seq.params.temperature
            block_tables[i, : len(seq.block_table)] = seq.block_table
        for buf, x in ((self.ch_pos, pos), (self.ch_temp, temp), (self.ch_block_tables, block_tables)):
            buf[:padded].copy_(torch.from_numpy(x))

    def draft_chain(self, padded: int):
        """Drafts num_spec_tokens tokens per row of ch_g, the draft's output at each sequence's last verified
        position: the first from ch_g, every next one after one more draft step at the following position.
        Greedy rows take the argmax; the others sample at their temperature and keep the distribution."""
        bs, g = self.cfg.block_size, self.cfg.num_spec_tokens
        h, pos, bt, temp = self.ch_g[:padded], self.ch_pos[:padded], self.ch_block_tables[:padded], self.ch_temp[:padded]
        sampled = temp > 0
        t = torch.where(sampled, temp, 1.0)[:, None]
        for k in range(g):
            logits = self.draft.logits(h).float()
            q = torch.softmax(logits / t, dim=-1)
            pick = torch.where(sampled, (q / torch.empty_like(q).exponential_()).argmax(-1), logits.argmax(-1))
            tok = self.draft.vocab_ids[pick]
            self.ch_tokens[:padded, k] = tok
            self.ch_probs[:padded, k] = q
            if k == g - 1:
                break
            p = pos + k
            slots = bt.gather(1, (p // bs)[:, None]).squeeze(1).long() * bs + p % bs
            meta = AttnMetadata(is_prefill=False, slot_mapping=slots, block_tables=bt, context_lens=(p + 1).int())
            h = self.draft(h, tok, p.clamp(max=self.cfg.max_model_len - 1), meta)

    @torch.inference_mode()
    def draft_next(self, entries):
        """EAGLE drafting after a step. entries is [(i, n)]: sequence i of the last forward continues, and its
        first n rows there are verified. Returns each sequence's num_spec_tokens drafts, and the distributions
        they were sampled from ([len(entries), num_spec_tokens, draft vocabulary]; None if all are greedy)."""
        seqs, hidden, starts, meta, positions = self.last
        if meta is not None:  # after a prefill: the draft's prefill, over every prompt position
            cu = meta.cu_seqlens
            ids = itertools.chain.from_iterable(seq.token_ids[1 : 1 + b - a] for seq, a, b in zip(seqs, cu[:-1], cu[1:]))
            out = self.draft(hidden, self.to_device(list(ids), np.int64), positions, meta)
            last_rows = [starts[i] + n - 1 for i, n in entries]
        else:  # after a decode or verify step: catch up on the verified positions
            rows = [starts[i] + j for i, n in entries for j in range(n)]
            padded = next(b for b in self.mixed_buckets if b >= len(rows))
            cu = self.fill_mixed_buffers(
                [(seqs[i], len(seqs[i]) - 1 - n, seqs[i].token_ids[len(seqs[i]) - n :]) for i, n in entries], padded)
            self.mix_feat[: len(rows)].copy_(hidden[self.to_device(rows, np.int64)])
            if self.use_cuda_graph:
                graph, out = self.draft_graphs[padded]
                graph.replay()
            else:
                out = self.draft_hidden(padded)
            last_rows = (cu[1:] - 1).tolist()
        chain = [seqs[i] for i, _ in entries]
        padded = next(b for b in self.buckets if b >= len(chain))
        self.fill_chain_buffers(chain, padded)
        self.ch_g[: len(chain)].copy_(out[self.to_device(last_rows, np.int64)])
        if self.use_cuda_graph:
            self.chain_graphs[padded].replay()
        else:
            self.draft_chain(padded)
        drafts = self.ch_tokens[: len(chain)].tolist()
        probs = self.ch_probs[: len(chain)].clone() if any(seq.params.temperature > 0 for seq in chain) else None
        return drafts, probs

    @torch.inference_mode()
    def capture_graphs(self):
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
        for padded in reversed(self.mixed_buckets):
            self.fill_mixed_buffers([], padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.mixed_hidden(padded)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                hidden = self.mixed_hidden(padded)
            self.mixed_graphs[padded] = (graph, hidden)
        for padded in reversed(self.mixed_buckets) if self.draft is not None else ():
            self.fill_mixed_buffers([], padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.draft_hidden(padded)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                out = self.draft_hidden(padded)
            self.draft_graphs[padded] = (graph, out)
        for padded in reversed(self.buckets) if self.draft is not None else ():
            self.fill_chain_buffers([], padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.draft_chain(padded)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                self.draft_chain(padded)
            self.chain_graphs[padded] = graph
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.graph_capture_s = time.perf_counter() - start
        self.graph_memory_bytes = free_before - torch.cuda.mem_get_info()[0]
        self.use_cuda_graph = True

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions, meta, last = self.prepare_prefill(seqs)
            hidden = self.model(input_ids, positions, meta)
            if self.draft is not None:
                self.last = (seqs, hidden, meta.cu_seqlens[:-1], meta, positions)
            hidden = hidden[last]
        else:
            padded = next(b for b in self.buckets if b >= len(seqs))
            max_context_len = self.fill_decode_buffers(seqs, padded)
            if self.use_cuda_graph:
                graph, hidden = self.graphs[padded]
                graph.replay()
            else:
                hidden = self.decode_hidden(padded, max_context_len)
            hidden = hidden[: len(seqs)]
            if self.draft is not None:
                self.last = (seqs, hidden, list(range(len(seqs))), None, None)
        logits = self.model.compute_logits(hidden)
        return self.sampler(logits, [seq.params for seq in seqs]).tolist()

    @torch.inference_mode()
    def run_chunked(self, plan) -> list:
        """Runs one step of chunked prefill; returns each sequence's next token, or None in the middle of a prompt."""
        if all(n == 1 for _, n in plan):  # only last tokens to compute: an ordinary decode step
            return self.run([seq for seq, _ in plan], is_prefill=False)
        padded = next(b for b in self.mixed_buckets if b >= sum(n for _, n in plan))
        entries = [(seq, seq.num_computed_tokens, seq.token_ids[seq.num_computed_tokens : seq.num_computed_tokens + n])
                   for seq, n in plan]
        cu = self.fill_mixed_buffers(entries, padded)
        hidden = self.mixed_forward(padded)
        out = [None] * len(plan)
        # a sequence whose step reaches its last known token samples the next one from it
        sampled = [i for i, (seq, n) in enumerate(plan) if seq.num_computed_tokens + n == len(seq)]
        if sampled:
            logits = self.model.compute_logits(hidden[self.to_device([cu[i + 1] - 1 for i in sampled], np.int64)])
            for i, tok in zip(sampled, self.sampler(logits, [plan[i][0].params for i in sampled]).tolist()):
                out[i] = tok
        return out

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence], drafts: list[list[int]], draft_probs=None) -> list[list[int]]:
        """A decode step that also checks each sequence's draft tokens; returns per sequence the accepted
        drafts followed by one token from the target model. draft_probs: per sequence, the distributions its
        drafts were sampled from (EAGLE), or None for deterministic drafts."""
        if not any(drafts):
            return [[tok] for tok in self.run(seqs, is_prefill=False)]
        t = len(seqs) + sum(map(len, drafts))
        padded = next(b for b in self.mixed_buckets if b >= t)
        cu = self.fill_mixed_buffers([(seq, len(seq) - 1, [seq.token_ids[-1]] + d) for seq, d in zip(seqs, drafts)], padded)
        hidden = self.mixed_forward(padded)
        if self.draft is not None:
            self.last = (seqs, hidden, cu[:-1].tolist(), None, None)
        logits = self.model.compute_logits(hidden[:t])
        q, vocab = None, None
        if draft_probs is not None:
            q, vocab = torch.cat([probs[: len(d)] for probs, d in zip(draft_probs, drafts) if d]), self.draft.vocab_ids
        return verify_drafts(self.sampler, logits, [seq.params for seq in seqs], drafts, q, vocab)

    def mixed_forward(self, padded: int):
        if self.use_cuda_graph:
            graph, hidden = self.mixed_graphs[padded]
            graph.replay()
            return hidden
        return self.mixed_hidden(padded)
