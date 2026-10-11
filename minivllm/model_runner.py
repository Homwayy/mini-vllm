"""Loads the model, owns the KV pool, and turns a scheduled batch into one forward pass plus sampling.

Decode batches are padded to a bucket size. A padding row is a dummy one-token sequence at position 0 whose
K/V goes to a scratch block outside the managed pool, so padding needs no special case anywhere. The decode
inputs live in static buffers that are refilled in place every step. With CUDA graphs, each bucket's decode
forward is captured once and replayed; logits are computed outside the graph for the real rows only.

With chunked prefill, a step that computes more than one token of some sequence (a mixed step) is padded to
a token-count bucket instead. Its sequences sit in a fixed number of slots: the padding tokens form one
more sequence that reads and writes only the scratch block, and unused slots have no tokens. The prefill
kernel then runs with a grid that depends only on the bucket, so mixed steps are captured as well. A bucket
of t tokens uses only min(max_num_seqs, t) + 1 slots, since the kernel runs programs for every slot.

Without chunked prefill, a prefill step of at most prefill_graph_tokens tokens takes the same layout, padded
to its own token buckets with prefill-sized tiles, and is captured too: eager, such a step is bound by
launching its kernels. Longer prefill steps run eagerly, unpadded.

With speculative decoding, a decode step that checks draft tokens (a verify step) is a mixed step too: each
sequence computes its last token and its drafts, so a sequence without drafts adds a single row.

With EAGLE drafting, after every step the draft model first catches up on the positions the target just
verified (again a mixed step, over the draft's own KV pool), then drafts num_spec_tokens tokens per sequence
in a chain of single-row steps; both are captured as CUDA graphs. After a verify step, both take their
inputs from the verify step's buffers and acceptance on the GPU. After a prefill, the catch-up is the
draft's prefill over the prompt, with the target's prefill layout (in its buffers, if padded).

With DSpark drafting, the target's forward itself writes the draft's context K/V for all its rows, so there is
no catch-up; drafting is one graph per batch bucket: the forward of a block of rows from each sequence's anchor,
and the sampling of its tokens with the Markov head (minivllm/spec/dspark.py).

Both draw each draft token with one kernel (minivllm/spec/draft_sampling.py), keyed by a seed that every
drafting redraws on the GPU, so a graph replay samples afresh.
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
from minivllm.paths import GPT2_CKPT
from minivllm.sampler import Sampler
from minivllm.sequence import Sequence
from minivllm.spec.draft_sampling import sample_drafts
from minivllm.spec.dspark import block_inputs_kernel, load_dspark
from minivllm.spec.eagle import draft_inputs_kernel, load_eagle
from minivllm.spec.rejection import verify_drafts

MODELS = {
    "qwen3": dict(path=os.path.join("checkpoints", "Qwen3-0.6B"), eos=(151643, 151645), vocab_limit=None),
    "gpt2": dict(path=GPT2_CKPT, eos=(50256,), vocab_limit=50257),
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


def prefill_buckets(max_tokens):
    # finer than powers of 2: a padded token costs a bit of GPU time, while a graph saves launch time per kernel
    sizes = [b for b in (16, 32) if b < max_tokens] + list(range(64, max_tokens, 64))
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
        self.model.fused = cfg.fused_kernels
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
        self.group_pad = triton.next_power_of_2(self.model.attn_layers[0].num_heads // self.model.num_kv_heads)
        # mixed steps: prompt chunks with decode rows, or (speculative decoding) decode rows with their drafts
        self.mixed_buckets, self.mixed_block_m = [], MIXED_BLOCK_M
        if cfg.chunked_prefill:
            self.mixed_buckets = mixed_buckets(cfg.max_num_batched_tokens)
        elif cfg.spec:
            self.mixed_buckets = mixed_buckets(cfg.max_num_seqs * (cfg.num_spec_tokens + 1), smallest=8)
            self.mixed_block_m = prefill_block_m(cfg.num_spec_tokens + 1, self.group_pad)
        # short prefill steps in the same buffers (chunked prefill has every prompt in mixed steps already)
        self.prefill_buckets = []
        if cfg.prefill_graph_tokens > 0 and cfg.attention == "triton" and not cfg.chunked_prefill:
            self.prefill_buckets = prefill_buckets(min(cfg.prefill_graph_tokens, cfg.max_num_batched_tokens))
        self.prefill_graphs = {}  # token bucket -> (CUDAGraph, hidden-state output)
        if self.mixed_buckets or self.prefill_buckets:
            t, n = max(self.mixed_buckets[-1:] + self.prefill_buckets[-1:]), cfg.max_num_seqs + 1  # one more slot for the padding tokens
            self.mix_ids = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_pos = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_slots = torch.zeros(t, dtype=torch.int64, device=device)
            self.mix_cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
            self.mix_seq_lens = torch.ones(n, dtype=torch.int32, device=device)
            # wide enough for the padding sequence, which can be longer than any real one
            w = max(w, cdiv(t, cfg.block_size))
            self.mix_block_tables = torch.zeros(n, w, dtype=torch.int32, device=device)
        self.mixed_graphs = {}  # token bucket -> (CUDAGraph, hidden-state output)
        # EAGLE or DSpark: the draft model, with one more KV pool in the target's layout (draft position t in the
        # slot of target position t), and the static buffers of its drafting steps (and EAGLE's catch-up)
        self.spec = cfg.spec
        self.draft = None
        self.draft_graphs = {}  # token bucket -> (CUDAGraph, draft output) for catching up
        self.draft_prefill_graphs = {}  # prefill token bucket -> (CUDAGraph, draft output)
        self.chain_graphs = {}  # batch bucket -> CUDAGraph that drafts num_spec_tokens tokens
        self.last = None  # the last forward, for drafting: (sequences, hidden states, first row of each, prefill meta and positions)
        self.last_verify = None  # the token bucket of the last forward if it was a verify step
        self.last_prefill = None  # the token bucket of the last forward if it was a padded prefill step
        if cfg.spec in ("eagle", "dspark"):
            path = cfg.draft_path or os.path.join("checkpoints", f"gpt2_{cfg.spec}.pt")
            self.draft = (load_eagle if cfg.spec == "eagle" else load_dspark)(path, self.model, attn, device)
            self.draft.fused = cfg.fused_kernels
            for a in self.draft.attn_layers:
                a.k_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
                a.v_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
                self.kv_bytes += 2 * a.k_cache.numel() * a.k_cache.element_size()
            d, s, g = self.model.cfg.n_embd, self.buckets[-1], cfg.num_spec_tokens
            self.ch_pos = torch.zeros(s, dtype=torch.int64, device=device)
            self.ch_temp = torch.zeros(s, dtype=torch.float32, device=device)
            # drafting writes up to num_spec_tokens - 1 positions past a sequence; past its blocks, into scratch
            self.ch_block_tables = torch.zeros(s, cdiv(cfg.max_model_len + g, cfg.block_size), dtype=torch.int32, device=device)
            self.ch_tokens = torch.zeros(s, g, dtype=torch.int64, device=device)
            self.ch_prev = torch.zeros(s, dtype=torch.int64, device=device)  # each row's latest draft
            self.draft_seed = torch.zeros(1, dtype=torch.int64, device=device)  # redrawn before each drafting
            self.ch_probs = torch.zeros(s, g, len(self.draft.vocab_ids), dtype=torch.float32, device=device)
            self.verify_out = torch.zeros(s, 2, dtype=torch.int64, device=device)  # per sequence: accepted drafts, next token
        if cfg.spec == "eagle":
            self.mix_feat = torch.zeros(self.mix_ids.shape[0], d, dtype=torch.float32, device=device)
            self.ch_g = torch.zeros(s, d, dtype=torch.float32, device=device)
            self.ch_rows = torch.zeros(s, dtype=torch.int64, device=device)  # the catch-up row each chain row starts from
        if cfg.spec == "dspark":
            assert g <= self.draft.block_len, f"the draft model was trained for blocks of up to {self.draft.block_len} rows"
            self.model.capture = self.draft.feature_layers
            self.ch_anchor = torch.zeros(s, dtype=torch.int64, device=device)  # each block's anchor token
            # a block's rows are a mixed step of g tokens per sequence: fixed cu_seqlens, tiles for g rows
            self.blk_cu = torch.arange(0, (s + 1) * g, g, dtype=torch.int32, device=device)
            self.blk_offsets = torch.arange(g, device=device)
            self.blk_block_m = prefill_block_m(g, self.group_pad)
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
        return self.target_forward(self.dec_ids[:padded], self.dec_pos[:padded], meta)

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

    def mixed_meta(self, padded: int, prefill: bool):
        # a padded prefill step has eager prefill's tiles (prefill_block_m); a mixed step keeps them small
        block_m = prefill_block_m(padded, self.group_pad) if prefill else self.mixed_block_m
        # a step of `padded` tokens has at most that many sequences, and the padding tokens take one more slot;
        # the prefill kernel runs programs for every slot, so the slots no step of this bucket can use are left out
        s = min(self.cfg.max_num_seqs, padded) + 1
        return AttnMetadata(
            is_prefill=True, slot_mapping=self.mix_slots[:padded], block_tables=self.mix_block_tables[:s],
            context_lens=self.mix_seq_lens[:s], backend_state=(self.mix_cu[: s + 1], block_m),
        )

    def mixed_hidden(self, padded: int, prefill=False):
        return self.target_forward(self.mix_ids[:padded], self.mix_pos[:padded], self.mixed_meta(padded, prefill))

    def draft_hidden(self, padded: int, prefill=False):
        """The draft model over a mixed step in the static buffers, with features from mix_feat."""
        return self.draft(self.mix_feat[:padded], self.mix_ids[:padded], self.mix_pos[:padded], self.mixed_meta(padded, prefill))

    def fill_chain_buffers(self, seqs: list[Sequence], padded: int):
        """Writes where each sequence's draft chain (or DSpark block) starts: its last position and token, block
        table and temperature."""
        pos = np.zeros(padded, np.int64)
        last = np.zeros(padded, np.int64)
        temp = np.zeros(padded, np.float32)
        block_tables = np.full((padded, self.ch_block_tables.shape[1]), self.scratch_block, np.int32)
        for i, seq in enumerate(seqs):
            pos[i], last[i], temp[i] = len(seq) - 1, seq.token_ids[-1], seq.params.temperature
            block_tables[i, : len(seq.block_table)] = seq.block_table
        bufs = [(self.ch_pos, pos), (self.ch_temp, temp), (self.ch_block_tables, block_tables)]
        if self.spec == "dspark":
            bufs.append((self.ch_anchor, last))
        for buf, x in bufs:
            buf[:padded].copy_(torch.from_numpy(x))

    def target_forward(self, input_ids, positions, meta: AttnMetadata):
        """The target model's forward. With DSpark drafting, it also writes the context K/V of every row into the
        draft model's pool, before acceptance is known (see minivllm/spec/dspark.py)."""
        hidden = self.model(input_ids, positions, meta)
        if self.spec == "dspark":
            self.draft.inject(self.model.captured + [hidden], positions, meta)
        return hidden

    def draft_chain(self, padded: int):
        """Drafts num_spec_tokens tokens per row of ch_g, the draft's output at each sequence's last verified
        position: the first from ch_g, every next one after one more draft step at the following position.
        Greedy rows take the argmax; the others sample at their temperature and keep the distribution."""
        bs, g = self.cfg.block_size, self.cfg.num_spec_tokens
        h, pos, bt, temp = self.ch_g[:padded], self.ch_pos[:padded], self.ch_block_tables[:padded], self.ch_temp[:padded]
        tok = self.ch_prev[:padded]
        self.draft_seed.random_(2**31 - 1)
        for k in range(g):
            sample_drafts(self.draft.logits(h), temp, self.draft_seed, self.draft.vocab_ids, self.ch_tokens[:padded, k], tok,
                          self.ch_probs[:padded, k], k)
            if k == g - 1:
                break
            p = pos + k
            slots = bt.gather(1, (p // bs)[:, None]).squeeze(1).long() * bs + p % bs
            meta = AttnMetadata(is_prefill=False, slot_mapping=slots, block_tables=bt, context_lens=(p + 1).int())
            h = self.draft(h, tok, p.clamp(max=self.cfg.max_model_len - 1), meta)

    def dspark_block(self, padded: int):
        """DSpark: drafts num_spec_tokens tokens per row of the chain buffers. One forward of the draft model gives
        base logits for a whole block, from the anchor ch_anchor at ch_pos and mask rows after it; then the tokens
        are drafted one by one, each with the Markov head's term for the token before. Greedy rows take the
        argmax; the others sample at their temperature and keep the distribution."""
        bs, g = self.cfg.block_size, self.cfg.num_spec_tokens
        bt, anchors = self.ch_block_tables[:padded], self.ch_anchor[:padded]
        pos = (self.ch_pos[:padded, None] + self.blk_offsets).view(-1)
        slots = bt.gather(1, (pos // bs).view(padded, g)).view(-1).long() * bs + pos % bs
        meta = AttnMetadata(is_prefill=True, slot_mapping=slots, block_tables=bt, context_lens=(self.ch_pos[:padded] + g).int(),
                            backend_state=(self.blk_cu[: padded + 1], self.blk_block_m))
        h = self.draft(anchors, pos.clamp(max=self.cfg.max_model_len - 1), meta)
        u = self.draft.logits(h).view(padded, g, -1)
        temp, prev = self.ch_temp[:padded], self.ch_prev[:padded]
        self.draft_seed.random_(2**31 - 1)
        for k in range(g):
            sample_drafts(u[:, k], temp, self.draft_seed, self.draft.vocab_ids, self.ch_tokens[:padded, k], prev,
                          self.ch_probs[:padded, k], k, bias=self.draft.transition(anchors if k == 0 else prev))

    @torch.inference_mode()
    def dspark_next(self, entries):
        """draft_next for DSpark: one block per sequence. The context K/V are in the pool already (target_forward),
        so only each block's anchor, position, block table and temperature are needed. After a verify step they
        come from its buffers and acceptance, on the GPU (block_inputs_kernel); a sequence not in entries gets
        temperature -1, which makes its block padding. After a prefill or a decode step, the CPU writes them."""
        seqs = self.last[0]
        if self.last_verify is not None:
            n = len(seqs)
            padded = next(b for b in self.buckets if b >= n)
            temps = np.full(n, -1.0, np.float32)
            for i, _ in entries:
                temps[i] = seqs[i].params.temperature
            temps = self.to_device(temps, np.float32)
            block_inputs_kernel[(padded,)](
                self.mix_pos, self.mix_cu, self.mix_block_tables, self.mix_block_tables.stride(0), self.mix_block_tables.shape[1],
                self.verify_out, temps, n, self.scratch_block, self.ch_anchor, self.ch_pos, self.ch_temp, self.ch_block_tables,
                self.ch_block_tables.stride(0), self.ch_block_tables.shape[1],
                BT_BLOCK=triton.next_power_of_2(self.ch_block_tables.shape[1]))
            rows = [i for i, _ in entries]
        else:
            seqs = [seqs[i] for i, _ in entries]
            n = len(seqs)
            padded = next(b for b in self.buckets if b >= n)
            self.fill_chain_buffers(seqs, padded)
            rows = list(range(n))
        if self.use_cuda_graph:
            self.chain_graphs[padded].replay()
        else:
            self.dspark_block(padded)
        drafts = self.ch_tokens[:n].tolist()
        probs = None
        if any(seqs[i].params.temperature > 0 for i in rows):
            probs = self.ch_probs[:n].clone() if len(rows) == n else self.ch_probs[self.to_device(rows, np.int64)]
        return [drafts[i] for i in rows], probs

    @torch.inference_mode()
    def draft_next(self, entries):
        """Drafting after a step. entries is [(i, n)]: sequence i of the last forward continues, and its first n rows
        there are verified. Returns each sequence's num_spec_tokens drafts, and the distributions they were
        sampled from ([len(entries), num_spec_tokens, draft vocabulary]; None if all are greedy)."""
        if self.spec == "dspark":
            return self.dspark_next(entries)
        if self.last_verify is not None:
            return self.draft_after_verify(entries)
        seqs, hidden, starts, meta, positions = self.last
        if self.last_prefill is not None:  # after a padded prefill: the draft's prefill in the same buffers
            padded = self.last_prefill
            # the row of position p takes token p + 1; the step has appended each sequence's next token
            ids = list(itertools.chain.from_iterable(seq.token_ids[seq.num_cached_tokens + 1 :] for seq in seqs))
            self.mix_ids[: len(ids)].copy_(torch.tensor(ids))
            self.mix_feat[:padded].copy_(hidden[:padded])
            if self.use_cuda_graph:
                graph, out = self.draft_prefill_graphs[padded]
                graph.replay()
            else:
                out = self.draft_hidden(padded, prefill=True)
            last_rows = [starts[i] + n - 1 for i, n in entries]
        elif meta is not None:  # after an eager prefill: the draft's prefill, over every prompt position
            cu = meta.cu_seqlens
            ids = itertools.chain.from_iterable(seq.token_ids[1 : 1 + b - a] for seq, a, b in zip(seqs, cu[:-1], cu[1:]))
            out = self.draft(hidden, self.to_device(list(ids), np.int64), positions, meta)
            last_rows = [starts[i] + n - 1 for i, n in entries]
        else:  # after a decode step without drafts: catch up on the verified positions
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
    def draft_after_verify(self, entries):
        """draft_next after a verify step, with no input built on the CPU. The catch-up reuses the verify step's
        buffers: each verified row gets the token after it, and the other rows write into the scratch block
        (draft_inputs_kernel). The chain then starts from each sequence's last verified row and uses the
        verify step's block tables, which have room for it (LLMEngine._step_spec). Both run for every
        sequence of the verify step; only those in entries keep their drafts."""
        seqs, hidden = self.last[0], self.last[1]
        padded, n = self.last_verify, len(seqs)
        chain_padded = next(b for b in self.buckets if b >= n)
        temps = self.to_device([seq.params.temperature for seq in seqs], np.float32)
        self.mix_feat[:padded].copy_(hidden[:padded])
        draft_inputs_kernel[(chain_padded,)](
            self.mix_ids, self.mix_pos, self.mix_slots, self.mix_cu, self.mix_block_tables, self.mix_block_tables.stride(0),
            self.mix_block_tables.shape[1], self.verify_out, temps, n, self.scratch_block, self.cfg.block_size,
            self.ch_rows, self.ch_pos, self.ch_temp, self.ch_block_tables, self.ch_block_tables.stride(0),
            self.ch_block_tables.shape[1], MAX_ROWS=triton.next_power_of_2(self.cfg.num_spec_tokens + 1),
            BT_BLOCK=triton.next_power_of_2(self.ch_block_tables.shape[1]))
        if self.use_cuda_graph:
            graph, g = self.draft_graphs[padded]
            graph.replay()
        else:
            g = self.draft_hidden(padded)
        torch.index_select(g, 0, self.ch_rows[:chain_padded], out=self.ch_g[:chain_padded])
        if self.use_cuda_graph:
            self.chain_graphs[chain_padded].replay()
        else:
            self.draft_chain(chain_padded)
        rows = [i for i, _ in entries]
        drafts = self.ch_tokens[:n].tolist()
        probs = None
        if any(seqs[i].params.temperature > 0 for i in rows):
            # usually every sequence keeps drafting, and a copy needs no row indices
            probs = self.ch_probs[:n].clone() if len(rows) == n else self.ch_probs[self.to_device(rows, np.int64)]
        return [drafts[i] for i in rows], probs

    @torch.inference_mode()
    def capture_graphs(self):
        # empty_cache before and after: compiling flex can leave large cached blocks that nothing uses
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]
        max_len = self.cfg.max_model_len  # graphs need a fixed shape for the gather backend's padding
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()

        def capture(fill, forward, padded):
            fill([], padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # compiles flex for this shape before capturing
                    forward(padded)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                out = forward(padded)
            return graph, out

        mix = self.fill_mixed_buffers
        # largest bucket first, so the smaller graphs reuse its memory
        for padded in reversed(self.buckets):
            self.graphs[padded] = capture(self.fill_decode_buffers, lambda p: self.decode_hidden(p, max_len), padded)
        for padded in reversed(self.mixed_buckets):
            self.mixed_graphs[padded] = capture(mix, self.mixed_hidden, padded)
        for padded in reversed(self.prefill_buckets):
            self.prefill_graphs[padded] = capture(mix, lambda p: self.mixed_hidden(p, prefill=True), padded)
        if self.spec == "dspark":
            for padded in reversed(self.buckets):
                self.chain_graphs[padded] = capture(self.fill_chain_buffers, self.dspark_block, padded)[0]
        if self.spec == "eagle":
            for padded in reversed(self.mixed_buckets):
                self.draft_graphs[padded] = capture(mix, self.draft_hidden, padded)
            for padded in reversed(self.prefill_buckets):
                self.draft_prefill_graphs[padded] = capture(mix, lambda p: self.draft_hidden(p, prefill=True), padded)
            for padded in reversed(self.buckets):
                self.chain_graphs[padded] = capture(self.fill_chain_buffers, self.draft_chain, padded)[0]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.graph_capture_s = time.perf_counter() - start
        self.graph_memory_bytes = free_before - torch.cuda.mem_get_info()[0]
        self.use_cuda_graph = True

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        num_new = sum(len(seq) - seq.num_cached_tokens for seq in seqs) if is_prefill else 0
        if is_prefill and self.prefill_buckets and num_new <= self.prefill_buckets[-1]:
            padded = next(b for b in self.prefill_buckets if b >= num_new)
            cu = self.fill_mixed_buffers([(seq, seq.num_cached_tokens, seq.token_ids[seq.num_cached_tokens :]) for seq in seqs], padded)
            if self.use_cuda_graph:
                graph, hidden = self.prefill_graphs[padded]
                graph.replay()
            else:
                hidden = self.mixed_hidden(padded, prefill=True)
            if self.draft is not None:
                self.last, self.last_verify, self.last_prefill = (seqs, hidden, cu[:-1].tolist(), None, None), None, padded
            hidden = hidden[self.to_device(cu[1:] - 1, np.int64)]
        elif is_prefill:
            input_ids, positions, meta, last = self.prepare_prefill(seqs)
            hidden = self.target_forward(input_ids, positions, meta)
            if self.draft is not None:
                self.last, self.last_verify, self.last_prefill = (seqs, hidden, meta.cu_seqlens[:-1], meta, positions), None, None
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
                self.last, self.last_verify, self.last_prefill = (seqs, hidden, list(range(len(seqs))), None, None), None, None
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
        self.fill_mixed_buffers([(seq, len(seq) - 1, [seq.token_ids[-1]] + d) for seq, d in zip(seqs, drafts)], padded)
        hidden = self.mixed_forward(padded)
        if self.draft is not None:
            self.last, self.last_verify, self.last_prefill = (seqs, hidden, None, None, None), padded, None
        logits = self.model.compute_logits(hidden[:t])
        q, vocab = None, None
        if draft_probs is not None:
            q, vocab = torch.cat([probs[: len(d)] for probs, d in zip(draft_probs, drafts) if d]), self.draft.vocab_ids
        return verify_drafts(self.sampler, logits, [seq.params for seq in seqs], drafts, q, vocab,
                             out=self.verify_out if self.draft is not None else None)

    def mixed_forward(self, padded: int):
        if self.use_cuda_graph:
            graph, hidden = self.mixed_graphs[padded]
            graph.replay()
            return hidden
        return self.mixed_hidden(padded)
