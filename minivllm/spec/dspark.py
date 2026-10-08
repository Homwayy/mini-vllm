"""DSpark-style draft model for the GPT-2 target (E8c): a block of drafts in one forward, plus a Markov head.

A block has one row per draft. Row 0 is the anchor, the last token, whose target features are not known yet; the
other rows are mask rows at the following positions. The rows run through one GPT-2 block, attending causally to
each other and to K/V computed from the target's features at every earlier position (KV injection): its hidden
states at the inputs of blocks feature_layers and its final hidden state, concatenated and projected. Row k gives
base logits U_k for the token at the anchor's position + k + 1, through the target's frozen LM head rows of the
draft vocabulary. Attention inside the block is causal, so a model trained with block_len rows drafts any
shorter block exactly as it would the first rows of a full one.

A low-rank Markov head adds what parallel rows miss, the token just before: the draft at row k is sampled from
softmax(U_k + B[x]), where x is the token before it (the anchor at row 0, then the drafts) and B[x] = W1[x] W2.
Only this cheap step is sequential.

The context K/V of position t depend only on the target's features at t, so the engine writes them inside the
target's own forward (inject), for every row it computes, before acceptance is known. A rejected row lies at or
after the next anchor, and the next block's rows overwrite it before anything reads it.

A confidence head, sigmoid(w . [h_k; W1[x]]), is trained to predict each draft's acceptance probability (for
confidence scheduling, E8d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from minivllm import fused
from minivllm.attention import AttnMetadata, store_kv
from minivllm.models.gpt2 import GPT2Block, GPT2Config, apply_rope

FEATURE_LAYERS = (4, 8)  # with the final hidden state: low, middle and last layers of the 12


class DSparkDraft(nn.Module):
    def __init__(self, cfg: GPT2Config, attn_backend, vocab_ids, block_len=8, rank=64, feature_layers=FEATURE_LAYERS):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.block_len = block_len
        self.feature_layers = tuple(feature_layers)
        self.fc = nn.Linear((len(self.feature_layers) + 1) * d, d)
        self.row_emb = nn.Parameter(0.02 * torch.randn(block_len, d))  # added to row k; row 0 also gets the anchor's embedding
        self.block = GPT2Block(cfg, attn_backend)
        self.ln_out = nn.LayerNorm(d)
        self.markov_in = nn.Embedding(cfg.vocab_size, rank)  # W1, indexed by the previous token
        self.markov_out = nn.Linear(rank, len(vocab_ids), bias=False)  # W2, onto the draft vocabulary
        self.conf = nn.Linear(d + rank, 1)
        nn.init.normal_(self.markov_in.weight, std=0.02)
        nn.init.zeros_(self.markov_out.weight)  # starts as parallel drafting alone
        self.register_buffer("vocab_ids", torch.as_tensor(vocab_ids, dtype=torch.long))
        self.target = ()  # (wte, cos, sin, LM head rows of vocab_ids), shared with the target and frozen
        self.fused = False  # run the block with the fused kernels of minivllm/fused.py

    def attach(self, target):
        rows = target.lm_head_weight[self.vocab_ids.to(target.lm_head_weight.device)]
        self.target = (target.wte, target.cos, target.sin, rows)

    def context(self, features):
        """features: the target's hidden states at the inputs of feature_layers, then its final one ([..., d]
        each). Returns the vectors whose K/V the block reads, [..., d] fp32."""
        x = torch.cat([f.to(self.fc.weight.dtype) for f in features], -1)
        return self.fc(x).float()

    def inject(self, features, positions, meta: AttnMetadata):
        """Writes the context K/V of the target's rows [T] into the draft's pool, at meta.slot_mapping."""
        _, cos, sin, _ = self.target
        attn, ln = self.block.attn, self.block.ln_1
        c = self.context(features)
        dtype = attn.c_attn.weight.dtype
        if self.fused:  # q is computed and dropped
            fused.rope_store(attn.c_attn(fused.add_layer_norm(c, None, ln, dtype)), attn.num_heads, positions, cos, sin,
                             attn.k_cache, attn.v_cache, meta.slot_mapping, False)
            return
        _, k, v = attn.c_attn(ln(c).to(dtype)).view(c.shape[0], 3, attn.num_heads, attn.head_dim).unbind(1)
        k = apply_rope(k, cos[positions].unsqueeze(1), sin[positions].unsqueeze(1))
        store_kv(k, v, attn.k_cache, attn.v_cache, meta.slot_mapping)

    def rows(self, anchors, n):
        """The inputs of n rows per anchor, [len(anchors) * n, d] fp32."""
        x = self.row_emb[:n].float().repeat(anchors.shape[0], 1)
        x[::n] += self.target[0](anchors)
        return x

    def forward(self, anchors, positions, meta: AttnMetadata):
        """anchors [P]; positions [P * n]: row k of block i at anchor i's position + k. The rows' K/V go to the pool,
        and attention reads the context there. Returns h [P * n, d] fp32."""
        _, cos, sin, _ = self.target
        x = self.rows(anchors, positions.shape[0] // anchors.shape[0])
        if self.fused:
            x = self.block.forward_fused(x, positions, cos, sin, meta)
        else:
            x = self.block(x, cos[positions].unsqueeze(1), sin[positions].unsqueeze(1), meta)
        return self.ln_out(x)

    def logits(self, h):
        """Base logits U over the draft vocabulary, [N, len(vocab_ids)]."""
        return F.linear(h.to(self.target[3].dtype), self.target[3])

    def transition(self, prev):
        """The Markov head's B[prev] over the draft vocabulary, [N, len(vocab_ids)]."""
        return self.markov_out(self.markov_in(prev).to(self.markov_out.weight.dtype))

    def confidence(self, h, prev):
        return torch.sigmoid(self.conf(torch.cat([h, self.markov_in(prev)], -1).to(self.conf.weight.dtype)).float()).squeeze(-1)

    def train_forward(self, features, ids, block_mask, flex):
        """Dense forward for training: every position a of the [B, T] windows is an anchor, and row k of its
        block sits at position a + k. Queries are pass-major (k * T + a); keys are the T context positions, then
        the rows in the same order, and block_mask lets row (k, a) see context before a and rows (k' <= k, a).
        Returns h [B, block_len, T, d]."""
        B, T, d = features[0].shape
        n, H = self.block_len, self.n_head
        D = d // H
        wte, cos, sin, _ = self.target
        blk = self.block
        pos = torch.arange(T, device=ids.device)
        _, kc, vc = blk.attn.c_attn(blk.ln_1(self.context(features))).view(B, T, 3, H, D).unbind(2)
        kc = apply_rope(kc, cos[pos].unsqueeze(1), sin[pos].unsqueeze(1))
        x = self.row_emb[:, None, :].float() + torch.zeros(B, n, T, d, device=ids.device)
        x[:, 0] += wte(ids)
        rpos = (pos[None, :] + torch.arange(n, device=ids.device)[:, None]).clamp(max=T - 1)  # rows past the window have no loss
        c, s = cos[rpos].unsqueeze(2), sin[rpos].unsqueeze(2)
        q, kb, vb = blk.attn.c_attn(blk.ln_1(x)).view(B, n, T, 3, H, D).unbind(3)
        q, kb = apply_rope(q, c, s), apply_rope(kb, c, s)
        o = flex(q.reshape(B, n * T, H, D).transpose(1, 2),
                 torch.cat([kc, kb.reshape(B, n * T, H, D)], 1).transpose(1, 2),
                 torch.cat([vc, vb.reshape(B, n * T, H, D)], 1).transpose(1, 2), block_mask=block_mask)
        x = x + blk.attn.c_proj(o.transpose(1, 2).reshape(B, n, T, d))
        x = x + blk.mlp(blk.ln_2(x))
        return self.ln_out(x)


def train_mask_mod(T):
    """block_mask for train_forward: row (k, a) sees context positions before a and rows (k' <= k, a)."""
    def mask_mod(b, h, q, kv):
        k, a = q // T, q % T
        kk, aa = (kv - T) // T, (kv - T) % T
        return ((kv < T) & (kv < a)) | ((kv >= T) & (aa == a) & (kk <= k))
    return mask_mod


@triton.jit(do_not_specialize=["num_seqs"])
def block_inputs_kernel(pos_ptr, cu_ptr, bt_ptr, bt_stride, bt_width, accepted_ptr, temp_ptr, num_seqs, scratch_block,
                        anchor_ptr, ch_pos_ptr, ch_temp_ptr, ch_bt_ptr, ch_bt_stride, ch_bt_width, BT_BLOCK: tl.constexpr):
    """One program per block after a verify step. Sequence i accepted a drafts: its anchor is the target's token
    after them, one position past its row a of the verify step; the block uses the verify step's block table.
    Rows past the sequences are padding: position 0, greedy, scratch block."""
    i = tl.program_id(0)
    cols = tl.arange(0, BT_BLOCK)
    if i < num_seqs:
        a = tl.load(accepted_ptr + 2 * i)
        tl.store(anchor_ptr + i, tl.load(accepted_ptr + 2 * i + 1))
        tl.store(ch_pos_ptr + i, tl.load(pos_ptr + tl.load(cu_ptr + i) + a) + 1)
        tl.store(ch_temp_ptr + i, tl.load(temp_ptr + i))
        bt = tl.load(bt_ptr + i * bt_stride + cols, mask=cols < bt_width, other=scratch_block)
        tl.store(ch_bt_ptr + i * ch_bt_stride + cols, bt, mask=cols < ch_bt_width)
    else:
        tl.store(anchor_ptr + i, 0)
        tl.store(ch_pos_ptr + i, 0)
        tl.store(ch_temp_ptr + i, 0.0)
        tl.store(ch_bt_ptr + i * ch_bt_stride + cols, scratch_block, mask=cols < ch_bt_width)


def load_dspark(path, target, attn_backend, device="cuda", dtype=torch.bfloat16):
    ckpt = torch.load(path, map_location="cpu")
    a = ckpt["args"]
    draft = DSparkDraft(target.cfg, attn_backend, ckpt["vocab_ids"], a["block_len"], a["rank"], ckpt["feature_layers"])
    draft.load_state_dict(ckpt["draft"])
    for m in (draft.fc, draft.block.attn, draft.block.mlp, draft.markov_out, draft.conf):
        m.to(dtype)
    draft.to(device)
    draft.attach(target)
    return draft.eval().requires_grad_(False)
