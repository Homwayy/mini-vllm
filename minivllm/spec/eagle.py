"""EAGLE-style draft model for the GPT-2 target (E8b).

The draft works on the target's features. At position t it takes the target's final hidden state f_t (after
the last LayerNorm, the input of the LM head) and the embedding of the next token x_{t+1}, projects the pair
to the model width and runs one GPT-2 block, whose attention reads the draft's own KV cache. The output g_t
predicts f_{t+1}; its logits use the target's frozen embedding rows of the most frequent tokens only, so a
drafting step does not read the whole 74 MB LM head. Each further draft token feeds the prediction back:
(g_t, d_1) at position t + 1 gives g_{t+1}, and so on.

The draft's KV cache has the target's layout and uses the target's block tables: draft position t sits in
the same slot as target position t, in one more pool.

After a verify step, the inputs of the draft's catch-up and of its drafting chain are built on the GPU from
the verify step's buffers and its acceptance (draft_inputs_kernel), so no buffer is filled on the CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from minivllm.attention import AttnMetadata
from minivllm.models.gpt2 import GPT2Block, GPT2Config, apply_rope


class EagleDraft(nn.Module):
    def __init__(self, cfg: GPT2Config, attn_backend, vocab_ids):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.fc = nn.Linear(2 * d, d)
        self.block = GPT2Block(cfg, attn_backend)
        self.register_buffer("vocab_ids", torch.as_tensor(vocab_ids, dtype=torch.long))
        self.target = ()  # (wte, cos, sin, LM head rows of vocab_ids), shared with the target and frozen
        self.fused = False  # run the block with the fused kernels of minivllm/fused.py

    def attach(self, target):
        rows = target.lm_head_weight[self.vocab_ids.to(target.lm_head_weight.device)]
        self.target = (target.wte, target.cos, target.sin, rows)

    @property
    def attn_layers(self):
        """The attention modules that need a KV pool in the engine."""
        return [self.block.attn]

    def init_as_copy(self):
        """Starts near g_t = f_t: the feature half of fc is the identity, and the block adds only the biases of
        its two output projections, whose weights are zero."""
        d = self.fc.out_features
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, d:] = torch.eye(d)
            self.fc.bias.zero_()
            self.block.attn.c_proj.weight.zero_()
            self.block.mlp.c_proj.weight.zero_()

    def forward(self, features, tokens, positions, meta: AttnMetadata):
        """features [T, d] fp32 and tokens [T]: (f_t, x_{t+1}) at each draft position t. Returns g [T, d] fp32."""
        wte, cos, sin, _ = self.target
        x = torch.cat([wte(tokens), features], dim=-1)
        x = self.fc(x.to(self.fc.weight.dtype)).float()
        if self.fused:
            return self.block.forward_fused(x, positions, cos, sin, meta)
        return self.block(x, cos[positions].unsqueeze(1), sin[positions].unsqueeze(1), meta)

    def logits(self, g):
        """Logits over the draft vocabulary, [T, len(vocab_ids)]."""
        return F.linear(g.to(self.target[3].dtype), self.target[3])

    def train_forward(self, f, x, passes=1, noise=0.0):
        """Dense forward for training, with the attention every drafting step sees.

        f [B, T, d]: target features at positions 0..T-1; x [B, T]: the token after each. Pass 0 takes the
        target's features. Pass j at position s takes pass j - 1's prediction from position s - 1 and attends
        to pass-0 keys up to s - j and to pass m's key at s - (j - m), like the j-th step of a draft chain
        that started after position s - j. Returns each pass's predictions [B, T, d].
        """
        B, T, d = f.shape
        H = self.n_head
        wte, cos, sin, _ = self.target
        e = wte(x)
        c, s = cos[:T].unsqueeze(1), sin[:T].unsqueeze(1)
        feat = f + noise * (2 * torch.rand_like(f) - 1) if noise else f
        blk, ks, vs, outs = self.block, [], [], []
        for j in range(passes):
            h = self.fc(torch.cat([e, feat], -1)).float()
            q, k, v = blk.attn.c_attn(blk.ln_1(h)).view(B, T, 3, H, d // H).unbind(2)
            q, k = apply_rope(q, c, s), apply_rope(k, c, s)
            ks.append(k)
            vs.append(v)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), torch.cat(ks, 1).transpose(1, 2), torch.cat(vs, 1).transpose(1, 2),
                attn_mask=chain_mask(j, T, f.device),
            )
            h = h + blk.attn.c_proj(o.transpose(1, 2).reshape(B, T, d))
            h = h + blk.mlp(blk.ln_2(h))
            outs.append(h)
            feat = torch.cat([f[:, :1], h[:, :-1]], 1)
        return outs


def chain_mask(j, T, device):
    """Keys visible to pass j: pass 0 up to position s - j, and pass m (1 <= m <= j) at s - (j - m)."""
    s = torch.arange(T, device=device)[:, None]
    t = torch.arange(T, device=device)[None, :]
    return torch.cat([t <= s - j] + [t == s - (j - m) for m in range(1, j + 1)], 1)


@triton.jit(do_not_specialize=["num_seqs"])
def draft_inputs_kernel(ids_ptr, pos_ptr, slots_ptr, cu_ptr, bt_ptr, bt_stride, bt_width, accepted_ptr, temp_ptr,
                        num_seqs, scratch_block, block_size, rows_ptr, ch_pos_ptr, ch_temp_ptr, ch_bt_ptr, ch_bt_stride,
                        ch_bt_width, MAX_ROWS: tl.constexpr, BT_BLOCK: tl.constexpr):
    """One program per row of the drafting chain. For sequence i of the verify step (rows start .. start +
    count - 1, of which the first a + 1 are verified): the catch-up gives verified row j the token after it
    (the next row's token, or the target's token at row a) and sends the other rows' K/V to the scratch
    block; the chain starts from the catch-up's row a, one position further, with the verify step's block
    table. Rows past the sequences are padding: position 0, greedy, scratch block."""
    i = tl.program_id(0)
    cols = tl.arange(0, BT_BLOCK)
    if i < num_seqs:
        start = tl.load(cu_ptr + i).to(tl.int64)
        count = tl.load(cu_ptr + i + 1) - tl.load(cu_ptr + i)
        a = tl.load(accepted_ptr + 2 * i)  # accepted drafts
        tok = tl.load(accepted_ptr + 2 * i + 1)  # the target's token after them
        j = tl.arange(0, MAX_ROWS)
        nxt = tl.load(ids_ptr + start + j + 1, mask=j < a, other=0)
        tl.store(ids_ptr + start + j, tl.where(j < a, nxt, tok), mask=j <= a)
        tl.store(slots_ptr + start + j, scratch_block * block_size, mask=(j > a) & (j < count))
        row = start + a
        tl.store(rows_ptr + i, row)
        tl.store(ch_pos_ptr + i, tl.load(pos_ptr + row) + 1)
        tl.store(ch_temp_ptr + i, tl.load(temp_ptr + i))
        bt = tl.load(bt_ptr + i * bt_stride + cols, mask=cols < bt_width, other=scratch_block)
        tl.store(ch_bt_ptr + i * ch_bt_stride + cols, bt, mask=cols < ch_bt_width)
    else:
        tl.store(rows_ptr + i, 0)
        tl.store(ch_pos_ptr + i, 0)
        tl.store(ch_temp_ptr + i, 0.0)
        tl.store(ch_bt_ptr + i * ch_bt_stride + cols, scratch_block, mask=cols < ch_bt_width)


def load_eagle(path, target, attn_backend, device="cuda", dtype=torch.bfloat16):
    ckpt = torch.load(path, map_location="cpu")
    draft = EagleDraft(target.cfg, attn_backend, ckpt["vocab_ids"])
    draft.load_state_dict(ckpt["draft"])
    draft.fc.to(dtype)
    draft.block.attn.to(dtype)
    draft.block.mlp.to(dtype)
    draft.to(device)
    draft.attach(target)
    return draft.eval().requires_grad_(False)
