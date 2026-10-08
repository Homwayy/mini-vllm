"""FlexAttention backend.

Prefill: the packed new tokens of all sequences in one call, with a document-causal block mask built
directly from the sequence boundaries (create_block_mask costs several milliseconds per call).
Decode: reads K/V straight from the paged pool. The block mask lists each sequence's pages from its
block table: all pages but the last are full blocks, and the last page is masked by how full it is.

Prefill is compiled with dynamic shapes, since the packed token count changes every step. Inductor only
uses flex's decoding kernel when the batch size is static, so decode is compiled per batch size; the model
runner pads decode batches to a few bucket sizes. The mask functions read persistent buffers, and each
decode batch size keeps one BlockMask whose tensors are rewritten in place every step, so nothing triggers
a recompile and the decode step can be captured in a CUDA graph.
"""

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from minivllm.attention import AttnMetadata

SPARSE_BLOCK = 128  # block size of the prefill mask, and the query block size for decode


class FlexAttention:
    reads_new_kv = True  # prefill attends over this step's k and v as tensors

    def __init__(self, block_size, max_num_batched_tokens, max_num_seqs, device):
        self.block_size = block_size
        self.decode_masks = {}  # (batch size, block table width) -> (BlockMask, its index tensors)
        padded = -(-max_num_batched_tokens // SPARSE_BLOCK) * SPARSE_BLOCK
        self.doc = torch.full((padded,), -1, dtype=torch.int32, device=device)  # sequence of each packed token
        self.last_len = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)  # tokens in each last page
        doc, last_len, P = self.doc, self.last_len, block_size

        def prefill_mask(b, h, q_idx, kv_idx):
            return (doc[q_idx] == doc[kv_idx]) & (q_idx >= kv_idx)

        def decode_mask(b, h, q_idx, kv_idx):
            return kv_idx % P < last_len[b]

        self.prefill_mask, self.decode_mask = prefill_mask, decode_mask
        # Past the recompile limit, dynamo would silently run flex_attention eagerly. That path ignores the
        # block sparsity (decode_mask relies on it) and materializes the full score matrix, so fail instead.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        self.flex_prefill = torch.compile(flex_attention, dynamic=True)
        self.flex_decode = torch.compile(flex_attention, dynamic=False)

    def __call__(self, q, k, v, k_cache, v_cache, meta: AttnMetadata):
        if meta.is_prefill:
            if meta.backend_state is None:
                meta.backend_state = self.prefill_block_mask(meta.cu_seqlens, q.device)
            q4, k4, v4 = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
            # below 128 tokens Inductor would pick the decoding kernel, which fails on these strided views
            out = self.flex_prefill(
                q4, k4, v4, block_mask=meta.backend_state, enable_gqa=True,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
            )
            return out[0].transpose(0, 1)
        if meta.backend_state is None:
            meta.backend_state = self.decode_block_mask(meta.block_tables, meta.context_lens, k_cache.shape[0])
        # pools [num_blocks, block_size, Hkv, D] seen as one long sequence [1, Hkv, num_slots, D]
        kc, vc = (c.view(1, -1, *c.shape[2:]).transpose(1, 2) for c in (k_cache, v_cache))
        return self.flex_decode(q.unsqueeze(2), kc, vc, block_mask=meta.backend_state, enable_gqa=True).squeeze(2)

    def prefill_block_mask(self, cu_seqlens, device):
        cu = np.asarray(cu_seqlens)
        T = int(cu[-1])
        self.doc.fill_(-1)
        self.doc[:T].copy_(torch.from_numpy(np.repeat(np.arange(len(cu) - 1, dtype=np.int32), np.diff(cu))))
        nb = -(-T // SPARSE_BLOCK)
        first = np.arange(nb) * SPARSE_BLOCK
        last = np.minimum(first + SPARSE_BLOCK, T) - 1
        d_first = np.searchsorted(cu, first, side="right") - 1  # sequence of each block's first token
        d_last = np.searchsorted(cu, last, side="right") - 1
        i, j = np.arange(nb)[:, None], np.arange(nb)[None, :]
        below = j < i
        # (q block i, kv block j) with j < i: every pair is visible if both blocks lie in one sequence,
        # and some pair is visible if kv block j ends in the sequence that q block i starts in
        full = below & (d_first[j] == d_last[i])
        partial = (j == i) | (below & (d_last[j] == d_first[i]) & ~full)

        def to_blocks(m):
            num = torch.from_numpy(m.sum(1).astype(np.int32)).view(1, 1, nb)
            idx = torch.from_numpy(np.argsort(~m, axis=1, kind="stable").astype(np.int32)).view(1, 1, nb, nb)
            return num.to(device), idx.to(device)

        kv_num, kv_idx = to_blocks(partial)
        full_num, full_idx = to_blocks(full)
        return BlockMask.from_kv_blocks(
            kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=SPARSE_BLOCK, mask_mod=self.prefill_mask,
            seq_lengths=(T, T), compute_q_blocks=False,
        )

    def decode_block_mask(self, block_tables, context_lens, num_blocks):
        # every row must have at least one token of context
        P = self.block_size
        B, width = block_tables.shape
        if (B, width) not in self.decode_masks:
            dev = block_tables.device
            kv_num = torch.ones(B, 1, 1, dtype=torch.int32, device=dev)  # the last page is the one partial block
            kv_idx = torch.zeros(B, 1, 1, width, dtype=torch.int32, device=dev)
            full_num = torch.zeros(B, 1, 1, dtype=torch.int32, device=dev)
            full_idx = torch.zeros(B, 1, 1, width, dtype=torch.int32, device=dev)
            block_mask = BlockMask.from_kv_blocks(
                kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=(SPARSE_BLOCK, P),
                mask_mod=self.decode_mask, seq_lengths=(1, num_blocks * P), compute_q_blocks=False,
            )
            self.decode_masks[B, width] = (block_mask, kv_idx, full_num, full_idx)
        block_mask, kv_idx, full_num, full_idx = self.decode_masks[B, width]
        npages = (context_lens + P - 1) // P
        self.last_len[:B] = context_lens - (npages - 1) * P
        full_num.view(B).copy_(npages - 1)
        full_idx.view(B, width).copy_(block_tables)
        kv_idx[:, 0, 0, 0] = block_tables.gather(1, (npages - 1).long().unsqueeze(1)).squeeze(1)
        return block_mask
