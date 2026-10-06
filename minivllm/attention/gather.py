"""Baseline attention: copy each sequence's KV out of the pool into a padded tensor, then call SDPA."""

import torch
import torch.nn.functional as F

from minivllm.attention import AttnMetadata


class GatherAttention:
    def __call__(self, q, k, v, k_cache, v_cache, meta: AttnMetadata):
        # q [T, Hq, D]; k, v [T, Hkv, D] are this step's new tokens, already stored in the pools
        if meta.is_prefill:
            return self.prefill(q, k, v, meta)
        return self.decode(q, k_cache, v_cache, meta)

    @staticmethod
    def prefill(q, k, v, meta):
        # no cached prefix yet: each sequence attends causally over its own new tokens
        group = q.shape[1] // k.shape[1]
        out = torch.empty_like(q)
        cu = meta.cu_seqlens
        for i in range(len(cu) - 1):
            a, b = cu[i], cu[i + 1]
            qi = q[a:b].transpose(0, 1).unsqueeze(0)
            ki = k[a:b].transpose(0, 1).repeat_interleave(group, 0).unsqueeze(0)
            vi = v[a:b].transpose(0, 1).repeat_interleave(group, 0).unsqueeze(0)
            out[a:b] = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)[0].transpose(0, 1)
        return out

    @staticmethod
    def decode(q, k_cache, v_cache, meta):
        B, Hq, D = q.shape
        block_size, Hkv = k_cache.shape[1], k_cache.shape[2]
        L = meta.max_context_len
        pos = torch.arange(L, device=q.device)
        slots = meta.block_tables.long()[:, pos // block_size] * block_size + pos % block_size  # [B, L]
        k = k_cache.view(-1, Hkv, D)[slots].transpose(1, 2)  # [B, Hkv, L, D]
        v = v_cache.view(-1, Hkv, D)[slots].transpose(1, 2)
        mask = (pos[None, :] < meta.context_lens[:, None]).view(B, 1, 1, L)
        # GQA without copying KV: the query heads sharing a KV head become that head's query rows
        qg = q.view(B, Hkv, Hq // Hkv, D)
        out = F.scaled_dot_product_attention(qg, k, v, attn_mask=mask)
        return out.reshape(B, Hq, D)
