from dataclasses import dataclass

import torch


@dataclass
class AttnMetadata:
    """Per-step attention inputs for the flat layout: all sequences' new tokens are concatenated."""

    is_prefill: bool
    slot_mapping: torch.Tensor  # [T] int64, pool slot that each new token's K/V is written to
    # prefill
    cu_seqlens: list[int] | None = None  # prefix sums of per-sequence new-token counts
    # decode
    block_tables: torch.Tensor | None = None  # [B, max_blocks] int32
    context_lens: torch.Tensor | None = None  # [B] int32, cached tokens including this step's
    max_context_len: int = 0


def store_kv(k, v, k_cache, v_cache, slot_mapping):
    """Write new K/V ([T, Hkv, D]) into the pools ([num_blocks, block_size, Hkv, D])."""
    k_cache.view(-1, *k.shape[1:]).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, *v.shape[1:]).index_copy_(0, slot_mapping, v)
