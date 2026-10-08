"""Helpers shared by the correctness tests: a scattered block layout, packed prefill and batched greedy decode."""

import argparse
import random

import numpy as np
import torch

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.attention.triton_paged import TritonAttention

VAL_BIN = r"A:\GitHub project\nanoGPT\data\fineweb_edu\val.bin"
EOT = 50256  # GPT-2 <|endoftext|>, placed before every document in val.bin


def val_documents(n_tokens=2_000_000):
    """FineWeb-Edu validation documents as GPT-2 token id lists (without the leading <|endoftext|>)."""
    data = np.array(np.memmap(VAL_BIN, dtype=np.uint16, mode="r")[:n_tokens])
    bounds = np.flatnonzero(data == EOT)
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield data[a + 1 : b].tolist()


def attention_from_args(block_size, max_tokens, max_seqs, device):
    """The attention backend named by --attn, and whether --fused asks for the fused kernels (minivllm/fused.py)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", choices=["gather", "flex", "triton", "triton-decode"], default="gather")
    parser.add_argument("--fused", action="store_true", help="run the model's fused forward")
    args = parser.parse_args()
    name = args.attn
    if name == "gather":
        return name, GatherAttention(), args.fused
    if name.startswith("triton"):
        return name, TritonAttention(block_size, max_tokens, max_seqs, device, flex_prefill=name == "triton-decode"), args.fused
    return name, FlexAttention(block_size, max_tokens, max_seqs, device), args.fused


def logit_stats(x, ref):
    x, ref = x.float(), ref.float()
    err = (x - ref).abs()
    choice = x.argmax(-1, keepdim=True)
    # how far below the reference's best logit the token picked from x is (0 when argmaxes agree)
    regret = ref.max(-1).values - ref.gather(-1, choice).squeeze(-1)
    return {
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "argmax_agree": (choice.squeeze(-1) == ref.argmax(-1)).float().mean().item(),
        "max_fp32_regret": regret.max().item(),
    }


class PagedSeqs:
    """Allocates a KV pool and hands out shuffled blocks, so each sequence's KV is scattered through it."""

    def __init__(self, model, max_lens, block_size, device):
        self.block_size = block_size
        nblk = [(n + block_size - 1) // block_size for n in max_lens]
        ids = list(range(sum(nblk)))
        random.Random(0).shuffle(ids)
        self.block_tables = torch.zeros(len(max_lens), max(nblk), dtype=torch.int32)
        for i, n in enumerate(nblk):
            self.block_tables[i, :n] = torch.tensor(ids[:n])
            ids = ids[n:]
        self.block_tables = self.block_tables.to(device)
        shape = (sum(nblk), block_size, model.num_kv_heads, model.head_dim)
        for attn in model.attn_layers:
            attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)

    def slots(self, seq_idx, positions):
        bs = self.block_size
        return self.block_tables[seq_idx, positions // bs].long() * bs + positions % bs


@torch.inference_mode()
def run_ours(model, prompts, new_tokens, block_size, device):
    """Packed prefill of all prompts, then batched greedy decode.

    Returns the prefill hidden states, cu_seqlens, generated tokens per prompt, and
    step_logits [B, new_tokens, V] where step_logits[:, i] are the logits that chose token i.
    """
    lens = [len(p) for p in prompts]
    seqs = PagedSeqs(model, [n + new_tokens for n in lens], block_size, device)

    input_ids = torch.tensor(sum(prompts, []), device=device)
    positions = torch.cat([torch.arange(n, device=device) for n in lens])
    seq_idx = torch.repeat_interleave(torch.arange(len(lens), device=device), torch.tensor(lens, device=device))
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    meta = AttnMetadata(
        is_prefill=True, slot_mapping=seqs.slots(seq_idx, positions), cu_seqlens=cu, block_tables=seqs.block_tables,
        context_lens=torch.tensor(lens, dtype=torch.int32, device=device),
    )
    hidden = model(input_ids, positions, meta)

    last = model.compute_logits(hidden[torch.tensor(cu[1:], device=device) - 1]).float()
    step_logits, tokens = [last], [last.argmax(-1)]
    pos = torch.tensor(lens, device=device)
    rows = torch.arange(len(lens), device=device)
    for _ in range(new_tokens - 1):
        meta = AttnMetadata(
            is_prefill=False, slot_mapping=seqs.slots(rows, pos), block_tables=seqs.block_tables,
            context_lens=(pos + 1).int(), max_context_len=int(pos.max()) + 1,
        )
        logits = model.compute_logits(model(tokens[-1], pos, meta)).float()
        step_logits.append(logits)
        tokens.append(logits.argmax(-1))
        pos = pos + 1
    return hidden, cu, torch.stack(tokens, 1).tolist(), torch.stack(step_logits, 1)


def print_stats(name, s):
    print(
        f"  {name:26s} max {s['max_abs_err']:.4f}  mean {s['mean_abs_err']:.5f}  "
        f"argmax agree {s['argmax_agree']:.4f}  max fp32 regret {s['max_fp32_regret']:.4f}"
    )
