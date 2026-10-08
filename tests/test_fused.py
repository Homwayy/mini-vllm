"""The fused kernels of minivllm/fused.py against the models' reference ops, on random data.

Each kernel must give the reference's bits, except where the order of the fp32 sums in a norm moves a
value across a bf16 rounding boundary. Such elements must be rare, and off by at most one ulp of the
tensor's largest value: right after a cancellation a value is tiny, and one step there is many of its own
ulps. RoPE, the residual adds, silu_mul and v must match bit for bit. The KV pools must hold the reference's k and v at the given slots and be untouched elsewhere.
Finally the fused forward of each model runs against its reference forward on random tokens, as a check
of the wiring only: the rare differences grow over the layers, so the outputs need not match closely.
The errors of the fused forward on real text, against fp32 references, are E0's: tests.test_qwen3_hf and
tests.test_gpt2_nanogpt with --fused.

Run from the repo root: python -m tests.test_fused
"""

import torch
import torch.nn.functional as F

from minivllm import fused
from minivllm.attention import AttnMetadata, store_kv
from minivllm.models import gpt2, qwen3

TOKENS = [1, 3, 64, 1000, 4096]
MAX_MISMATCH = 1e-3  # share of elements allowed to differ
NUM_SLOTS = 8192


def ordered(x):
    """bf16 bits as integers in the order of the values, so a difference counts ulps."""
    i = x.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def compare(name, out, ref, exact=False):
    if out.dtype == torch.float32:
        err = ((out - ref).abs() / (ref.abs() + 1e-3)).max().item()
        ok = err < 1e-4
        print(f"  {name:28s} max relative error {err:.1e}  {'ok' if ok else 'WRONG'}")
        return ok
    d = (ordered(out) - ordered(ref)).abs()
    share, worst = (d > 0).float().mean().item(), d.max().item()
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    # one bf16 ulp of the largest value is at most 2^-7 of it
    ok = worst == 0 if exact else (share <= MAX_MISMATCH and err <= 2**-7)
    print(f"  {name:28s} {share:.2e} of elements differ (at most {worst} ulp; {err:.1e} of the largest value)  "
          f"{'ok' if ok else 'WRONG'}")
    return ok


def pools(shape):
    k = torch.randn(NUM_SLOTS, *shape, device="cuda").to(torch.bfloat16)
    return k, torch.randn_like(k)


def test_gpt2(T):
    ok = True
    N, H, D = 768, 12, 64
    ln = torch.nn.LayerNorm(N).cuda()
    with torch.no_grad():
        ln.weight.copy_(1 + 0.2 * torch.randn(N))
        ln.bias.copy_(0.1 * torch.randn(N))
    x = 3 * torch.randn(T, N, device="cuda")
    branch = torch.randn(T, N, device="cuda").to(torch.bfloat16)
    ref_x = x + branch
    for dtype in (torch.bfloat16, torch.float32):
        xi = x.clone()
        out = fused.add_layer_norm(xi, branch, ln, dtype)
        ok &= compare(f"add_layer_norm {str(dtype)[6:]}", out, ln(ref_x).to(dtype))
        ok &= compare("  residual stream", xi, ref_x)
    ok &= compare("layer_norm (no branch)", fused.add_layer_norm(x.clone(), None, ln, torch.bfloat16), ln(x).to(torch.bfloat16))

    cos, sin = gpt2.rope_cache(D, 10000.0, 1024)
    cos, sin = cos.cuda(), sin.cuda()
    positions = torch.randint(0, 1024, (T,), device="cuda")
    slots = torch.randperm(NUM_SLOTS, device="cuda")[:T]
    qkv = torch.randn(T, 3 * N, device="cuda").to(torch.bfloat16)
    q, k, v = qkv.view(T, 3, H, D).unbind(1)
    c, s = cos[positions].unsqueeze(1), sin[positions].unsqueeze(1)
    rq, rk = gpt2.apply_rope(q, c, s), gpt2.apply_rope(k, c, s)
    kc, vc = pools((H, D))
    rkc, rvc = kc.clone(), vc.clone()
    store_kv(rk, v, rkc, rvc, slots)
    for need_kv in (False, True):
        kc2, vc2 = kc.clone(), vc.clone()
        fq, fk, fv = fused.rope_store(qkv, H, positions, cos, sin, kc2, vc2, slots, need_kv)
        ok &= compare(f"rope_store q (need_kv {need_kv})", fq, rq, exact=True)
        ok &= compare("  k pool", kc2, rkc, exact=True) & compare("  v pool", vc2, rvc, exact=True)
        if need_kv:
            ok &= compare("  k", fk, rk, exact=True) & compare("  v", fv, v, exact=True)
    return ok


def test_qwen3(T):
    ok = True
    N, H, HKV, D, I, eps = 1024, 16, 8, 128, 3072, 1e-6
    norm = qwen3.RMSNorm(N, eps).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.2 * torch.randn(N))
    x = torch.randn(T, N, device="cuda").to(torch.bfloat16)
    branch = torch.randn(T, N, device="cuda").to(torch.bfloat16)
    ref_x = x + branch
    xi = x.clone()
    ok &= compare("add_rms_norm", fused.add_rms_norm(xi, branch, norm), norm(ref_x))
    ok &= compare("  residual stream", xi, ref_x, exact=True)
    ok &= compare("rms_norm (no branch)", fused.add_rms_norm(x.clone(), None, norm), norm(x))

    q_norm, k_norm = qwen3.RMSNorm(D, eps).cuda().to(torch.bfloat16), qwen3.RMSNorm(D, eps).cuda().to(torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(1 + 0.3 * torch.randn(D))
        k_norm.weight.copy_(1 + 0.3 * torch.randn(D))
    cos, sin = qwen3.rope_cache(D, 1e6, 2048, "cuda")
    positions = torch.randint(0, 2048, (T,), device="cuda")
    slots = torch.randperm(NUM_SLOTS, device="cuda")[:T]
    qkv = (2 * torch.randn(T, (H + 2 * HKV) * D, device="cuda")).to(torch.bfloat16)
    q, k, v = qkv.split([H * D, HKV * D, HKV * D], dim=-1)
    c, s = cos[positions].to(torch.bfloat16).unsqueeze(1), sin[positions].to(torch.bfloat16).unsqueeze(1)
    rq = qwen3.apply_rope(q_norm(q.view(T, H, D)), c, s)
    rk = qwen3.apply_rope(k_norm(k.view(T, HKV, D)), c, s)
    rv = v.reshape(T, HKV, D)
    kc, vc = pools((HKV, D))
    rkc, rvc = kc.clone(), vc.clone()
    store_kv(rk, rv, rkc, rvc, slots)
    for need_kv in (False, True):
        kc2, vc2 = kc.clone(), vc.clone()
        fq, fk, fv = fused.qk_norm_rope_store(qkv, H, HKV, q_norm, k_norm, positions, cos.to(torch.bfloat16),
                                              sin.to(torch.bfloat16), kc2, vc2, slots, need_kv)
        ok &= compare(f"qk_norm_rope_store q (need_kv {need_kv})", fq, rq)
        ok &= compare("  k pool", kc2, rkc) & compare("  v pool", vc2, rvc, exact=True)
        if need_kv:
            ok &= compare("  k", fk, rk) & compare("  v", fv, rv, exact=True)

    gate_up = (3 * torch.randn(T, 2 * I, device="cuda")).to(torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    ok &= compare("silu_mul", fused.silu_mul(gate_up), F.silu(gate) * up, exact=True)
    return ok


def test_model(name, model, vocab):
    """Fused against reference forward of a whole model: one packed prefill of random tokens."""
    from minivllm.attention.triton_paged import TritonAttention

    T = 300
    attn = TritonAttention(16, 4096, 8, "cuda")
    for a in model.attn_layers:
        a.attn = attn
        a.k_cache, a.v_cache = pools((model.num_kv_heads, model.head_dim))
        a.k_cache = a.k_cache.view(-1, 16, model.num_kv_heads, model.head_dim)
        a.v_cache = a.v_cache.view(-1, 16, model.num_kv_heads, model.head_dim)
    ids = torch.randint(0, vocab, (T,), device="cuda")
    positions = torch.arange(T, device="cuda")
    out = []
    for f in (False, True):
        model.fused = f
        meta = AttnMetadata(is_prefill=True, slot_mapping=positions.clone(), cu_seqlens=[0, T],
                            block_tables=torch.arange(-(-T // 16), dtype=torch.int32, device="cuda")[None],
                            context_lens=torch.tensor([T], dtype=torch.int32, device="cuda"))
        with torch.inference_mode():
            out.append(model.compute_logits(model(ids, positions, meta)).float())
    model.fused = False
    ref, got = out
    err = (got - ref).abs().max().item()
    same = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    ok = same >= 0.9  # a miswired layer agrees almost nowhere
    print(f"{name} forward, fused vs reference: max logit difference {err:.3f}, same argmax at {same:.1%} of positions  "
          f"{'ok' if ok else 'WRONG'}")
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    for T in TOKENS:
        print(f"T = {T}")
        ok &= test_gpt2(T)
        ok &= test_qwen3(T)
    from minivllm.attention.gather import GatherAttention
    from minivllm.model_runner import MODELS

    ok &= test_model("gpt2", gpt2.load_gpt2(MODELS["gpt2"]["path"], GatherAttention()), 50257)
    torch.cuda.empty_cache()
    ok &= test_model("qwen3", qwen3.load_qwen3(MODELS["qwen3"]["path"], GatherAttention()), 151643)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
